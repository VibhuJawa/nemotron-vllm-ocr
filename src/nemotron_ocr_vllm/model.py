from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces import IsAttentionFree
from vllm.model_executor.models.interfaces_base import attn_type
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .codec import json_to_tensor, token_ids_to_paths


class NemotronOCRPayloadPooler(Pooler):
    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        num_reqs = len(pooling_metadata.prompt_lens)
        if hidden_states.dtype == torch.uint8 and hidden_states.ndim == 2:
            if hidden_states.shape[0] >= num_reqs:
                return [hidden_states[index] for index in range(num_reqs)]
            return [hidden_states[0] for _ in range(num_reqs)]
        return [
            torch.zeros(1, dtype=torch.uint8, device=hidden_states.device)
            for _ in range(num_reqs)
        ]


@attn_type("attention_free")
class NemotronOCRVllmForPooling(nn.Module, IsAttentionFree):
    is_pooling_model = True
    is_attention_free = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.pooler = NemotronOCRPayloadPooler()
        self._ocr = None

        hf_config = vllm_config.model_config.hf_config
        self.merge_level = getattr(hf_config, "nemotron_ocr_merge_level", "paragraph")
        self.lang = getattr(hf_config, "nemotron_ocr_lang", None)
        self.model_dir = getattr(hf_config, "nemotron_ocr_model_dir", None)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.empty((input_ids.shape[0], 0), device=input_ids.device)

    def _load_ocr(self):
        if self._ocr is not None:
            return self._ocr

        source_dir = os.environ.get("NEMOTRON_OCR_SOURCE")
        if source_dir:
            import sys

            src_path = str(Path(source_dir))
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

        kwargs: dict[str, Any] = {}
        if self.model_dir:
            kwargs["model_dir"] = self.model_dir
        elif self.lang:
            kwargs["lang"] = self.lang
        self._ocr = NemotronOCRV2(**kwargs)
        return self._ocr

    def _dummy_payload(self, image_path: str, mode: str) -> dict[str, Any]:
        if mode == "profile":
            return {
                "backend": "vllm",
                "mode": "profile",
                "image_path": image_path,
                "regions": [],
            }
        return {
            "backend": "vllm",
            "mode": mode,
            "image_path": image_path,
            "regions": [
                {
                    "text": f"dummy OCR for {Path(image_path).name}",
                    "confidence": 1.0,
                    "left": 0.0,
                    "upper": 0.0,
                    "right": 1.0,
                    "lower": 1.0,
                }
            ],
        }

    def _run_ocr_batch(self, image_paths: list[str]) -> list[dict[str, Any]]:
        if not image_paths:
            return []

        if os.environ.get("NEMOTRON_OCR_VLLM_DUMMY"):
            mode = "dummy"
            return [self._dummy_payload(path, mode) for path in image_paths]

        if all(not path for path in image_paths):
            return [self._dummy_payload(path, "profile") for path in image_paths]

        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                "Nemotron OCR image path does not exist: " + ", ".join(missing)
            )

        ocr = self._load_ocr()
        raw_predictions = ocr(image_paths, merge_level=self.merge_level)
        if len(image_paths) == 1 and raw_predictions and isinstance(raw_predictions[0], dict):
            predictions_by_image = [raw_predictions]
        else:
            predictions_by_image = raw_predictions
        return [
            self._payload_from_predictions(path, predictions)
            for path, predictions in zip(image_paths, predictions_by_image, strict=True)
        ]

    def _payload_from_predictions(
        self,
        image_path: str,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "backend": "vllm",
            "mode": "nemotron-ocr-v2",
            "image_path": image_path,
            "merge_level": self.merge_level,
            "regions": predictions,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("Nemotron OCR vLLM wrapper requires token IDs.")
        image_paths = token_ids_to_paths(input_ids, positions)
        payloads = self._run_ocr_batch(image_paths)
        encoded_rows = [
            json_to_tensor(payload, device=input_ids.device).squeeze(0)
            for payload in payloads
        ]
        if not encoded_rows:
            raise ValueError("Nemotron OCR vLLM wrapper received no image paths.")
        max_len = max(row.shape[0] for row in encoded_rows)
        output = torch.zeros(
            (len(encoded_rows), max_len),
            dtype=torch.uint8,
            device=input_ids.device,
        )
        for index, row in enumerate(encoded_rows):
            output[index, : row.shape[0]] = row
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return set()
