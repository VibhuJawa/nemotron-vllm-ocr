from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vllm import PoolingParams
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor

from .codec import path_to_token_ids, tensor_to_json


class NemotronOCRV2IOProcessor(IOProcessor[str | list[str], Any]):
    def parse_data(self, data: object) -> str | list[str]:
        if isinstance(data, (str, Path)):
            return self._validate_path(data)
        if isinstance(data, list) and all(isinstance(item, (str, Path)) for item in data):
            return [self._validate_path(item) for item in data]
        raise TypeError("Nemotron OCR vLLM input data must be an image path or list of image paths.")

    def _validate_path(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve()
        if os.environ.get("NEMOTRON_OCR_VLLM_DUMMY"):
            return str(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"Nemotron OCR image path does not exist: {resolved}")
        return str(resolved)

    def merge_pooling_params(self, params: PoolingParams | None = None) -> PoolingParams:
        merged = params or PoolingParams()
        merged.task = "plugin"
        return merged

    def pre_process(self, prompt: str | list[str], request_id: str | None = None, **kwargs):
        prompts = prompt if isinstance(prompt, list) else [prompt]
        processed = [
            {"prompt_token_ids": path_to_token_ids(path), "prompt": path}
            for path in prompts
        ]
        return processed[0] if len(processed) == 1 else processed

    def post_process(
        self,
        model_output: list[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> Any:
        payloads = []
        for output_item in model_output:
            output = output_item.outputs
            if not isinstance(output, PoolingOutput):
                raise TypeError(f"Expected PoolingOutput, got {type(output).__name__}.")
            payloads.append(tensor_to_json(output.data))
        return payloads[0] if len(payloads) == 1 else payloads


def get_io_processor_class() -> str:
    return "nemotron_ocr_vllm.io_processor.NemotronOCRV2IOProcessor"
