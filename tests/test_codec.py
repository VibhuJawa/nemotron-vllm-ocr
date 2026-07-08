from __future__ import annotations

import torch
import pytest

from nemotron_ocr_vllm.codec import (
    json_to_tensor,
    path_to_token_ids,
    tensor_to_json,
    token_ids_to_path,
    token_ids_to_paths,
)
from nemotron_ocr_vllm.io_processor import NemotronOCRV2IOProcessor


class _DummyIOProcessor(NemotronOCRV2IOProcessor):
    def __init__(self) -> None:
        pass


def test_path_token_roundtrip_single() -> None:
    path = "/tmp/nemotron sample.png"
    assert token_ids_to_path(torch.tensor(path_to_token_ids(path))) == path


def test_path_token_roundtrip_packed_batch() -> None:
    paths = ["/tmp/one.png", "/tmp/two.png"]
    ids = path_to_token_ids(paths[0]) + path_to_token_ids(paths[1])
    positions = list(range(len(path_to_token_ids(paths[0])))) + list(
        range(len(path_to_token_ids(paths[1])))
    )
    assert token_ids_to_paths(torch.tensor(ids), torch.tensor(positions)) == paths


def test_json_tensor_roundtrip() -> None:
    payload = {"backend": "vllm", "regions": [{"text": "Invoice total: $42.19"}]}
    tensor = json_to_tensor(payload, device=torch.device("cpu"))
    assert tensor_to_json(tensor) == payload


def test_io_processor_rejects_missing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMOTRON_OCR_VLLM_DUMMY", raising=False)
    processor = _DummyIOProcessor()
    with pytest.raises(FileNotFoundError):
        processor.parse_data("/tmp/definitely-missing-nemotron-ocr-image.png")
