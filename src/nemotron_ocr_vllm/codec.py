"""Token and tensor codecs shared by the vLLM model and IO processor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

TOKEN_OFFSET = 2
MAX_OUTPUT_BYTES = 1024 * 1024


def path_to_token_ids(path: str | Path) -> list[int]:
    payload = str(Path(path)).encode("utf-8")
    return [1, *(byte + TOKEN_OFFSET for byte in payload)]


def _decode_token_ids(ids: list[int]) -> str:
    payload = bytes((int(token) - TOKEN_OFFSET) % 256 for token in ids if token > 1)
    return payload.decode("utf-8", errors="ignore")


def token_ids_to_paths(
    token_ids: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> list[str]:
    ids_tensor = token_ids.detach().to("cpu", dtype=torch.int64)
    if ids_tensor.ndim == 2:
        return [_decode_token_ids(row.tolist()) for row in ids_tensor]

    ids = ids_tensor.flatten().tolist()
    if not ids:
        return []

    if positions is None:
        return [_decode_token_ids(ids)]

    pos = positions.detach().to("cpu", dtype=torch.int64).flatten().tolist()
    if len(pos) != len(ids):
        return [_decode_token_ids(ids)]

    chunks: list[list[int]] = []
    start = 0
    for index, value in enumerate(pos):
        if index > 0 and value == 0:
            chunks.append(ids[start:index])
            start = index
    chunks.append(ids[start:])
    return [_decode_token_ids(chunk) for chunk in chunks if chunk]


def token_ids_to_path(token_ids: torch.Tensor) -> str:
    paths = token_ids_to_paths(token_ids)
    return paths[0] if paths else ""


def json_to_tensor(payload: Any, *, device: torch.device) -> torch.Tensor:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError(
            f"OCR payload is {len(raw)} bytes; max supported is {MAX_OUTPUT_BYTES}."
        )
    prefix = len(raw).to_bytes(4, "little")
    data = torch.tensor(list(prefix + raw), dtype=torch.uint8, device=device)
    return data.unsqueeze(0)


def tensor_to_json(data: torch.Tensor) -> Any:
    flat = data.detach().to("cpu", dtype=torch.uint8).flatten().tolist()
    if len(flat) < 4:
        raise ValueError("vLLM OCR output tensor is too short to contain a length.")
    size = int.from_bytes(bytes(flat[:4]), "little")
    raw = bytes(flat[4 : 4 + size])
    return json.loads(raw.decode("utf-8"))
