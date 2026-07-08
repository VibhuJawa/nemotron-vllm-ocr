from __future__ import annotations


def register() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "NemotronOCRVllmForPooling",
        "nemotron_ocr_vllm.model:NemotronOCRVllmForPooling",
    )

