#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image path(s) to OCR through vLLM.")
    parser.add_argument(
        "--model-config",
        default=str(Path(__file__).resolve().parent / "model-config"),
        help="Path to the tiny vLLM wrapper model config.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for writing the OCR JSON payload. Useful because vLLM logs may share stdout.",
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model_config,
        runner="pooling",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        load_format="dummy",
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        max_model_len=4096,
    )
    requests = [str(Path(image).resolve()) for image in args.images]
    outputs = llm.encode(
        {"data": requests[0] if len(requests) == 1 else requests},
        pooling_task="plugin",
        use_tqdm=False,
    )
    payload = json.dumps(outputs[0].outputs, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
