#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Nemotron OCR direct vs vLLM wrapper.")
    parser.add_argument("--backend", choices=["direct", "vllm", "both"], default="both")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--merge-level", default="paragraph")
    parser.add_argument("--model-config", default=str(REPO_ROOT / "model-config"))
    parser.add_argument("--lang", default="multi")
    parser.add_argument("--direct-model-dir")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output")
    return parser.parse_args()


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def environment() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        data["torch"] = torch.__version__
        data["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            data["gpu"] = torch.cuda.get_device_name(0)
            data["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        data["torch_error"] = repr(exc)
    try:
        import vllm

        data["vllm"] = vllm.__version__
    except Exception as exc:
        data["vllm_error"] = repr(exc)
    return data


def summarize_latencies(latencies: list[float], image_count: int) -> dict[str, float]:
    mean = statistics.fmean(latencies)
    return {
        "latency_mean_seconds": mean,
        "latency_p50_seconds": statistics.median(latencies),
        "latency_min_seconds": min(latencies),
        "latency_max_seconds": max(latencies),
        "throughput_images_per_second": image_count / mean,
    }


def regions_by_image(raw: Any, image_count: int, backend: str) -> list[list[dict[str, Any]]]:
    if backend == "vllm":
        payloads = raw if isinstance(raw, list) else [raw]
        return [payload["regions"] for payload in payloads]

    if image_count == 1:
        return [raw]
    return raw


def text_signature(regions: list[list[dict[str, Any]]]) -> list[str]:
    return [
        " ".join(str(region.get("text", "")).strip() for region in image_regions).strip()
        for image_regions in regions
    ]


def timed_iterations(call, warmup: int, iterations: int) -> tuple[list[float], Any]:
    last_output = None
    for _ in range(warmup):
        last_output = call()
    latencies: list[float] = []
    for _ in range(iterations):
        sync_cuda()
        start = time.perf_counter()
        last_output = call()
        sync_cuda()
        latencies.append(time.perf_counter() - start)
    return latencies, last_output


def benchmark_direct(args: argparse.Namespace) -> dict[str, Any]:
    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

    images = [str(Path(image).resolve()) for image in args.images]
    kwargs: dict[str, Any] = {}
    if args.direct_model_dir:
        kwargs["model_dir"] = args.direct_model_dir
    elif args.lang:
        kwargs["lang"] = args.lang

    start = time.perf_counter()
    ocr = NemotronOCRV2(**kwargs)
    sync_cuda()
    init_seconds = time.perf_counter() - start

    def call():
        image_input = images[0] if len(images) == 1 else images
        return ocr(image_input, merge_level=args.merge_level)

    latencies, raw_output = timed_iterations(call, args.warmup, args.iterations)
    regions = regions_by_image(raw_output, len(images), "direct")
    return {
        "backend": "direct",
        "image_count": len(images),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "init_seconds": init_seconds,
        "latencies_seconds": latencies,
        **summarize_latencies(latencies, len(images)),
        "region_counts": [len(item) for item in regions],
        "text_signature": text_signature(regions),
        "sample_regions": regions[0] if regions else [],
        "environment": environment(),
    }


def benchmark_vllm(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM

    images = [str(Path(image).resolve()) for image in args.images]
    start = time.perf_counter()
    llm = LLM(
        model=args.model_config,
        runner="pooling",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        load_format="dummy",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sync_cuda()
    init_seconds = time.perf_counter() - start

    def call():
        data = images[0] if len(images) == 1 else images
        outputs = llm.encode({"data": data}, pooling_task="plugin", use_tqdm=False)
        return outputs[0].outputs

    latencies, raw_output = timed_iterations(call, args.warmup, args.iterations)
    regions = regions_by_image(raw_output, len(images), "vllm")
    return {
        "backend": "vllm",
        "image_count": len(images),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "init_seconds": init_seconds,
        "latencies_seconds": latencies,
        **summarize_latencies(latencies, len(images)),
        "region_counts": [len(item) for item in regions],
        "text_signature": text_signature(regions),
        "sample_regions": regions[0] if regions else [],
        "environment": environment(),
    }


def run_backend_subprocess(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / f"{backend}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--merge-level",
            args.merge_level,
            "--model-config",
            args.model_config,
            "--lang",
            args.lang,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--output",
            str(output),
        ]
        if args.direct_model_dir:
            cmd.extend(["--direct-model-dir", args.direct_model_dir])
        cmd.append("--images")
        cmd.extend(args.images)
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            print(result.stdout, file=sys.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
        return json.loads(output.read_text())


def compare(direct: dict[str, Any], vllm: dict[str, Any]) -> dict[str, Any]:
    direct_latency = direct["latency_mean_seconds"]
    vllm_latency = vllm["latency_mean_seconds"]
    return {
        "texts_match": direct["text_signature"] == vllm["text_signature"],
        "region_counts_match": direct["region_counts"] == vllm["region_counts"],
        "vllm_latency_over_direct": vllm_latency / direct_latency,
        "vllm_init_over_direct": vllm["init_seconds"] / direct["init_seconds"],
    }


def write_or_print(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


def main() -> None:
    args = parse_args()
    if args.backend == "direct":
        write_or_print(benchmark_direct(args), args.output)
        return
    if args.backend == "vllm":
        write_or_print(benchmark_vllm(args), args.output)
        return

    direct = run_backend_subprocess(args, "direct")
    vllm = run_backend_subprocess(args, "vllm")
    payload = {
        "benchmark": "nemotron-ocr-v2-direct-vs-vllm",
        "images": [str(Path(image).resolve()) for image in args.images],
        "direct": direct,
        "vllm": vllm,
        "comparison": compare(direct, vllm),
    }
    write_or_print(payload, args.output)


if __name__ == "__main__":
    main()
