#!/usr/bin/env python3
"""Run reproducible multi-replica native-vLLM Nemotron OCR sweeps.

The harness deliberately owns a unique CUDA MPS control directory for every
configuration.  It never sends commands to the default/global MPS control
socket, and it refuses to start on a busy target GPU unless explicitly told
otherwise.  Work dispatch remains in ``benchmark_multi_endpoint_pooling.py``
so requests flow through vLLM's native pooling API and continuous scheduler.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO


SCHEMA_VERSION = 1
JPEG_DATA_URI_PREFIX = b"data:image/jpeg;base64,"
RESERVED_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "NEMOTRON_OCR_SOURCE",
    "PYTHONPATH",
}
MPS_ENV_KEYS = {
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
}
ENV_CAPTURE_EXACT = {"PATH", "LD_LIBRARY_PATH", "CPATH", "LIBRARY_PATH"}
ENV_CAPTURE_PREFIXES = (
    "CUDA_",
    "CUDNN_",
    "HF_",
    "MKL_",
    "NCCL_",
    "NEMOTRON_OCR_",
    "OMP_",
    "PYTORCH_",
    "TOKENIZERS_",
    "TORCH_",
    "TRITON_",
    "VLLM_",
)
COMMON_KEYS = {
    "gpu",
    "python",
    "vllm_cli",
    "vllm_root",
    "model",
    "model_source",
    "dataset",
    "benchmark_script",
    "output_root",
    "mps_root",
    "host",
    "base_port",
    "ready_timeout_s",
    "health_poll_interval_s",
    "health_settle_s",
    "shutdown_timeout_s",
    "benchmark_timeout_s",
    "cuda_home",
    "hf_home",
    "hf_hub_cache",
    "torchinductor_cache_dir",
    "triton_cache_dir",
    "omp_num_threads",
    "expected_dataset_sha256",
    "expected_model_extension_sha256",
    "expected_vllm_commit",
    "expected_model_commit",
    "require_clean_repositories",
    "env",
    "server",
    "benchmark",
}
SERVER_KEYS = {
    "replicas",
    "max_num_seqs",
    "gpu_memory_utilization",
    "renderer_num_workers",
    "detector_max_batch_size",
    "recognizer_chunk_size",
    "relational_chunk_size",
    "infer_length",
    "mps_enabled",
    "mps_active_thread_percentage",
    "mm_processor_cache_gb",
    "enforce_eager",
    "async_scheduling",
    "disable_access_log",
    "disable_log_stats",
    "hf_overrides",
    "extra_args",
}
BENCHMARK_KEYS = {
    "num_prompts",
    "replay_count",
    "concurrency_per_endpoint",
    "warmups_per_endpoint",
    "trace_interval_ms",
    "extra_args",
}
SERVER_OWNED_FLAGS = {
    "--runner",
    "--skip-tokenizer-init",
    "--io-processor-plugin",
    "--max-num-seqs",
    "--gpu-memory-utilization",
    "--mm-processor-cache-gb",
    "--renderer-num-workers",
    "--host",
    "--port",
    "--hf-overrides",
    "--enforce-eager",
    "--no-enforce-eager",
    "--async-scheduling",
    "--no-async-scheduling",
    "--disable-uvicorn-access-log",
    "--enable-uvicorn-access-log",
    "--disable-log-stats",
    "--enable-log-stats",
}
BENCHMARK_OWNED_FLAGS = {
    "--endpoint",
    "--dataset",
    "--output-dir",
    "--model",
    "--num-prompts",
    "--replay-count",
    "--infer-length",
    "--concurrency-per-endpoint",
    "--warmups-per-endpoint",
    "--gpu",
    "--trace-interval-ms",
}
HF_OVERRIDE_OWNED_KEYS = {
    "model_type",
    "architectures",
    "nemotron_ocr_model_subdir",
    "nemotron_ocr_merge_level",
    "nemotron_ocr_detector_max_batch_size",
    "nemotron_ocr_recognizer_chunk_size",
    "nemotron_ocr_relational_chunk_size",
    "nemotron_ocr_verbose_post",
    "nemotron_ocr_infer_length",
}
DEFAULT_SERVER = {
    "replicas": 1,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
    "renderer_num_workers": 1,
    "detector_max_batch_size": 8,
    "recognizer_chunk_size": 128,
    "relational_chunk_size": 128,
    "infer_length": 1024,
    "mps_enabled": True,
    "mps_active_thread_percentage": 100,
    "mm_processor_cache_gb": 0,
    "enforce_eager": True,
    "async_scheduling": False,
    "disable_access_log": True,
    "disable_log_stats": True,
    "hf_overrides": {},
    "extra_args": [],
}
DEFAULT_BENCHMARK = {
    "replay_count": 10,
    "concurrency_per_endpoint": 96,
    "warmups_per_endpoint": 32,
    "trace_interval_ms": 250,
    "extra_args": [],
}


class ConfigError(ValueError):
    """Raised for invalid sweep input."""


@dataclass(frozen=True)
class DatasetInfo:
    path: str
    size_bytes: int
    mtime_ns: int
    rows: int
    jpeg_data_uri_rows: int
    jpeg_payload_bytes: int
    sha256: str


@dataclass(frozen=True)
class RunPlan:
    name: str
    config: dict[str, Any]
    run_dir: Path
    mps_runtime_dir: Path
    endpoints: list[str]
    server_commands: list[list[str]]
    server_env_overrides: list[dict[str, str]]
    benchmark_command: list[str]
    benchmark_env_overrides: dict[str, str]
    mps_env: dict[str, str]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    """Convert harness termination into normal process/MPS cleanup."""
    raise KeyboardInterrupt


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def require_keys(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown {label} keys: {', '.join(unknown)}")


def positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{label} must be {qualifier}")
    return value


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{label} must be positive")
    return float(value)


def nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{label} must be non-negative")
    return float(value)


def bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be true or false")
    return value


def string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{label} must be a list of strings")
    return value


def reject_owned_flags(values: list[str], owned: set[str], label: str) -> None:
    candidates = {value.split("=", 1)[0] for value in values if value.startswith("--")}
    conflicts = sorted(
        candidate
        for candidate in candidates
        if candidate in owned or any(flag.startswith(candidate) for flag in owned)
    )
    if conflicts:
        raise ConfigError(
            f"{label} cannot override harness-owned flags: {', '.join(conflicts)}"
        )


def sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    return (
        upper.endswith(("_TOKEN", "_KEY", "_PASSWORD", "_SECRET"))
        or "ACCESS_TOKEN" in upper
        or "CREDENTIAL" in upper
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.")
    if not slug:
        raise ConfigError("run name must contain a letter or digit")
    return slug


def resolve_path(value: Any, base_dir: Path, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty path string")
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    # Keep venv launcher symlinks intact: resolving ``venv/bin/python`` to the
    # system interpreter would discard the virtual environment at execution.
    return os.path.abspath(expanded)


def validate_mps_percentages(value: Any, replicas: int) -> list[int]:
    values = value if isinstance(value, list) else [value] * replicas
    if len(values) != replicas:
        raise ConfigError(
            "server.mps_active_thread_percentage list length must equal replicas"
        )
    percentages = [
        positive_int(item, "server.mps_active_thread_percentage") for item in values
    ]
    if any(item > 100 for item in percentages):
        raise ConfigError("MPS active thread percentages must be in [1, 100]")
    return percentages


def normalize_run(
    name: str,
    raw: dict[str, Any],
    *,
    config_dir: Path,
    output_root_override: Path | None,
    gpu_override: str | None,
) -> dict[str, Any]:
    require_keys(raw, COMMON_KEYS, f"run {name!r}")
    required = {
        "gpu",
        "python",
        "vllm_cli",
        "vllm_root",
        "model",
        "model_source",
        "dataset",
        "benchmark_script",
        "output_root",
        "mps_root",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"run {name!r} is missing: {', '.join(missing)}")

    config = copy.deepcopy(raw)
    for key in (
        "python",
        "vllm_cli",
        "vllm_root",
        "model",
        "model_source",
        "dataset",
        "benchmark_script",
        "output_root",
        "mps_root",
        "cuda_home",
        "hf_home",
        "hf_hub_cache",
        "torchinductor_cache_dir",
        "triton_cache_dir",
    ):
        if key in config:
            config[key] = resolve_path(config[key], config_dir, key)
    if output_root_override is not None:
        config["output_root"] = str(output_root_override.resolve())
    if gpu_override is not None:
        config["gpu"] = gpu_override
    if not isinstance(config["gpu"], str) or not config["gpu"]:
        raise ConfigError("gpu must be a non-empty index or UUID string")

    config.setdefault("host", "127.0.0.1")
    if config["host"] not in {"127.0.0.1", "localhost"}:
        raise ConfigError("host must be localhost or 127.0.0.1")
    config.setdefault("base_port", 8300)
    config.setdefault("ready_timeout_s", 900)
    config.setdefault("health_poll_interval_s", 1)
    config.setdefault("health_settle_s", 2)
    config.setdefault("shutdown_timeout_s", 30)
    config.setdefault("benchmark_timeout_s", 3600)
    config.setdefault("cuda_home", "/usr/local/cuda")
    config.setdefault("omp_num_threads", 4)
    config.setdefault("require_clean_repositories", False)
    config.setdefault("env", {})
    positive_int(config["base_port"], "base_port")
    for key in (
        "ready_timeout_s",
        "health_poll_interval_s",
        "shutdown_timeout_s",
        "benchmark_timeout_s",
    ):
        positive_number(config[key], key)
    nonnegative_number(config["health_settle_s"], "health_settle_s")
    positive_int(config["omp_num_threads"], "omp_num_threads")
    bool_value(config["require_clean_repositories"], "require_clean_repositories")
    for key in (
        "expected_dataset_sha256",
        "expected_model_extension_sha256",
        "expected_vllm_commit",
        "expected_model_commit",
    ):
        if key not in config:
            continue
        value = config[key]
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-fA-F]{40,64}", value) is None
        ):
            raise ConfigError(f"{key} must be a hexadecimal commit or SHA-256")
        required_length = 40 if key.endswith("_commit") else 64
        if len(value) != required_length:
            raise ConfigError(
                f"{key} must contain {required_length} hexadecimal digits"
            )
        config[key] = value.lower()
    if not isinstance(config["env"], dict):
        raise ConfigError("env must be an object")
    overlap = RESERVED_ENV_KEYS.intersection(config["env"])
    if overlap:
        raise ConfigError(
            "env cannot override harness-owned keys: " + ", ".join(sorted(overlap))
        )
    if not all(
        isinstance(key, str) and isinstance(value, (str, int, float, bool))
        for key, value in config["env"].items()
    ):
        raise ConfigError("env keys must be strings and values must be scalar")
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in config["env"].values()
    ):
        raise ConfigError("env numeric values must be finite")
    secret_keys = sorted(key for key in config["env"] if sensitive_env_key(key))
    if secret_keys:
        raise ConfigError(
            "env may not contain secrets because provenance records it: "
            + ", ".join(secret_keys)
        )

    server_raw = config.get("server", {})
    if not isinstance(server_raw, dict):
        raise ConfigError("server must be an object")
    server = deep_merge(DEFAULT_SERVER, server_raw)
    require_keys(server, SERVER_KEYS, "server")
    for key in (
        "replicas",
        "max_num_seqs",
        "renderer_num_workers",
        "detector_max_batch_size",
        "recognizer_chunk_size",
        "relational_chunk_size",
        "infer_length",
    ):
        positive_int(server[key], f"server.{key}")
    gmem = positive_number(
        server["gpu_memory_utilization"], "server.gpu_memory_utilization"
    )
    if gmem > 1:
        raise ConfigError("server.gpu_memory_utilization must be at most 1")
    nonnegative_number(server["mm_processor_cache_gb"], "server.mm_processor_cache_gb")
    for key in (
        "mps_enabled",
        "enforce_eager",
        "async_scheduling",
        "disable_access_log",
        "disable_log_stats",
    ):
        bool_value(server[key], f"server.{key}")
    if not isinstance(server["hf_overrides"], dict):
        raise ConfigError("server.hf_overrides must be an object")
    hf_conflicts = sorted(HF_OVERRIDE_OWNED_KEYS.intersection(server["hf_overrides"]))
    if hf_conflicts:
        raise ConfigError(
            "server.hf_overrides cannot replace canonical OCR settings: "
            + ", ".join(hf_conflicts)
        )
    server_extra_args = string_list(server["extra_args"], "server.extra_args")
    reject_owned_flags(server_extra_args, SERVER_OWNED_FLAGS, "server.extra_args")
    percentages = validate_mps_percentages(
        server["mps_active_thread_percentage"], server["replicas"]
    )
    server["mps_active_thread_percentage"] = percentages
    if config["base_port"] + server["replicas"] - 1 > 65535:
        raise ConfigError("replica ports exceed 65535")
    config["server"] = server

    benchmark_raw = config.get("benchmark", {})
    if not isinstance(benchmark_raw, dict):
        raise ConfigError("benchmark must be an object")
    benchmark = deep_merge(DEFAULT_BENCHMARK, benchmark_raw)
    require_keys(benchmark, BENCHMARK_KEYS, "benchmark")
    if "num_prompts" not in benchmark:
        raise ConfigError("benchmark.num_prompts is required")
    for key in (
        "num_prompts",
        "replay_count",
        "concurrency_per_endpoint",
        "trace_interval_ms",
    ):
        positive_int(benchmark[key], f"benchmark.{key}")
    positive_int(
        benchmark["warmups_per_endpoint"],
        "benchmark.warmups_per_endpoint",
        allow_zero=True,
    )
    benchmark_extra_args = string_list(benchmark["extra_args"], "benchmark.extra_args")
    reject_owned_flags(
        benchmark_extra_args, BENCHMARK_OWNED_FLAGS, "benchmark.extra_args"
    )
    if benchmark["num_prompts"] % benchmark["replay_count"]:
        raise ConfigError("benchmark.num_prompts must be divisible by replay_count")
    if benchmark["num_prompts"] < server["replicas"]:
        raise ConfigError("benchmark.num_prompts must be at least server.replicas")
    config["benchmark"] = benchmark
    config["name"] = slugify(name)
    return config


def load_config(
    path: Path,
    *,
    output_root_override: Path | None = None,
    gpu_override: str | None = None,
) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    require_keys(raw, {"schema_version", "defaults", "sweep"}, "top-level")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    defaults = raw.get("defaults")
    sweep = raw.get("sweep")
    if not isinstance(defaults, dict):
        raise ConfigError("defaults must be an object")
    require_keys(defaults, COMMON_KEYS, "defaults")
    if not isinstance(sweep, list) or not sweep:
        raise ConfigError("sweep must be a non-empty list")

    runs = []
    names: set[str] = set()
    for index, item in enumerate(sweep):
        if not isinstance(item, dict):
            raise ConfigError(f"sweep item {index} must be an object")
        require_keys(item, COMMON_KEYS | {"name"}, f"sweep item {index}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"sweep item {index} needs a non-empty name")
        slug = slugify(name)
        if slug in names:
            raise ConfigError(f"duplicate run name: {slug}")
        names.add(slug)
        override = {key: value for key, value in item.items() if key != "name"}
        runs.append(
            normalize_run(
                slug,
                deep_merge(defaults, override),
                config_dir=path.parent.resolve(),
                output_root_override=output_root_override,
                gpu_override=gpu_override,
            )
        )
    return runs


def decode_prompt_jpeg(line: bytes, line_number: int) -> bytes:
    """Parse one custom-dataset row and return its exact JPEG prompt bytes."""
    try:
        row = json.loads(line)
        prompt = row["prompt"]
        data = prompt["data"]
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid custom dataset row {line_number}") from exc
    if not isinstance(prompt, dict) or not isinstance(data, str):
        raise ConfigError(f"dataset row {line_number} prompt.data must be a string")
    prefix = JPEG_DATA_URI_PREFIX.decode()
    if not data.startswith(prefix):
        raise ConfigError(
            f"dataset row {line_number} prompt.data is not a JPEG-byte data URI"
        )
    try:
        payload = base64.b64decode(data[len(prefix) :], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ConfigError(
            f"dataset row {line_number} prompt.data has invalid base64"
        ) from exc
    if not payload.startswith(b"\xff\xd8\xff") or not payload.endswith(b"\xff\xd9"):
        raise ConfigError(
            f"dataset row {line_number} prompt.data lacks JPEG SOI/EOI bytes"
        )
    return payload


def inspect_dataset(path: Path) -> DatasetInfo:
    """Hash the dataset and fully validate every prompt's JPEG bytes."""
    digest = hashlib.sha256()
    rows = 0
    jpeg_rows = 0
    jpeg_payload_bytes = 0
    with path.open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            payload = decode_prompt_jpeg(line, line_number)
            jpeg_payload_bytes += len(payload)
            jpeg_rows += 1
    if not rows:
        raise ConfigError(f"dataset is empty: {path}")
    stat = path.stat()
    return DatasetInfo(
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        rows=rows,
        jpeg_data_uri_rows=jpeg_rows,
        jpeg_payload_bytes=jpeg_payload_bytes,
        sha256=digest.hexdigest(),
    )


def inspect_dataset_head(path: Path) -> None:
    """Cheap JPEG contract check for dry runs."""
    with path.open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            decode_prompt_jpeg(line, line_number)
            return
    raise ConfigError(f"dataset is empty: {path}")


def hf_overrides(config: dict[str, Any]) -> dict[str, Any]:
    server = config["server"]
    base = {
        "model_type": "nemotron_ocr_v2",
        "architectures": ["NemotronOCRV2ForImageToText"],
        "nemotron_ocr_model_subdir": "v2_multilingual",
        "nemotron_ocr_merge_level": "paragraph",
        "nemotron_ocr_detector_max_batch_size": server["detector_max_batch_size"],
        "nemotron_ocr_recognizer_chunk_size": server["recognizer_chunk_size"],
        "nemotron_ocr_relational_chunk_size": server["relational_chunk_size"],
        "nemotron_ocr_verbose_post": False,
        "nemotron_ocr_infer_length": server["infer_length"],
    }
    return deep_merge(base, server["hf_overrides"])


def base_environment(config: dict[str, Any]) -> dict[str, str]:
    env = {
        "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
        "CUDA_HOME": config["cuda_home"],
        "HF_HUB_OFFLINE": "1",
        "NEMOTRON_OCR_SOURCE": config["model_source"],
        "OMP_NUM_THREADS": str(config["omp_num_threads"]),
        "MKL_NUM_THREADS": str(config["omp_num_threads"]),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_LOGGING_LEVEL": "WARNING",
    }
    for key in (
        "hf_home",
        "hf_hub_cache",
        "torchinductor_cache_dir",
        "triton_cache_dir",
    ):
        if key in config:
            env[key.upper()] = config[key]
    env["PYTHONPATH"] = f"{config['vllm_root']}:{config['model_source']}"
    env.update({key: str(value) for key, value in config["env"].items()})
    return env


def process_environment(overrides: dict[str, str]) -> dict[str, str]:
    """Build a config-authoritative child environment.

    CUDA/framework/model tuning inherited from an interactive shell can make
    a nominal baseline non-reproducible. Remove those namespaces and
    PYTHONPATH first, then reapply only the values built from this run's config.
    """
    env = os.environ.copy()
    for key in list(env):
        if (
            key == "PYTHONPATH"
            or key in MPS_ENV_KEYS
            or key.startswith(ENV_CAPTURE_PREFIXES)
        ):
            env.pop(key, None)
    env.update(overrides)
    return env


def build_plan(
    config: dict[str, Any],
    *,
    invocation_dir: Path,
    mps_nonce: str,
) -> RunPlan:
    name = config["name"]
    server = config["server"]
    benchmark = config["benchmark"]
    run_dir = invocation_dir / name
    mps_digest = hashlib.sha256(f"{name}-{mps_nonce}".encode()).hexdigest()[:12]
    mps_runtime_dir = Path(config["mps_root"]) / f"{name[:24]}-{mps_digest}"
    pipe_dir = mps_runtime_dir / "pipe"
    log_dir = mps_runtime_dir / "log"
    if len(os.fsencode(pipe_dir / "control")) >= 100:
        raise ConfigError(
            "CUDA MPS control path is too long; shorten mps_root: "
            f"{pipe_dir / 'control'}"
        )
    common_env = base_environment(config)
    mps_env = {
        "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
        "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
        "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
    }
    if server["mps_enabled"]:
        common_env.update(mps_env)

    overrides_json = json.dumps(
        hf_overrides(config), separators=(",", ":"), sort_keys=True
    )
    endpoints: list[str] = []
    server_commands: list[list[str]] = []
    server_envs: list[dict[str, str]] = []
    for replica in range(server["replicas"]):
        port = config["base_port"] + replica
        endpoints.append(f"http://{config['host']}:{port}")
        command = [
            config["vllm_cli"],
            "serve",
            config["model"],
            "--runner",
            "pooling",
            "--skip-tokenizer-init",
            "--io-processor-plugin",
            "nemotron_ocr_v2",
            "--max-num-seqs",
            str(server["max_num_seqs"]),
            "--gpu-memory-utilization",
            str(server["gpu_memory_utilization"]),
            "--mm-processor-cache-gb",
            str(server["mm_processor_cache_gb"]),
            "--renderer-num-workers",
            str(server["renderer_num_workers"]),
            "--host",
            config["host"],
            "--port",
            str(port),
            "--hf-overrides",
            overrides_json,
        ]
        if server["enforce_eager"]:
            command.append("--enforce-eager")
        if server["async_scheduling"]:
            command.append("--async-scheduling")
        else:
            command.append("--no-async-scheduling")
        if server["disable_access_log"]:
            command.append("--disable-uvicorn-access-log")
        if server["disable_log_stats"]:
            command.append("--disable-log-stats")
        command.extend(server["extra_args"])
        replica_env = dict(common_env)
        if server["mps_enabled"]:
            replica_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                server["mps_active_thread_percentage"][replica]
            )
        server_commands.append(command)
        server_envs.append(replica_env)

    benchmark_command = [config["python"], config["benchmark_script"]]
    for endpoint in endpoints:
        benchmark_command.extend(["--endpoint", endpoint])
    benchmark_command.extend(
        [
            "--dataset",
            config["dataset"],
            "--output-dir",
            str(run_dir / "benchmark"),
            "--model",
            config["model"],
            "--num-prompts",
            str(benchmark["num_prompts"]),
            "--replay-count",
            str(benchmark["replay_count"]),
            "--infer-length",
            str(server["infer_length"]),
            "--concurrency-per-endpoint",
            str(benchmark["concurrency_per_endpoint"]),
            "--warmups-per-endpoint",
            str(benchmark["warmups_per_endpoint"]),
            "--gpu",
            str(config["gpu"]),
            "--trace-interval-ms",
            str(benchmark["trace_interval_ms"]),
        ]
    )
    benchmark_command.extend(benchmark["extra_args"])
    benchmark_env = base_environment(config)
    return RunPlan(
        name=name,
        config=config,
        run_dir=run_dir,
        mps_runtime_dir=mps_runtime_dir,
        endpoints=endpoints,
        server_commands=server_commands,
        server_env_overrides=server_envs,
        benchmark_command=benchmark_command,
        benchmark_env_overrides=benchmark_env,
        mps_env=mps_env,
    )


def selected_environment(overrides: dict[str, str]) -> dict[str, str]:
    effective = process_environment(overrides)
    keys = {
        key
        for key in effective
        if key in overrides
        or key in ENV_CAPTURE_EXACT
        or key.startswith(ENV_CAPTURE_PREFIXES)
    }
    return {
        key: "<redacted>" if sensitive_env_key(key) else effective[key]
        for key in sorted(keys)
    }


def command_record(command: list[str]) -> dict[str, Any]:
    return {"argv": command, "shell_escaped": " ".join(map(shlex_quote, command))}


def shlex_quote(value: str) -> str:
    # Importing shlex lazily keeps command construction dependency-free.
    import shlex

    return shlex.quote(value)


def plan_record(plan: RunPlan) -> dict[str, Any]:
    mps_enabled = plan.config["server"]["mps_enabled"]
    return {
        "name": plan.name,
        "run_dir": str(plan.run_dir),
        "mps_runtime_dir": str(plan.mps_runtime_dir),
        "config": plan.config,
        "endpoints": plan.endpoints,
        "mps": {
            "enabled": mps_enabled,
            "start": (
                command_record(["nvidia-cuda-mps-control", "-d"])
                if mps_enabled
                else None
            ),
            "stop_stdin": "quit\\n" if mps_enabled else None,
            "environment": plan.mps_env if mps_enabled else {},
        },
        "servers": [
            {
                **command_record(command),
                "environment": selected_environment(environment),
            }
            for command, environment in zip(
                plan.server_commands, plan.server_env_overrides
            )
        ],
        "benchmark": {
            **command_record(plan.benchmark_command),
            "environment": selected_environment(plan.benchmark_env_overrides),
        },
    }


def run_command(
    command: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        command,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def resolve_gpu(identifier: str) -> dict[str, Any]:
    result = run_command(
        [
            "nvidia-smi",
            "-i",
            identifier,
            "--query-gpu=index,uuid,name,pci.bus_id,memory.total,compute_mode,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"GPU selector {identifier!r} resolved to {len(rows)} rows")
    fields = [field.strip() for field in rows[0].split(",", 6)]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]!r}")
    return dict(
        zip(
            (
                "index",
                "uuid",
                "name",
                "pci_bus_id",
                "memory_total_mib",
                "compute_mode",
                "driver_version",
            ),
            fields,
        )
    )


def gpu_processes(gpu_uuid: str) -> list[dict[str, str]]:
    result = run_command(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = []
    for line in result.stdout.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3:
            processes.append(
                {"pid": fields[0], "process_name": fields[1], "memory_mib": fields[2]}
            )
    return processes


def acquire_gpu_lock(gpu_uuid: str) -> BinaryIO:
    safe_uuid = re.sub(r"[^a-zA-Z0-9_.-]", "-", gpu_uuid)
    path = Path("/tmp") / f"nemotron-ocr-sweep-{safe_uuid}.lock"
    lock = path.open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f"another sweep holds {path}") from exc
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} started={utc_now()}\n".encode())
    lock.flush()
    return lock


def check_ports_available(host: str, ports: list[int]) -> None:
    unavailable = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        except OSError as exc:
            unavailable.append((port, str(exc)))
        finally:
            sock.close()
    if unavailable:
        raise RuntimeError(f"server ports are unavailable: {unavailable}")


def git_provenance(path: Path, artifact_dir: Path, label: str) -> dict[str, Any]:
    """Capture a commit plus the complete dirty source state needed to replay it."""
    record: dict[str, Any] = {"path": str(path.resolve())}
    try:
        sha = run_command(["git", "-C", str(path), "rev-parse", "HEAD"])
        branch = run_command(["git", "-C", str(path), "branch", "--show-current"])
        status = run_command(["git", "-C", str(path), "status", "--short"], timeout=60)
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--binary", "HEAD", "--"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    changes = status.stdout.splitlines()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    patch_path = artifact_dir / f"{label}.patch"
    patch_path.write_bytes(diff.stdout)
    untracked_root = artifact_dir / f"{label}_untracked"
    untracked_records = []
    for raw_relative in untracked.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = Path(os.fsdecode(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe untracked Git path: {relative}")
        source = path / relative
        record_item: dict[str, Any] = {"relative_path": str(relative)}
        if source.is_symlink():
            target = os.readlink(source)
            record_item.update(
                {
                    "type": "symlink",
                    "target": target,
                    "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
            )
        elif source.is_file():
            destination = untracked_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            captured = file_provenance(destination)
            source_state = file_provenance(source)
            if captured["sha256"] != source_state["sha256"]:
                raise RuntimeError(f"untracked file changed during capture: {relative}")
            record_item.update(
                {
                    "type": "file",
                    "captured_copy": captured,
                    "source": source_state,
                }
            )
        else:
            record_item["type"] = "missing_or_special"
        untracked_records.append(record_item)
    record.update(
        {
            "commit": sha.stdout.strip(),
            "branch": branch.stdout.strip(),
            "dirty": bool(changes),
            "status_short": changes,
            "tracked_binary_patch": file_provenance(patch_path),
            "untracked_files": untracked_records,
        }
    )
    return record


def file_provenance(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def verify_source_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Verify pinned clean source and extension inputs before touching a GPU."""
    repositories = {}
    for label, path_key, expected_key in (
        ("vllm", "vllm_root", "expected_vllm_commit"),
        ("model", "model", "expected_model_commit"),
    ):
        path = Path(config[path_key])
        commit = run_command(
            ["git", "-C", str(path), "rev-parse", "HEAD"]
        ).stdout.strip()
        status = run_command(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            timeout=60,
        ).stdout.splitlines()
        expected = config.get(expected_key)
        if expected is not None and commit != expected:
            raise ConfigError(
                f"{label} commit mismatch: expected {expected}, found {commit}"
            )
        if config["require_clean_repositories"] and status:
            raise ConfigError(
                f"{label} repository must be clean; changes: {status[:20]}"
            )
        repositories[label] = {
            "path": str(path),
            "commit": commit,
            "expected_commit": expected,
            "clean": not status,
            "status_short": status,
        }

    extension_record = None
    expected_extension = config.get("expected_model_extension_sha256")
    if expected_extension is not None:
        extension_dir = Path(config["model_source"]) / "nemotron_ocr_cpp"
        extensions = sorted(extension_dir.glob("_nemotron_ocr_cpp*.so"))
        if len(extensions) != 1:
            raise ConfigError(
                "expected exactly one model extension for strict provenance; "
                f"found {len(extensions)} in {extension_dir}"
            )
        extension_record = file_provenance(extensions[0])
        if extension_record["sha256"] != expected_extension:
            raise ConfigError(
                "model extension SHA-256 mismatch: expected "
                f"{expected_extension}, found {extension_record['sha256']}"
            )
    return {
        "require_clean_repositories": config["require_clean_repositories"],
        "repositories": repositories,
        "model_extension": extension_record,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def stop_process_group(process: subprocess.Popen[Any], timeout: float) -> None:
    # The launcher may have exited while vLLM multiprocessing children remain.
    # Address its original process group even when the leader is already gone.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait(timeout=5)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait(timeout=5)
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait(timeout=5)


class ActiveRun:
    """Own all processes and MPS state for one configuration."""

    def __init__(self, plan: RunPlan) -> None:
        self.plan = plan
        self.server_processes: list[subprocess.Popen[Any]] = []
        self.server_logs: list[TextIO] = []
        self.benchmark_process: subprocess.Popen[Any] | None = None
        self.benchmark_log: TextIO | None = None
        self.mps_runtime_owned = False
        self.cleanup_warnings: list[str] = []

    def start_mps(self) -> None:
        if not self.plan.config["server"]["mps_enabled"]:
            return
        pipe_dir = Path(self.plan.mps_env["CUDA_MPS_PIPE_DIRECTORY"])
        log_dir = Path(self.plan.mps_env["CUDA_MPS_LOG_DIRECTORY"])
        self.plan.mps_runtime_dir.mkdir(parents=True, exist_ok=False)
        self.mps_runtime_owned = True
        pipe_dir.mkdir()
        log_dir.mkdir()
        start_log_path = self.plan.run_dir / "mps_start.log"
        env = process_environment(self.plan.mps_env)
        with start_log_path.open("w", encoding="utf-8") as start_log:
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                check=True,
                text=True,
                stdout=start_log,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        deadline = time.monotonic() + 15
        while not (pipe_dir / "control").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("owned CUDA MPS control socket did not appear")
            time.sleep(0.1)

    def start_servers(self) -> None:
        server_dir = self.plan.run_dir / "servers"
        server_dir.mkdir(parents=True, exist_ok=True)
        for index, (command, overrides) in enumerate(
            zip(self.plan.server_commands, self.plan.server_env_overrides)
        ):
            log = (server_dir / f"server_{index:02d}.log").open("w", encoding="utf-8")
            self.server_logs.append(log)
            env = process_environment(overrides)
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.server_processes.append(process)

    def wait_for_health(self) -> None:
        config = self.plan.config
        deadline = time.monotonic() + float(config["ready_timeout_s"])
        pending = set(range(len(self.plan.endpoints)))
        next_update = time.monotonic()
        while pending:
            for index in list(pending):
                process = self.server_processes[index]
                code = process.poll()
                if code is not None:
                    log_path = self.plan.run_dir / "servers" / f"server_{index:02d}.log"
                    raise RuntimeError(
                        f"server {index} exited with {code} before health; "
                        f"log tail:\n{tail(log_path)}"
                    )
                url = self.plan.endpoints[index] + "/health"
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if response.status == 200:
                            pending.remove(index)
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
            now = time.monotonic()
            if not pending:
                break
            if now >= deadline:
                raise TimeoutError(
                    f"servers not healthy before timeout: {sorted(pending)}"
                )
            if now >= next_update:
                print(
                    f"[{self.plan.name}] waiting for {len(pending)} "
                    "native vLLM server(s)",
                    flush=True,
                )
                next_update = now + 15
            time.sleep(float(config["health_poll_interval_s"]))
        if config["health_settle_s"]:
            time.sleep(float(config["health_settle_s"]))

    def run_benchmark(self) -> dict[str, Any]:
        benchmark_dir = self.plan.run_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_log = (self.plan.run_dir / "benchmark.log").open(
            "w", encoding="utf-8"
        )
        env = process_environment(self.plan.benchmark_env_overrides)
        self.benchmark_process = subprocess.Popen(
            self.plan.benchmark_command,
            env=env,
            stdout=self.benchmark_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = self.benchmark_process.wait(
                timeout=float(self.plan.config["benchmark_timeout_s"])
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("native pooling benchmark timed out") from exc
        if code:
            raise RuntimeError(
                f"native pooling benchmark exited with {code}; "
                f"log tail:\n{tail(self.plan.run_dir / 'benchmark.log')}"
            )
        summary_path = benchmark_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("benchmark completed without summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("failed", -1)) != 0:
            raise RuntimeError(f"benchmark recorded failures: {summary.get('failed')}")
        expected = self.plan.config["benchmark"]["num_prompts"]
        if int(summary.get("completed", -1)) != expected:
            raise RuntimeError(
                f"benchmark completed {summary.get('completed')}; expected {expected}"
            )
        return summary

    def cleanup(self) -> None:
        timeout = float(self.plan.config["shutdown_timeout_s"])
        if self.benchmark_process is not None:
            try:
                stop_process_group(self.benchmark_process, timeout)
            except Exception as exc:  # noqa: BLE001 - continue cleanup
                self.cleanup_warnings.append(f"benchmark cleanup: {exc!r}")
        if self.benchmark_log is not None and not self.benchmark_log.closed:
            try:
                self.benchmark_log.close()
            except OSError as exc:
                self.cleanup_warnings.append(f"benchmark log cleanup: {exc!r}")
        for process in reversed(self.server_processes):
            try:
                stop_process_group(process, timeout)
            except Exception as exc:  # noqa: BLE001 - continue cleanup
                self.cleanup_warnings.append(f"server cleanup: {exc!r}")
        for log in self.server_logs:
            if not log.closed:
                try:
                    log.close()
                except OSError as exc:
                    self.cleanup_warnings.append(f"server log cleanup: {exc!r}")

        if self.mps_runtime_owned:
            pipe_dir = Path(self.plan.mps_env["CUDA_MPS_PIPE_DIRECTORY"])
            safe_to_remove_runtime = not (pipe_dir / "control").exists()
            if (pipe_dir / "control").exists():
                env = process_environment(self.plan.mps_env)
                try:
                    result = subprocess.run(
                        ["nvidia-cuda-mps-control"],
                        env=env,
                        input="quit\n",
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=30,
                        check=False,
                    )
                    (self.plan.run_dir / "mps_stop.log").write_text(
                        result.stdout, encoding="utf-8"
                    )
                    if result.returncode:
                        self.cleanup_warnings.append(
                            f"owned MPS control quit returned {result.returncode}"
                        )
                    else:
                        safe_to_remove_runtime = True
                except Exception as exc:  # noqa: BLE001 - continue cleanup
                    self.cleanup_warnings.append(f"owned MPS cleanup: {exc!r}")
            log_dir = Path(self.plan.mps_env["CUDA_MPS_LOG_DIRECTORY"])
            if log_dir.is_dir():
                try:
                    shutil.copytree(
                        log_dir,
                        self.plan.run_dir / "mps_logs",
                        dirs_exist_ok=True,
                    )
                except (OSError, shutil.Error) as exc:
                    self.cleanup_warnings.append(f"MPS log copy: {exc!r}")
            if safe_to_remove_runtime:
                try:
                    shutil.rmtree(self.plan.mps_runtime_dir)
                except OSError as exc:
                    self.cleanup_warnings.append(f"MPS runtime removal: {exc!r}")
            else:
                self.cleanup_warnings.append(
                    "preserved owned MPS runtime after unsuccessful quit: "
                    f"{self.plan.mps_runtime_dir}"
                )


def validate_input_paths(config: dict[str, Any]) -> None:
    files = ("python", "vllm_cli", "dataset", "benchmark_script")
    directories = ("vllm_root", "model", "model_source")
    for key in files:
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key}: {config[key]}")
    for key in directories:
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(f"{key}: {config[key]}")


def execute_plan(
    plan: RunPlan,
    *,
    dataset_info: DatasetInfo,
    gpu_info: dict[str, Any],
    config_provenance: dict[str, Any],
) -> dict[str, Any]:
    plan.run_dir.mkdir(parents=True, exist_ok=False)
    extension_dir = Path(plan.config["model_source"]) / "nemotron_ocr_cpp"
    extension_artifacts = [
        file_provenance(path)
        for path in sorted(extension_dir.glob("_nemotron_ocr_cpp*.so"))
    ]
    source_state_dir = plan.run_dir / "source_state"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "started_at": utc_now(),
        "harness_argv": sys.argv,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions(
                ("torch", "vllm", "triton", "orjson", "aiohttp")
            ),
        },
        "target_gpu": gpu_info,
        "dataset": dataset_info.__dict__,
        "config_source": config_provenance,
        "repositories": {
            "vllm": git_provenance(
                Path(plan.config["vllm_root"]), source_state_dir, "vllm"
            ),
            "model": git_provenance(
                Path(plan.config["model"]), source_state_dir, "model"
            ),
        },
        "artifacts": {
            "harness": file_provenance(Path(__file__)),
            "benchmark_client": file_provenance(Path(plan.config["benchmark_script"])),
            "vllm_cli": file_provenance(Path(plan.config["vllm_cli"]).resolve()),
            "model_extensions": extension_artifacts,
        },
        "plan": plan_record(plan),
    }
    provenance_path = plan.run_dir / "provenance.json"
    write_json(provenance_path, provenance)
    active = ActiveRun(plan)
    started = time.time()
    try:
        ports = [plan.config["base_port"] + i for i in range(len(plan.endpoints))]
        check_ports_available(plan.config["host"], ports)
        provenance["status"] = "starting_mps"
        write_json(provenance_path, provenance)
        active.start_mps()
        provenance["status"] = "loading_servers"
        write_json(provenance_path, provenance)
        active.start_servers()
        provenance["server_pids"] = [process.pid for process in active.server_processes]
        write_json(provenance_path, provenance)
        active.wait_for_health()
        provenance["servers_ready_at"] = utc_now()
        provenance["status"] = "benchmarking"
        write_json(provenance_path, provenance)
        benchmark_summary = active.run_benchmark()
        status = "complete"
        error = None
    except BaseException as exc:
        benchmark_summary = None
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        active.cleanup()
        if status == "complete" and active.cleanup_warnings:
            status = "failed"
            error = "cleanup did not complete cleanly: " + "; ".join(
                active.cleanup_warnings
            )
        finished = time.time()
        provenance.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "wall_elapsed_s": finished - started,
                "error": error,
                "cleanup_warnings": active.cleanup_warnings,
            }
        )
        write_json(provenance_path, provenance)
        run_summary = {
            "name": plan.name,
            "status": status,
            "error": error,
            "cleanup_warnings": active.cleanup_warnings,
            "provenance": str(provenance_path),
            "benchmark_summary": benchmark_summary,
        }
        write_json(plan.run_dir / "run_summary.json", run_summary)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", action="append", help="Run only this named config.")
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", help="Override the config GPU index or UUID.")
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Run despite existing compute processes on the selected GPU.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print expanded commands; start nothing and write nothing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_hash = file_provenance(config_path)
    runs = load_config(
        config_path,
        output_root_override=args.output_root,
        gpu_override=args.gpu,
    )
    if args.only:
        wanted = {slugify(name) for name in args.only}
        runs = [run for run in runs if run["name"] in wanted]
        missing = wanted - {run["name"] for run in runs}
        if missing:
            raise ConfigError(f"unknown --only configs: {', '.join(sorted(missing))}")
    if not runs:
        raise ConfigError("no sweep configs selected")
    source_contract_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    dataset_contract_cache: dict[str, dict[str, Any]] = {}
    strict_preflight: dict[str, Any] = {}
    for run in runs:
        validate_input_paths(run)
        inspect_dataset_head(Path(run["dataset"]))
        source_key = (
            run["vllm_root"],
            run["model"],
            run["model_source"],
            run.get("expected_vllm_commit"),
            run.get("expected_model_commit"),
            run.get("expected_model_extension_sha256"),
            run["require_clean_repositories"],
        )
        if source_key not in source_contract_cache:
            source_contract_cache[source_key] = verify_source_contract(run)
        expected_dataset = run.get("expected_dataset_sha256")
        dataset_record = None
        if expected_dataset is not None:
            if run["dataset"] not in dataset_contract_cache:
                dataset_contract_cache[run["dataset"]] = file_provenance(
                    Path(run["dataset"])
                )
            dataset_record = dataset_contract_cache[run["dataset"]]
            if dataset_record["sha256"] != expected_dataset:
                raise ConfigError(
                    "dataset SHA-256 mismatch: expected "
                    f"{expected_dataset}, found {dataset_record['sha256']}"
                )
        strict_preflight[run["name"]] = {
            "source": source_contract_cache[source_key],
            "dataset": dataset_record,
        }

    label = slugify(args.label)
    output_roots = {run["output_root"] for run in runs}
    if len(output_roots) != 1:
        raise ConfigError("all selected runs must use the same output_root")
    invocation_dir = Path(output_roots.pop()) / label
    nonce = f"{label}-{os.getpid()}"
    plans = [
        build_plan(run, invocation_dir=invocation_dir, mps_nonce=nonce) for run in runs
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config_source": config_hash,
                    "invocation_dir": str(invocation_dir),
                    "strict_preflight": strict_preflight,
                    "plans": [plan_record(plan) for plan in plans],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if invocation_dir.exists():
        raise FileExistsError(f"invocation output already exists: {invocation_dir}")
    gpu_selectors = {run["gpu"] for run in runs}
    if len(gpu_selectors) != 1:
        raise ConfigError("all selected runs must target the same GPU")
    requested_gpu = gpu_selectors.pop()
    gpu_info = resolve_gpu(requested_gpu)
    gpu_info["requested_selector"] = requested_gpu
    # UUIDs avoid CUDA index remapping ambiguity between an MPS control daemon
    # and its clients. Dry-run remains GPU-free and therefore shows the input
    # selector; every real command uses the resolved physical UUID.
    for run in runs:
        run["gpu"] = gpu_info["uuid"]
    plans = [
        build_plan(run, invocation_dir=invocation_dir, mps_nonce=nonce) for run in runs
    ]
    gpu_lock = acquire_gpu_lock(gpu_info["uuid"])
    try:
        existing = gpu_processes(gpu_info["uuid"])
        if existing and not args.allow_busy_gpu:
            raise RuntimeError(
                "target GPU has compute processes; refusing to alter MPS state: "
                + json.dumps(existing)
            )
        dataset_cache: dict[str, DatasetInfo] = {}
        for run in runs:
            dataset = run["dataset"]
            if dataset not in dataset_cache:
                print(f"Inspecting JPEG-byte dataset: {dataset}", flush=True)
                dataset_cache[dataset] = inspect_dataset(Path(dataset))
            expected_rows = (
                run["benchmark"]["num_prompts"] // run["benchmark"]["replay_count"]
            )
            if dataset_cache[dataset].rows != expected_rows:
                raise ConfigError(
                    f"{run['name']}: dataset has {dataset_cache[dataset].rows} rows; "
                    f"num_prompts/replay_count requires {expected_rows}"
                )
            expected_sha256 = run.get("expected_dataset_sha256")
            if (
                expected_sha256 is not None
                and dataset_cache[dataset].sha256 != expected_sha256
            ):
                raise ConfigError(
                    f"{run['name']}: dataset SHA-256 changed after preflight; "
                    f"expected {expected_sha256}, found {dataset_cache[dataset].sha256}"
                )

        invocation_dir.mkdir(parents=True, exist_ok=False)
        invocation = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at": utc_now(),
            "argv": sys.argv,
            "config_source": config_hash,
            "target_gpu": gpu_info,
            "allow_busy_gpu": args.allow_busy_gpu,
            "preexisting_gpu_processes": existing,
            "strict_preflight": strict_preflight,
            "datasets": {key: value.__dict__ for key, value in dataset_cache.items()},
            "plans": [plan_record(plan) for plan in plans],
            "results": [],
        }
        invocation_path = invocation_dir / "sweep_summary.json"
        write_json(invocation_path, invocation)
        failed = False
        for plan in plans:
            print(f"[{plan.name}] starting native vLLM sweep config", flush=True)
            try:
                current_processes = gpu_processes(gpu_info["uuid"])
                if current_processes and not args.allow_busy_gpu:
                    raise RuntimeError(
                        "target GPU became busy before configuration start: "
                        + json.dumps(current_processes)
                    )
                result = execute_plan(
                    plan,
                    dataset_info=dataset_cache[plan.config["dataset"]],
                    gpu_info=gpu_info,
                    config_provenance=config_hash,
                )
                if result.get("status") != "complete":
                    failed = True
                    print(
                        f"[{plan.name}] failed: {result.get('error')}",
                        file=sys.stderr,
                    )
            except KeyboardInterrupt:
                invocation["status"] = "interrupted"
                invocation["finished_at"] = utc_now()
                write_json(invocation_path, invocation)
                raise
            except BaseException as exc:
                failed = True
                result = {
                    "name": plan.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[{plan.name}] failed: {result['error']}", file=sys.stderr)
            invocation["results"].append(result)
            write_json(invocation_path, invocation)
            if failed and not args.continue_on_error:
                break
        invocation["status"] = "failed" if failed else "complete"
        invocation["finished_at"] = utc_now()
        write_json(invocation_path, invocation)
        print(f"Sweep summary: {invocation_path}", flush=True)
        return 1 if failed else 0
    finally:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_UN)
        gpu_lock.close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
