#!/usr/bin/env python3
"""Generate deterministic Nemotron OCR benchmark charts and a Markdown report.

The generator reads a curated manifest rather than guessing which benchmark run is
publishable.  This keeps 768-token experiments, failed runs, partial client logs,
and sustained replay workloads out of the 1024-resolution / 1K-unique headline.

SVG output uses a tiny built-in vector renderer.  PNG output uses Pillow, which is
already part of the benchmark environment; matplotlib is intentionally not needed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only on an incomplete host
    raise SystemExit(
        "Pillow is required for PNG output. Install Pillow or use the benchmark venv."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "ocr_benchmark_report.json"
DEFAULT_RESULTS_ROOT = Path("/raid/vjawa/tmp/ocr_optimization/results")
DEFAULT_OUTPUT_DIR = ROOT / "results" / "publication"

GPU_COMPARISON_OVERRIDE_LABELS = {
    "official_hf": (
        "Official NVIDIA/HF in-process baseline",
        "Official NVIDIA/HF baseline",
    ),
    "vllm_baseline": (
        "Tuned clean-model vLLM baseline",
        "Tuned clean vLLM baseline",
    ),
    "optimized_native_vllm": (
        "Optimized native vLLM",
        "Optimized vLLM",
    ),
}

WIDTH = 1600
HEIGHT = 900
PNG_SCALE = 2

COLORS = {
    "ink": "#172B4D",
    "muted": "#64748B",
    "grid": "#DCE3EC",
    "blue": "#2F6BFF",
    "teal": "#009E9A",
    "orange": "#F59E0B",
    "light_blue": "#DCE7FF",
    "light_teal": "#D9F4F1",
    "light_orange": "#FEF0CE",
    "panel": "#F7F9FC",
    "white": "#FFFFFF",
    "red": "#C2413B",
}

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


class ReportError(RuntimeError):
    """A source artifact violates the report contract."""


@dataclass
class Point:
    label: str
    throughput: float
    count: int
    infer_length: int
    workload_kind: str
    source: str
    source_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hf-baseline-json",
        type=Path,
        help="Optional completed 1024-resolution, 1K-unique HF result JSON.",
    )
    parser.add_argument(
        "--hf-baseline-metric",
        default=None,
        help="Metric key for --hf-baseline-json (default: manifest value).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Validate all required sources; missing optional sources remain pending.",
    )
    parser.add_argument(
        "--gpu-rolling-window-seconds",
        type=float,
        default=None,
        help=(
            "Override the trailing rolling-mean window used by the GPU utilization "
            "and power comparison (manifest default: 15 seconds)."
        ),
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help=(
            "Generate only the matched throughput/speedup and GPU-active "
            "utilization/power comparison. This path does not load legacy sweep "
            "or publication inputs."
        ),
    )
    parser.add_argument(
        "--official-hf-result-json",
        type=Path,
        help="Explicit official NVIDIA/HF in-process result JSON.",
    )
    parser.add_argument(
        "--official-hf-trace-csv",
        type=Path,
        help="Raw nvidia-smi CSV paired with --official-hf-result-json.",
    )
    parser.add_argument(
        "--tuned-clean-vllm-result-json",
        type=Path,
        help="Explicit tuned clean-model vLLM result JSON.",
    )
    parser.add_argument(
        "--tuned-clean-vllm-trace-csv",
        type=Path,
        help="Raw nvidia-smi CSV paired with --tuned-clean-vllm-result-json.",
    )
    parser.add_argument(
        "--optimized-vllm-result-json",
        type=Path,
        help="Explicit optimized native-vLLM result JSON.",
    )
    parser.add_argument(
        "--optimized-vllm-trace-csv",
        type=Path,
        help="Raw nvidia-smi CSV paired with --optimized-vllm-result-json.",
    )
    return parser.parse_args()


def gpu_comparison_overrides_from_args(
    args: argparse.Namespace,
) -> dict[str, dict[str, Path]]:
    configured = {
        "official_hf": (
            args.official_hf_result_json,
            args.official_hf_trace_csv,
        ),
        "vllm_baseline": (
            args.tuned_clean_vllm_result_json,
            args.tuned_clean_vllm_trace_csv,
        ),
        "optimized_native_vllm": (
            args.optimized_vllm_result_json,
            args.optimized_vllm_trace_csv,
        ),
    }
    overrides: dict[str, dict[str, Path]] = {}
    for run_id, (result_path, trace_path) in configured.items():
        if (result_path is None) != (trace_path is None):
            label = GPU_COMPARISON_OVERRIDE_LABELS[run_id][0]
            raise ReportError(
                f"{label} requires both an explicit result JSON and trace CSV"
            )
        if result_path is not None and trace_path is not None:
            overrides[run_id] = {
                "result_path": result_path.resolve(),
                "trace_path": trace_path.resolve(),
            }
    return overrides


def apply_gpu_comparison_overrides(
    spec: dict[str, Any] | None,
    overrides: dict[str, dict[str, Path]] | None,
) -> dict[str, Any] | None:
    if spec is None or not overrides:
        return copy.deepcopy(spec)
    updated = copy.deepcopy(spec)
    runs = updated.get("runs")
    if not isinstance(runs, list):
        raise ReportError("gpu_active_comparison.runs must be a list")
    runs_by_id = {run.get("id"): run for run in runs if isinstance(run, dict)}
    unknown = sorted(set(overrides) - set(runs_by_id))
    if unknown:
        raise ReportError(f"unknown GPU comparison override ids: {unknown}")
    for run_id, paths in overrides.items():
        run = runs_by_id[run_id]
        result_path = paths.get("result_path")
        trace_path = paths.get("trace_path")
        if result_path is None or trace_path is None:
            raise ReportError(
                f"GPU comparison override {run_id} requires result_path and trace_path"
            )
        label, chart_label = GPU_COMPARISON_OVERRIDE_LABELS[run_id]
        run["result_path"] = str(Path(result_path).resolve())
        run["trace_path"] = str(Path(trace_path).resolve())
        run["label"] = label
        run["chart_label"] = chart_label
    return updated


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ReportError(f"source artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"invalid JSON in {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(results_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else results_root / path


def source_name(results_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(results_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def nested_get(value: Any, dotted_key: str, default: Any = None) -> Any:
    current = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def number(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{context} must be numeric; got {value!r}") from exc
    if not math.isfinite(result):
        raise ReportError(f"{context} must be finite; got {value!r}")
    return result


def integer(value: Any, context: str) -> int:
    result = number(value, context)
    if not result.is_integer():
        raise ReportError(f"{context} must be an integer; got {value!r}")
    return int(result)


def select_json_record(payload: Any, path: Path) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list):
        raise ReportError(f"expected a JSON object or list in {path}")
    records = [item for item in payload if isinstance(item, dict)]
    complete = [item for item in records if item.get("status") == "complete"]
    if complete:
        records = complete
    if len(records) != 1:
        raise ReportError(
            f"expected exactly one eligible record in {path}; found {len(records)}"
        )
    return records[0]


def infer_length_for(record: dict[str, Any], spec: dict[str, Any]) -> int:
    candidates = [
        record.get("infer_length"),
        nested_get(record, "config.infer_length"),
        nested_get(record, "workers.0.infer_length"),
        spec.get("infer_length"),
    ]
    workers = record.get("workers")
    if isinstance(workers, list) and workers and isinstance(workers[0], dict):
        candidates.insert(2, workers[0].get("infer_length"))
    for candidate in candidates:
        if candidate is not None:
            return integer(candidate, "infer_length")
    raise ReportError(f"no infer_length in source or manifest entry {spec.get('path')}")


def validate_workload(
    record: dict[str, Any],
    spec: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    kind = str(spec["workload_kind"])
    artifact_unique = record.get("unique_image_count")
    artifact_replay = record.get("replay_count")
    artifact_timed = record.get("timed_workload_image_count")
    manifest_unique = spec.get("unique_image_count")
    manifest_replay = spec.get("replay_count")
    manifest_timed = spec.get("timed_workload_image_count")
    unique = artifact_unique if artifact_unique is not None else manifest_unique
    replay = artifact_replay if artifact_replay is not None else manifest_replay
    timed = (
        artifact_timed
        if artifact_timed is not None
        else manifest_timed
        if manifest_timed is not None
        else count
    )
    metadata: dict[str, Any] = {"metadata_sources": {}}
    if unique is not None:
        unique = integer(unique, "unique_image_count")
        metadata["unique_image_count"] = unique
        metadata["metadata_sources"]["unique_image_count"] = (
            "artifact" if artifact_unique is not None else "manifest"
        )
    if replay is not None:
        replay = integer(replay, "replay_count")
        metadata["replay_count"] = replay
        metadata["metadata_sources"]["replay_count"] = (
            "artifact" if artifact_replay is not None else "manifest"
        )
    if timed is not None:
        timed = integer(timed, "timed_workload_image_count")
        metadata["timed_workload_image_count"] = timed
        metadata["metadata_sources"]["timed_workload_image_count"] = (
            "artifact"
            if artifact_timed is not None
            else "manifest"
            if manifest_timed is not None
            else "count_field"
        )

    for name, artifact_value, manifest_value in (
        ("unique_image_count", artifact_unique, manifest_unique),
        ("replay_count", artifact_replay, manifest_replay),
        ("timed_workload_image_count", artifact_timed, manifest_timed),
    ):
        if artifact_value is not None and manifest_value is not None:
            if integer(artifact_value, name) != integer(manifest_value, name):
                raise ReportError(
                    f"artifact {name} disagrees with curated manifest: {spec['path']}"
                )

    if kind == "sustained_replay":
        if replay is None or replay <= 1:
            raise ReportError(
                f"sustained replay source must report replay_count > 1: {spec['path']}"
            )
        if unique is None or timed <= unique:
            raise ReportError(
                f"sustained replay source must report timed workload > unique images: "
                f"{spec['path']}"
            )
    elif kind.startswith("unique_"):
        if replay is not None and replay != 1:
            raise ReportError(
                f"unique workload unexpectedly reports replay_count={replay}: {spec['path']}"
            )
        if unique is not None and timed != unique:
            raise ReportError(
                f"unique workload timed count ({timed}) differs from unique count "
                f"({unique}): {spec['path']}"
            )
    else:
        raise ReportError(f"unknown workload_kind {kind!r}")
    return metadata


def load_point(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> Point:
    path = resolve_source(results_root, spec["path"])
    record = select_json_record(read_json(path), path)
    if record.get("status") not in (None, "complete"):
        raise ReportError(f"source is not complete: {path}")
    if integer(record.get("failed", 0), "failed") != 0:
        raise ReportError(f"source reports failed requests: {path}")

    infer_length = infer_length_for(record, spec)
    if infer_length != required_infer_length:
        raise ReportError(
            f"infer_length={infer_length} is ineligible; required "
            f"{required_infer_length}: {path}"
        )

    count_field = spec.get("count_field", "count")
    raw_count = nested_get(record, count_field)
    if raw_count is None:
        raw_count = record.get("timed_workload_image_count")
    if raw_count is None:
        raise ReportError(f"missing count field {count_field!r}: {path}")
    count = integer(raw_count, f"{count_field} in {path}")
    expected_count = spec.get("expected_count")
    if expected_count is not None and count != int(expected_count):
        raise ReportError(
            f"expected {expected_count} completed images, found {count}: {path}"
        )

    throughput = number(nested_get(record, spec["metric"]), spec["metric"])
    if throughput <= 0:
        raise ReportError(f"throughput must be positive: {path}")
    metadata = validate_workload(record, spec, count)
    for key in (
        "replicas",
        "max_num_seqs",
        "renderer_workers",
        "detector_batch",
    ):
        if key in spec:
            metadata[key] = spec[key]
    config = record.get("config")
    if isinstance(config, dict):
        metadata["config"] = {
            key: config[key]
            for key in (
                "replicas",
                "max_num_seqs",
                "recognizer_chunk_size",
                "gpu_memory_utilization",
                "detector_max_batch_size",
            )
            if key in config
        }
    return Point(
        label=str(spec["label"]),
        throughput=throughput,
        count=count,
        infer_length=infer_length,
        workload_kind=str(spec["workload_kind"]),
        source=source_name(results_root, path),
        source_sha256=sha256(path),
        metadata=metadata,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise ReportError(f"source artifact not found: {path}") from exc


def eligible_csv_rows(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = resolve_source(results_root, spec["path"])
    expected_count = int(spec["expected_count"])
    fixed_concurrency = int(spec["fixed_concurrency"])
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        completed = integer(raw.get("completed"), f"completed in {path}")
        concurrency = integer(raw.get("max_concurrency"), f"max_concurrency in {path}")
        if completed != expected_count or concurrency != fixed_concurrency:
            continue
        parsed = dict(raw)
        overrides_raw = raw.get("hf_overrides")
        overrides: dict[str, Any] = {}
        if overrides_raw:
            try:
                candidate = ast.literal_eval(overrides_raw)
            except (SyntaxError, ValueError) as exc:
                raise ReportError(f"invalid hf_overrides in {path}") from exc
            if isinstance(candidate, dict):
                overrides = candidate
        inferred = overrides.get("nemotron_ocr_infer_length", spec.get("infer_length"))
        if integer(inferred, f"infer_length in {path}") != required_infer_length:
            continue
        parsed["_throughput"] = number(raw.get(spec["metric"]), spec["metric"])
        parsed["_max_num_seqs"] = integer(raw.get("max_num_seqs"), "max_num_seqs")
        parsed["_renderer_workers"] = integer(
            raw.get("renderer_num_workers"), "renderer_num_workers"
        )
        parsed["_overrides"] = overrides
        rows.append(parsed)
    if not rows:
        raise ReportError(f"no eligible completed rows in {path}")
    provenance = {"source": source_name(results_root, path), "sha256": sha256(path)}
    return rows, provenance


def unique_sorted_points(
    rows: Iterable[dict[str, Any]],
    x_key: str,
    context: str,
) -> list[tuple[int, float]]:
    points: dict[int, float] = {}
    for row in rows:
        x = int(row[x_key])
        if x in points:
            raise ReportError(f"duplicate {context} point at {x}")
        points[x] = float(row["_throughput"])
    if not points:
        raise ReportError(f"no points for {context}")
    return sorted(points.items())


def load_settings_sweep(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> dict[str, Any]:
    rows, provenance = eligible_csv_rows(results_root, spec, required_infer_length)
    seq_rows = [
        row
        for row in rows
        if row["_renderer_workers"] == 4
        and integer(row["max_concurrency"], "max_concurrency") == 128
    ]
    renderer_rows = [
        row
        for row in rows
        if row["_max_num_seqs"] == 64
        and integer(row["max_concurrency"], "max_concurrency") == 128
    ]
    return {
        "max_num_seqs": unique_sorted_points(seq_rows, "_max_num_seqs", "max_num_seqs"),
        "renderer_workers": unique_sorted_points(
            renderer_rows, "_renderer_workers", "renderer workers"
        ),
        "count": int(spec["expected_count"]),
        "infer_length": required_infer_length,
        "workload_kind": spec["workload_kind"],
        **provenance,
    }


def load_detector_sweep(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> dict[str, Any]:
    rows, provenance = eligible_csv_rows(results_root, spec, required_infer_length)
    detector_rows = []
    for row in rows:
        overrides = row["_overrides"]
        batch = overrides.get("nemotron_ocr_detector_max_batch_size")
        if batch is None:
            raise ReportError("detector sweep row is missing detector batch override")
        row["_detector_batch"] = integer(batch, "detector batch")
        detector_rows.append(row)
    return {
        "points": unique_sorted_points(
            detector_rows, "_detector_batch", "detector batch"
        ),
        "count": int(spec["expected_count"]),
        "infer_length": required_infer_length,
        "workload_kind": spec["workload_kind"],
        **provenance,
    }


def load_deployment_optimization_curve(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> dict[str, Any]:
    defaults = spec.get("source_defaults")
    if not isinstance(defaults, dict):
        raise ReportError("deployment optimization curve requires source_defaults")
    groups = []
    all_signatures = set()
    seen_ids: set[str] = set()
    for group_spec in spec.get("groups", []):
        group_runs = []
        expected_dispatcher = group_spec.get("artifact_dispatcher")
        for run_spec in group_spec.get("runs", []):
            run_id = str(run_spec["id"])
            if run_id in seen_ids:
                raise ReportError(f"duplicate deployment optimization run id: {run_id}")
            seen_ids.add(run_id)
            source_spec = {**defaults, **run_spec}
            point = load_point(results_root, source_spec, required_infer_length)
            source_path = resolve_source(results_root, source_spec["path"])
            record = select_json_record(read_json(source_path), source_path)
            artifact_dispatcher = record.get("dispatcher")
            if (
                expected_dispatcher is not None
                and artifact_dispatcher != expected_dispatcher
            ):
                raise ReportError(
                    f"dispatcher mismatch for {run_id}: expected {expected_dispatcher!r}, "
                    f"found {artifact_dispatcher!r}"
                )
            artifact_replicas = record.get("replicas")
            if artifact_replicas is not None and integer(
                artifact_replicas, "replicas"
            ) != int(run_spec["replicas"]):
                raise ReportError(f"replica count mismatch for {run_id}")
            signature = (
                point.count,
                point.infer_length,
                point.workload_kind,
                point.metadata.get("unique_image_count"),
                point.metadata.get("replay_count"),
            )
            all_signatures.add(signature)
            group_runs.append(
                {
                    "id": run_id,
                    "label": run_spec["label"],
                    "throughput": point.throughput,
                    "count": point.count,
                    "infer_length": point.infer_length,
                    "workload_kind": point.workload_kind,
                    "workload_metadata": point.metadata,
                    "source": point.source,
                    "source_sha256": point.source_sha256,
                    "replicas": int(run_spec["replicas"]),
                    "recognizer_chunk_size": int(run_spec["recognizer_chunk_size"]),
                    "mps": str(run_spec["mps"]),
                    "access_log": str(run_spec["access_log"]),
                    "cudnn_benchmark": str(run_spec["cudnn_benchmark"]),
                }
            )
        if not group_runs:
            raise ReportError(
                f"deployment optimization group {group_spec.get('id')!r} is empty"
            )
        reference_id = str(group_spec["reference_id"])
        reference = next((run for run in group_runs if run["id"] == reference_id), None)
        if reference is None:
            raise ReportError(
                f"deployment optimization reference {reference_id!r} not in group"
            )
        for run in group_runs:
            run["delta_vs_group_reference_images_per_second"] = (
                run["throughput"] - reference["throughput"]
            )
            run["delta_vs_group_reference_pct"] = (
                run["throughput"] / reference["throughput"] - 1
            ) * 100
        groups.append(
            {
                "id": group_spec["id"],
                "label": group_spec["label"],
                "reference_id": reference_id,
                "artifact_dispatcher": expected_dispatcher,
                "runs": group_runs,
            }
        )
    if not groups:
        raise ReportError("deployment optimization curve has no groups")
    if len(all_signatures) != 1:
        raise ReportError(
            "deployment optimization runs do not share count, infer_length, workload, "
            "unique-image count, and replay count"
        )
    all_runs = [run for group in groups for run in group["runs"]]
    best = max(all_runs, key=lambda run: run["throughput"])
    return {
        "comparison_contract": spec["comparison_contract"],
        "groups": groups,
        "best_run_id": best["id"],
        "best_throughput": best["throughput"],
    }


def load_optimized_30k_variants(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> dict[str, Any]:
    defaults = spec.get("source_defaults")
    if not isinstance(defaults, dict):
        raise ReportError("optimized_30k_variants requires source_defaults")
    runs = []
    datasets = set()
    models = set()
    for run_spec in spec.get("runs", []):
        source_spec = {**defaults, **run_spec}
        point = load_point(results_root, source_spec, required_infer_length)
        result_path = resolve_source(results_root, source_spec["path"])
        record = select_json_record(read_json(result_path), result_path)
        if record.get("dispatcher") != "work_conserving_client_side_queue":
            raise ReportError(f"unexpected dispatcher in {result_path}")
        if integer(record.get("replicas"), "replicas") != 8:
            raise ReportError(
                f"optimized 30K profile must use eight replicas: {result_path}"
            )
        trace_summary = record.get("gpu_trace")
        if not isinstance(trace_summary, dict):
            raise ReportError(f"missing gpu_trace summary: {result_path}")
        trace_path = resolve_source(results_root, run_spec["trace_path"])
        if not trace_path.is_file():
            raise ReportError(f"optimized 30K GPU trace not found: {trace_path}")
        recorded_trace_path = record.get("gpu_trace_csv")
        if (
            recorded_trace_path
            and Path(recorded_trace_path).resolve() != trace_path.resolve()
        ):
            raise ReportError(f"GPU trace path mismatch in {result_path}")
        datasets.add(record.get("dataset"))
        models.add(record.get("model"))
        runs.append(
            {
                "id": run_spec["id"],
                "label": run_spec["label"],
                "role": run_spec["role"],
                "recognizer_chunk_size": int(run_spec["recognizer_chunk_size"]),
                "throughput": point.throughput,
                "completed": point.count,
                "failed": integer(record.get("failed", 0), "failed"),
                "elapsed_seconds": number(record.get("elapsed_s"), "elapsed_s"),
                "gpu_utilization_pct_avg": number(
                    trace_summary.get("gpu_util_pct_avg"), "gpu_util_pct_avg"
                ),
                "gpu_power_w_avg": number(
                    trace_summary.get("gpu_power_w_avg"), "gpu_power_w_avg"
                ),
                "gpu_trace_samples": integer(
                    trace_summary.get("samples"), "gpu_trace samples"
                ),
                "result_source": point.source,
                "result_sha256": point.source_sha256,
                "trace_source": source_name(results_root, trace_path),
                "trace_sha256": sha256(trace_path),
                "workload_metadata": point.metadata,
            }
        )
    if [run["id"] for run in runs] != [
        "rec64_publication",
        "rec128_conservative",
    ]:
        raise ReportError(
            "optimized_30k_variants must contain rec64 publication then rec128 conservative"
        )
    if len(datasets) != 1 or None in datasets or len(models) != 1 or None in models:
        raise ReportError(
            "optimized 30K profiles must share one recorded dataset and model"
        )
    publication = runs[0]
    for run in runs:
        run["delta_vs_publication_images_per_second"] = (
            run["throughput"] - publication["throughput"]
        )
        run["delta_vs_publication_pct"] = (
            run["throughput"] / publication["throughput"] - 1
        ) * 100
    return {
        "comparison_contract": spec["comparison_contract"],
        "dataset": next(iter(datasets)),
        "model": next(iter(models)),
        "runs": runs,
    }


def load_nondeterminism_comparisons(
    results_root: Path,
    specs: Sequence[dict[str, Any]],
    required_infer_length: int,
) -> dict[str, Any]:
    comparisons = []
    for spec in specs:
        if int(spec["infer_length"]) != required_infer_length:
            raise ReportError(f"ineligible nondeterminism comparison: {spec['path']}")
        path = resolve_source(results_root, spec["path"])
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ReportError(f"comparison must be a JSON object: {path}")
        comparisons.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "comparison_type": spec["comparison_type"],
                "passed": bool(payload.get("passed")),
                "reference_images": integer(
                    payload.get("reference_images"), "reference_images"
                ),
                "candidate_images": integer(
                    payload.get("candidate_images"), "candidate_images"
                ),
                "region_count_mismatches": integer(
                    payload.get("region_count_mismatches"),
                    "region_count_mismatches",
                ),
                "paired_regions": integer(
                    payload.get("paired_regions"), "paired_regions"
                ),
                "text_mismatches": integer(
                    payload.get("text_mismatches"), "text_mismatches"
                ),
                "text_exact_rate": number(
                    payload.get("text_exact_rate"), "text_exact_rate"
                ),
                "max_coordinate_abs_error": number(
                    payload.get("max_coordinate_abs_error"),
                    "max_coordinate_abs_error",
                ),
                "max_confidence_abs_error": number(
                    payload.get("max_confidence_abs_error"),
                    "max_confidence_abs_error",
                ),
                "source": source_name(results_root, path),
                "sha256": sha256(path),
            }
        )
    same_config_count = sum(
        item["comparison_type"] == "same_config_repeat" for item in comparisons
    )
    batching_count = sum(
        item["comparison_type"] == "batching_variant" for item in comparisons
    )
    if same_config_count != 2 or batching_count != 2:
        raise ReportError(
            "nondeterminism table requires two same-config and two batching comparisons"
        )
    return {
        "comparisons": comparisons,
        "interpretation_scope": (
            "Mismatch documentation only. Region sequence shifts make raw paired-region "
            "coordinate/confidence maxima unsuitable as quality evidence."
        ),
    }


def sequence_levenshtein(first: Sequence[str], second: Sequence[str]) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_value in enumerate(first, 1):
        current = [first_index]
        for second_index, second_value in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_value != second_value),
                )
            )
        previous = current
    return previous[-1]


def load_sequence_aware_audit(
    results_root: Path,
    spec: dict[str, Any],
    required_infer_length: int,
) -> dict[str, Any]:
    if int(spec["infer_length"]) != required_infer_length:
        raise ReportError("sequence-aware audit uses an ineligible infer_length")
    profiles: dict[str, dict[str, Any]] = {}
    for profile_spec in spec.get("profiles", []):
        profile_id = str(profile_spec["id"])
        path = resolve_source(results_root, profile_spec["path"])
        payload = read_json(path)
        if not isinstance(payload, list) or not payload:
            raise ReportError(
                f"prediction audit source must be a non-empty list: {path}"
            )
        pages: dict[int, list[str]] = {}
        for page in payload:
            if not isinstance(page, dict) or not isinstance(page.get("regions"), list):
                raise ReportError(f"malformed prediction page in {path}")
            image_index = integer(page.get("image_index"), "image_index")
            if image_index in pages:
                raise ReportError(f"duplicate image_index {image_index} in {path}")
            texts = []
            for region in page["regions"]:
                if not isinstance(region, dict) or not isinstance(
                    region.get("text"), str
                ):
                    raise ReportError(f"malformed prediction region in {path}")
                texts.append(region["text"])
            pages[image_index] = texts
        profiles[profile_id] = {
            "id": profile_id,
            "label": profile_spec["label"],
            "pages": pages,
            "source": source_name(results_root, path),
            "sha256": sha256(path),
        }
    if list(profiles) != ["A", "B", "C"]:
        raise ReportError("sequence-aware audit profiles must be ordered A, B, C")
    page_indexes = [set(profile["pages"]) for profile in profiles.values()]
    if any(indexes != page_indexes[0] for indexes in page_indexes[1:]):
        raise ReportError("sequence-aware audit profiles do not share image indexes")
    ordered_indexes = sorted(page_indexes[0])
    pairs = []
    for first_id, second_id in (("A", "B"), ("A", "C"), ("B", "C")):
        distances = [
            sequence_levenshtein(
                profiles[first_id]["pages"][index],
                profiles[second_id]["pages"][index],
            )
            for index in ordered_indexes
        ]
        pairs.append(
            {
                "pair": f"{first_id}-{second_id}",
                "first_label": profiles[first_id]["label"],
                "second_label": profiles[second_id]["label"],
                "page_text_sequence_levenshtein_total": sum(distances),
                "exact_page_sequences": sum(distance == 0 for distance in distances),
                "page_count": len(distances),
            }
        )

    agreement = {
        "all_same": 0,
        "A_equals_B_not_C": 0,
        "B_equals_C_not_A": 0,
        "A_equals_C_not_B": 0,
        "all_different": 0,
    }
    aligned_pages = 0
    for index in ordered_indexes:
        first = profiles["A"]["pages"][index]
        second = profiles["B"]["pages"][index]
        third = profiles["C"]["pages"][index]
        if not (len(first) == len(second) == len(third)):
            continue
        aligned_pages += 1
        for first_text, second_text, third_text in zip(first, second, third):
            if first_text == second_text == third_text:
                agreement["all_same"] += 1
            elif first_text == second_text:
                agreement["A_equals_B_not_C"] += 1
            elif second_text == third_text:
                agreement["B_equals_C_not_A"] += 1
            elif first_text == third_text:
                agreement["A_equals_C_not_B"] += 1
            else:
                agreement["all_different"] += 1
    return {
        "profiles": [
            {key: value for key, value in profile.items() if key not in {"pages"}}
            for profile in profiles.values()
        ],
        "pairs": pairs,
        "aligned_equal_length_page_count": aligned_pages,
        "total_page_count": len(ordered_indexes),
        "aligned_region_count": sum(agreement.values()),
        "aligned_region_agreement": agreement,
        "method": (
            "Levenshtein distance over each page's ordered region-text sequence; "
            "triple agreement is position-wise only on pages where A, B, and C have "
            "equal region counts."
        ),
    }


def load_hf_output_agreement(
    results_root: Path,
    specs: Sequence[dict[str, Any]],
    required_infer_length: int,
) -> dict[str, Any]:
    comparisons = []
    references = set()
    for spec in specs:
        if int(spec["infer_length"]) != required_infer_length:
            raise ReportError(f"ineligible HF agreement comparison: {spec['path']}")
        path = resolve_source(results_root, spec["path"])
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ReportError(f"HF agreement comparison must be an object: {path}")
        references.add(payload.get("reference"))
        comparisons.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "candidate": spec["candidate"],
                "passed": bool(payload.get("passed")),
                "reference_images": integer(
                    payload.get("reference_images"), "reference_images"
                ),
                "candidate_images": integer(
                    payload.get("candidate_images"), "candidate_images"
                ),
                "region_count_mismatches": integer(
                    payload.get("region_count_mismatches"),
                    "region_count_mismatches",
                ),
                "paired_regions": integer(
                    payload.get("paired_regions"), "paired_regions"
                ),
                "text_mismatches": integer(
                    payload.get("text_mismatches"), "text_mismatches"
                ),
                "zipped_text_exact_rate": number(
                    payload.get("text_exact_rate"), "text_exact_rate"
                ),
                "max_coordinate_abs_error": number(
                    payload.get("max_coordinate_abs_error"),
                    "max_coordinate_abs_error",
                ),
                "max_confidence_abs_error": number(
                    payload.get("max_confidence_abs_error"),
                    "max_confidence_abs_error",
                ),
                "source": source_name(results_root, path),
                "sha256": sha256(path),
            }
        )
    expected_ids = ["hf_vs_vllm_baseline", "hf_vs_rec128", "hf_vs_rec64"]
    if [item["id"] for item in comparisons] != expected_ids:
        raise ReportError(f"HF agreement comparisons must be ordered {expected_ids}")
    if len(references) != 1 or None in references:
        raise ReportError(
            "HF agreement comparisons must share one recorded reference run"
        )
    return {
        "comparisons": comparisons,
        "reference_prediction_path": next(iter(references)),
        "interpretation_scope": (
            "Agreement with one stochastic official-HF output run; no ground-truth labels, "
            "and positional zipping is sensitive to region insertions/deletions."
        ),
    }


def load_accuracy_checks(
    results_root: Path,
    specs: Sequence[dict[str, Any]],
    required_infer_length: int,
) -> list[dict[str, Any]]:
    checks = []
    for spec in specs:
        if int(spec["infer_length"]) != required_infer_length:
            raise ReportError(f"ineligible accuracy check: {spec['path']}")
        path = resolve_source(results_root, spec["path"])
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ReportError(f"accuracy comparison must be a JSON object: {path}")
        checks.append(
            {
                "label": spec["label"],
                "role": spec["role"],
                "passed": bool(payload.get("passed")),
                "reference_images": integer(
                    payload.get("reference_images"), "reference_images"
                ),
                "candidate_images": integer(
                    payload.get("candidate_images"), "candidate_images"
                ),
                "region_count_mismatches": integer(
                    payload.get("region_count_mismatches"),
                    "region_count_mismatches",
                ),
                "text_mismatches": integer(
                    payload.get("text_mismatches"), "text_mismatches"
                ),
                "text_exact_rate": number(
                    payload.get("text_exact_rate"), "text_exact_rate"
                ),
                "max_coordinate_abs_error": number(
                    payload.get("max_coordinate_abs_error"),
                    "max_coordinate_abs_error",
                ),
                "max_confidence_abs_error": number(
                    payload.get("max_confidence_abs_error"),
                    "max_confidence_abs_error",
                ),
                "source": source_name(results_root, path),
                "sha256": sha256(path),
            }
        )
    return checks


def parse_trace_timestamp(value: str) -> float | None:
    value = value.strip()
    for fmt in (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    return None


def parse_trace_metric(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "N/A", "[Not Supported]", "[N/A]"}:
        return None
    try:
        parsed = float(stripped.split()[0])
    except (IndexError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_gpu_trace(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReportError(f"GPU trace not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ReportError(f"GPU trace has no CSV header: {path}")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        samples_by_timestamp: dict[float, dict[str, Any]] = {}
        for raw in reader:
            row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in raw.items()
                if key is not None
            }
            timestamp_label = str(row.get("timestamp", ""))
            timestamp_s = parse_trace_timestamp(timestamp_label)
            gpu_util = parse_trace_metric(row.get("utilization.gpu [%]"))
            power_w = parse_trace_metric(row.get("power.draw [W]"))
            memory_used_mib = parse_trace_metric(row.get("memory.used [MiB]"))
            if (
                timestamp_s is None
                or gpu_util is None
                or power_w is None
                or memory_used_mib is None
            ):
                continue
            samples_by_timestamp[timestamp_s] = {
                "timestamp_s": timestamp_s,
                "timestamp": timestamp_label,
                "gpu_utilization_pct": gpu_util,
                "power_w": power_w,
                "memory_used_mib": memory_used_mib,
            }
    samples = [samples_by_timestamp[key] for key in sorted(samples_by_timestamp)]
    if len(samples) < 3:
        raise ReportError(
            "GPU trace must contain at least three timestamp/utilization/power/"
            f"memory rows: {path}"
        )
    return samples


def first_sustained_active_index(
    samples: Sequence[dict[str, Any]],
    threshold_pct: float,
    detection_window_seconds: float,
    minimum_active_fraction: float,
) -> int:
    """Return the first sample starting a sustained forward-looking active window."""

    end = 0
    for start, sample in enumerate(samples):
        if float(sample["gpu_utilization_pct"]) < threshold_pct:
            continue
        target = float(sample["timestamp_s"]) + detection_window_seconds
        end = max(end, start)
        while end < len(samples) and float(samples[end]["timestamp_s"]) < target:
            end += 1
        if end >= len(samples):
            break
        window = samples[start : end + 1]
        active_count = sum(
            float(item["gpu_utilization_pct"]) >= threshold_pct for item in window
        )
        fraction = active_count / len(window)
        mean_util = sum(float(item["gpu_utilization_pct"]) for item in window) / len(
            window
        )
        if fraction >= minimum_active_fraction and mean_util >= threshold_pct:
            return start
    raise ReportError(
        "no sustained GPU-active point found using "
        f">={threshold_pct:g}% utilization, a {detection_window_seconds:g}s window, "
        f"and active fraction >={minimum_active_fraction:g}"
    )


def trailing_rolling_trace(
    samples: Sequence[dict[str, Any]],
    active_start_index: int,
    rolling_window_seconds: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    active_samples = samples[active_start_index:]
    active_start_s = float(active_samples[0]["timestamp_s"])
    left = 0
    util_sum = 0.0
    power_sum = 0.0
    memory_sum = 0.0
    series: list[dict[str, float]] = []
    for index, sample in enumerate(active_samples):
        timestamp_s = float(sample["timestamp_s"])
        util_sum += float(sample["gpu_utilization_pct"])
        power_sum += float(sample["power_w"])
        memory_sum += float(sample["memory_used_mib"])
        cutoff = timestamp_s - rolling_window_seconds
        while left < index and float(active_samples[left]["timestamp_s"]) < cutoff:
            util_sum -= float(active_samples[left]["gpu_utilization_pct"])
            power_sum -= float(active_samples[left]["power_w"])
            memory_sum -= float(active_samples[left]["memory_used_mib"])
            left += 1
        sample_count = index - left + 1
        series.append(
            {
                "time_s": round(timestamp_s - active_start_s, 6),
                "gpu_utilization_pct": round(util_sum / sample_count, 6),
                "power_w": round(power_sum / sample_count, 6),
                "memory_used_mib": round(memory_sum / sample_count, 6),
            }
        )
    raw_util = [float(sample["gpu_utilization_pct"]) for sample in active_samples]
    raw_power = [float(sample["power_w"]) for sample in active_samples]
    raw_memory = [float(sample["memory_used_mib"]) for sample in active_samples]
    alignment = {
        "raw_sample_count": len(samples),
        "aligned_sample_count": len(active_samples),
        "active_start_timestamp": active_samples[0]["timestamp"],
        "leading_trim_seconds": round(
            active_start_s - float(samples[0]["timestamp_s"]), 6
        ),
        "aligned_duration_seconds": series[-1]["time_s"],
        "active_raw_gpu_utilization_pct_mean": sum(raw_util) / len(raw_util),
        "active_raw_gpu_utilization_pct_max": max(raw_util),
        "active_raw_power_w_mean": sum(raw_power) / len(raw_power),
        "active_raw_power_w_max": max(raw_power),
        "active_raw_memory_used_mib_mean": sum(raw_memory) / len(raw_memory),
        "active_raw_memory_used_mib_max": max(raw_memory),
    }
    return series, alignment


def load_gpu_active_comparison(
    results_root: Path,
    spec: dict[str, Any] | None,
    required_infer_length: int,
    rolling_window_override: float | None,
) -> dict[str, Any]:
    if spec is None:
        return {
            "available": False,
            "missing_inputs": ["gpu_active_comparison manifest section"],
        }
    rolling_window_seconds = (
        rolling_window_override
        if rolling_window_override is not None
        else number(spec.get("rolling_window_seconds", 15.0), "rolling window")
    )
    detection = spec.get("active_detection", {})
    threshold_pct = number(
        detection.get("gpu_utilization_threshold_pct", 20.0),
        "GPU active threshold",
    )
    detection_window_seconds = number(
        detection.get("window_seconds", 5.0), "active detection window"
    )
    minimum_active_fraction = number(
        detection.get("minimum_active_sample_fraction", 0.6),
        "minimum active sample fraction",
    )
    max_plot_points = integer(
        spec.get("max_plot_points_per_run", 1200), "max plot points per run"
    )
    if rolling_window_seconds <= 0 or detection_window_seconds <= 0:
        raise ReportError("GPU rolling and active-detection windows must be positive")
    if not 0 <= threshold_pct <= 100:
        raise ReportError("GPU active utilization threshold must be in [0, 100]")
    if not 0 < minimum_active_fraction <= 1:
        raise ReportError("minimum active sample fraction must be in (0, 1]")
    if max_plot_points < 2:
        raise ReportError("max plot points per run must be at least two")

    run_specs = spec.get("runs")
    if not isinstance(run_specs, list):
        raise ReportError("gpu_active_comparison.runs must be a list")
    expected_ids = ["official_hf", "vllm_baseline", "optimized_native_vllm"]
    actual_ids = [run.get("id") for run in run_specs if isinstance(run, dict)]
    if actual_ids != expected_ids:
        raise ReportError(
            f"GPU comparison runs must be ordered as {expected_ids}; got {actual_ids}"
        )

    missing_inputs: list[str] = []
    input_status = []
    for run in run_specs:
        missing_fields = []
        status = {"id": run["id"], "label": run["label"]}
        for field_name in ("result_path", "trace_path"):
            configured = run.get(field_name)
            status[field_name] = configured
            status[f"{field_name}_exists"] = bool(
                configured and resolve_source(results_root, configured).is_file()
            )
            if not configured:
                missing_fields.append(field_name)
            elif not resolve_source(results_root, configured).is_file():
                missing_fields.append(f"{field_name} (file not found)")
        input_status.append(status)
        if missing_fields:
            missing_inputs.append(f"{run['label']}: {', '.join(missing_fields)}")

    algorithm = {
        "comparison_contract": spec.get(
            "comparison_contract",
            "All runs must use the same model, input corpus, and timed workload.",
        ),
        "gpu_utilization_threshold_pct": threshold_pct,
        "active_detection_window_seconds": detection_window_seconds,
        "minimum_active_sample_fraction": minimum_active_fraction,
        "rolling_window_seconds": rolling_window_seconds,
        "rolling_mean_direction": "trailing",
        "alignment": "each run's first sustained GPU-active sample becomes t=0",
        "trailing_trace_policy": "retain all samples through raw trace end",
        "max_plot_points_per_run": max_plot_points,
    }
    chart_context = spec.get("chart_context", {})
    if missing_inputs:
        return {
            "available": False,
            "missing_inputs": missing_inputs,
            "algorithm": algorithm,
            "publication_methodology": spec.get("publication_methodology", {}),
            "chart_context": chart_context,
            "input_status": input_status,
            "runs": [],
        }

    runs = []
    for run_spec in run_specs:
        point_spec = dict(run_spec)
        point_spec["path"] = run_spec["result_path"]
        point = load_point(results_root, point_spec, required_infer_length)
        result_path = resolve_source(results_root, run_spec["result_path"])
        result_record = select_json_record(read_json(result_path), result_path)
        trace_path = resolve_source(results_root, run_spec["trace_path"])
        recorded_trace_path = result_record.get("gpu_trace_csv")
        if (
            recorded_trace_path
            and Path(recorded_trace_path).resolve() != trace_path.resolve()
        ):
            raise ReportError(
                f"{run_spec['label']}: result JSON records GPU trace "
                f"{recorded_trace_path}, but comparison selected {trace_path}"
            )
        if (
            result_record.get("failed") is not None
            and integer(result_record["failed"], "failed") != 0
        ):
            raise ReportError(
                f"{run_spec['label']}: comparison result contains failed requests"
            )
        if run_spec["id"] != "official_hf" and result_record.get("failed") is None:
            raise ReportError(
                f"{run_spec['label']}: native vLLM result must record failed=0"
            )
        samples = read_gpu_trace(trace_path)
        try:
            active_start_index = first_sustained_active_index(
                samples,
                threshold_pct,
                detection_window_seconds,
                minimum_active_fraction,
            )
        except ReportError as exc:
            raise ReportError(f"{run_spec['label']}: {exc}") from exc
        series, alignment = trailing_rolling_trace(
            samples, active_start_index, rolling_window_seconds
        )
        runs.append(
            {
                "id": run_spec["id"],
                "label": run_spec["label"],
                "chart_label": run_spec.get("chart_label", run_spec["label"]),
                "workflow": run_spec.get("workflow", ""),
                "note": run_spec.get("note", ""),
                "throughput": point.throughput,
                "count": point.count,
                "infer_length": point.infer_length,
                "workload_kind": point.workload_kind,
                "workload_metadata": point.metadata,
                "result_source": point.source,
                "result_sha256": point.source_sha256,
                "trace_source": source_name(results_root, trace_path),
                "trace_sha256": sha256(trace_path),
                "failed": (
                    integer(result_record["failed"], "failed")
                    if result_record.get("failed") is not None
                    else None
                ),
                "alignment": alignment,
                "series": series,
                "artifact_configuration": {
                    key: result_record[key]
                    for key in (
                        "backend",
                        "replicas",
                        "total_concurrency",
                        "concurrency_per_endpoint",
                        "batch_size",
                        "detector_max_batch_size",
                        "recognizer_chunk_size",
                        "relational_chunk_size",
                    )
                    if key in result_record
                },
            }
        )

    signatures = {
        (
            run["count"],
            run["infer_length"],
            run["workload_kind"],
            run["workload_metadata"].get("unique_image_count"),
            run["workload_metadata"].get("replay_count"),
        )
        for run in runs
    }
    if len(signatures) != 1:
        raise ReportError(
            "GPU comparison runs do not share count, infer_length, workload kind, "
            "unique-image count, and replay count"
        )
    official_throughput = runs[0]["throughput"]
    clean_vllm_throughput = runs[1]["throughput"]
    for run in runs:
        run["speedup_vs_official_hf"] = run["throughput"] / official_throughput
        run["speedup_vs_clean_vllm"] = run["throughput"] / clean_vllm_throughput
    return {
        "available": True,
        "missing_inputs": [],
        "algorithm": algorithm,
        "publication_methodology": spec.get("publication_methodology", {}),
        "chart_context": chart_context,
        "input_status": input_status,
        "runs": runs,
    }


def collect_report_data(
    manifest: dict[str, Any],
    results_root: Path,
    hf_override: Path | None,
    hf_metric: str | None,
    gpu_rolling_window_seconds: float | None,
    strict: bool,
    gpu_comparison_overrides: dict[str, dict[str, Path]] | None = None,
) -> dict[str, Any]:
    required_infer_length = int(manifest["accuracy_contract"]["infer_length"])
    comparison = manifest["baseline_comparison"]
    baseline = load_point(results_root, comparison["baseline"], required_infer_length)
    optimized = load_point(results_root, comparison["optimized"], required_infer_length)

    hf_spec = dict(comparison["hf_baseline"])
    if hf_override is not None:
        hf_spec["path"] = str(hf_override.resolve())
    if hf_metric is not None:
        hf_spec["metric"] = hf_metric
    hf_baseline = None
    if hf_spec.get("path"):
        path = resolve_source(results_root, hf_spec["path"])
        if path.exists() or strict:
            hf_baseline = load_point(results_root, hf_spec, required_infer_length)

    native_scaling = [
        load_point(results_root, spec, required_infer_length)
        for spec in manifest["native_scaling"]
    ]
    native_scaling.sort(key=lambda point: int(point.metadata["replicas"]))
    expected_replicas = [1, 2, 4, 8]
    actual_replicas = [int(point.metadata["replicas"]) for point in native_scaling]
    if actual_replicas != expected_replicas:
        raise ReportError(
            f"native scaling must contain replicas {expected_replicas}; got {actual_replicas}"
        )

    sustained = None
    sustained_spec = manifest.get("sustained_replay")
    if sustained_spec and sustained_spec.get("path"):
        path = resolve_source(results_root, sustained_spec["path"])
        if path.exists():
            sustained = load_point(results_root, sustained_spec, required_infer_length)
        elif strict and not sustained_spec.get("optional", False):
            raise ReportError(f"required sustained source missing: {path}")

    settings = load_settings_sweep(
        results_root, manifest["settings_sweep"], required_infer_length
    )
    detector = load_detector_sweep(
        results_root, manifest["detector_sweep"], required_infer_length
    )
    deployment_curve = load_deployment_optimization_curve(
        results_root,
        manifest["deployment_optimization_curve"],
        required_infer_length,
    )
    optimized_30k_variants = load_optimized_30k_variants(
        results_root,
        manifest["optimized_30k_variants"],
        required_infer_length,
    )
    accuracy_checks = load_accuracy_checks(
        results_root, manifest["accuracy_checks"], required_infer_length
    )
    gpu_comparison_spec = apply_gpu_comparison_overrides(
        manifest.get("gpu_active_comparison"), gpu_comparison_overrides
    )
    gpu_active_comparison = load_gpu_active_comparison(
        results_root,
        gpu_comparison_spec,
        required_infer_length,
        gpu_rolling_window_seconds,
    )
    nondeterminism_comparisons = load_nondeterminism_comparisons(
        results_root,
        manifest["nondeterminism_comparisons"],
        required_infer_length,
    )
    sequence_aware_audit = load_sequence_aware_audit(
        results_root,
        manifest["sequence_aware_audit"],
        required_infer_length,
    )
    hf_output_agreement = load_hf_output_agreement(
        results_root,
        manifest["hf_output_agreement"],
        required_infer_length,
    )

    return {
        "schema_version": 1,
        "title": manifest["title"],
        "accuracy_contract": manifest["accuracy_contract"],
        "benchmark_dataset": manifest["benchmark_dataset"],
        "methodology": manifest["methodology"],
        "baseline": asdict(baseline),
        "optimized": asdict(optimized),
        "hf_baseline": asdict(hf_baseline) if hf_baseline else None,
        "native_scaling": [asdict(point) for point in native_scaling],
        "sustained_replay": asdict(sustained) if sustained else None,
        "settings_sweep": settings,
        "detector_sweep": detector,
        "deployment_optimization_curve": deployment_curve,
        "optimized_30k_variants": optimized_30k_variants,
        "accuracy_checks": accuracy_checks,
        "gpu_active_comparison": gpu_active_comparison,
        "nondeterminism_comparisons": nondeterminism_comparisons,
        "sequence_aware_audit": sequence_aware_audit,
        "hf_output_agreement": hf_output_agreement,
    }


def collect_comparison_report_data(
    manifest: dict[str, Any],
    results_root: Path,
    gpu_rolling_window_seconds: float | None,
    gpu_comparison_overrides: dict[str, dict[str, Path]] | None,
) -> dict[str, Any]:
    """Load only the matched three-system publication comparison.

    This intentionally bypasses legacy 1K sweeps and older optimized-30K
    variants, so an explicitly selected result can never be mixed with a stale
    throughput headline elsewhere in the generated output.
    """

    required_infer_length = int(manifest["accuracy_contract"]["infer_length"])
    spec = apply_gpu_comparison_overrides(
        manifest.get("gpu_active_comparison"), gpu_comparison_overrides
    )
    comparison = load_gpu_active_comparison(
        results_root,
        spec,
        required_infer_length,
        gpu_rolling_window_seconds,
    )
    if not comparison["available"]:
        raise ReportError(
            "matched comparison inputs are incomplete: "
            + "; ".join(comparison["missing_inputs"])
        )
    return {
        "schema_version": 2,
        "report_kind": "matched_gpu_active_comparison",
        "title": manifest["title"],
        "accuracy_contract": manifest["accuracy_contract"],
        "benchmark_dataset": manifest["benchmark_dataset"],
        "methodology": manifest["methodology"],
        "gpu_active_comparison": comparison,
    }


class Scene:
    """Shared primitive scene rendered to both SVG and antialiased PNG."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.width = width
        self.height = height
        self.commands: list[tuple[str, dict[str, Any]]] = []

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1,
        radius: float = 0,
    ) -> None:
        self.commands.append(
            (
                "rect",
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "fill": fill,
                    "stroke": stroke,
                    "stroke_width": stroke_width,
                    "radius": radius,
                },
            )
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        width: float = 2,
        dash: tuple[float, float] | None = None,
    ) -> None:
        self.commands.append(
            (
                "line",
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "color": color,
                    "width": width,
                    "dash": dash,
                },
            )
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: str,
        width: float = 3,
        fill: str = "none",
        dash: tuple[float, float] | None = None,
    ) -> None:
        self.commands.append(
            (
                "polyline",
                {
                    "points": list(points),
                    "color": color,
                    "width": width,
                    "fill": fill,
                    "dash": dash,
                },
            )
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1,
    ) -> None:
        self.commands.append(
            (
                "circle",
                {
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "fill": fill,
                    "stroke": stroke,
                    "stroke_width": stroke_width,
                },
            )
        )

    def polygon(
        self,
        points: Sequence[tuple[float, float]],
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1,
    ) -> None:
        self.commands.append(
            (
                "polygon",
                {
                    "points": list(points),
                    "fill": fill,
                    "stroke": stroke,
                    "stroke_width": stroke_width,
                },
            )
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 24,
        color: str = COLORS["ink"],
        anchor: str = "start",
        weight: str = "normal",
    ) -> None:
        self.commands.append(
            (
                "text",
                {
                    "x": x,
                    "y": y,
                    "text": str(text),
                    "size": size,
                    "color": color,
                    "anchor": anchor,
                    "weight": weight,
                },
            )
        )

    def save_svg(self, path: Path) -> None:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
            ),
            "<title>Nemotron OCR v2 benchmark chart</title>",
        ]
        for kind, args in self.commands:
            if kind == "rect":
                stroke = (
                    f' stroke="{args["stroke"]}" stroke-width="{args["stroke_width"]}"'
                    if args["stroke"]
                    else ""
                )
                lines.append(
                    f'<rect x="{args["x"]}" y="{args["y"]}" '
                    f'width="{args["width"]}" height="{args["height"]}" '
                    f'rx="{args["radius"]}" fill="{args["fill"]}"{stroke}/>'
                )
            elif kind == "line":
                dash = (
                    f' stroke-dasharray="{args["dash"][0]} {args["dash"][1]}"'
                    if args["dash"]
                    else ""
                )
                lines.append(
                    f'<line x1="{args["x1"]}" y1="{args["y1"]}" '
                    f'x2="{args["x2"]}" y2="{args["y2"]}" '
                    f'stroke="{args["color"]}" stroke-width="{args["width"]}"{dash}/>'
                )
            elif kind == "polyline":
                points = " ".join(f"{x},{y}" for x, y in args["points"])
                dash = (
                    f' stroke-dasharray="{args["dash"][0]} {args["dash"][1]}"'
                    if args["dash"]
                    else ""
                )
                lines.append(
                    f'<polyline points="{points}" fill="{args["fill"]}" '
                    f'stroke="{args["color"]}" stroke-width="{args["width"]}" '
                    f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
                )
            elif kind == "circle":
                stroke = (
                    f' stroke="{args["stroke"]}" stroke-width="{args["stroke_width"]}"'
                    if args["stroke"]
                    else ""
                )
                lines.append(
                    f'<circle cx="{args["x"]}" cy="{args["y"]}" '
                    f'r="{args["radius"]}" fill="{args["fill"]}"{stroke}/>'
                )
            elif kind == "polygon":
                points = " ".join(f"{x},{y}" for x, y in args["points"])
                stroke = (
                    f' stroke="{args["stroke"]}" stroke-width="{args["stroke_width"]}"'
                    if args["stroke"]
                    else ""
                )
                lines.append(
                    f'<polygon points="{points}" fill="{args["fill"]}"{stroke}/>'
                )
            elif kind == "text":
                anchor = {"start": "start", "middle": "middle", "end": "end"}[
                    args["anchor"]
                ]
                lines.append(
                    f'<text x="{args["x"]}" y="{args["y"]}" '
                    f'font-family="DejaVu Sans, sans-serif" font-size="{args["size"]}" '
                    f'font-weight="{args["weight"]}" fill="{args["color"]}" '
                    f'text-anchor="{anchor}">{escape(args["text"])}</text>'
                )
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n")

    def save_png(self, path: Path) -> None:
        scale = PNG_SCALE
        image = Image.new(
            "RGB", (self.width * scale, self.height * scale), COLORS["white"]
        )
        draw = ImageDraw.Draw(image)
        font_cache: dict[
            tuple[int, str], ImageFont.FreeTypeFont | ImageFont.ImageFont
        ] = {}

        def font(
            size: float, weight: str
        ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            key = (round(size * scale), weight)
            if key not in font_cache:
                candidate = FONT_BOLD if weight == "bold" else FONT_REGULAR
                if candidate.exists():
                    font_cache[key] = ImageFont.truetype(str(candidate), key[0])
                else:  # pragma: no cover - standard Linux images have DejaVu
                    font_cache[key] = ImageFont.load_default()
            return font_cache[key]

        def scaled_points(
            points: Sequence[tuple[float, float]],
        ) -> list[tuple[int, int]]:
            return [(round(x * scale), round(y * scale)) for x, y in points]

        def dashed_line(args: dict[str, Any]) -> None:
            x1, y1, x2, y2 = (
                args["x1"] * scale,
                args["y1"] * scale,
                args["x2"] * scale,
                args["y2"] * scale,
            )
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            dash, gap = (part * scale for part in args["dash"])
            cursor = 0.0
            while cursor < length:
                end = min(cursor + dash, length)
                start_fraction = cursor / length if length else 0
                end_fraction = end / length if length else 0
                draw.line(
                    (
                        x1 + dx * start_fraction,
                        y1 + dy * start_fraction,
                        x1 + dx * end_fraction,
                        y1 + dy * end_fraction,
                    ),
                    fill=args["color"],
                    width=max(1, round(args["width"] * scale)),
                )
                cursor += dash + gap

        for kind, args in self.commands:
            if kind == "rect":
                box = (
                    round(args["x"] * scale),
                    round(args["y"] * scale),
                    round((args["x"] + args["width"]) * scale),
                    round((args["y"] + args["height"]) * scale),
                )
                kwargs: dict[str, Any] = {"fill": args["fill"]}
                if args["stroke"]:
                    kwargs.update(
                        outline=args["stroke"],
                        width=max(1, round(args["stroke_width"] * scale)),
                    )
                if args["radius"]:
                    draw.rounded_rectangle(
                        box, radius=round(args["radius"] * scale), **kwargs
                    )
                else:
                    draw.rectangle(box, **kwargs)
            elif kind == "line":
                if args["dash"]:
                    dashed_line(args)
                else:
                    draw.line(
                        (
                            round(args["x1"] * scale),
                            round(args["y1"] * scale),
                            round(args["x2"] * scale),
                            round(args["y2"] * scale),
                        ),
                        fill=args["color"],
                        width=max(1, round(args["width"] * scale)),
                    )
            elif kind == "polyline":
                points = scaled_points(args["points"])
                if args["fill"] != "none":
                    draw.polygon(points, fill=args["fill"])
                if args["dash"]:
                    for first, second in zip(args["points"], args["points"][1:]):
                        dashed_line(
                            {
                                "x1": first[0],
                                "y1": first[1],
                                "x2": second[0],
                                "y2": second[1],
                                "color": args["color"],
                                "width": args["width"],
                                "dash": args["dash"],
                            }
                        )
                else:
                    draw.line(
                        points,
                        fill=args["color"],
                        width=max(1, round(args["width"] * scale)),
                        joint="curve",
                    )
            elif kind == "circle":
                box = (
                    round((args["x"] - args["radius"]) * scale),
                    round((args["y"] - args["radius"]) * scale),
                    round((args["x"] + args["radius"]) * scale),
                    round((args["y"] + args["radius"]) * scale),
                )
                draw.ellipse(
                    box,
                    fill=args["fill"],
                    outline=args["stroke"],
                    width=max(1, round(args["stroke_width"] * scale)),
                )
            elif kind == "polygon":
                draw.polygon(
                    scaled_points(args["points"]),
                    fill=args["fill"],
                    outline=args["stroke"],
                )
                if args["stroke"] and args["stroke_width"] > 1:
                    closed = list(args["points"]) + [args["points"][0]]
                    draw.line(
                        scaled_points(closed),
                        fill=args["stroke"],
                        width=round(args["stroke_width"] * scale),
                    )
            elif kind == "text":
                active_font = font(args["size"], args["weight"])
                bbox = draw.textbbox((0, 0), args["text"], font=active_font)
                text_width = bbox[2] - bbox[0]
                x = args["x"] * scale
                if args["anchor"] == "middle":
                    x -= text_width / 2
                elif args["anchor"] == "end":
                    x -= text_width
                y = args["y"] * scale - bbox[3]
                draw.text(
                    (round(x), round(y)),
                    args["text"],
                    fill=args["color"],
                    font=active_font,
                )
        image.resize((self.width, self.height), Image.Resampling.LANCZOS).save(
            path, optimize=True
        )


def chart_header(scene: Scene, title: str, subtitle: str) -> None:
    scene.rect(0, 0, scene.width, scene.height, COLORS["white"])
    scene.rect(0, 0, 18, scene.height, COLORS["blue"])
    scene.text(72, 82, title, 42, weight="bold")
    scene.text(72, 128, subtitle, 21, COLORS["muted"])


def draw_arrow(
    scene: Scene,
    points: Sequence[tuple[float, float]],
    color: str,
    width: float = 4,
    head_size: float = 13,
    dash: tuple[float, float] | None = None,
) -> None:
    """Draw a deterministic polyline arrow in both SVG and PNG output."""

    if len(points) < 2:
        raise ValueError("an arrow requires at least two points")
    scene.polyline(points, color, width, dash=dash)
    before_x, before_y = points[-2]
    tip_x, tip_y = points[-1]
    dx = tip_x - before_x
    dy = tip_y - before_y
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("an arrow's final segment must have non-zero length")
    unit_x, unit_y = dx / length, dy / length
    side_x, side_y = -unit_y, unit_x
    base_x = tip_x - unit_x * head_size
    base_y = tip_y - unit_y * head_size
    half_width = head_size * 0.52
    scene.polygon(
        [
            (tip_x, tip_y),
            (base_x + side_x * half_width, base_y + side_y * half_width),
            (base_x - side_x * half_width, base_y - side_y * half_width),
        ],
        color,
    )


def draw_rocket(scene: Scene, x: float, y: float, scale: float = 1.0) -> None:
    """Draw a small up-and-right vector rocket without font/emoji dependencies."""

    forward = (math.sqrt(0.5), -math.sqrt(0.5))
    side = (math.sqrt(0.5), math.sqrt(0.5))

    def point(along: float, across: float) -> tuple[float, float]:
        return (
            x + scale * (along * forward[0] + across * side[0]),
            y + scale * (along * forward[1] + across * side[1]),
        )

    # Flame is drawn first so it sits behind the rocket body.
    scene.polygon(
        [point(-17, -5), point(-38, 0), point(-17, 5)],
        COLORS["orange"],
        COLORS["red"],
        2,
    )
    scene.polygon(
        [point(-18, -2), point(-30, 0), point(-18, 2)],
        "#FFE08A",
    )
    scene.polygon(
        [point(-8, 8), point(-20, 19), point(-17, 6)],
        COLORS["blue"],
        COLORS["ink"],
        2,
    )
    scene.polygon(
        [point(-8, -8), point(-20, -19), point(-17, -6)],
        COLORS["blue"],
        COLORS["ink"],
        2,
    )
    scene.polygon(
        [
            point(30, 0),
            point(11, 9),
            point(-17, 9),
            point(-20, -9),
            point(11, -9),
        ],
        COLORS["white"],
        COLORS["ink"],
        2.5,
    )
    scene.polygon(
        [point(30, 0), point(11, 9), point(11, -9)],
        COLORS["teal"],
        COLORS["ink"],
        2,
    )
    window_x, window_y = point(1, 0)
    scene.circle(
        window_x,
        window_y,
        5.5 * scale,
        COLORS["light_blue"],
        COLORS["blue"],
        2,
    )


def nice_upper(value: float, step: float = 10.0) -> float:
    return max(step, math.ceil(value * 1.12 / step) * step)


def draw_axes(
    scene: Scene,
    bounds: tuple[float, float, float, float],
    y_min: float,
    y_max: float,
    ticks: int = 5,
    y_label: str = "Throughput (images/s)",
) -> Callable[[float], float]:
    x0, y0, x1, y1 = bounds
    scene.text(x0, y0 - 22, y_label, 18, COLORS["muted"], weight="bold")
    for index in range(ticks + 1):
        value = y_min + (y_max - y_min) * index / ticks
        y = y1 - (y1 - y0) * index / ticks
        scene.line(x0, y, x1, y, COLORS["grid"], 1)
        label = f"{value:.0f}" if abs(value - round(value)) < 0.01 else f"{value:.1f}"
        scene.text(x0 - 18, y + 7, label, 17, COLORS["muted"], anchor="end")
    scene.line(x0, y0, x0, y1, COLORS["ink"], 2)
    scene.line(x0, y1, x1, y1, COLORS["ink"], 2)

    def y_for(value: float) -> float:
        return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)

    return y_for


def save_chart(scene: Scene, output_dir: Path, stem: str) -> None:
    scene.save_svg(output_dir / f"{stem}.svg")
    scene.save_png(output_dir / f"{stem}.png")


def baseline_chart(data: dict[str, Any], output_dir: Path) -> None:
    baseline = data["baseline"]
    optimized = data["optimized"]
    hf = data["hf_baseline"]
    bars = []
    if hf:
        bars.append((hf, COLORS["orange"], "HF baseline"))
    bars.extend(
        [
            (baseline, COLORS["blue"], "vLLM baseline"),
            (optimized, COLORS["teal"], "vLLM optimized"),
        ]
    )
    scene = Scene()
    chart_header(
        scene,
        "Historical offline replica-harness throughput",
        "A100 · infer_length=1024 · 1K unique PNG pages · separate from matched 30K headline",
    )
    bounds = (170, 230, 1450, 700)
    y_max = nice_upper(max(item[0]["throughput"] for item in bars))
    y_for = draw_axes(scene, bounds, 0, y_max)
    x0, _, x1, y1 = bounds
    spacing = (x1 - x0) / len(bars)
    bar_width = min(250, spacing * 0.55)
    for index, (point, color, short_label) in enumerate(bars):
        x = x0 + spacing * (index + 0.5)
        y = y_for(point["throughput"])
        scene.rect(x - bar_width / 2, y, bar_width, y1 - y, color, radius=8)
        scene.text(
            x,
            y - 18,
            f"{point['throughput']:.2f}",
            29,
            COLORS["ink"],
            anchor="middle",
            weight="bold",
        )
        scene.text(x, y1 + 40, short_label, 20, anchor="middle", weight="bold")
        scene.text(
            x,
            y1 + 72,
            point["label"],
            17,
            COLORS["muted"],
            anchor="middle",
        )
    speedup = optimized["throughput"] / baseline["throughput"]
    scene.rect(1090, 160, 360, 66, COLORS["light_teal"], radius=14)
    scene.text(
        1270,
        202,
        f"{speedup:.2f}× vs vLLM baseline",
        23,
        anchor="middle",
        weight="bold",
    )
    if not hf:
        scene.text(
            170,
            820,
            "1K offline HF baseline pending — no 30K value is substituted.",
            18,
            COLORS["orange"],
            weight="bold",
        )
    scene.text(
        1450,
        820,
        "Steady-state metric; offline replica harness (PNG inputs)",
        17,
        COLORS["muted"],
        anchor="end",
    )
    save_chart(scene, output_dir, "baseline_vs_optimized")


def matched_30k_speedup_chart(data: dict[str, Any], output_dir: Path) -> None:
    comparison = data["gpu_active_comparison"]
    if not comparison["available"]:
        return
    runs = comparison["runs"]
    colors = [COLORS["orange"], COLORS["blue"], COLORS["teal"]]
    scene = Scene()
    chart_header(
        scene,
        "Matched 30K throughput and speedup",
        "A100 · JPEG bytes · infer_length=1024 · 30K = 1K unique × 30",
    )
    bounds = (165, 235, 1450, 700)
    y_max = nice_upper(max(run["throughput"] for run in runs))
    y_for = draw_axes(scene, bounds, 0, y_max)
    x0, _, x1, y1 = bounds
    spacing = (x1 - x0) / len(runs)
    bar_width = min(230, spacing * 0.55)
    hf_throughput = runs[0]["throughput"]
    baseline_throughput = runs[1]["throughput"]
    reference_y = y_for(hf_throughput)
    scene.line(x0, reference_y, x1, reference_y, COLORS["orange"], 2, dash=(8, 6))
    optimized_bar_left = x0 + spacing * 2.5 - bar_width / 2
    scene.text(
        optimized_bar_left - 12,
        reference_y - 10,
        "Official HF reference",
        15,
        COLORS["orange"],
        anchor="end",
        weight="bold",
    )
    centers = [x0 + spacing * (index + 0.5) for index in range(len(runs))]
    tops = [y_for(run["throughput"]) for run in runs]
    for index, (run, color) in enumerate(zip(runs, colors)):
        x = centers[index]
        y = tops[index]
        scene.rect(x - bar_width / 2, y, bar_width, y1 - y, color, radius=8)
        scene.text(
            x,
            y - 18,
            f"{run['throughput']:.2f}",
            28,
            anchor="middle",
            weight="bold",
        )
        scene.text(
            x,
            y1 + 40,
            run["chart_label"],
            18,
            anchor="middle",
            weight="bold",
        )
        scene.text(
            x,
            y1 + 72,
            f"{run['throughput'] / hf_throughput:.2f}× vs official HF",
            17,
            COLORS["muted"],
            anchor="middle",
        )
        if run["id"] == "optimized_native_vllm":
            scene.text(
                x,
                y1 + 102,
                f"{run['throughput'] / baseline_throughput:.2f}× vs clean vLLM",
                17,
                COLORS["teal"],
                anchor="middle",
                weight="bold",
            )
    arrow_gap = 14
    arrow_specs = [
        (
            (centers[0] + bar_width / 2 + arrow_gap, tops[0] + 20),
            (centers[1] - bar_width / 2 - arrow_gap, tops[1] + 20),
            COLORS["blue"],
            f"{runs[1]['throughput'] / runs[0]['throughput']:.2f}× · HF→clean vLLM",
        ),
        (
            (centers[1] + bar_width / 2 + arrow_gap, tops[1] + 20),
            (centers[2] - bar_width / 2 - arrow_gap, tops[2] + 28),
            COLORS["teal"],
            f"{runs[2]['throughput'] / runs[1]['throughput']:.2f}× · clean→optimized",
        ),
    ]
    for start, end, color, label in arrow_specs:
        draw_arrow(scene, [start, end], color, 5, 16)
        scene.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 - 13,
            label,
            15,
            color,
            anchor="middle",
            weight="bold",
        )
    draw_arrow(
        scene,
        [
            (centers[2] + bar_width / 2 + 12, tops[2] + 18),
            (1342, 273),
        ],
        COLORS["teal"],
        3,
        11,
        dash=(8, 7),
    )
    draw_rocket(scene, 1372, 244, 1.15)
    scene.text(
        165,
        840,
        "Matched workload and quality configuration; engine startup and warmup excluded from timed spans.",
        17,
        COLORS["muted"],
    )
    save_chart(scene, output_dir, "matched_30k_speedup")


def native_scaling_chart(data: dict[str, Any], output_dir: Path) -> None:
    points = data["native_scaling"]
    sustained = data["sustained_replay"]
    all_values = [point["throughput"] for point in points]
    if sustained:
        all_values.append(sustained["throughput"])
    scene = Scene()
    chart_header(
        scene,
        "Native vLLM deployment scaling",
        "A100 · infer_length=1024 · JPEG Q100 4:4:4 · 1K unique workload shown in blue",
    )
    bounds = (170, 230, 1450, 700)
    y_max = nice_upper(max(all_values))
    y_for = draw_axes(scene, bounds, 0, y_max)
    x0, _, x1, y1 = bounds
    xs = [x0 + (x1 - x0) * index / (len(points) - 1) for index in range(len(points))]
    curve = [(x, y_for(point["throughput"])) for x, point in zip(xs, points)]
    scene.polyline(curve, COLORS["blue"], 6)
    baseline = points[0]["throughput"]
    for x, y, point in zip(xs, (pair[1] for pair in curve), points):
        scene.circle(x, y, 12, COLORS["white"], COLORS["blue"], 6)
        scene.text(
            x,
            y - 31,
            f"{point['throughput']:.2f}",
            24,
            anchor="middle",
            weight="bold",
        )
        scene.text(
            x,
            y - 62,
            f"{point['throughput'] / baseline:.2f}×",
            17,
            COLORS["blue"],
            anchor="middle",
            weight="bold",
        )
        replicas = int(point["metadata"]["replicas"])
        scene.text(x, y1 + 42, str(replicas), 23, anchor="middle", weight="bold")
    scene.text(
        (x0 + x1) / 2,
        y1 + 82,
        "Native vLLM replicas",
        19,
        COLORS["muted"],
        anchor="middle",
    )

    legend_x = 1065
    scene.line(legend_x, 180, legend_x + 56, 180, COLORS["blue"], 5)
    scene.circle(legend_x + 28, 180, 7, COLORS["white"], COLORS["blue"], 4)
    scene.text(legend_x + 72, 187, "1K unique", 18, weight="bold")
    if sustained:
        x = xs[-1] + 28
        y = y_for(sustained["throughput"])
        diamond = [(x, y - 14), (x + 14, y), (x, y + 14), (x - 14, y)]
        scene.polygon(diamond, COLORS["white"], COLORS["orange"], 5)
        scene.text(
            x - 18,
            y + 48,
            f"{sustained['throughput']:.2f} sustained",
            17,
            COLORS["orange"],
            anchor="end",
            weight="bold",
        )
        scene.polygon(
            [
                (legend_x + 28, 199),
                (legend_x + 36, 207),
                (legend_x + 28, 215),
                (legend_x + 20, 207),
            ],
            COLORS["white"],
            COLORS["orange"],
            3,
        )
        scene.text(legend_x + 72, 214, "Sustained replay", 18, weight="bold")
    else:
        scene.text(
            legend_x,
            214,
            "Sustained replay: pending",
            18,
            COLORS["orange"],
            weight="bold",
        )
    scene.text(
        170,
        820,
        "Configuration note: r8 uses detector batch 16; r1/r2/r4 use detector batch 8.",
        17,
        COLORS["muted"],
    )
    save_chart(scene, output_dir, "native_vllm_scaling")


def panel_line_chart(
    scene: Scene,
    bounds: tuple[float, float, float, float],
    points: Sequence[tuple[int, float]],
    title: str,
    x_label: str,
    y_min: float,
    y_max: float,
    color: str,
) -> None:
    x0, y0, x1, y1 = bounds
    scene.rect(
        x0 - 55, y0 - 95, x1 - x0 + 95, y1 - y0 + 180, COLORS["panel"], radius=18
    )
    scene.text(x0, y0 - 56, title, 24, weight="bold")
    y_for = draw_axes(scene, bounds, y_min, y_max, ticks=5)
    if len(points) == 1:
        xs = [(x0 + x1) / 2]
    else:
        xs = [
            x0 + (x1 - x0) * index / (len(points) - 1) for index in range(len(points))
        ]
    curve = [(x, y_for(value)) for x, (_, value) in zip(xs, points)]
    scene.polyline(curve, color, 5)
    for index, (x, y, (setting, value)) in enumerate(
        zip(xs, (pair[1] for pair in curve), points)
    ):
        scene.circle(x, y, 9, COLORS["white"], color, 5)
        label_x = x
        label_anchor = "middle"
        if index == 0:
            label_x += 10
            label_anchor = "start"
        elif index == len(points) - 1:
            label_x -= 10
            label_anchor = "end"
        scene.text(
            label_x,
            y - 23,
            f"{value:.2f}",
            17,
            anchor=label_anchor,
            weight="bold",
        )
        scene.text(x, y1 + 34, str(setting), 18, anchor="middle", weight="bold")
    scene.text((x0 + x1) / 2, y1 + 70, x_label, 17, COLORS["muted"], anchor="middle")


def settings_chart(data: dict[str, Any], output_dir: Path) -> None:
    sweep = data["settings_sweep"]
    values = [value for _, value in sweep["max_num_seqs"] + sweep["renderer_workers"]]
    y_min = math.floor((min(values) - 1.0) / 2.0) * 2.0
    y_max = math.ceil((max(values) + 1.0) / 2.0) * 2.0
    scene = Scene()
    chart_header(
        scene,
        "Native vLLM setting sweeps",
        "A100 · infer_length=1024 · 512 unique document pages · concurrency 128 · truncated y-axis",
    )
    panel_line_chart(
        scene,
        (150, 300, 760, 690),
        sweep["max_num_seqs"],
        "Scheduler capacity (renderer workers=4)",
        "max_num_seqs",
        y_min,
        y_max,
        COLORS["blue"],
    )
    panel_line_chart(
        scene,
        (900, 300, 1450, 690),
        sweep["renderer_workers"],
        "Renderer parallelism (max_num_seqs=64)",
        "renderer workers",
        y_min,
        y_max,
        COLORS["teal"],
    )
    scene.text(
        150,
        830,
        "Configuration sweep only; headline throughput uses the separate 1K-unique runs.",
        17,
        COLORS["muted"],
    )
    save_chart(scene, output_dir, "native_settings_sweep")


def detector_chart(data: dict[str, Any], output_dir: Path) -> None:
    points = data["detector_sweep"]["points"]
    values = [value for _, value in points]
    y_min = math.floor((min(values) - 0.75) / 1.0)
    y_max = math.ceil((max(values) + 0.75) / 1.0)
    scene = Scene()
    chart_header(
        scene,
        "Detector batch-size sweep",
        "A100 · infer_length=1024 · 512 unique document pages · s64 / rw4 / c128 · truncated y-axis",
    )
    bounds = (180, 250, 1440, 700)
    y_for = draw_axes(scene, bounds, y_min, y_max, ticks=int(y_max - y_min))
    x0, _, x1, y1 = bounds
    xs = [x0 + (x1 - x0) * index / (len(points) - 1) for index in range(len(points))]
    curve = [(x, y_for(value)) for x, (_, value) in zip(xs, points)]
    scene.polyline(curve, COLORS["teal"], 6)
    best_batch, best_value = max(points, key=lambda item: item[1])
    for index, (x, y, (batch, value)) in enumerate(
        zip(xs, (pair[1] for pair in curve), points)
    ):
        color = COLORS["orange"] if batch == best_batch else COLORS["teal"]
        scene.circle(x, y, 11, COLORS["white"], color, 6)
        label_x = x
        label_anchor = "middle"
        if index == 0:
            label_x += 12
            label_anchor = "start"
        elif index == len(points) - 1:
            label_x -= 12
            label_anchor = "end"
        scene.text(
            label_x,
            y - 28,
            f"{value:.2f}",
            22,
            anchor=label_anchor,
            weight="bold",
        )
        scene.text(x, y1 + 42, str(batch), 22, anchor="middle", weight="bold")
    scene.text(
        (x0 + x1) / 2,
        y1 + 83,
        "detector max batch size",
        18,
        COLORS["muted"],
        anchor="middle",
    )
    gain = (best_value / points[0][1] - 1) * 100
    scene.rect(1070, 165, 370, 62, COLORS["light_orange"], radius=14)
    scene.text(
        1255,
        205,
        f"Best: batch {best_batch} · +{gain:.1f}% vs batch {points[0][0]}",
        20,
        anchor="middle",
        weight="bold",
    )
    scene.text(
        180,
        825,
        "Configuration sweep only; each point is a completed 512-page run.",
        17,
        COLORS["muted"],
    )
    save_chart(scene, output_dir, "detector_batch_sweep")


def deployment_optimization_chart(data: dict[str, Any], output_dir: Path) -> None:
    curve = data["deployment_optimization_curve"]
    all_runs = [run for group in curve["groups"] for run in group["runs"]]
    values = [run["throughput"] for run in all_runs]
    x_min = math.floor(min(values) - 0.5)
    x_max = math.ceil(max(values) + 0.5)
    if x_max - x_min < 2:
        x_max = x_min + 2
    colors = {
        "work_conserving": COLORS["teal"],
        "fixed_shard": COLORS["blue"],
    }
    scene = Scene(width=1600, height=980)
    chart_header(
        scene,
        "Deployment optimization curve",
        (
            "A100 · 10K timed (1K unique × 10) · infer_length=1024 · "
            f"truncated x-axis ({x_min}–{x_max} images/s)"
        ),
    )
    plot_x0, plot_x1 = 610.0, 1450.0
    plot_y0, plot_y1 = 215.0, 835.0
    scene.rect(565, 165, 925, 710, COLORS["panel"], radius=18)
    for tick in range(x_min, x_max + 1):
        x = plot_x0 + (tick - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)
        scene.line(x, plot_y0, x, plot_y1, COLORS["grid"], 1)
        scene.text(x, 885, str(tick), 17, COLORS["muted"], anchor="middle")
    scene.line(plot_x0, plot_y1, plot_x1, plot_y1, COLORS["ink"], 2)
    scene.text(
        (plot_x0 + plot_x1) / 2,
        923,
        "Aggregate throughput (images/s)",
        18,
        COLORS["muted"],
        anchor="middle",
    )

    best_run_id = curve["best_run_id"]
    y = 185.0
    for group_index, group in enumerate(curve["groups"]):
        color = colors.get(group["id"], COLORS["muted"])
        scene.text(80, y + 8, group["label"], 21, color, weight="bold")
        y += 50
        row_positions = [y + index * 58 for index in range(len(group["runs"]))]
        reference = next(
            run for run in group["runs"] if run["id"] == group["reference_id"]
        )
        reference_x = plot_x0 + (
            (reference["throughput"] - x_min) / (x_max - x_min)
        ) * (plot_x1 - plot_x0)
        scene.line(
            reference_x,
            row_positions[0] - 23,
            reference_x,
            row_positions[-1] + 23,
            color,
            2,
            dash=(7, 6),
        )
        for row_y, run in zip(row_positions, group["runs"]):
            scene.line(plot_x0, row_y, plot_x1, row_y, COLORS["grid"], 1)
            scene.text(525, row_y + 7, run["label"], 18, anchor="end", weight="bold")
            x = plot_x0 + ((run["throughput"] - x_min) / (x_max - x_min)) * (
                plot_x1 - plot_x0
            )
            scene.line(reference_x, row_y, x, row_y, color, 5)
            marker_color = COLORS["orange"] if run["id"] == best_run_id else color
            scene.circle(x, row_y, 10, COLORS["white"], marker_color, 5)
            delta = run["delta_vs_group_reference_pct"]
            suffix = "ref" if run["id"] == group["reference_id"] else f"{delta:+.2f}%"
            label_x = x + 17
            anchor = "start"
            if run["throughput"] < reference["throughput"] or x > plot_x1 - 125:
                label_x = x - 17
                anchor = "end"
            scene.text(
                label_x,
                row_y + 7,
                f"{run['throughput']:.2f}  ({suffix})",
                17,
                marker_color,
                anchor=anchor,
                weight="bold",
            )
        y = row_positions[-1] + 82
        if group_index < len(curve["groups"]) - 1:
            scene.line(80, y - 34, 1490, y - 34, COLORS["grid"], 2)

    best = next(run for run in all_runs if run["id"] == best_run_id)
    scene.rect(1110, 145, 380, 56, COLORS["light_orange"], radius=12)
    scene.text(
        1300,
        181,
        f"Best: {best['label']} · {best['throughput']:.2f} img/s",
        17,
        anchor="middle",
        weight="bold",
    )
    scene.text(
        80,
        955,
        "Dashed lines are per-dispatcher references; percentages are deltas within each dispatcher group.",
        16,
        COLORS["muted"],
    )
    save_chart(scene, output_dir, "deployment_optimization_curve")


def downsample_series(
    series: Sequence[dict[str, float]], max_points: int
) -> list[dict[str, float]]:
    if len(series) <= max_points:
        return list(series)
    bucket_size = math.ceil(len(series) / max_points)
    reduced = []
    for start in range(0, len(series), bucket_size):
        bucket = series[start : start + bucket_size]
        reduced.append(
            {
                "time_s": sum(point["time_s"] for point in bucket) / len(bucket),
                "gpu_utilization_pct": sum(
                    point["gpu_utilization_pct"] for point in bucket
                )
                / len(bucket),
                "power_w": sum(point["power_w"] for point in bucket) / len(bucket),
                "memory_used_gib": sum(point["memory_used_gib"] for point in bucket)
                / len(bucket),
            }
        )
    return reduced


def nice_time_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    rough_step = value / 5
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return math.ceil(value / step) * step


def draw_gpu_time_panel(
    scene: Scene,
    bounds: tuple[float, float, float, float],
    runs: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    y_max: float,
    x_max: float,
    colors: dict[str, str],
    max_plot_points: int,
    show_x_labels: bool,
    x_label_divisor: float = 1.0,
) -> None:
    x0, y0, x1, y1 = bounds
    scene.rect(x0 - 70, y0 - 58, x1 - x0 + 95, y1 - y0 + 92, COLORS["panel"], radius=18)
    scene.text(x0, y0 - 20, title, 22, weight="bold")
    y_for = draw_axes(scene, bounds, 0, y_max, ticks=5, y_label="")
    for tick in range(6):
        seconds = x_max * tick / 5
        x = x0 + (x1 - x0) * tick / 5
        scene.line(x, y0, x, y1, COLORS["grid"], 1)
        if show_x_labels:
            display_time = seconds / x_label_divisor
            label = (
                f"{display_time:.0f}"
                if abs(display_time - round(display_time)) < 0.05
                else f"{display_time:.1f}"
            )
            scene.text(x, y1 + 31, label, 16, COLORS["muted"], anchor="middle")
    for run in runs:
        display_series = downsample_series(run["series"], max_plot_points)
        points = [
            (
                x0 + point["time_s"] / x_max * (x1 - x0),
                y_for(point[metric]),
            )
            for point in display_series
            if point["time_s"] <= x_max
        ]
        if len(points) >= 2:
            scene.polyline(points, colors[run["id"]], 4)


def gpu_active_comparison_chart(data: dict[str, Any], output_dir: Path) -> None:
    comparison = data["gpu_active_comparison"]
    if not comparison["available"]:
        return
    runs = comparison["runs"]
    algorithm = comparison["algorithm"]
    colors = {
        "official_hf": COLORS["orange"],
        "vllm_baseline": COLORS["blue"],
        "optimized_native_vllm": COLORS["teal"],
    }
    max_duration = max(run["series"][-1]["time_s"] for run in runs)
    x_max = nice_time_upper(max_duration / 60.0) * 60.0
    max_power = max(point["power_w"] for run in runs for point in run["series"])
    power_y_max = nice_upper(max_power, step=50)
    for run in runs:
        for point in run["series"]:
            point["memory_used_gib"] = point["memory_used_mib"] / 1024
    scene = Scene(width=1600, height=1080)
    rolling = algorithm["rolling_window_seconds"]
    context = comparison.get("chart_context", {})
    hardware = context.get("hardware", "1× NVIDIA A100 80GB")
    workload = context.get(
        "workload",
        "30,000 JPEG-byte requests · 1,000 Bo767 pages × 30 · infer_length=1024",
    )
    chart_header(
        scene,
        "Nemotron OCR v2: GPU-active aligned utilization and power",
        f"{hardware} · {workload}",
    )
    scene.text(
        72,
        162,
        (
            "Startup/model-load trimmed per run; lines are trailing "
            f"{rolling:g}s means. Legend uses measured timed images/s."
        ),
        18,
        COLORS["muted"],
    )

    legend_starts = [100, 600, 1100]
    for x, run in zip(legend_starts, runs):
        color = colors[run["id"]]
        scene.line(x, 213, x + 48, 213, color, 6)
        scene.circle(x + 24, 213, 7, COLORS["white"], color, 4)
        scene.text(x + 62, 208, run["chart_label"], 16, weight="bold")
        scene.text(
            x + 62,
            237,
            f"{run['throughput']:.2f} img/s · {run['speedup_vs_official_hf']:.2f}× vs official HF",
            15,
            COLORS["muted"],
        )

    draw_gpu_time_panel(
        scene,
        (145, 315, 1510, 535),
        runs,
        "gpu_utilization_pct",
        "Average GPU utilization (%)",
        100,
        x_max,
        colors,
        int(algorithm["max_plot_points_per_run"]),
        False,
    )
    draw_gpu_time_panel(
        scene,
        (145, 660, 1510, 880),
        runs,
        "power_w",
        "GPU power draw (W)",
        power_y_max,
        x_max,
        colors,
        int(algorithm["max_plot_points_per_run"]),
        True,
        x_label_divisor=60.0,
    )
    scene.text(
        828,
        942,
        "Minutes since sustained GPU activity began",
        19,
        COLORS["muted"],
        anchor="middle",
    )
    threshold = algorithm["gpu_utilization_threshold_pct"]
    active_window = algorithm["active_detection_window_seconds"]
    fraction = algorithm["minimum_active_sample_fraction"] * 100
    scene.text(
        145,
        1022,
        (
            f"Alignment gate: first ≥{threshold:g}% sample starting a {active_window:g}s window "
            f"with mean ≥{threshold:g}% and ≥{fraction:g}% active samples; raw trace tail retained."
        ),
        16,
        COLORS["muted"],
    )
    save_chart(scene, output_dir, "gpu_active_comparison")


def workload_label(kind: str) -> str:
    return {
        "unique_1k": "1K unique",
        "unique_512": "512 unique",
        "sustained_replay": "sustained replay",
    }[kind]


def markdown_report(data: dict[str, Any]) -> str:
    baseline = data["baseline"]
    optimized = data["optimized"]
    native = data["native_scaling"]
    sustained = data["sustained_replay"]
    settings = data["settings_sweep"]
    detector = data["detector_sweep"]
    deployment_curve = data["deployment_optimization_curve"]
    optimized_30k = data["optimized_30k_variants"]
    nondeterminism = data["nondeterminism_comparisons"]
    sequence_audit = data["sequence_aware_audit"]
    hf_agreement = data["hf_output_agreement"]
    checks = data["accuracy_checks"]
    gpu_comparison = data["gpu_active_comparison"]
    benchmark_dataset = data["benchmark_dataset"]
    offline_speedup = optimized["throughput"] / baseline["throughput"]
    native_speedup = native[-1]["throughput"] / native[0]["throughput"]
    best_seq = max(settings["max_num_seqs"], key=lambda item: item[1])
    best_renderer = max(settings["renderer_workers"], key=lambda item: item[1])
    best_detector = max(detector["points"], key=lambda item: item[1])
    deployment_runs = [
        run for group in deployment_curve["groups"] for run in group["runs"]
    ]
    best_deployment = next(
        run for run in deployment_runs if run["id"] == deployment_curve["best_run_id"]
    )
    rec64_publication = optimized_30k["runs"][0]
    rec128_conservative = optimized_30k["runs"][1]
    optimized_check = next(
        check for check in checks if check["role"] == "optimized_validation"
    )
    repeat_check = next(
        check for check in checks if check["role"] == "repeatability_context"
    )

    lines = [
        f"# {data['title']}",
        "",
        (
            "This report includes only `infer_length=1024` throughput artifacts. "
            "It keeps the matched 30K sustained headline (1K distinct document pages × 30) "
            "separate from the 1K-unique sweeps and never substitutes one workload's "
            "throughput for the other."
        ),
        "",
        "## Summary",
        "",
        (
            f"- The offline vLLM replica harness rises from **{baseline['throughput']:.2f}** "
            f"to **{optimized['throughput']:.2f} images/s**, a **{offline_speedup:.2f}×** "
            "steady-state speedup on the 1K-unique workload."
        ),
        (
            f"- Native vLLM serving scales from **{native[0]['throughput']:.2f}** at one "
            f"replica to **{native[-1]['throughput']:.2f} images/s** at eight replicas "
            f"(**{native_speedup:.2f}×**) on the separate 1K-unique JPEG workload."
        ),
        (
            f"- The 512-page settings sweeps peak at `max_num_seqs={best_seq[0]}` "
            f"(**{best_seq[1]:.2f} images/s**), renderer workers={best_renderer[0]} "
            f"(**{best_renderer[1]:.2f} images/s**), and detector batch={best_detector[0]} "
            f"(**{best_detector[1]:.2f} images/s**)."
        ),
        (
            f"- The matched 10K deployment sweep peaks at **{best_deployment['throughput']:.2f} "
            f"images/s** with {best_deployment['label']}; dynamic and fixed-shard "
            "dispatchers are reported as separate groups."
        ),
        (
            f"- The completed 30K optimized publication run processed "
            f"**{rec64_publication['completed']:,}/"
            f"{rec64_publication['workload_metadata']['timed_workload_image_count']:,}** images at "
            f"**{rec64_publication['throughput']:.4f} images/s** with rec64; the "
            f"conservative rec128 comparison measured **{rec128_conservative['throughput']:.4f} "
            f"images/s** ({rec128_conservative['delta_vs_publication_pct']:+.2f}%)."
        ),
        (
            "- **Rec64 quality impact remains unresolved.** Rec64 is the throughput "
            "publication profile; conservative rec128 is retained as the accuracy profile "
            "until labeled evaluation isolates batching from run-to-run scheduling effects."
        ),
    ]
    if gpu_comparison["available"]:
        gpu_runs = gpu_comparison["runs"]
        lines.insert(
            6,
            (
                f"- **Matched 30K headline:** official HF **{gpu_runs[0]['throughput']:.4f} "
                f"images/s** (1.00×), clean vLLM **{gpu_runs[1]['throughput']:.4f}** "
                f"({gpu_runs[1]['speedup_vs_official_hf']:.2f}× HF), and optimized vLLM "
                f"**{gpu_runs[2]['throughput']:.4f}** "
                f"({gpu_runs[2]['speedup_vs_official_hf']:.2f}× HF; "
                f"{gpu_runs[2]['throughput'] / gpu_runs[1]['throughput']:.2f}× clean vLLM)."
            ),
        )
    if data["hf_baseline"]:
        hf = data["hf_baseline"]
        lines.append(
            f"- The measured HF baseline is **{hf['throughput']:.2f} images/s**; the "
            f"optimized offline result is **{optimized['throughput'] / hf['throughput']:.2f}×** faster."
        )
    else:
        lines.append(
            "- **The separate 1K-unique offline HF baseline remains pending.** No value is "
            "inferred from the completed 30K official-HF run or copied across workloads. "
            "Add its result path to "
            "`configs/ocr_benchmark_report.json` (or pass `--hf-baseline-json`) after the "
            "1024-resolution 1K-unique run completes."
        )
    if sustained:
        replay = sustained["metadata"]["replay_count"]
        unique = sustained["metadata"]["unique_image_count"]
        timed = sustained["metadata"]["timed_workload_image_count"]
        lines.append(
            f"- Sustained replay is **{sustained['throughput']:.2f} images/s** over "
            f"{timed:,} timed requests ({unique:,} unique images × {replay} replays)."
        )
    else:
        lines.append(
            "- **Sustained replay pending.** Partial client logs are intentionally ignored; "
            "the report will include the result only when the aggregate `summary.json` "
            "contains explicit unique/replay/timed counts."
        )
    if not gpu_comparison["available"]:
        lines.append(
            "- **GPU-active comparison pending.** The generator will not draw the "
            "three-run chart until every measured result and raw trace is present."
        )
    else:
        gpu_runs = gpu_comparison["runs"]
        hf_throughput = gpu_runs[0]["throughput"]
        vllm_baseline_throughput = gpu_runs[1]["throughput"]
        lines.extend(
            [
                "",
                "## Matched 30K headline comparison",
                "",
                "![Matched 30K throughput and speedup](matched_30k_speedup.png)",
                "",
                "[Vector chart](matched_30k_speedup.svg)",
                "",
                "A100, JPEG byte inputs, `infer_length=1024`, and 30K timed images (1K unique × 30) for every system.",
                "",
                "| System | Throughput | Speedup vs official HF | Speedup vs clean vLLM |",
                "|---|---:|---:|---:|",
            ]
        )
        for run in gpu_runs:
            lines.append(
                f"| {run['label']} | {run['throughput']:.7f} images/s | "
                f"{run['throughput'] / hf_throughput:.2f}× | "
                f"{run['throughput'] / vllm_baseline_throughput:.2f}× |"
            )
        if gpu_runs[1].get("note"):
            lines.extend(["", f"**Baseline audit:** {gpu_runs[1]['note']}"])

        lines.extend(
            [
                "",
                "## Benchmark page corpus",
                "",
                (
                    f"The matched 30K runs use **{benchmark_dataset['name']}**, the "
                    f"{benchmark_dataset['source_pdf_count']}-PDF collection used by NVIDIA "
                    f"[NeMo Retriever benchmark examples]({benchmark_dataset['nvidia_usage_url']}). "
                    f"The selected pool contains {benchmark_dataset['selected_page_count']:,} "
                    "document pages, not 1,000 separate PDFs."
                ),
                "",
                f"- **Selection:** {benchmark_dataset['selection']}",
                f"- **Rendering:** {benchmark_dataset['rendering']}",
                f"- **Request payload:** {benchmark_dataset['payload']}",
                (
                    f"- **Timed workload:** the ordered page pool replayed "
                    f"{benchmark_dataset['timed_replay_count']} times."
                ),
                f"- **Input artifact:** `{benchmark_dataset['dataset_artifact']}`",
                f"- **Input SHA-256:** `{benchmark_dataset['dataset_sha256']}`",
                f"- **Verification:** {benchmark_dataset['verification']}",
                "",
                (
                    "The older L4 benchmark used a different SAFEDOCS `page1_png` pool; "
                    "that input is not used for this final A100 30K comparison."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Historical offline replica-harness comparison (1K unique)",
            "",
            "![Baseline versus optimized throughput](baseline_vs_optimized.png)",
            "",
            "[Vector chart](baseline_vs_optimized.svg)",
            "",
            "| Run | Throughput | Speedup vs baseline | Workload | Configuration |",
            "|---|---:|---:|---|---|",
            (
                f"| {baseline['label']} | {baseline['throughput']:.2f} images/s | "
                f"1.00× | {workload_label(baseline['workload_kind'])} | "
                f"1 replica, s64, detector batch 8 |"
            ),
            (
                f"| {optimized['label']} | {optimized['throughput']:.2f} images/s | "
                f"{offline_speedup:.2f}× | {workload_label(optimized['workload_kind'])} | "
                f"16 replicas, s32, MPS 25%, queue chunk 32 |"
            ),
            "",
            (
                "These two values come from the offline replica harness and PNG inputs. "
                "They are directly comparable to each other, but are not the final matched "
                "30K headline and must not be compared directly to the native HTTP/JPEG "
                "serving series."
            ),
            "",
            "## Native vLLM replica scaling",
            "",
            "![Native vLLM scaling](native_vllm_scaling.png)",
            "",
            "[Vector chart](native_vllm_scaling.svg)",
            "",
            "| Replicas | Throughput | Speedup vs r1 | Workload | Detector batch |",
            "|---:|---:|---:|---|---:|",
        ]
    )
    for point in native:
        lines.append(
            f"| {point['metadata']['replicas']} | {point['throughput']:.2f} images/s | "
            f"{point['throughput'] / native[0]['throughput']:.2f}× | "
            f"{workload_label(point['workload_kind'])} | "
            f"{point['metadata']['detector_batch']} |"
        )
    lines.extend(
        [
            "",
            (
                "The 8-replica point also changes detector batch from 8 to 16, so this is "
                "a deployment scaling curve rather than a strictly replica-only ablation."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Deployment optimization curve (10K sustained)",
            "",
            "![Deployment optimization curve](deployment_optimization_curve.png)",
            "",
            "[Vector chart](deployment_optimization_curve.svg)",
            "",
            deployment_curve["comparison_contract"],
            "",
            "| Dispatcher | Configuration | Replicas | Rec chunk | MPS | Access log | cuDNN benchmark | Throughput | Δ vs group reference |",
            "|---|---|---:|---:|---|---|---|---:|---:|",
        ]
    )
    manifest_metadata_runs = []
    for group in deployment_curve["groups"]:
        for run in group["runs"]:
            delta = run["delta_vs_group_reference_pct"]
            delta_label = (
                "reference" if run["id"] == group["reference_id"] else f"{delta:+.2f}%"
            )
            lines.append(
                f"| {group['label']} | {run['label']} | {run['replicas']} | "
                f"{run['recognizer_chunk_size']} | {run['mps']} | "
                f"{run['access_log']} | {run['cudnn_benchmark']} | "
                f"{run['throughput']:.4f} images/s | {delta_label} |"
            )
            sources = run["workload_metadata"].get("metadata_sources", {})
            if any(value == "manifest" for value in sources.values()):
                manifest_metadata_runs.append(run["label"])
    lines.extend(
        [
            "",
            (
                "Dynamic work-conserving results and fixed-shard results use different "
                "dispatch policies. Their group-relative deltas are informative; treating "
                "the two groups as a controlled dispatcher-only ranking is not."
            ),
        ]
    )
    if manifest_metadata_runs:
        lines.extend(
            [
                "",
                (
                    "Some early dynamic summaries omitted explicit replay metadata. For "
                    f"{', '.join(manifest_metadata_runs)}, the curated manifest supplies "
                    "the 1K × 10 run contract while the artifact itself must still report "
                    "1,000 unique images, 10,000 timed/completed images, zero failures, and "
                    "the expected dispatcher."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## GPU-active aligned utilization, power, and memory",
            "",
            "### 30K publication methodology template",
            "",
            gpu_comparison["algorithm"]["comparison_contract"],
            "",
        ]
    )
    publication_methodology = gpu_comparison["publication_methodology"]
    for key in (
        "hardware",
        "workload",
        "quality_contract",
        "throughput_metric",
        "telemetry",
    ):
        if key in publication_methodology:
            lines.append(
                f"- **{key.replace('_', ' ').title()}:** {publication_methodology[key]}"
            )
    for control in publication_methodology.get("controls", []):
        lines.append(f"- **Acceptance control:** {control}")
    lines.extend(
        [
            "",
            "### Completed optimized 30K profiles",
            "",
            optimized_30k["comparison_contract"],
            "",
            "| Role | Rec chunk | Completed | Throughput | Δ vs rec64 | Avg GPU util | Avg power | Trace samples |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in optimized_30k["runs"]:
        delta_label = (
            "reference"
            if run["id"] == "rec64_publication"
            else f"{run['delta_vs_publication_pct']:+.3f}%"
        )
        lines.append(
            f"| {run['label']} | {run['recognizer_chunk_size']} | "
            f"{run['completed']:,}/{run['workload_metadata']['timed_workload_image_count']:,} | "
            f"{run['throughput']:.7f} images/s | {delta_label} | "
            f"{run['gpu_utilization_pct_avg']:.4f}% | "
            f"{run['gpu_power_w_avg']:.4f} W | {run['gpu_trace_samples']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "The rec64 run remains the throughput publication pair. Rec128 is retained "
                "as the conservative accuracy profile and is not added as a fourth system to "
                "the official-HF/vLLM-baseline/optimized telemetry chart. GPU utilization "
                "and power in this table are the averages recorded in each completed summary; "
                "the aligned chart reads the raw CSV samples. These profile labels separate "
                "deployment roles; they do not assert that rec64 preserves OCR quality."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "| System | Result JSON | Raw GPU trace | Input state |",
            "|---|---|---|---|",
        ]
    )
    for status in gpu_comparison["input_status"]:
        result_path = status["result_path"] or "TBD"
        trace_path = status["trace_path"] or "TBD"
        if status["result_path_exists"] and status["trace_path_exists"]:
            state = "ready"
        elif status["result_path_exists"] or status["trace_path_exists"]:
            state = "partial"
        else:
            state = "pending"
        lines.append(
            f"| {status['label']} | `{result_path}` | `{trace_path}` | {state} |"
        )
    lines.extend(["", "### Aligned telemetry result", ""])
    algorithm = gpu_comparison["algorithm"]
    if gpu_comparison["available"]:
        lines.extend(
            [
                "![GPU-active aligned utilization, power, and memory](gpu_active_comparison.png)",
                "",
                "[Vector chart](gpu_active_comparison.svg)",
                "",
                "| Run | Measured throughput | Speedup vs official HF | Active start | Leading trim | Aligned trace |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for run in gpu_comparison["runs"]:
            alignment = run["alignment"]
            lines.append(
                f"| {run['label']} | {run['throughput']:.2f} images/s | "
                f"{run['speedup_vs_official_hf']:.2f}× | "
                f"`{alignment['active_start_timestamp']}` | "
                f"{alignment['leading_trim_seconds']:.2f}s | "
                f"{alignment['aligned_duration_seconds']:.2f}s |"
            )
    else:
        lines.extend(
            [
                (
                    "**Pending measured inputs; no chart or placeholder values are emitted.** "
                    "Populate the following manifest fields after matched runs complete:"
                ),
                "",
            ]
        )
        lines.extend(f"- {missing}" for missing in gpu_comparison["missing_inputs"])
    threshold = algorithm["gpu_utilization_threshold_pct"]
    detection_window = algorithm["active_detection_window_seconds"]
    active_fraction = algorithm["minimum_active_sample_fraction"] * 100
    rolling_window = algorithm["rolling_window_seconds"]
    lines.extend(
        [
            "",
            f"Manifest comparison contract: {algorithm['comparison_contract']}",
            "",
            "Alignment and smoothing algorithm:",
            "",
            "1. Parse timestamp, `utilization.gpu [%]`, `power.draw [W]`, and `memory.used [MiB]` directly from each raw `nvidia-smi` CSV; invalid metric rows are excluded.",
            (
                f"2. Scan forward for the earliest sample at or above {threshold:g}% whose next "
                f"{detection_window:g}s window has mean GPU utilization at least {threshold:g}% and at least "
                f"{active_fraction:g}% of samples at or above {threshold:g}%."
            ),
            "3. Discard only samples before that point and independently set each run's detected point to `t=0`; retain the raw trace tail so workload completion remains visible.",
            (
                f"4. Plot a sample-based trailing {rolling_window:g}s arithmetic mean for "
                "GPU utilization, power draw, and memory used. The window is configurable with "
                "`--gpu-rolling-window-seconds`."
            ),
            "",
            "The chart is gated on all three runs sharing the same completed count, `infer_length`, workload kind, unique-image count, and replay count.",
            "",
            "## Setting sweeps",
            "",
            "![Native setting sweeps](native_settings_sweep.png)",
            "",
            "[Vector chart](native_settings_sweep.svg)",
            "",
            "| Sweep | Setting | Throughput |",
            "|---|---:|---:|",
        ]
    )
    for setting, throughput in settings["max_num_seqs"]:
        lines.append(f"| max_num_seqs (rw4) | {setting} | {throughput:.2f} images/s |")
    for setting, throughput in settings["renderer_workers"]:
        lines.append(
            f"| renderer workers (s64) | {setting} | {throughput:.2f} images/s |"
        )
    lines.extend(
        [
            "",
            "![Detector batch sweep](detector_batch_sweep.png)",
            "",
            "[Vector chart](detector_batch_sweep.svg)",
            "",
            "| Detector batch | Throughput |",
            "|---:|---:|",
        ]
    )
    for batch, throughput in detector["points"]:
        lines.append(f"| {batch} | {throughput:.2f} images/s |")

    lines.extend(
        [
            "",
            "Both settings charts use 512 distinct document pages and a deliberately truncated "
            "y-axis to make small tuning differences visible. They are ablations, not the "
            "1K-unique headline.",
            "",
            "## Output agreement and workload gates",
            "",
            (
                f"- The controlled optimized OpenAI Triton-kernel comparison **{'passed' if optimized_check['passed'] else 'failed'}**: "
                f"{optimized_check['region_count_mismatches']} region-count mismatches, "
                f"{optimized_check['text_mismatches']} text mismatches, "
                f"{optimized_check['text_exact_rate'] * 100:.1f}% text exact rate across "
                f"{optimized_check['candidate_images']} images. This validates that specific "
                "controlled kernel-path check; it does not establish rec64 batching quality."
            ),
            (
                f"- A repeated baseline run itself **{'passed' if repeat_check['passed'] else 'did not pass'}** "
                f"the strict bitwise-style comparison ({repeat_check['region_count_mismatches']} "
                f"region-count mismatches; {repeat_check['text_exact_rate'] * 100:.1f}% text exact). "
                "This documents run-to-run nondeterminism. `infer_length=1024` is a "
                "configuration eligibility rule for this report, not evidence that an "
                "optimization preserves OCR accuracy."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "### Batching and nondeterminism diagnostics",
            "",
            "| Comparison | Type | Strict pass | Region-count mismatches | Text mismatches / paired | Text exact |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for comparison in nondeterminism["comparisons"]:
        comparison_type = (
            "same-config repeat"
            if comparison["comparison_type"] == "same_config_repeat"
            else "rec128/rec64 batching"
        )
        lines.append(
            f"| {comparison['label']} | {comparison_type} | "
            f"{'yes' if comparison['passed'] else 'no'} | "
            f"{comparison['region_count_mismatches']} | "
            f"{comparison['text_mismatches']} / {comparison['paired_regions']} | "
            f"{comparison['text_exact_rate'] * 100:.3f}% |"
        )
    lines.extend(
        [
            "",
            (
                "This mismatch table is retained as nondeterminism documentation. Raw "
                "paired-region coordinate and confidence maxima are intentionally not used "
                "as quality evidence: insertions/deletions shift region sequences and can "
                "make subsequent positional pairings incomparable."
            ),
            "",
            "### Sequence-aware JPEG diagnostics",
            "",
            "Profiles: **A** = rec128 reference, **B** = rec128 repeat, **C** = rec64.",
            "",
            "| Pair | Page-text sequence Levenshtein total | Exact page sequences |",
            "|---|---:|---:|",
        ]
    )
    for pair in sequence_audit["pairs"]:
        lines.append(
            f"| {pair['pair']}: {pair['first_label']} vs {pair['second_label']} | "
            f"{pair['page_text_sequence_levenshtein_total']} | "
            f"{pair['exact_page_sequences']}/{pair['page_count']} |"
        )
    agreement = sequence_audit["aligned_region_agreement"]
    agreement_labels = (
        ("all_same", "A = B = C"),
        ("A_equals_B_not_C", "A = B ≠ C"),
        ("B_equals_C_not_A", "B = C ≠ A"),
        ("A_equals_C_not_B", "A = C ≠ B"),
        ("all_different", "A, B, C all different"),
    )
    lines.extend(
        [
            "",
            (
                f"Position-wise agreement covers {sequence_audit['aligned_region_count']} "
                f"regions on {sequence_audit['aligned_equal_length_page_count']}/"
                f"{sequence_audit['total_page_count']} pages where all three runs emitted "
                "the same region count:"
            ),
            "",
            "| Agreement category | Regions | Share of aligned regions |",
            "|---|---:|---:|",
        ]
    )
    for key, label in agreement_labels:
        count = agreement[key]
        lines.append(
            f"| {label} | {count} | "
            f"{count / sequence_audit['aligned_region_count'] * 100:.3f}% |"
        )
    lines.extend(
        [
            "",
            f"Method: {sequence_audit['method']}",
            "",
            "### Agreement with one official HF output run",
            "",
            "| Candidate compared with official HF output | Strict pass | Region-count mismatches | Text mismatches / zipped pairs | Zipped text exact |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in hf_agreement["comparisons"]:
        lines.append(
            f"| {comparison['candidate']} | "
            f"{'yes' if comparison['passed'] else 'no'} | "
            f"{comparison['region_count_mismatches']} | "
            f"{comparison['text_mismatches']} / {comparison['paired_regions']} | "
            f"{comparison['zipped_text_exact_rate'] * 100:.3f}% |"
        )
    lines.extend(
        [
            "",
            (
                "Rec64 happened to be closer to this particular official-HF output run "
                "than rec128 by positional/zipped text agreement (96.774% versus 92.194%; "
                "vLLM baseline was 93.826%). This is agreement, not accuracy: the official "
                "HF output is another model run rather than labeled ground truth, every "
                "strict comparison fails, and region insertions/deletions can shift zipped "
                "pair positions."
            ),
            "",
            (
                "**Conclusion: rec64 quality impact is unresolved; conservative rec128 is "
                "retained as the accuracy profile.** There are no labeled reference "
                "transcriptions or boxes, so agreement is not ground-truth accuracy. In "
                "addition, work-conserving scheduling can place a page at different dynamic "
                "batch positions across runs, confounding recognizer chunk size with ordinary "
                "run-to-run variation. The diagnostics therefore establish neither bitwise "
                "identity nor accuracy preservation."
            ),
            "",
            "Workload labels are enforced from artifact counts and the curated source manifest:",
            "",
            "- **1K unique:** 1,000 distinct page images, processed once.",
            "- **512 unique:** 512 distinct page images, processed once, used only for tuning sweeps.",
            "- **Sustained replay:** requires `replay_count > 1` and timed request count greater than the unique-image count.",
            "",
            "## Methodology and provenance",
            "",
            f"- Hardware: {data['methodology']['hardware']}.",
            "- Resolution contract: `infer_length=1024`; no 768 results are ingested.",
            f"- Offline input: {data['methodology']['offline_input']}.",
            f"- Native serving input: {data['methodology']['native_input']}.",
            "- Throughput values are read directly from the listed JSON/CSV artifacts.",
            "- Full normalized data and SHA-256 provenance are in `report_data.json`.",
            "",
            "| Use | Source artifact | SHA-256 (first 12) |",
            "|---|---|---|",
        ]
    )
    provenance = [
        ("Offline baseline", baseline["source"], baseline["source_sha256"]),
        ("Offline optimized", optimized["source"], optimized["source_sha256"]),
    ]
    if data["hf_baseline"]:
        provenance.append(
            (
                "HF baseline",
                data["hf_baseline"]["source"],
                data["hf_baseline"]["source_sha256"],
            )
        )
    provenance.extend(
        (
            f"Native r{point['metadata']['replicas']}",
            point["source"],
            point["source_sha256"],
        )
        for point in native
    )
    if sustained:
        provenance.append(
            ("Sustained replay", sustained["source"], sustained["source_sha256"])
        )
    for group in deployment_curve["groups"]:
        for run in group["runs"]:
            provenance.append(
                (
                    f"Deployment: {run['label']}",
                    run["source"],
                    run["source_sha256"],
                )
            )
    for run in optimized_30k["runs"]:
        provenance.extend(
            [
                (
                    f"30K {run['label']} result",
                    run["result_source"],
                    run["result_sha256"],
                ),
                (
                    f"30K {run['label']} trace",
                    run["trace_source"],
                    run["trace_sha256"],
                ),
            ]
        )
    if gpu_comparison["available"]:
        for run in gpu_comparison["runs"]:
            provenance.extend(
                [
                    (
                        f"{run['label']} throughput",
                        run["result_source"],
                        run["result_sha256"],
                    ),
                    (
                        f"{run['label']} GPU trace",
                        run["trace_source"],
                        run["trace_sha256"],
                    ),
                ]
            )
    provenance.extend(
        [
            ("Settings sweep", settings["source"], settings["sha256"]),
            ("Detector sweep", detector["source"], detector["sha256"]),
        ]
    )
    provenance.extend(
        (check["label"], check["source"], check["sha256"]) for check in checks
    )
    provenance.extend(
        (
            f"Nondeterminism: {comparison['label']}",
            comparison["source"],
            comparison["sha256"],
        )
        for comparison in nondeterminism["comparisons"]
    )
    provenance.extend(
        (
            f"Sequence audit {profile['id']}: {profile['label']}",
            profile["source"],
            profile["sha256"],
        )
        for profile in sequence_audit["profiles"]
    )
    provenance.extend(
        (
            f"HF agreement: {comparison['candidate']}",
            comparison["source"],
            comparison["sha256"],
        )
        for comparison in hf_agreement["comparisons"]
    )
    for label, source, digest in provenance:
        lines.append(f"| {label} | `{source}` | `{digest[:12]}` |")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "/raid/vjawa/tmp/ocr_optimization/venv/bin/python \\",
            "  /home/nfs/vjawa/ocr_optimization/scripts/generate_ocr_benchmark_report.py \\",
            "  --strict --gpu-rolling-window-seconds 15",
            "```",
            "",
            (
                "The output is deterministic for a fixed manifest and source artifacts. "
                "PNG and SVG charts are produced from the same geometry."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def comparison_markdown_report(data: dict[str, Any]) -> str:
    comparison = data["gpu_active_comparison"]
    runs = comparison["runs"]
    official, clean_vllm, optimized = runs
    lines = [
        "# Nemotron OCR v2: matched HF and vLLM comparison",
        "",
        (
            f"The optimized run measured **{optimized['throughput']:.4f} images/s**, "
            f"**{optimized['speedup_vs_official_hf']:.2f}×** the official NVIDIA/HF "
            f"in-process baseline and **{optimized['speedup_vs_clean_vllm']:.2f}×** "
            "the selected clean-model vLLM baseline."
        ),
        "",
        "All values below are loaded from the selected result artifacts; "
        "the generator does not embed a throughput constant.",
        "",
        "![Matched throughput and speedup](matched_30k_speedup.png)",
        "",
        "[Vector chart](matched_30k_speedup.svg)",
        "",
        "![GPU-active aligned utilization and power](gpu_active_comparison.png)",
        "",
        "[Vector chart](gpu_active_comparison.svg)",
        "",
        "## Matched throughput",
        "",
        "| System | Workflow | Timed images | Throughput | vs official HF | vs clean vLLM | Failures field |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        failure_value = (
            str(run["failed"]) if run["failed"] is not None else "not encoded"
        )
        lines.append(
            f"| {run['label']} | {run['workflow']} | {run['count']:,} | "
            f"{run['throughput']:.4f} img/s | "
            f"{run['speedup_vs_official_hf']:.3f}× | "
            f"{run['speedup_vs_clean_vllm']:.3f}× | {failure_value} |"
        )
    lines.extend(
        [
            "",
            "The official HF artifact reports the complete timed workload count but "
            "does not contain a separate failed-request field; native vLLM summaries "
            "must explicitly report zero failures.",
            "",
            "## GPU-active alignment",
            "",
            (
                f"Traces use trailing **{comparison['algorithm']['rolling_window_seconds']:g}s** "
                "rolling means. Each run's first sustained GPU-active sample is aligned "
                "to `t=0`; the raw tail is retained so completion remains visible."
            ),
            "",
            "| System | Active start | Leading trim | Aligned duration | Mean active GPU | Mean active power |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        alignment = run["alignment"]
        lines.append(
            f"| {run['label']} | `{alignment['active_start_timestamp']}` | "
            f"{alignment['leading_trim_seconds']:.2f}s | "
            f"{alignment['aligned_duration_seconds'] / 60:.2f} min | "
            f"{alignment['active_raw_gpu_utilization_pct_mean']:.2f}% | "
            f"{alignment['active_raw_power_w_mean']:.2f} W |"
        )
    lines.extend(
        [
            "",
            "## Artifact provenance",
            "",
            "| System | Result JSON | Result SHA-256 | GPU trace | Trace SHA-256 |",
            "|---|---|---|---|---|",
        ]
    )
    for run in runs:
        lines.append(
            f"| {run['label']} | `{run['result_source']}` | "
            f"`{run['result_sha256']}` | `{run['trace_source']}` | "
            f"`{run['trace_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Comparison contract",
            "",
            comparison["algorithm"]["comparison_contract"],
            "",
            f"- Infer length: `{official['infer_length']}`",
            f"- Timed workload: `{official['count']:,}` images",
            (
                "- Unique/replay workload: "
                f"`{official['workload_metadata'].get('unique_image_count')}` unique × "
                f"`{official['workload_metadata'].get('replay_count')}` replays"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_comparison_outputs(data: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_30k_speedup_chart(data, output_dir)
    gpu_active_comparison_chart(data, output_dir)
    (output_dir / "report_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "NEMOTRON_OCR_COMPARISON.md").write_text(
        comparison_markdown_report(data)
    )


def write_outputs(data: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_chart(data, output_dir)
    native_scaling_chart(data, output_dir)
    settings_chart(data, output_dir)
    detector_chart(data, output_dir)
    deployment_optimization_chart(data, output_dir)
    gpu_comparison = data["gpu_active_comparison"]
    if gpu_comparison["available"]:
        matched_30k_speedup_chart(data, output_dir)
        gpu_active_comparison_chart(data, output_dir)
    else:
        for stem in ("matched_30k_speedup", "gpu_active_comparison"):
            for suffix in ("png", "svg"):
                stale_path = output_dir / f"{stem}.{suffix}"
                if stale_path.exists():
                    stale_path.unlink()
    (output_dir / "report_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "NEMOTRON_OCR_VLLM_REPORT.md").write_text(markdown_report(data))


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest.resolve())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReportError("manifest must be an object with schema_version=1")
    gpu_overrides = gpu_comparison_overrides_from_args(args)
    if args.comparison_only:
        data = collect_comparison_report_data(
            manifest,
            args.results_root.resolve(),
            args.gpu_rolling_window_seconds,
            gpu_overrides,
        )
        write_comparison_outputs(data, args.output_dir.resolve())
    else:
        data = collect_report_data(
            manifest,
            args.results_root.resolve(),
            args.hf_baseline_json,
            args.hf_baseline_metric,
            args.gpu_rolling_window_seconds,
            args.strict,
            gpu_overrides,
        )
        write_outputs(data, args.output_dir.resolve())
    print(f"Wrote benchmark report to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
