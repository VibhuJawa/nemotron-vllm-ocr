from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_ocr_benchmark_report.py"
SPEC = importlib.util.spec_from_file_location("generate_ocr_benchmark_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _run(run_id: str, label: str, throughput: float) -> dict[str, object]:
    official = 10.0
    clean = 20.0
    return {
        "id": run_id,
        "label": label,
        "chart_label": label,
        "throughput": throughput,
        "speedup_vs_official_hf": throughput / official,
        "speedup_vs_clean_vllm": throughput / clean,
        "workflow": "test workflow",
        "series": [
            {
                "time_s": 0.0,
                "gpu_utilization_pct": 25.0,
                "power_w": 100.0,
                "memory_used_mib": 1024.0,
            },
            {
                "time_s": 60.0,
                "gpu_utilization_pct": 90.0,
                "power_w": 250.0,
                "memory_used_mib": 2048.0,
            },
        ],
    }


def _comparison_data() -> dict[str, object]:
    return {
        "gpu_active_comparison": {
            "available": True,
            "chart_context": {
                "hardware": "1× Test GPU",
                "workload": "matched test workload",
            },
            "algorithm": {
                "rolling_window_seconds": 15.0,
                "max_plot_points_per_run": 100,
                "gpu_utilization_threshold_pct": 20.0,
                "active_detection_window_seconds": 5.0,
                "minimum_active_sample_fraction": 0.6,
            },
            "runs": [
                _run("official_hf", "Official HF", 10.0),
                _run("vllm_baseline", "Tuned clean vLLM", 20.0),
                _run("optimized_native_vllm", "Optimized vLLM", 50.0),
            ],
        }
    }


def test_apply_gpu_comparison_overrides_is_explicit_and_nonmutating(
    tmp_path: Path,
) -> None:
    spec = {
        "runs": [
            {
                "id": run_id,
                "label": "old label",
                "result_path": "old.json",
                "trace_path": "old.csv",
            }
            for run_id in (
                "official_hf",
                "vllm_baseline",
                "optimized_native_vllm",
            )
        ]
    }
    result_path = tmp_path / "winner.json"
    trace_path = tmp_path / "winner.csv"
    updated = REPORT.apply_gpu_comparison_overrides(
        spec,
        {
            "vllm_baseline": {
                "result_path": result_path,
                "trace_path": trace_path,
            }
        },
    )

    assert spec["runs"][1]["result_path"] == "old.json"
    assert updated["runs"][1]["result_path"] == str(result_path.resolve())
    assert updated["runs"][1]["trace_path"] == str(trace_path.resolve())
    assert updated["runs"][1]["label"] == "Tuned clean-model vLLM baseline"
    assert updated["runs"][1]["chart_label"] == "Tuned clean vLLM baseline"


def test_cli_override_requires_result_and_trace_pair(tmp_path: Path) -> None:
    args = argparse.Namespace(
        official_hf_result_json=tmp_path / "official.json",
        official_hf_trace_csv=None,
        tuned_clean_vllm_result_json=None,
        tuned_clean_vllm_trace_csv=None,
        optimized_vllm_result_json=None,
        optimized_vllm_trace_csv=None,
    )

    with pytest.raises(REPORT.ReportError, match="requires both"):
        REPORT.gpu_comparison_overrides_from_args(args)


def test_speedup_chart_uses_artifact_values_not_static_labels(tmp_path: Path) -> None:
    REPORT.matched_30k_speedup_chart(_comparison_data(), tmp_path)

    svg = (tmp_path / "matched_30k_speedup.svg").read_text()
    assert "2.00× · HF→clean vLLM" in svg
    assert "2.50× · clean→optimized" in svg
    assert "1.40× · native queue" not in svg
    assert "1.60× · replica boost" not in svg


def test_gpu_active_chart_matches_two_panel_requested_style(tmp_path: Path) -> None:
    REPORT.gpu_active_comparison_chart(_comparison_data(), tmp_path)

    svg = (tmp_path / "gpu_active_comparison.svg").read_text()
    assert "GPU-active aligned utilization and power" in svg
    assert "Average GPU utilization (%)" in svg
    assert "GPU power draw (W)" in svg
    assert "Minutes since sustained GPU activity began" in svg
    assert "GPU memory used" not in svg
