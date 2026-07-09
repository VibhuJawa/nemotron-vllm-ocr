#!/usr/bin/env python3
"""Aggregate and plot the isolated Hugging Face model PR #8 A/B benchmark."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BASELINE_COMMIT = "0e83e83f17943524b90afa6c0fd82ac2bc1a40ca"
PATCHED_COMMIT = "bb392d494b616d3a1692c3dbe59f63c1d2a8a7fa"
DATASET_SHA256 = "139c96ef75a85da440350722a95d9eb3bd21dd4155d43f7281253f63c07eaa16"
T_CRIT_95_DF2 = 4.302652729911275


def load_runs(prefix: str) -> list[dict]:
    return [
        json.loads((ROOT / f"{prefix}-r{index}.json").read_text())
        for index in range(1, 4)
    ]


def weighted_trace_mean(runs: list[dict], key: str) -> float:
    numerator = sum(
        run["gpu_trace_timed"][key] * run["gpu_trace_timed"]["samples"]
        for run in runs
    )
    denominator = sum(run["gpu_trace_timed"]["samples"] for run in runs)
    return numerator / denominator


def aggregate(runs: list[dict]) -> dict:
    rates = [run["images_per_second"] for run in runs]
    total_images = sum(run["timed_workload_image_count"] for run in runs)
    total_elapsed_s = sum(run["elapsed_s"] for run in runs)
    return {
        "repetitions": len(runs),
        "total_timed_images": total_images,
        "total_elapsed_s": total_elapsed_s,
        "images_per_second_mean": statistics.mean(rates),
        "images_per_second_stdev": statistics.stdev(rates),
        "images_per_second_pooled": total_images / total_elapsed_s,
        "ms_per_image_pooled": 1000.0 * total_elapsed_s / total_images,
        "warmup_s_mean": statistics.mean(run["warmup_s"] for run in runs),
        "gpu_util_pct_timed_weighted_mean": weighted_trace_mean(
            runs, "gpu_util_pct_avg"
        ),
        "gpu_power_w_timed_weighted_mean": weighted_trace_mean(
            runs, "gpu_power_w_avg"
        ),
        "gpu_power_w_max": max(
            run["gpu_trace_timed"]["gpu_power_w_max"] for run in runs
        ),
        "gpu_memory_used_mib_max": max(
            run["gpu_trace_timed"]["gpu_memory_used_mib_max"] for run in runs
        ),
        "pytorch_peak_allocated_gb_max": max(
            run["peak_gpu_memory_allocated_gb"] for run in runs
        ),
        "pytorch_peak_reserved_gb_max": max(
            run["peak_gpu_memory_reserved_gb"] for run in runs
        ),
    }


def validate(baseline: list[dict], patched: list[dict]) -> None:
    all_runs = baseline + patched
    expected = {
        "backend": "official_nvidia_pipeline",
        "uses_vllm": False,
        "dataset_kind": "jpeg_byte_jsonl",
        "unique_image_count": 1000,
        "replay_count": 10,
        "timed_workload_image_count": 10000,
        "batch_size": 64,
        "warmup_image_count": 128,
        "infer_length": 1024,
        "merge_level": "paragraph",
        "detector_max_batch_size": 32,
        "recognizer_chunk_size": 128,
        "relational_chunk_size": 128,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
    }
    for run in all_runs:
        for key, value in expected.items():
            assert run[key] == value, (key, run[key], value)
        assert run["model_repo_provenance"]["dirty"] is False
    assert {
        run["model_repo_provenance"]["commit"] for run in baseline
    } == {BASELINE_COMMIT}
    assert {
        run["model_repo_provenance"]["commit"] for run in patched
    } == {PATCHED_COMMIT}
    assert len({run["cuda_visible_devices"] for run in all_runs}) == 1


def main() -> None:
    baseline = load_runs("baseline")
    patched = load_runs("patched")
    validate(baseline, patched)

    baseline_rates = [run["images_per_second"] for run in baseline]
    patched_rates = [run["images_per_second"] for run in patched]
    paired_uplifts = [
        100.0 * (patched_rate / baseline_rate - 1.0)
        for baseline_rate, patched_rate in zip(baseline_rates, patched_rates)
    ]
    paired_mean = statistics.mean(paired_uplifts)
    paired_stdev = statistics.stdev(paired_uplifts)
    paired_half_width = T_CRIT_95_DF2 * paired_stdev / math.sqrt(3)
    paired_t = paired_mean / (paired_stdev / math.sqrt(3))
    # The two-sided Student-t p-value has this closed form for two degrees of freedom.
    paired_p = 1.0 - abs(paired_t) / math.sqrt(paired_t**2 + 2.0)

    baseline_aggregate = aggregate(baseline)
    patched_aggregate = aggregate(patched)
    pooled_uplift = 100.0 * (
        patched_aggregate["images_per_second_pooled"]
        / baseline_aggregate["images_per_second_pooled"]
        - 1.0
    )

    run_order = [
        "baseline-r1",
        "patched-r1",
        "patched-r2",
        "baseline-r2",
        "baseline-r3",
        "patched-r3",
    ]
    summary = {
        "schema_version": 1,
        "benchmark_date": "2026-07-09",
        "comparison": "isolated Hugging Face in-process model-source A/B",
        "hardware": {
            "device": baseline[0]["cuda_device_name"],
            "gpu_uuid": baseline[0]["cuda_visible_devices"],
        },
        "workload": {
            "dataset": "bo767_1k_pooling_jpeg_q100_444.jsonl",
            "dataset_sha256": DATASET_SHA256,
            "dataset_jsonl_bytes": 856608912,
            "decoded_jpeg_bytes": baseline[0]["dataset_compressed_bytes"],
            "encoding": "JPEG quality 100, 4:4:4, base64 bytes in JSONL",
            "unique_images_per_run": 1000,
            "replay_count_per_run": 10,
            "timed_images_per_run": 10000,
            "repetitions_per_condition": 3,
            "total_timed_images_per_condition": 30000,
            "batch_size": 64,
            "warmup_images_per_run": 128,
            "infer_length": 1024,
            "merge_level": "paragraph",
            "detector_max_batch_size": 32,
            "recognizer_chunk_size": 128,
            "relational_chunk_size": 128,
        },
        "software": {
            "baseline_model_commit": BASELINE_COMMIT,
            "patched_model_commit": PATCHED_COMMIT,
            "model_pr": "https://huggingface.co/nvidia/nemotron-ocr-v2/discussions/8",
            "benchmark_driver_commit": "ec7b7fb48c8dad71722d16d0b901804daea72cc3",
            "benchmark_driver_sha256": "2f79e737083a4a67f1e83243cc7958bbd3d79ba06dd4a7ab2e4e533a225a7350",
            "baseline_extension_sha256": "05ac1b8d35a36b82595257ec85f081adb922c088249da9a83eaf6bc2b963ec1e",
            "patched_extension_sha256": "9e36cfbc656fb91d3dbafa6cf2aeb7e47c02477f763b063ba0a2bff711fe7794",
            "torch_version": baseline[0]["torch_version"],
            "torch_cuda_version": baseline[0]["torch_cuda_version"],
        },
        "execution_order": run_order,
        "runs": [
            {
                "id": f"{kind}-r{index}",
                "result": f"{kind}-r{index}.json",
                "gpu_trace": f"{kind}-r{index}-gpu-trace.csv",
                "images_per_second": run["images_per_second"],
                "elapsed_s": run["elapsed_s"],
                "gpu_util_pct_timed_avg": run["gpu_trace_timed"][
                    "gpu_util_pct_avg"
                ],
                "gpu_power_w_timed_avg": run["gpu_trace_timed"][
                    "gpu_power_w_avg"
                ],
            }
            for kind, runs in (("baseline", baseline), ("patched", patched))
            for index, run in enumerate(runs, 1)
        ],
        "aggregate": {
            "baseline": baseline_aggregate,
            "patched": patched_aggregate,
            "pooled_uplift_pct": pooled_uplift,
            "paired_uplift_pct_by_repetition": paired_uplifts,
            "paired_uplift_pct_mean": paired_mean,
            "paired_uplift_pct_stdev": paired_stdev,
            "paired_uplift_pct_95_ci": [
                paired_mean - paired_half_width,
                paired_mean + paired_half_width,
            ],
            "paired_t_statistic": paired_t,
            "paired_t_degrees_of_freedom": 2,
            "paired_t_two_sided_p": paired_p,
        },
        "limitations": [
            "The confidence interval uses a paired Student-t calculation with only three repetitions.",
            "The same deterministic 1,000-page workload is replayed; this measures timing, not OCR accuracy.",
            "Absolute throughput should be compared within this A/B because it used a different physical A100 from the earlier deployment study.",
            "The patched sm_80 extension binary was rebuilt earlier from the exact committed C++ sources and is pinned by SHA-256; this repackaging host lacked Python development headers.",
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "figure.facecolor": "#fbfbfc",
            "axes.facecolor": "#ffffff",
        }
    )
    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    ax_rate = fig.add_subplot(grid[:, 0])
    ax_uplift = fig.add_subplot(grid[0, 1])
    ax_metrics = fig.add_subplot(grid[1, 1])

    pair_colors = ["#4c78a8", "#72b7b2", "#f58518"]
    for index, (base_rate, patch_rate, color) in enumerate(
        zip(baseline_rates, patched_rates, pair_colors), 1
    ):
        ax_rate.plot(
            [0, 1],
            [base_rate, patch_rate],
            marker="o",
            markersize=9,
            linewidth=2.2,
            color=color,
            label=f"Repetition {index}",
        )
        ax_rate.text(-0.04, base_rate, f"{base_rate:.3f}", ha="right", va="center")
        ax_rate.text(1.04, patch_rate, f"{patch_rate:.3f}", ha="left", va="center")

    mean_base = baseline_aggregate["images_per_second_mean"]
    mean_patch = patched_aggregate["images_per_second_mean"]
    ax_rate.plot(
        [0, 1],
        [mean_base, mean_patch],
        color="#1f2937",
        linewidth=4,
        marker="D",
        markersize=9,
        label="Mean",
        zorder=5,
    )
    ax_rate.annotate(
        f"+{pooled_uplift:.3f}% ↑",
        xy=(1, mean_patch),
        xytext=(0.54, mean_patch + 0.075),
        arrowprops={"arrowstyle": "->", "color": "#d1495b", "lw": 2.4},
        color="#d1495b",
        fontsize=16,
        fontweight="bold",
        ha="center",
    )
    ax_rate.set_xticks([0, 1], ["Clean upstream\n0e83e83f", "Model PR #8\nbb392d4"])
    ax_rate.set_ylabel("Timed throughput (images/s)")
    ax_rate.set_title("Every patched run is faster than its clean control", loc="left")
    ax_rate.set_xlim(-0.28, 1.30)
    ax_rate.set_ylim(30.85, 31.80)
    ax_rate.grid(axis="y", color="#d8dee8", linewidth=0.8)
    ax_rate.legend(loc="lower right", frameon=False)

    bars = ax_uplift.bar(
        [1, 2, 3], paired_uplifts, color=pair_colors, width=0.62
    )
    ax_uplift.axhspan(
        paired_mean - paired_half_width,
        paired_mean + paired_half_width,
        color="#d1495b",
        alpha=0.12,
        label="95% paired CI",
    )
    ax_uplift.axhline(
        paired_mean,
        color="#d1495b",
        linewidth=2,
        linestyle="--",
        label=f"Mean {paired_mean:.3f}%",
    )
    ax_uplift.bar_label(bars, labels=[f"+{value:.2f}%" for value in paired_uplifts])
    ax_uplift.set_xticks([1, 2, 3], ["Pair 1", "Pair 2", "Pair 3"])
    ax_uplift.set_ylabel("Patched uplift (%)")
    ax_uplift.set_title("Paired uplift is consistent", loc="left")
    ax_uplift.set_ylim(0, 2.75)
    ax_uplift.grid(axis="y", color="#d8dee8", linewidth=0.8)
    ax_uplift.legend(loc="lower right", frameon=False, fontsize=10)

    ax_metrics.axis("off")
    metric_text = (
        "POOLED RESULT\n"
        f"Clean HF       {baseline_aggregate['images_per_second_pooled']:.4f} images/s\n"
        f"Patched HF     {patched_aggregate['images_per_second_pooled']:.4f} images/s\n"
        f"Model-only lift  +{pooled_uplift:.3f}%\n\n"
        "PAIRED UNCERTAINTY\n"
        f"95% interval   +{paired_mean - paired_half_width:.2f}% to "
        f"+{paired_mean + paired_half_width:.2f}%\n"
        f"Paired t test  p={paired_p:.4f} (n=3)\n\n"
        "GPU TELEMETRY (clean → patched)\n"
        f"Avg util       {baseline_aggregate['gpu_util_pct_timed_weighted_mean']:.2f}% → "
        f"{patched_aggregate['gpu_util_pct_timed_weighted_mean']:.2f}%\n"
        f"Avg power      {baseline_aggregate['gpu_power_w_timed_weighted_mean']:.2f} W → "
        f"{patched_aggregate['gpu_power_w_timed_weighted_mean']:.2f} W\n"
        f"Warmup         {baseline_aggregate['warmup_s_mean']:.2f} s → "
        f"{patched_aggregate['warmup_s_mean']:.2f} s"
    )
    ax_metrics.text(
        0.02,
        0.98,
        metric_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=12,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#f3f6fa",
            "edgecolor": "#c7d0dd",
        },
    )

    fig.suptitle(
        "Nemotron OCR v2 model PR #8: isolated Hugging Face in-process A/B",
        fontsize=21,
        fontweight="bold",
        ha="left",
        x=0.03,
        y=0.982,
    )
    fig.text(
        0.03,
        0.936,
        "One A100-SXM4-80GB · same 1,000 Bo767 JPEG-byte pages · 3 × 10K timed images per condition · infer_length=1024",
        fontsize=13,
        color="#3f4a5a",
    )
    fig.text(
        0.03,
        0.025,
        "Model patch only; no vLLM in either condition. Startup and warmup excluded from timed throughput. Raw JSON/CSV traces are published beside this chart.",
        fontsize=10.5,
        color="#596579",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.875,
        bottom=0.125,
        wspace=0.11,
        hspace=0.16,
    )
    fig.savefig(ROOT / "model_pr8_hf_ab.png", dpi=180)
    svg_path = ROOT / "model_pr8_hf_ab.svg"
    fig.savefig(svg_path)
    # Matplotlib preserves spaces at the ends of multiline SVG path commands.
    # They are semantically irrelevant but make Git's whitespace check noisy.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
        + "\n"
    )


if __name__ == "__main__":
    main()
