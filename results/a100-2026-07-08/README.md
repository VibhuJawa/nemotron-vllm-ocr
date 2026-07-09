# Nemotron OCR v2 on A100: interim vLLM throughput

This report includes only `infer_length=1024` throughput artifacts. Headline results use 1,000 distinct document images exactly once; sustained replay is reported separately and never substituted for the 1K-unique result.

## Summary

- The offline vLLM replica harness rises from **15.28** to **63.76 images/s**, a **4.17×** steady-state speedup on the 1K-unique workload.
- Native vLLM serving scales from **44.72** at one replica to **65.49 images/s** at eight replicas (**1.46×**) on the separate 1K-unique JPEG workload.
- The 512-document settings sweeps peak at `max_num_seqs=64` (**46.21 images/s**), renderer workers=4 (**46.21 images/s**), and detector batch=16 (**47.69 images/s**).
- **HF baseline pending.** No HF throughput value is inferred, copied from a different resolution, or otherwise fabricated. Add its result path to `configs/ocr_benchmark_report.json` (or pass `--hf-baseline-json`) after the 1024-resolution 1K-unique run completes.
- The final long work-conserving native serving run reaches **70.12 images/s**
  over 30,000 timed requests (1,000 unique JPEG images × 30 replays), with
  30,000/30,000 successful responses, **99.74%** average GPU utilization, and
  **393.17 W** average GPU power. This crosses the 70 images/s target without
  changing `infer_length=1024`.
- The earlier fixed-shard sustained replay run is **68.62 images/s**. The
  work-conserving dispatcher removes slow-shard tail imbalance while each vLLM
  replica retains its native queue and continuous batching.
- The rec128 conservative profile completed the same 30K workload at
  **69.54 images/s**. Rec64 remains labeled as the throughput profile because
  its batching-quality impact is not separable from observed run-to-run OCR
  nondeterminism with the current 32-image diagnostic.
- **GPU-active comparison pending.** The generator will not draw the three-run chart until every measured result and raw trace is present.

## Baseline vs optimized

![Baseline versus optimized throughput](baseline_vs_optimized.png)

[Vector chart](baseline_vs_optimized.svg)

| Run | Throughput | Speedup vs baseline | Workload | Configuration |
|---|---:|---:|---|---|
| Single replica baseline | 15.28 images/s | 1.00× | 1K unique | 1 replica, s64, detector batch 8 |
| Optimized 16 replicas | 63.76 images/s | 4.17× | 1K unique | 16 replicas, s32, MPS 25%, queue chunk 32 |

These two values come from the offline replica harness and PNG inputs. They are directly comparable to each other, not to the native HTTP/JPEG serving series below.

## Native vLLM replica scaling

![Native vLLM scaling](native_vllm_scaling.png)

[Vector chart](native_vllm_scaling.svg)

| Replicas | Throughput | Speedup vs r1 | Workload | Detector batch |
|---:|---:|---:|---|---:|
| 1 | 44.72 images/s | 1.00× | 1K unique | 8 |
| 2 | 57.15 images/s | 1.28× | 1K unique | 8 |
| 4 | 59.53 images/s | 1.33× | 1K unique | 8 |
| 8 | 65.49 images/s | 1.46× | 1K unique | 16 |

The 8-replica point also changes detector batch from 8 to 16, so this is a deployment scaling curve rather than a strictly replica-only ablation.

## Latest sustained native serving run

| Metric | Result |
|---|---:|
| Completed requests | 30,000 / 30,000 |
| Aggregate throughput | **70.12 images/s** |
| Timed duration | 427.86 s |
| Unique documents | 1,000 JPEG Q100 4:4:4 images |
| Replays | 30× |
| vLLM replicas | 8 |
| Concurrency per replica | 128 |
| Average / maximum GPU utilization | 99.74% / 100% |
| Average / maximum GPU power | 393.17 W / 446.51 W |
| Peak observed GPU memory | 67,555 MiB |

This run uses a client-side, work-conserving dispatcher that sends the next
request to whichever native vLLM `/pooling` replica becomes free. It introduces
no intermediate HTTP proxy. Global dispatch prevents a statically assigned slow
shard from determining the whole run, while vLLM remains responsible for the
queue and batching inside every replica.

Artifacts: [`optimized-vllm-r8-rec64-30k.json`](optimized-vllm-r8-rec64-30k.json)
and
[`optimized-vllm-r8-rec64-30k-gpu-trace.csv`](optimized-vllm-r8-rec64-30k-gpu-trace.csv).
Exact settings, hardware, source state, hashes, and pending items are recorded
in [`run-metadata.json`](run-metadata.json).

### Queueing architecture and Ray Serve

The measured deployment does not use Ray. Each replica is a native vLLM
pooling server: vLLM performs per-replica admission, continuous batching, and
execution scheduling, while a lightweight client-side dispatcher supplies the
next request to the first available replica.

Ray Serve is a sensible next experiment for production global queueing. The
recommended shape is one Serve ingress deployment routing by deployment handle
to long-lived replicas that embed vLLM `AsyncLLM`; this avoids adding a second
HTTP serialization hop. Ray Serve would own global admission, backpressure,
routing, and autoscaling, while vLLM retains continuous batching within each
replica. This has not yet been benchmarked and is not included in the throughput
claims above. Ray Serve also does not by itself provide arbitrary CUDA-tensor
handoff between separately served OCR stages.

## GPU-active aligned utilization and power

**Pending measured inputs; no chart or placeholder values are emitted.** Populate the following manifest fields after matched runs complete:

- Official NVIDIA/HF in-process: result_path, trace_path
- vLLM baseline: result_path, trace_path

Manifest comparison contract: Matched A100 runs over the same 1,000-image JPEG Q100 4:4:4 corpus, repeated 10 times at infer_length=1024.

Alignment and smoothing algorithm:

1. Parse timestamp, `utilization.gpu [%]`, and `power.draw [W]` directly from each raw `nvidia-smi` CSV; invalid metric rows are excluded.
2. Scan forward for the earliest sample at or above 20% whose next 5s window has mean GPU utilization at least 20% and at least 60% of samples at or above 20%.
3. Discard only samples before that point and independently set each run's detected point to `t=0`; retain the raw trace tail so workload completion remains visible.
4. Plot a sample-based trailing 15s arithmetic mean for both GPU utilization and power draw. The window is configurable with `--gpu-rolling-window-seconds`.

The chart is gated on all three runs sharing the same completed count, `infer_length`, workload kind, unique-image count, and replay count.

## Setting sweeps

![Native setting sweeps](native_settings_sweep.png)

[Vector chart](native_settings_sweep.svg)

| Sweep | Setting | Throughput |
|---|---:|---:|
| max_num_seqs (rw4) | 32 | 45.28 images/s |
| max_num_seqs (rw4) | 64 | 46.21 images/s |
| max_num_seqs (rw4) | 96 | 46.20 images/s |
| max_num_seqs (rw4) | 128 | 45.92 images/s |
| renderer workers (s64) | 1 | 44.10 images/s |
| renderer workers (s64) | 2 | 45.54 images/s |
| renderer workers (s64) | 4 | 46.21 images/s |

![Detector batch sweep](detector_batch_sweep.png)

[Vector chart](detector_batch_sweep.svg)

| Detector batch | Throughput |
|---:|---:|
| 8 | 46.51 images/s |
| 12 | 47.22 images/s |
| 16 | 47.69 images/s |
| 24 | 47.54 images/s |
| 32 | 47.22 images/s |

Both settings charts use 512 distinct documents and a deliberately truncated y-axis to make small tuning differences visible. They are ablations, not the 1K-unique headline.

## Accuracy and workload gates

- The controlled optimized Triton comparison **passed**: 0 region-count mismatches, 0 text mismatches, 100.0% text exact rate across 32 images, and maximum coordinate error 2.05e-07.
- A repeated baseline run itself **did not pass** the strict bitwise-style comparison (2 region-count mismatches; 93.5% text exact). This documents the observed nondeterminism envelope; `accuracy-valid` here means default `infer_length=1024` plus the passing controlled optimized-path check, not that every independent run is bitwise stable.

Workload labels are enforced from artifact counts and the curated source manifest:

- **1K unique:** 1,000 distinct page images, processed once.
- **512 unique:** 512 distinct page images, processed once, used only for tuning sweeps.
- **Sustained replay:** requires `replay_count > 1` and timed request count greater than the unique-image count.

## Methodology and provenance

- Hardware: NVIDIA A100 80GB.
- Resolution contract: `infer_length=1024`; no 768 results are ingested.
- Offline input: PNG source pages.
- Native serving input: JPEG quality 100, 4:4:4.
- Throughput values are read directly from the listed JSON/CSV artifacts.
- Full normalized data and SHA-256 provenance are in `report_data.json`.

| Use | Source artifact | SHA-256 (first 12) |
|---|---|---|
| Offline baseline | `a100_replica_sweeps/accuracy_baseline_1024/sweep_summary.json` | `293ba2ef6874` |
| Offline optimized | `a100_replica_sweeps/accuracy_optimized2_1024_mps_inproc_pct25_r16_q32_s32/sweep_summary.json` | `4cb686027ff4` |
| Native r1 | `native_vllm/native_single_s64_c128_jpeg_q100_444_1k.json` | `f3c030f2e03c` |
| Native r2 | `native_vllm_replicas/r2_s64_rw4_c128_mps50_jpeg_q100_1k/summary.json` | `6dbd4cac707f` |
| Native r4 | `native_vllm_replicas/r4_s64_rw4_c128_mps25_jpeg_q100_1k/summary.json` | `f6c852e66026` |
| Native r8 | `native_vllm_replicas/r8_s64_rw4_c128_det16_mps25_jpeg_q100_1k/summary.json` | `c328b3cb3533` |
| Sustained replay | `native_vllm_replicas/r8_s64_rw4_c128_det16_mps25_orjson_jpeg_q100_replay10/summary.json` | `2ab679589b5e` |
| Work-conserving sustained replay | `native-r8-dynamic-10k.json` | `b142ba4099e4` |
| Work-conserving GPU trace | `native-r8-dynamic-10k-gpu-trace.csv` | `65bd0593d5ae` |
| Settings sweep | `native_vllm_sweeps/native_jpeg_q100_targeted_20260708/summary.csv` | `bc2c62a7634f` |
| Detector sweep | `native_vllm_sweeps/native_detector_jpeg_q100_20260708/summary.csv` | `ba8fc77590b6` |
| Optimized Triton path vs repeated baseline | `equivalence_optimized_triton_1024_32_comparison.json` | `a7e52556f73c` |
| Repeated baseline vs baseline | `equivalence_baseline_repeat_1024_32_comparison.json` | `7c34bc3e291a` |

## Publication status

This checkpoint intentionally publishes only completed measurements. The clean
official NVIDIA/Hugging Face in-process baseline and the matched three-run
GPU-active utilization/power chart are still pending; no value is inferred from
the model card or a different workload. This report will be regenerated when
those measured artifacts are complete.
