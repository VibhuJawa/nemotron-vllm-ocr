# Nemotron OCR v2: matched HF and vLLM comparison

The optimized run measured **85.3941 images/s**, **2.73×** the official NVIDIA/HF in-process baseline and **1.89×** the selected clean-model vLLM baseline.

All values below are loaded from the selected result artifacts; the generator does not embed a throughput constant.

![Matched throughput and speedup](matched_30k_speedup.png)

[Vector chart](matched_30k_speedup.svg)

![GPU-active aligned utilization and power](gpu_active_comparison.png)

[Vector chart](gpu_active_comparison.svg)

## Matched throughput

| System | Workflow | Timed images | Throughput | vs official HF | vs clean vLLM | Failures field |
|---|---|---:|---:|---:|---:|---:|
| Official NVIDIA/HF in-process baseline | direct NVIDIA/HF in-process pipeline | 30,000 | 31.2457 img/s | 1.000× | 0.692× | not encoded |
| Tuned clean-model vLLM baseline | JPEG → one native vLLM /pooling queue | 30,000 | 45.1634 img/s | 1.445× | 1.000× | 0 |
| Optimized native vLLM | dispatcher → eight MPS vLLM replicas | 30,000 | 85.3941 img/s | 2.733× | 1.891× | 0 |

The official HF artifact reports the complete timed workload count but does not contain a separate failed-request field; native vLLM summaries must explicitly report zero failures.

## GPU-active alignment

Traces use trailing **15s** rolling means. Each run's first sustained GPU-active sample is aligned to `t=0`; the raw tail is retained so completion remains visible.

| System | Active start | Leading trim | Aligned duration | Mean active GPU | Mean active power |
|---|---|---:|---:|---:|---:|
| Official NVIDIA/HF in-process baseline | `2026/07/08 23:28:13.671` | 1.28s | 15.98 min | 50.79% | 235.23 W |
| Tuned clean-model vLLM baseline | `2026/07/09 04:35:42.017` | 3.52s | 11.15 min | 73.80% | 300.80 W |
| Optimized native vLLM | `2026/07/09 05:11:44.499` | 0.77s | 5.91 min | 99.93% | 395.92 W |

## Artifact provenance

| System | Result JSON | Result SHA-256 | GPU trace | Trace SHA-256 |
|---|---|---|---|---|
| Official NVIDIA/HF in-process baseline | `publication_runs/hf_official_inprocess_b64_d32_jpeg_q100_30k/result.json` | `4c0343d3ae08dbf1ca06bf992e05301913463c534139a9d55ff1d3d53954f4e7` | `publication_runs/hf_official_inprocess_b64_d32_jpeg_q100_30k/gpu_trace.csv` | `12bce0eed91f359b4195c2e4a3b3c8d372b35ae11099754a0eb126212c79cddd` |
| Tuned clean-model vLLM baseline | `native_vllm_clean_baseline_validation/clean-baseline-winner-30k/joint-d16-rw4-s64-c128/benchmark/summary.json` | `3045455fb8727d6ded88ec8b6c33d03c20a8a660c151a26d02703be82b7d9ddf` | `native_vllm_clean_baseline_validation/clean-baseline-winner-30k/joint-d16-rw4-s64-c128/benchmark/gpu_trace.csv` | `6bc1641d94a5097f91c4a53bbc98286322675d75abf1ed30f24ba41c29147f14` |
| Optimized native vLLM | `native_vllm_config_sweeps/final-optimized-30k/r8-seq40-c64-mps25-rec128-exact-fusions/benchmark/summary.json` | `4661f889a476adbc12eeaa3dea19da6058b5032bedf55cace1d4ddbf352e8252` | `native_vllm_config_sweeps/final-optimized-30k/r8-seq40-c64-mps25-rec128-exact-fusions/benchmark/gpu_trace.csv` | `cc4c18b0eabae7ef527a59ba8962539ed0684bf334c388b3d368b49cadf6824d` |

## Comparison contract

Matched A100 publication runs over the same 1,000-image JPEG Q100 4:4:4 corpus, repeated 30 times (30,000 timed images) at infer_length=1024.

- Infer length: `1024`
- Timed workload: `30,000` images
- Unique/replay workload: `1000` unique × `30` replays
