# Portable chart inputs

The benchmark result JSON files record absolute lab paths for their paired GPU
traces. The report generator deliberately rejects a result/trace pair when that
recorded path does not resolve to the selected trace. These three derived JSON
files make chart regeneration portable by changing only `gpu_trace_csv` to the
corresponding bundle-relative path:

| Portable input | Exact raw source | Raw source SHA-256 |
| --- | --- | --- |
| `hf-result.json` | [`../../runs/hf-official/result.json`](../../runs/hf-official/result.json) | `4c0343d3ae08dbf1ca06bf992e05301913463c534139a9d55ff1d3d53954f4e7` |
| `clean-vllm-summary.json` | [`../../runs/clean-vllm/summary.json`](../../runs/clean-vllm/summary.json) | `3045455fb8727d6ded88ec8b6c33d03c20a8a660c151a26d02703be82b7d9ddf` |
| `optimized-vllm-summary.json` | [`../../runs/optimized-vllm/summary.json`](../../runs/optimized-vllm/summary.json) | `4661f889a476adbc12eeaa3dea19da6058b5032bedf55cace1d4ddbf352e8252` |

No throughput, timing, GPU summary, workload, endpoint, or model field is
changed. The exact raw artifacts remain authoritative for benchmark provenance.
