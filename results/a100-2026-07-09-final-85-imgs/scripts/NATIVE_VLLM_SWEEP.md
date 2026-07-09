# Native vLLM OCR replica sweeps

`run_native_vllm_sweep.py` launches one or more native vLLM pooling servers,
then drives them through the work-conserving queue in
`benchmark_multi_endpoint_pooling.py`. The supplied workload is JSONL whose
`prompt.data` values are `data:image/jpeg;base64,...`; the runner verifies the
full base64 payload and JPEG SOI/EOI bytes for every row before a real run.

Inspect the fully expanded commands without starting MPS, a server, a GPU
trace, or a client:

```bash
/raid/vjawa/tmp/ocr_optimization/venv/bin/python \
  /home/nfs/vjawa/ocr_optimization/scripts/run_native_vllm_sweep.py \
  --config /home/nfs/vjawa/ocr_optimization/configs/native_vllm_replica_sweep.json \
  --only r8-c96-mps25-rec128 \
  --label rec128-dry-run \
  --dry-run
```

Run all configurations sequentially:

```bash
/raid/vjawa/tmp/ocr_optimization/venv/bin/python \
  /home/nfs/vjawa/ocr_optimization/scripts/run_native_vllm_sweep.py \
  --config /home/nfs/vjawa/ocr_optimization/configs/native_vllm_replica_sweep.json \
  --label rec128-concurrency-sweep
```

The runner takes a per-GPU advisory lock and refuses a GPU with existing
compute processes. `--allow-busy-gpu` is an explicit escape hatch, not the
default. Each configuration gets a unique local MPS pipe directory. Cleanup
sends `quit` only through that owned directory; it never addresses the global
MPS control socket. Servers and the benchmark run in separate process groups,
which are terminated on failure, timeout, Ctrl-C, or SIGTERM.
Inherited CUDA/framework/model tuning variables and `PYTHONPATH` are scrubbed
before constructing every child environment, including configurations with
MPS disabled; only config-declared values are reapplied. Harness-owned server
and client flags cannot be replaced through `extra_args`.
Real runs resolve an index or UUID selector to the physical GPU UUID before
constructing MPS and server environments, avoiding CUDA's MPS index remapping.

Each result directory contains:

- `provenance.json`: exact argv, relevant environment, expanded configuration,
  target GPU, dataset SHA-256 and JPEG row count, vLLM/model Git commits and
  replayable binary patches and copies of untracked files, plus
  benchmark/client/extension hashes;
- `servers/server_*.log`, `benchmark.log`, and copied MPS logs;
- `benchmark/summary.json` and `benchmark/gpu_trace.csv` from the native queue;
- `run_summary.json`, with success/failure and cleanup status.

The config has `defaults` plus a `sweep` list. Nested `server`, `benchmark`,
and `env` objects are deep-merged for each named run. The high-value sweep
fields are:

- `server.replicas`, `server.max_num_seqs`,
  `server.renderer_num_workers`, and
  `server.mps_active_thread_percentage` (one integer or a list per replica);
- detector, recognizer, relational chunk sizes and `infer_length` under
  `server`;
- `benchmark.concurrency_per_endpoint`, warmups, replay count, and total
  prompts.

`benchmark.num_prompts / benchmark.replay_count` must equal the JSONL row
count. This makes sustained replay runs explicit and prevents accidentally
timing too little work.
