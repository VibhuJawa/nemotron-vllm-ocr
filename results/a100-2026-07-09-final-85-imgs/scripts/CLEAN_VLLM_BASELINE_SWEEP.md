# Clean native-vLLM baseline tuning

This sweep tunes the corrected, isolated vLLM baseline without importing any
optimized model source. It uses one native vLLM pooling server, the official
model worktree, the clean vLLM OCR worktree, and the same 1,000-document JPEG
byte dataset used by the optimized deployment.

## Pinned inputs

| Input | Path | Required identity |
|---|---|---|
| vLLM server code | `/raid/vjawa/tmp/ocr_optimization/vllm-baseline` | clean commit `62c2813a5af52bb2d60b1a2ee47bbfcf91280b9f` |
| Official model | `/raid/vjawa/tmp/ocr_optimization/nemotron-ocr-v2-baseline` | clean commit `0e83e83f17943524b90afa6c0fd82ac2bc1a40ca` |
| Official CUDA extension | `nemotron-ocr/src/nemotron_ocr_cpp/_nemotron_ocr_cpp*.so` | SHA-256 `05ac1b8d35a36b82595257ec85f081adb922c088249da9a83eaf6bc2b963ec1e` |
| JPEG-byte JSONL | `/raid/vjawa/tmp/ocr_optimization/data/bo767_1k_pooling_jpeg_q100_444.jsonl` | 1,000 rows; SHA-256 `139c96ef75a85da440350722a95d9eb3bd21dd4155d43f7281253f63c07eaa16` |

The runner checks these identities and repository cleanliness before GPU
resolution. A mismatch fails the run. Every child environment is built with
inherited MPS routing removed; this single-replica baseline has MPS explicitly
disabled. `env` is empty, so no optimized Nemotron kernel flags are enabled.

## Screen design

[`native_vllm_clean_baseline_sweep.json`](../configs/native_vllm_clean_baseline_sweep.json)
uses 10 complete replays: 10,000 timed JPEG requests plus 32 warmups for every
configuration. At the corrected baseline's 43.89 images/s, each timed window
lasts about 3.8 minutes, which is long enough to avoid a short-burst result.

The control is detector batch 8, one renderer worker, `max_num_seqs=64`, and
client concurrency 128. One-dimensional screens hold the other values at that
control:

- detector batch: 8, 12, 16, 24, 32;
- renderer workers: 1, 2, 4;
- `max_num_seqs`: 32, 64, 96, 128;
- client concurrency: 64, 96, 128, 192.

Three joint candidates test likely interactions: d16/rw4/s64/c128,
d32/rw4/s64/c128, and d16/rw4/s96/c192. All runs retain infer length 1024,
recognizer and relational chunks 128, eager execution, disabled async
scheduling, disabled access/stat logs, and the native `/pooling` queue.

Validate the complete command matrix without querying or initializing a GPU:

```bash
/raid/vjawa/tmp/ocr_optimization/venv/bin/python \
  /home/nfs/vjawa/ocr_optimization/scripts/run_native_vllm_sweep.py \
  --config /home/nfs/vjawa/ocr_optimization/configs/native_vllm_clean_baseline_sweep.json \
  --label clean-baseline-screen-dry \
  --dry-run > /tmp/clean-baseline-screen-plan.json
```

Run the 10K screen sequentially:

```bash
/raid/vjawa/tmp/ocr_optimization/venv/bin/python \
  /home/nfs/vjawa/ocr_optimization/scripts/run_native_vllm_sweep.py \
  --config /home/nfs/vjawa/ocr_optimization/configs/native_vllm_clean_baseline_sweep.json \
  --label clean-baseline-screen-10k
```

The output root is
`/raid/vjawa/tmp/ocr_optimization/results/native_vllm_clean_baseline_sweeps`.
Each configuration restarts the clean server, waits for `/health`, runs the
same work-conserving native queue, records the GPU trace, and tears the server
down before moving to the next configuration.

## Matched 30K validation

After ranking the screen, use
[`native_vllm_clean_baseline_validation.json`](../configs/native_vllm_clean_baseline_validation.json).
It contains the same named matrix at 30 full replays, so the selected winner
can be compared to a fresh corrected control with exactly 30,000 timed images:

```bash
WINNER=joint-d16-rw4-s64-c128  # replace with the actual 10K winner
/raid/vjawa/tmp/ocr_optimization/venv/bin/python \
  /home/nfs/vjawa/ocr_optimization/scripts/run_native_vllm_sweep.py \
  --config /home/nfs/vjawa/ocr_optimization/configs/native_vllm_clean_baseline_validation.json \
  --only control-d8-rw1-s64-c128 \
  --only "$WINNER" \
  --label clean-baseline-control-vs-winner-30k
```

The prior corrected control was 43.8927 images/s over 30,000 images. It should
not be replaced in publication artifacts until the fresh control and selected
winner both complete with 30,000 responses, zero failures, valid OCR response
envelopes, matching pinned inputs, and uncontaminated GPU traces.
