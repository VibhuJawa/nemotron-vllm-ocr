# Exact and portable sweep configs

`exact/` contains byte-for-byte copies used by the benchmark harness. They keep
the original A100 UUID and `/raid`/`/home` paths so their SHA-256 values match
the provenance records. In particular:

- `native_vllm_exact_bn_gate.json` SHA-256:
  `8a84a01f6943384bea20b504fa27c241ec026b6ac0b8554a8c721ff0dbc551d6`.
- `native_vllm_clean_baseline_validation.json` SHA-256:
  `8a39cb0d0e547502e585e62d32f827c8bdab2e7c50c53dad16241f1bd322f95d`.
- `native_vllm_clean_baseline_sweep.json` SHA-256:
  `79e9892164858e2044820ddf5eeaaba55c331290ec7cdf7f36d84bd1df8e21a7`.

`portable/` contains readable one-run configs for the two publication vLLM
conditions. Replace every `/path/to/...` value and `GPU-REPLACE-WITH-UUID`
before execution. Paths are resolved relative to each config file, so the
bundled benchmark client uses `../../scripts/benchmark_multi_endpoint_pooling.py`.

Both portable configs require clean repositories. The optimized portable config
pins the follow-up vLLM and model PR commits into which the benchmark-time
changes were folded. The original optimized run intentionally recorded a dirty
source state on top of earlier base commits; its exact `provenance.json`
contains binary patch and untracked-file hashes, and those captured files remain
under `../runs/optimized-vllm/source_state/` for byte-level auditability.

Run from any directory with:

```bash
python results/a100-2026-07-09-final-85-imgs/scripts/run_native_vllm_sweep.py \
  --config results/a100-2026-07-09-final-85-imgs/configs/portable/final-optimized-30k.json
```
