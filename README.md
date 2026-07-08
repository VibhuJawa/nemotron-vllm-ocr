# Nemotron OCR v2 on vLLM

This repository contains a small vLLM pooling/plugin wrapper for
[nvidia/nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2).

The important caveat: Nemotron OCR v2 is not a native causal language model that
vLLM can load directly. The Hugging Face repository ships a Python OCR pipeline
with detector, recognizer, relational model, and custom CUDA extension. This
project registers a tiny vLLM pooling model plus an IO processor plugin; the
vLLM engine receives image paths, batches them as plugin prompts, and calls
`NemotronOCRV2` inside the vLLM worker.

That makes sense when you need a vLLM-shaped integration point or want to prove
the OCR pipeline can live behind vLLM's plugin API. It is not expected to beat
the direct `NemotronOCRV2` Python API for standalone OCR latency because the
actual OCR compute remains the NVIDIA pipeline and vLLM adds scheduling and
serialization overhead.

## Repository Layout

- `src/nemotron_ocr_vllm/` registers the vLLM model and IO processor plugins.
- `model-config/config.json` is the minimal vLLM model config used by the wrapper.
- `run_vllm_ocr.py` runs OCR through the vLLM plugin path.
- `benchmarks/benchmark_ocr.py` compares direct Nemotron OCR with the vLLM wrapper.
- `notebooks/nemotron_ocr_vllm_demo.ipynb` is a runnable notebook version.
- `examples/make_sample_image.py` generates a small OCR sample image.
- `docs/design.md` explains why this uses vLLM's plugin API rather than the
  native VLM model path used by Nemotron Parse.

## Setup

Use Python 3.12 on a Linux machine with an NVIDIA GPU and a working CUDA toolkit.
Nemotron OCR builds a CUDA extension, so PyTorch's CUDA major version and the
available `nvcc` toolkit major version must match.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "vllm>=0.24.0"
```

Clone and install the NVIDIA OCR package:

```bash
git lfs install
git clone https://huggingface.co/nvidia/nemotron-ocr-v2
cd nemotron-ocr-v2/nemotron-ocr
pip install --no-build-isolation -v .
```

Then install this wrapper:

```bash
cd /path/to/nemotron-vllm-ocr
pip install -e ".[dev]"
```

If your Hugging Face cache points at a restricted location, set a writable cache:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HOME/.cache/huggingface/hub"
```

## Run

Generate a sample image:

```bash
python examples/make_sample_image.py
```

Run the vLLM-backed OCR path:

```bash
python run_vllm_ocr.py examples/sample_invoice.png
```

Run two images in one vLLM plugin call:

```bash
python run_vllm_ocr.py examples/sample_invoice.png examples/sample_invoice.png
```

## Benchmark

The benchmark measures initialization separately from loaded-engine inference.
For `--backend both`, it runs direct OCR and vLLM OCR in separate subprocesses so
GPU memory and module state do not leak between backends.

```bash
python benchmarks/benchmark_ocr.py \
  --backend both \
  --images examples/sample_invoice.png \
  --warmup 1 \
  --iterations 3 \
  --output results/benchmark-l4-single.json
```

For a small batch-like plugin call:

```bash
python benchmarks/benchmark_ocr.py \
  --backend both \
  --images examples/sample_invoice.png examples/sample_invoice.png examples/sample_invoice.png examples/sample_invoice.png \
  --warmup 1 \
  --iterations 3 \
  --output results/benchmark-l4-batch4.json
```

Interpretation guidelines:

- `texts_match=true` verifies the vLLM wrapper returns the same recognized text
  as direct Nemotron OCR for the benchmark images.
- `vllm_latency_over_direct` is expected to be above `1.0` for standalone OCR.
- Use direct `NemotronOCRV2` for the lowest local latency.
- Use this wrapper when the integration surface needs to be vLLM.

### Local Benchmark Results

These results were collected on an NVIDIA L4 with Python 3.12.13, Torch
2.11.0+cu130, and vLLM 0.24.0. See `results/benchmark-l4-single.json` and
`results/benchmark-l4-batch4.json` for the full payloads.

| Case | Direct Mean | vLLM Mean | Direct Throughput | vLLM Throughput | Texts Match | Region Counts Match |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 image | 0.1066 s | 0.1090 s | 9.38 img/s | 9.17 img/s | yes | yes |
| 4 images in one plugin call | 0.2548 s | 2.0716 s | 15.70 img/s | 1.93 img/s | yes | yes |

The single-image result shows the wrapper can preserve OCR behavior with little
loaded-call overhead for this sample. The four-image plugin result also preserves
OCR behavior, but shows high vLLM worker/scheduler variance for this adapter.
That reinforces the support boundary: this is a useful vLLM integration surface,
not a faster replacement for direct `NemotronOCRV2`.

## Notes

- The wrapper serializes OCR JSON into a `uint8` pooling tensor and decodes it in
  the IO processor. The default payload limit is 1 MiB.
- No model weights are redistributed here. Nemotron OCR v2 remains governed by
  NVIDIA's model terms and package requirements.
- vLLM may emit non-fatal warnings about optional CUDA kernels such as DeepGEMM
  depending on the local CUDA environment.
