# Design Notes

## What vLLM's Nemotron Parse Support Teaches Us

vLLM PR [#30864](https://github.com/vllm-project/vllm/pull/30864) added
first-class support for NVIDIA Nemotron Parse. That model fits vLLM's native VLM
shape:

- it is a Hugging Face generative model with a registered architecture,
- it has a multimodal processor that converts images into `pixel_values`,
- it embeds images through a vision tower,
- it decodes output tokens with a language decoder,
- and its vLLM test compares Hugging Face generation/logprobs against vLLM
  generation/logprobs.

Nemotron OCR v2 is structurally different. The Hugging Face repository exposes a
Python OCR pipeline (`NemotronOCRV2`) that coordinates a detector, recognizer,
relational model, image preprocessing, non-max suppression, quadrangle
rectification, text grouping, and custom CUDA/C++ post-processing kernels. Its
public output is already structured OCR JSON-like dictionaries, not generated
token IDs.

## Why This Wrapper Uses vLLM Pooling Plugins

Because Nemotron OCR v2 is not distributed as a normal `AutoModelFor...`
generative VLM, a faithful native vLLM port would mean reimplementing the entire
OCR pipeline as vLLM model-executor code:

- model construction for detector, recognizer, and relational modules,
- checkpoint loading and key mapping for three separate weight files,
- CUDA extension availability inside the worker,
- pre/post-processing parity,
- structured OCR output serialization,
- and HF/direct parity tests.

That is possible, but it is a model port, not a light adapter.

The implemented wrapper therefore uses vLLM's documented IO processor plugin
path for pooling models:

1. The user calls `LLM.encode({"data": image_path_or_paths}, pooling_task="plugin")`.
2. The IO processor validates local image paths and encodes them as token IDs so
   they can cross the vLLM scheduler/worker boundary.
3. The custom attention-free pooling model runs `NemotronOCRV2` inside the vLLM
   worker.
4. OCR payloads are serialized into `uint8` pooling tensors.
5. The IO processor decodes the tensors back into Python dictionaries.

This is proper as a vLLM plugin integration: requests enter and leave through
vLLM, batching is handled through one plugin call, and the OCR model executes in
the vLLM worker process. It is not proper to market this as native vLLM
acceleration of Nemotron OCR internals.

## Support Boundary

Supported:

- local image paths,
- one or more image paths in a single plugin request,
- structured OCR output matching direct `NemotronOCRV2` text for tested images,
- explicit dummy mode for cheap plugin plumbing tests via
  `NEMOTRON_OCR_VLLM_DUMMY=1`.

Not supported:

- arbitrary image bytes/PIL objects over the offline API,
- OpenAI-compatible `/v1/chat/completions` generation,
- vLLM KV-cache acceleration of the OCR pipeline,
- tensor-parallel sharding of detector/recognizer/relational internals,
- native logprob parity because OCR v2 is not a token-generating API.

## Verification Standard

The benchmark follows the spirit of the Nemotron Parse test by comparing direct
and vLLM-backed outputs. For OCR v2, the relevant parity check is:

- recognized text signatures match,
- region counts match,
- benchmark latencies are reported with initialization separated from loaded
  inference.

Small confidence differences are expected because both paths run GPU kernels in
separate processes.
