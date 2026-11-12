# kimi_pixel_norm

## Description

- **Function**: Converts Kimi uint8 image patches into the BF16/FP16 input of the
  vision patch projection. `KimiPixelNormLut` is installed by the Kimi-K3/K3.1
  constructor after the projection device and dtype are known. Other models do
  not install this module, and missing vision towers skip construction.
- **Formula**: `output[i] = lut[pixels[i]]`. The CPU processor builds the reference
  in its original FP32 order, `(value / 255.0 - 0.5) * (1.0 / 0.5)`, before one
  conversion to the projection dtype. The device kernel only looks up the
  rounded table entries; it does not reassociate the formula into an affine
  expression.
- **Algorithm flow**:
  1. Select a tile of 1024, 4096 or 8192 scalar elements for small, medium or large
     inputs. Dispatch boundaries are 196608 and 2408448 elements, corresponding
     to 334 and 4096 complete patches.
  2. Start at most the actual number of Vector cores and assign tiles by a
     grid-stride loop. Every tile uses the same 256-entry table.
  3. Convert uint8 pixels through FP16 to int32 indices, then gather the
     corresponding table entries. Every uint8 value is represented exactly by
     FP16, so the conversion preserves the index and the rounded table value.
  4. Mask the last tile; empty input returns an empty output without a launch.
- **Supported modes**: A3 and 950PR&950DT Products, eager execution, dynamic Torch
  compile and ACL graph capture/replay. Hardware validation used Ascend950DT_9582
  and Ascend910_9382. A2 has not been validated.

## Parameters

The launch wrapper is `kimi_pixel_norm_lut(pixels, lut, vector_cores)` in
`vllm_ascend/ops/triton/kimi_pixel_norm.py`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `pixels` | Input | Contiguous NPU tensor; Kimi supplies `[P,588]` patches | uint8 | ND |
| `lut` | Input | Contiguous shared-channel table on the same device, `[256]` | BF16 / FP16 | 1D |
| `vector_cores` | Attribute | Actual Vector core count read during module initialization | host int | scalar |
| return value | Output | Newly allocated tensor with the input shape and table dtype | BF16 / FP16 | ND |

## Constraints

- Model integration is limited to Kimi-K3/K3.1, patch size 14 and RGB mean/std
  `[0.5, 0.5, 0.5]`. These conditions are checked once during construction.
- Legacy `preprocessor_config.json` metadata is also read when processor-class
  resolution and cached image metadata do not provide `media_proc_cfg`. A
  `processor_class` field is not required there; missing media settings still
  raise an error.
- RGB channels use one table built from the checked processor configuration.
  The module checks BF16/FP16 support when constructed or moved and retains
  only 16-bit recovery data outside checkpoint state; its
  temporary FP32 reference is not stored permanently.
- Table shape and layout are fixed by module construction. The Vector core
  count is checked when device properties are read. Runtime checks cover pixel
  dtype, device and contiguity, matching devices, and the element count.
- Output dtype must match the actual patch projection dtype. Supported dtype and
  device moves restore the corresponding table bits instead of converting a
  previously rounded BF16 table to FP16.
- Element count must be below `2**31 - 1`. Input and output storage are separate.
- A2 support, full-model answer accuracy and generation throughput are not
  established by the single-operator tests.

## JIT warmup

The model constructor registers the module inside the runner's existing model
load registry window. `KimiPixelNormKernel` derives from
`AscendTritonJitKernel`; when the shared registry expands its registrations after
weight loading, it restores the constant table and selects the module's final
dtype. The three compile keys are `(dtype, BLOCK)`; `count` is a runtime scalar
with `do_not_specialize`.

`TritonWarmupTensor` supplies aligned pointer metadata, and the existing registry
compiles the production kernel without launching image payloads. Runtime uses the
same kernel and launch constructor. With startup warmup enabled, new patch counts
reuse these variants. Disabling the shared warmup option retains the framework's
normal lazy compilation behavior.

## Validation

The nightly E2E case is
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kimi_pixel_norm.py`.
It checks both output dtypes, all pixel values against the CPU reference, empty
inputs, both dispatch boundaries, large inputs, invalid metadata, dtype/device
lifecycle, changing graph inputs and a shared dynamic Torch compile graph.
The warmup coverage case uses the real registry, then prohibits further JIT
compilation while executing multiple patch counts and graph replay. The nightly
file does not clear or isolate Triton compilation caches, so this case does not
independently establish cold-cache compilation coverage.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kimi_pixel_norm.py
```

CPU unit tests are in
`tests/ut/model_executor/warmup/test_jit_warmup_kimi_pixel_norm.py`. They check
scoped and deduplicated registration, deferred final-dtype key expansion and
table restoration, metadata-only compilation of all three BLOCK variants, and
runtime launch coverage at both element-count dispatch boundaries. Compiler and
device launches are mocked; these tests do not establish NPU output correctness
or actual cold-cache compilation behavior.

```bash
pytest -q tests/ut/model_executor/warmup/test_jit_warmup_kimi_pixel_norm.py
```

Recorded A3 and A5 hardware runs of the FP16-index kernel passed all 10 nightly
cases on each device. Real-image validation used 1730 MMMU_Pro requests and 2015
images; all 2,735,879,328 BF16 projection-input elements matched the actual CPU
processor on each device. These are existing hardware results; the latest
checkout has not been rerun on NPU.

The C=1 preprocessing comparison includes decoded RGB resize/pad/patchify,
pinning, H2D and synchronization to BF16 readiness, with one consumer, four Numba
threads and three alternating CPU/device pairs over the same request order.

| Hardware | CPU requests/s | NPU requests/s | Throughput gain |
| --- | ---: | ---: | ---: |
| A3 | 148.93 | 171.62 | 15.24% |
| A5 | 126.83 | 159.65 | 25.87% |

Environment: CANN 9.2.1, torch 2.10.0, torch-npu 2.10.0.post7 and triton-ascend
3.6.0.post2. These measurements cover image preprocessing, not full-model answer
accuracy, TTFT or generation throughput. A5 used a shared card, and aggregate
throughput gains do not establish a latency improvement for every input shape.
