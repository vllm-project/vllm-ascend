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
     grid-stride loop. Small tiles use the 256-entry table directly; medium and
     large tiles expand it into a repeated 512-entry table on chip.
  3. Small tiles convert uint8 pixels through FP16 to int32 indices, avoiding
     table expansion overhead. Medium and large tiles widen pixels to integers
     and add 256 before Gather. Indices 256 through 511 avoid the tested A5
     compiler's 8-bit Gather layout failure without floating-point index
     conversions. Both paths preserve the table values and their rounding.
  4. Mask the last tile; empty input returns an empty output without a launch.
- **Supported modes**: A3 and 950PR&950DT Products, eager execution, dynamic Torch
  compile and ACL graph capture/replay. The integer-index update was validated on
  Ascend950DT_9582 and Ascend910_9382. A3 quick checks covered both output dtypes,
  precision, cold-cache warmup and changed-input graph replay. A3 preprocessing
  throughput below still uses the previous kernel. A2 has not been validated.

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
Cold-cache cases compile through the real registry, then prohibit further JIT
compilation while executing previously unseen patch counts and graph replay.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kimi_pixel_norm.py
```

A5 real-image validation was refreshed after the integer-index update using
1730 MMMU_Pro requests and 2015 images; all 2,735,879,328 BF16 projection-input
elements matched the CPU processor. Manual checks covered both output dtypes,
all pixel values, empty inputs, tails and dispatch boundaries. Cold-cache shared
registry warmup compiled three keys per dtype; a compiler-denying guard passed
for new shapes and graph replay with changed inputs. The nightly file was not
rerun for this update.

The C=1 comparison included decoded RGB preprocessing, pinning, H2D and
synchronization, with one consumer, four Numba threads and three alternating
CPU/device pairs. A3 measurements are retained as historical evidence for the
previous kernel.

| Hardware / kernel | CPU requests/s | NPU requests/s | Throughput gain |
| --- | ---: | ---: | ---: |
| A3 / previous FP16-index kernel | 148.93 | 171.62 | 15.24% |
| A5 / current kernel | 120.28 | 146.02 | 21.41% |

A5 pure-kernel comparisons reused preallocated buffers in graph replay and
bracketed each current-kernel measurement with the previous FP16-index kernel.
Both used the same shape, dtype, BLOCK and program count. Compilation, allocation
and H2D were excluded. The BF16 results below use nine paired rounds, with 200
launches per event sample and three samples per section.

| Patches | BLOCK | Current kernel (us) | Median paired latency reduction |
| ---: | ---: | ---: | ---: |
| 1064 | 4096 | 2.15 | 1.29% |
| 1152 | 4096 | 2.17 | 2.17% |
| 20000 | 8192 | 8.95 | 2.76% |

These are preprocessing observations, not a full-model serving result or a
hardware-only speed ratio. A5 used a shared card, and pure-kernel measurements
reused cached buffers. The refreshed run is a new CPU/device comparison; changes
from historical throughput cannot be attributed solely to index optimization.
Per-shape and paired-round variation must be considered separately; aggregate
throughput improvement does not establish every latency gate.
