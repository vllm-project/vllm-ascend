# prepare_qsa_lightning_v2_inputs

## Description

- **Function:** Prepare native LightningIndexerV2 inputs for Qwen QSA in one
  Triton launch. The production caller is
  `models/qwen4_exp/lightning_indexer.py::qsa_select_paged_tokens_lightning_v2`.
- **Formula:** For request length `L`, compression ratio `r` and query row count
  `T`, produce `L // r`, `L % r` and `min(query_start_loc, T)`. Fill weights with
  `float32(bfloat16(1 / sqrt(128)))`, preserving the existing effective value.
- **Algorithm flow:** Each vector tile independently writes request lengths,
  query boundaries and query-head weights with separate bounds masks.
- **Supported modes:** Intended for Atlas A2, Atlas A3 and 950PR/950DT through
  Triton Ascend. Manual operator correctness, graph replay and preparation
  performance have been validated on A3. Other devices remain unverified;
  CPU contract checks do not establish device support.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query` | Input | Only shape and device are used | Model BF16; contents are not read | `[T, H, 128]` |
| `sequence_lengths` | Input | Nonnegative original-token lengths | INT32 or INT64 | `[B]`, strided allowed |
| `query_start_loc` | Input | Packed query boundaries | INT32 or INT64 | At least `[B + 1]`, strided allowed |
| `compress_ratio` | Attribute | Positive tokens per group; QSA uses 4 | Integer | Scalar |
| `weights` | Output | Uniform score weights | FP32 | `[T, H]`, contiguous |
| `cu_query` | Output | Boundaries clamped to T | Boundary input dtype | `[B + 1]`, contiguous |
| `compressed` | Output | Complete compressed-group counts | Length input dtype | `[B]`, contiguous |
| `residual` | Output | Remaining original-token counts | Length input dtype | `[B]`, contiguous |

Outputs are returned in table order: weights, cu_query, compressed, residual.

## Constraints

- Inputs share a device; metadata vectors may be strided. `H` must be positive.
  Zero query rows and zero requests are supported by the wrapper. The model
  caller bypasses preparation entirely for zero query rows.
- Outputs are fresh contiguous allocations. Inputs are not mutated. There is
  no head padding, query copy or per-token block-table gather.
- Valid model metadata describes contiguous query suffixes per request. Padded
  boundaries are clamped, without discarding the original request capacity.
- The wrapper reads no device values on the host. Graph replay must observe
  updated lengths and boundaries at the captured addresses.
- CANN schedule generation and reuse remain unchanged. The input adapter still
  prepares the reusable schedule outside the layer loop. A trimmed query causes
  the selector to regenerate it from the same fused outputs passed to Indexer.
- Output index-buffer initialization and group packing remain separate.

## Origin and Differences

- **Origin:** Replaces four PyTorch operations in the native V2 selector:
  `floor_divide`, `remainder`, `clamp_max` and `full`.
- **Differences:** V2 keeps original heads and request-level page tables.
  This operator replaces input
  preparation only; CANN still performs scoring and TopK.
- **JIT:** Compression ratio, score weight and vector tile size are compile-time
  parameters. Counts and strides are runtime integers, subject to the backend's
  ordinary scalar specialization classes. Metadata contents do not specialize.
  Pointer dtypes/alignment and compiler options can create variants. The fixed
  256-element tile is a starting configuration, not a measured optimum. No
  explicit warmup registration is added; actual compilation counts are pending.

## Test Cases

CPU tests cover decode/prefill row counts, padded requests, empty inputs,
both integer dtypes, strided metadata and schedule reuse/regeneration.
Repository NPU e2e cases are deferred. The manual A3 validation below is
separate from these CPU tests and is not an automated CI test entry point.

```bash
pytest -sv tests/ut/ops/test_qsa_lightning_v2_prepare_contract.py
pytest -sv tests/ut/models/qwen4_exp/test_lightning_indexer_v2.py
```

CPU tests cover allocation, launch wiring, dtype/stride contracts, schedule
reuse, trimming and output-buffer behavior. Device arithmetic is not emulated.

### Manual A3 operator validation

Validated implementation: `a5e9acf3a9580afd8e5c9375f0ee1fb6cc750761`.
All four outputs matched the original PyTorch expressions exactly
(`rtol=0`, `atol=0`), including the BF16-rounded weight promoted to FP32.
The 96 cases covered heads 2/4/8, INT32/INT64 metadata, contiguous/strided
metadata, empty inputs and padded request boundaries. Three captured-graph
replays after updating lengths and boundaries also matched exactly.

Performance setup: one Ascend A3 device (`Ascend910_9382`), 1109-r89 base image,
CANN 9.2.1, PyTorch 2.10.0, torch_npu 2.10.0.post7 and
triton-ascend 3.6.0.post2. Inputs were BF16 queries `[T, 4, 128]`, INT32
metadata and compression ratio 4. Compare the four original PyTorch operations
against the fused wrapper using 128 calls per captured NPUGraph, 10 replays
per timed sample, seven samples per arm and five A/B/A trials. Values below
are median per-call NPU-event latency, excluding compilation and warmup.

| Query rows | Requests | Baseline (us) | Fused (us) | Latency reduction |
| --- | --- | --- | --- | --- |
| 32 | 1 | 8.377 | 2.238 | 73.29% |
| 32 | 8 | 7.974 | 2.233 | 72.00% |
| 32 | 32 | 8.463 | 2.241 | 73.51% |
| 128 | 1 | 8.394 | 2.239 | 73.33% |
| 128 | 8 | 8.000 | 2.244 | 71.96% |
| 128 | 32 | 8.418 | 2.242 | 73.37% |

These results measure input preparation only, excluding CANN schedule metadata,
scoring/TopK, index packing and model execution. They do not establish
end-to-end model speedup or model accuracy non-regression. Other-device
coverage, repository NPU e2e cases, model accuracy/performance comparisons and
actual startup/serving compilation counts remain pending.
