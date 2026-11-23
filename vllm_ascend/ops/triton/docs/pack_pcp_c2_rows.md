# pack_pcp_c2_rows

## Description

- **Function**: Pack selected FP32 KV and gate-score projections directly into
  a caller-provided C2 communication buffer. This is a model-specific operator
  used by `AscendDSAV41PCPImpl._compress_local_c2` for DeepSeek V4.1 PCP with
  compression ratio 2.
- **Formula**: For output row `r`, let `i = send_indices[r]` and
  `p = valid_mask[r] and 0 <= i < T`. If `p`, write
  `output[r] = concat(kv[i], scores[i])`; otherwise write an all-zero row.
- **Algorithm flow**: A program processes one send row and one column tile;
  load its index and validity predicate, mask both projection loads, and write
  the KV and score halves. Invalid rows are overwritten, including when T is 0.
- **Supported modes**: Intended for applicable Atlas A2, A3 and 950PR/950DT
  configurations using the project's Ascend Triton backend. Actual device
  validation for this proposed revision has not been run. Graph capture/replay
  is supported for fixed tensor addresses, shapes, strides and launch settings.

## Parameters

All parameters are required. All tensors are on the same NPU.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| kv | Input | KV projections, T >= 0, D > 0 | FP32 | Strided [T, D] |
| scores | Input | Gate-score projections | FP32 | Strided [T, D] |
| send_indices | Input | Arbitrary INT64 source indices; out-of-range indices produce zero rows | INT64 | Contiguous [S] |
| valid_mask | Input | Whether the send row is enabled | BOOL | Contiguous [S] |
| packed_output | Output | Caller-owned storage; fully overwritten for S > 0 | FP32 | Contiguous [S, 2*D] |

## Constraints

- T and S may be zero. S=0 returns the same empty output without launching.
- Projection layouts may differ. Positive strides, nonzero storage offsets,
  and zero strides supported by PyTorch strided tensors are accepted. Tests
  cover contiguous, transposed, column-strided, offset and broadcast views.
- Nonempty output must not share underlying storage with any nonempty input,
  including disjoint views of the same storage. This conservative constraint
  is checked using host-side storage metadata.
- No input is modified. Only the specified output view is written. Output
  allocation and its lifetime belong to the caller.
- Invalid rows must be zero on every call, even if output contains stale data.
- Replayed graphs may observe new tensor contents, including indices and masks.
  Changing tensor metadata requires the corresponding capture/dispatch path.
- Workspace views in the production PCP path are borrowed storage shared by
  serial layers on the main stream. Concurrent forwards/replays need separate
  workspaces or explicit lifetime synchronization.
- Warmup: no new explicit registry wrapper is introduced in this proposal.
  Validate coverage of existing dummy-run/capture initialization before serving;
  add explicit registration if required variants remain uncovered.

## Origin and Differences

- **Origin**: Developed for the existing DeepSeek V4.1 PCP PyTorch sequence
  `cat -> index_select -> copy -> masked_fill`, rather than ported from an
  upstream vLLM Triton operator.
- **Differences**: Fuse selected-row packing into a single launch; use the
  caller's preallocated output; avoid materializing concatenated projections;
  explicitly form a predicate from Ascend BOOL loads; zero negative and
  out-of-range indices without dereferencing them.

## Test Cases

Reference: independent CPU row selection and concatenation. Because this is
pure FP32 data movement, require bit-exact valid-row copies and exact zeroes
for invalid rows. Floating point arithmetic tolerance tables do not relax this.

Coverage includes empty tokens/send rows, indices -1/T/large INT64 values,
duplicates, all-valid/all-invalid controls, tile tails, projection layouts,
output guard regions, immutable inputs, interface rejections and graph replay
after changing projections and controls. Real model shapes must be collected
from the target run and provided separately; synthetic boundaries are not
evidence of production-shape coverage.

```bash
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compressor_triton.py -k pack_pcp_c2_rows
VLLM_PCP_C2_CASES_JSON=/path/to/observed_cases.json python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compressor_triton.py -k observed_model_cases
```

### Specialization analysis

| Dimension | Treatment | Remaining compilation condition |
| --- | --- | --- |
| T | Runtime, do_not_specialize | Inferred scalar type changes can still specialize |
| S | Launch grid only | Grid changes alone do not specialize |
| Projection strides | KV_ROW_STRIDE, KV_COL_STRIDE, SCORE_ROW_STRIDE, SCORE_COL_STRIDE constexpr | A different stride tuple creates a specialization |
| D | WIDTH constexpr | Different widths compile different variants |
| Tile | BLOCK_SIZE=128 | A tuning change compiles another variant |
| Tensor dtypes | Fixed FP32 / INT64 / BOOL | Different signatures are unsupported |
| Tensor addresses | Default backend alignment specialization | Alignment-class changes can specialize |
| Compiler/target | Backend defaults | Effective option, source or target changes can compile |

The production caller obtains both projections directly from `nn.Linear` in
the ratio-2 compressor, with width fixed by `config.head_dim`. Their expected
contiguous strides are `(D, 1)`, independent of T. Retain constexpr strides so
the compiler can exploit fixed address arithmetic and unit column stride.
Confirm the observed stride domain with the target workload; this is not a
claim that every backend output layout has already been measured. Strided
views remain supported and may compile additional variants; boundary tests of
such views do not imply that production needs runtime stride arguments.

The 128-element tile is inherited as a conservative starting point, not a
measured optimum. Benchmark this constexpr-stride implementation against both
the old PyTorch path and the original PR kernel. Consider runtime strides only
if observed production layout changes cause unnecessary variants, then validate
accuracy and performance before adopting that alternative.

Black-box compilation counts, target-device accuracy, operator timings and
model accuracy/performance for this proposed revision: **not run**. Record
dummy run, explicit warmup, graph capture and serving separately. Python launch
counts and unique shapes are not actual compilation counts.
