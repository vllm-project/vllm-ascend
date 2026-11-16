# record_expert_tokens

## Description

- **Function**: Accumulate operator-provided local physical-expert token counts into the corresponding slice of the EPLB load view when device-side recording is enabled. This shared MoE operator is called by `AscendEplbLayerState.record_load` through `torch.ops.vllm.ascend_eplb_record_expert_tokens` and the public `record_expert_tokens` wrapper in `ops/fused_moe/eplb.py`.
- **Formula**: For local expert `i`, let `c[i] = expert_tokens[i]` when `group_list_type == 1`; otherwise let `c[0] = expert_tokens[0]` and `c[i] = expert_tokens[i] - expert_tokens[i - 1]`. When enabled, update `expert_load_view[local_expert_start + i] += c[i]`. Integer subtraction follows the count dtype and accumulation follows the load dtype, including modular overflow.
- **Algorithm flow**: Split local experts into independent programs of at most 256 elements. Each program reads the device-side recording flag, loads counts (and the preceding prefix value in cumulative mode), and updates its disjoint load slice. Mask the last block and preserve load elements outside the local range. Narrow wider counts before arithmetic when the load dtype is narrower; subtraction and addition give the same result modulo the destination width.
- **Supported modes**: The implementation has no device-specific dispatch and retains the existing Ascend Triton path. Validation of this extracted change on Atlas A2, Atlas A3, and 950PR/950DT remains pending.

## Parameters

All parameters are required. The wrapper returns `None`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `expert_tokens` | Input | Local counts (type 1) or cumulative local counts (type 0), shape `[num_local_experts]` | INT32, INT64 | Contiguous ND |
| `expert_load_view` | Input/Output | Global physical-expert load view, updated in place in the selected local slice | INT32, INT64 | Contiguous 1D ND |
| `record_enabled` | Input | Single-element device tensor; recording is enabled when nonzero | BOOL | Contiguous ND |
| `group_list_type` | Attribute | `0` for cumulative counts, `1` for per-expert counts | Python `int` | N/A |
| `local_expert_start` | Attribute | Start of the local physical-expert slice in the load view | Python `int` | N/A |

## Constraints

- Tensors reside on the same NPU. Count and load dtypes can differ. Inputs follow the contiguous layouts above; strided tensors are outside this contract.
- `0 <= local_expert_start` and `local_expert_start + expert_tokens.numel() <= expert_load_view.numel()`. The public wrapper checks the local range, the count rank and the group-list type.
- Count and flag storage must not overlap the load view. Concurrent calls must not update overlapping load slices; this kernel uses disjoint stores rather than atomic accumulation.
- Empty counts are a no-op. A disabled recording flag leaves the entire load view unchanged. Count and flag tensors are never modified.
- Graph replay reads count and flag contents from their existing device addresses. Updating these contents does not require recapture; tensor addresses, shapes and Python attributes must remain compatible with the captured invocation.
- `GROUP_LIST_TYPE` is constexpr because it selects direct counts or prefix differences. `BLOCK_SIZE` is constexpr for `tl.arange` and is the next power of two of the expert count, capped by `RECORD_EXPERTS_PER_BLOCK = 256`. The cap bounds each program's work without selecting a particular device's core count.
- Count/load/flag pointer types, constexpr values, target/compiler options, and default runtime-integer specialization classes for expert count and local start can create JIT variants. Tensor contents and grid changes alone do not. This extraction adds no warmup registration; cold/warm-cache and serving compilation coverage remain to be measured for the independent record change.

## Origin and Differences

- **Origin**: Extracted from the record portion of [PR #17579](https://github.com/vllm-project/vllm-ascend/pull/17579), source commit `736c4efa8138809611fa8025ef2f2793ab3bada3`, on top of the existing shared EPLB implementation.
- **Differences**: Replace the single program with independently masked expert blocks and use destination-width arithmetic when counts are wider than the load view. Use a named block-size cap and uppercase constexpr names. The mapping kernel, router, count generation, public API, production call chain and padding rules are unchanged.
- **Performance**: N/A for measured results in this extraction. Source-PR map-plus-record and model measurements do not isolate the record change; separate record-only benchmarks and model validation are still required before merge.

## Test Cases

The reference is CPU PyTorch: decode cumulative counts in the count dtype, cast to the destination dtype, and add to a copy of the initial local load slice. All integer results require exact equality (`rtol=0`, `atol=0`).

- Parameterized synthetic boundary cases: 0, 1, 63, 64, 112, 255, 256, 257, 511, 512 and 896 local experts; both group-list formats; all INT32/INT64 count/load combinations; prefix and destination overflow; nonzero local start; unchanged count inputs and surrounding load elements; recording on/off.
- Record-only graph replay through the production custom op: 257 experts, both count formats, mutable count contents, load reset, and a recording flag toggled off and back on.
- These cases are boundary and API tests. They do not establish real-model shape provenance or replace model-level accuracy/performance validation. The existing combined map/count PR test remains available for integration coverage.

Run in a prepared NPU environment:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_record_expert_tokens.py
pytest -sv tests/e2e/pull_request/one_card/test_eplb_map_to_physical_npu.py
```
