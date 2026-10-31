# EPLB mapping and expert-load kernels

## Description

- **Function**: Provides two independent operators in `vllm_ascend/ops/triton/eplb.py`: `map_to_physical_triton` maps logical expert IDs to physical IDs; `record_expert_tokens_triton` accumulates counts returned by the existing MoE operator. Recording does not consume mapping outputs. Existing EPLB wrappers and routing/dispatch integration are unchanged.
- **Formula**:
    - Mapping: `physical_ids[t,j] = routing_table[t % R, topk_ids[t,j]]` for `0 <= topk_ids[t,j] < E`; otherwise output `-1`. `R` and `E` are the routing-table row count and logical expert count.
    - Recording: let `C = expert_tokens.numel()` and `s = local_expert_start`. For `group_list_type=1`, `count[i] = expert_tokens[i]`. For type 0, `count[0] = expert_tokens[0]` and `count[i] = expert_tokens[i] - expert_tokens[i-1]`. When `record_enabled != 0`, `expert_load_view[s+i] += count[i]`; otherwise load is unchanged. Integer-width semantics are described below.
- **Algorithm flow**:
    1. Mapping launches `G = min(T, vector_core_num)` programs. With `base = T // G` and `extra = T % G`, program `pid` owns `base + (pid < extra)` consecutive tokens starting at `pid * base + min(pid, extra)`.
    2. Each program processes its flattened assignments in fixed 512-element tiles, performs the runtime table lookup, and stores physical IDs. Tail masking respects program ownership. Grid selection is independent of tile size.
    3. Existing MoE dispatch and computation produce `expert_tokens`. Recording uses blocks of `B = min(256, next_power_of_2(C))` experts and `ceil(C / B)` programs. Each program reads the device-side flag, decodes counts and updates its disjoint load range. Type 0 reads the previous global element across block boundaries. Neither kernel uses global atomics.
- **Supported modes**: Eager execution and ACL graph capture/replay on Ascend NPU. Hardware validation for this change was on Atlas A3; Atlas A2 and 950PR&950DT validation is N/A. This change does not alter gating, routing weights, model support or communication modes.

## Parameters

All public entry-point parameters below are required. `T` is the token count, `K` is the routing slot count, `P` is the physical expert-load size, and `C` is the local expert count.

### map_to_physical_triton

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `topk_ids` | Input | Logical expert IDs `[T, K]` | int32 / int64 | Contiguous ND |
| `expert_replica_routing_table` | Input | Runtime logical-to-physical table `[R, E]` | int32 | Contiguous ND |
| Return value | Output | Physical IDs `[T, K]`; invalid logical IDs produce `-1` | Same as `topk_ids` | Contiguous ND |

### record_expert_tokens_triton

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `expert_tokens` | Input | MoE-provided counts or cumulative counts `[C]` | int32 / int64 | Contiguous 1-D |
| `expert_load_view` | Input / Output | Cumulative physical-domain load `[P]`; only `[s, s+C)` is updated | int32 / int64 | Contiguous 1-D |
| `record_enabled` | Input | Device scalar; false disables load updates | bool | Scalar |
| `group_list_type` | Attribute | `1` for counts; `0` for cumulative counts | Python int | Scalar |
| `local_expert_start` | Attribute | Start `s` of the local physical-expert range | Python int | Scalar |
| Return value | Output | N/A; updates the load in place | None | N/A |

## Constraints

- Tensors must use contiguous storage on the same NPU. Arbitrary strides and overlapping source/destination buffers are not supported.
- Mapping requires 2-D IDs and a 2-D table. For nonempty IDs, `R > 0` and `E > 0`; physical IDs in the table must be representable by the output dtype. The kernel masks invalid logical IDs but does not validate table contents. Empty IDs return without a kernel launch.
- Recording requires 1-D counts/load, `group_list_type` in `{0, 1}`, and `0 <= s <= s+C <= P`. The flag is a device scalar, not a host boolean. Empty counts return without a kernel launch.
- If counts are wider than the load dtype, operands are narrowed before arithmetic; add/subtract is equivalent modulo the destination width. Otherwise cumulative-count subtraction retains the input width before adding to the destination, preserving existing overflow semantics. Floating-point tolerance is N/A.
- Mapping specializes `TOPK=K` and fixes `BLOCK_SIZE=512`. `tokens` and `num_grids` are marked `do_not_specialize`; dtype, `K` and backend argument specialization can still select separate compiled variants. Recording specializes `group_list_type` and its block size.
- Graph replay retains captured shapes, addresses, launch dimensions and host scalar arguments. Routing-table contents and the device-side record flag may change in the same buffers before replay; both are read on device without `.item()`.
- Recording consumes MoE-provided counts rather than mapped-ID histograms. Padding exclusion is N/A to this optimization: padding-aware correction is a separate correctness change. This kernel cannot remove padding already included in its input counts.

## Origin and Differences

- **Origin**: Optimizes the existing `_map_to_physical_kernel` and `_record_expert_tokens_kernel` in `vllm_ascend/ops/triton/eplb.py`, called through generic EPLB wrappers in `vllm_ascend/ops/fused_moe/eplb.py`. These are Ascend EPLB operators, not upstream vLLM gating/TopK ports.
- **Differences**:
    - Mapping balances contiguous token ownership independently of its fixed tile, retains the input ID width instead of unconditionally widening to int64, and avoids token/grid value specialization.
    - Recording splits experts into blocks of at most 256, preserves the cumulative-count dependency across blocks, and avoids wider-than-needed arithmetic when the load dtype is narrower.
    - The public ABI, runtime table lookup, cumulative load updates, record-off behavior and post-MoE recording boundary are unchanged. No model-specific integration is added.

## Test Cases

The existing NPU suite checks exact integer equality (`rtol=0`, `atol=0`) against independent indexing and count-decoding references:

- Mapping: `T` from 1 to 65536; `(E, K) = (128, 8)` and `(896, 16)`; int32/int64 IDs; invalid IDs; balanced ownership, tails and tile boundaries.
- Recording: `C = 255/256/257/896`; both group-list types; all int32/int64 count/load combinations; nonzero initial loads and local starts; record-off behavior.
- Graph replay: routing-table contents and the record flag change between replays.

The model workload recorded in the PR was complete Qwen3-30B-A3B W8A8, TP2/EP2. Observed mapping inputs were int32 `[33, 8]` / `[32, 8]` in prefill and `[1, 8]` in decode; recording used int64 `[64]` counts and int32 `[128]` load with local starts 0/64. This describes existing validation, not a new model-specific implementation or a new device run for this documentation update. Performance results remain in the PR description.

```bash
pytest -sv tests/e2e/pull_request/one_card/test_eplb_map_to_physical_npu.py
```
