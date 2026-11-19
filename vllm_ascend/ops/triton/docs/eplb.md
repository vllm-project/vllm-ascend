# EPLB mapping and expert-load kernels

## Description

- **Function**: Model Runner V2 EPLB uses `map_to_physical_triton` from the Ascend MoE router to map logical expert IDs to physical IDs. The EPLB layer state calls `record_expert_tokens_triton` after MoE execution to accumulate local expert counts.
- **Formula**:
    - Mapping: `physical_ids[t,j] = routing_table[t % R, topk_ids[t,j]]` for `0 <= topk_ids[t,j] < E`; otherwise output `-1`. `R` and `E` are the routing-table row count and logical expert count.
    - Recording: let `C = expert_tokens.numel()` and `s = local_expert_start`. For `group_list_type=1`, `count[i] = expert_tokens[i]`. For type 0, `count[0] = expert_tokens[0]` and `count[i] = expert_tokens[i] - expert_tokens[i-1]`. When `record_enabled != 0`, `expert_load_view[s+i] += count[i]`; otherwise load is unchanged. Integer-width semantics are described below.
- **Algorithm flow**:
    1. Each program maps one contiguous tile of flattened token and top-k assignments. For up to 2048 assignments the tile size is 64, for up to 8192 it is 128, and for larger inputs it is 1024.
    2. Each lane derives the routing row from its token index, masks invalid logical expert IDs, and writes one physical ID. The last tile masks assignments beyond the end of the input.
    3. Existing MoE dispatch and computation produce `expert_tokens`. Recording uses blocks of `B = min(256, next_power_of_2(C))` experts and `ceil(C / B)` programs. Each program reads the device-side flag, decodes counts and updates its disjoint load range. Type 0 reads the previous global element across block boundaries. Neither kernel uses global atomics.
- **Supported modes**: Eager execution and ACL graph capture/replay on Ascend NPU. The optimized kernels were validated on Atlas A3 and Ascend 950DT.

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
- Mapping specializes `TOPK` and `BLOCK_SIZE` to simplify token indexing and set the vector width. `numel` is a runtime bound with value specialization disabled. Triton still distinguishes pointer dtype and alignment, integer types and the default scalar classes of `num_logical_experts` and `routing_table_rows`.
- The MoE factory registers the map wrapper for EPLB-enabled V2 routers inside the model runner's JIT registry window. The worker compiles six keys per logical expert count and top-k, covering three tiles and int32/int64 IDs for the standard aligned, contiguous int32 routing table with 1024 rows. The same wrapper and launch specification serve runtime calls. Other pointer alignment classes or routing-table scalar classes can compile on first use.
- `kernel_config.enable_jit_warmup` controls startup compilation and defaults to `True`. See [Triton JIT warmup on Ascend](../../../../docs/source/developer_guide/Design_Documents/triton_jit_warmup.md) for registration and validation details. Recording uses lazy compilation with `group_list_type` and its block size as compile-time parameters.
- Graph replay retains captured shapes, addresses, launch dimensions and host scalar arguments. Routing-table contents and the device-side record flag may change in the same buffers before replay; both are read on device without `.item()`.
- Recording accumulates the counts supplied by EPLB. Padding exclusion, where applicable, takes place before the operator receives those counts.

## Origin and Differences

- **Origin**: Optimizes the existing Ascend EPLB kernels in `vllm_ascend/ops/triton/eplb.py`, called through the shared EPLB custom operators in `vllm_ascend/ops/fused_moe/eplb.py`.
- **Differences**:
    - The flat map distributes contiguous assignments across programs and keeps the input ID width. Assignment counts of the same scalar type share a variant within a tile bucket when the other specialization inputs match.
    - Recording splits experts into blocks of at most 256, preserves the cumulative-count dependency across blocks, and avoids wider-than-needed arithmetic when the load dtype is narrower.
    - The public ABI, runtime table lookup, cumulative load updates, record-off behavior and post-MoE recording boundary are unchanged. No model-specific integration is added.

## Test Cases

The nightly NPU suite checks exact integer equality (`rtol=0`, `atol=0`) against independent PyTorch indexing and count-decoding references.

- Mapping covers Qwen3-30B-A3B decode and prefill, additional `(E, K) = (896, 16)` geometry, both ID dtypes, invalid IDs, table wrapping, and tile tails and transitions.
- Recording covers `C = 64/255/256/257/896`, both group-list types, all int32/int64 count/load combinations, integer wraparound, nonzero initial loads and local starts, and record-off behavior.
- Graph replay reads updated routing-table contents and the record flag from the same device buffers.
- CPU warmup tests cover enabled-router registration, registry-window scope, deduplication, metadata-only compilation and runtime dispatch at the tile boundaries.

The Qwen3-30B-A3B W8A8 workload used TP2/EP2. Observed mapping inputs were int32 `[33, 8]` and `[32, 8]` in prefill and `[1, 8]` in decode. Recording used int64 `[64]` counts and int32 `[128]` load with local starts 0 and 64. The PR description records the measurement revisions and conditions.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_eplb.py
pytest -q tests/ut/model_executor/warmup/test_eplb_map_warmup.py
```
