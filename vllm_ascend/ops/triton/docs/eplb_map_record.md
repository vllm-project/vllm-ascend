# eplb_map_and_record

## Description

- **Function**: Map Router-produced logical TopK expert IDs through the runtime EPLB table and add valid-token assignment counts to the cumulative physical-expert load. Scoring, gating, TopK and routing weights remain with the existing Router.
- **Formula**: `physical_ids[t, j] = routing_table[t % table_rows, logical_ids[t, j]]`. For each local physical expert `e`, `expert_load[e] += sum(t < valid_tokens, j: physical_ids[t, j] == e)` when recording is enabled. Every assignment contributes one, independent of its weight.
- **Algorithm flow**: The first Triton kernel launches `min(T, vector_core_num)` programs and assigns each a balanced contiguous token range using quotient/remainder partitioning. Within that range, it maps every row (including padding rows) in contiguous flattened assignment tiles, accumulating one program-private physical-expert histogram over valid tokens. A token's slots remain owned by the same program even when an inner tile ends between slots. Each program stores one disjoint histogram row. The unchanged second kernel reduces these rows and adds the result to `expert_load`. Neither kernel uses a global atomic.
- **Supported modes**: The restored assignment-tile source passed eager and operator graph-replay tests on Atlas A3 (Ascend910_9362, Triton-Ascend 3.2.2). Previous-source tests passed on Atlas A2 (910B4-1); the current source has not been rerun on A2 or 950PR&950DT Products. No dispatch guard is based solely on ALLGATHER, MC2, FUSED_MC2, ALLTOALL, DP or PCP; their valid-token layout must still satisfy the prefix contract below. Standalone operator validation does not establish distributed model load equivalence.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logical_ids` | Input | Existing Router output, shape `[T, K]`; noncontiguous input is copied before the kernels | int32/int64 | ND |
| `routing_table` | Input | Mutable EPLB table, shape `[R, E_logical]` | int32 | contiguous ND |
| `expert_load` | Input/Output | Cumulative load in the global physical-expert domain; only the specified local range is updated | int32 | contiguous 1-D ND |
| `record_enabled` | Input | Mutable device scalar; false leaves `expert_load` unchanged | bool/integer | scalar |
| `valid_tokens` | Input | Number of valid rows at the beginning of `logical_ids`; device scalar or host integer | int32/int64 or Python `int` | scalar |
| `local_expert_start` | Attribute | Start of this rank's range in the global physical-expert domain | Python `int` | scalar |
| `local_expert_count` | Attribute | Number of local physical experts, independent of `E_logical` | Python `int` | scalar |
| `physical_ids` | Output | Mapped IDs for all T rows, including padding rows; same shape and dtype as `logical_ids` | int32/int64 | ND |

## Constraints

- `T >= 0`, `K >= 1`, `R >= 1`, `E_logical >= 1`. All tensors are on the same device. The routing table and load use int32; the load is a contiguous 1-D view.
- Logical IDs outside `[0, E_logical)` map to `-1`. The local range must lie within `expert_load` and have positive length. `BLOCK_P = next_power_of_2(local_expert_count)` and `BLOCK * BLOCK_P <= 8192`, where `BLOCK` counts contiguous assignments, not tokens. It is the minimum of 512, `8192 / BLOCK_P`, and the power-of-two padded largest program-owned assignment range, with a minimum of 2 to avoid singleton-index lowering. Extra lanes are masked out. The 8192 elements are an operator-internal comparison working-set safety budget—not a factor in grid selection. A local range whose `2 * BLOCK_P > 8192` cannot fit the minimum tile and is rejected internally.
- Logical IDs retain their input integer width in the kernel: int32 is not unconditionally widened, and int64 is not truncated. Invalid-ID checks happen before the routing-table lookup.
- The record contract is a **valid prefix**: rows `0 <= t < valid_tokens` are real and later rows are padding. The count can change on-device during graph replay. Mapping still produces output for padded rows, but they do not contribute to load.
- Communication mode names are not a correctness guard. The MC2 mask count and the existing step-level unpadded count supply the valid-prefix value; no new ALLTOALL partition arithmetic is introduced. A DP/PCP AllGather with unequal per-rank valid lengths can place padding between valid rows, and a scalar prefix cannot represent that layout. This remains an unknown/inherited #17574 prefix-contract risk, not a problem created or solved by replacing global atomics with grid-private reduction.
- Post-Router ID rewrites (`log2phy`, mixed placement, forced EPLB or forced load balance) use the existing downstream record path, because recording before the final ID rewrite would count the wrong assignment.

## Integration Reference Boundary

PR #17574 (`55b07ac7417b8d6c6d499769ae7cfca15f9613af`) records each valid token's mapped assignments into the same local physical-expert range using atomics. Given identical logical IDs, routing table, valid count, flag and local range, the grid-private histogram and reduction express the same integer sum. This does not imply identical end-to-end dispatch:

- #17574 fuses selection, mapping and recording for supported decode batches up to 512 tokens. This operator consumes the existing Router's logical IDs at the mapping hook and does not select experts or restrict the routing phase.
- #17574 obtains the valid count from the MC2 mask when present, otherwise from the prepared hidden-state row count. This integration uses the existing step-level unpadded count when no MC2 mask is present. These counts are equivalent only when they describe the same valid prefix of the mapping input.
- Both paths skip the V2 downstream record after mapping-stage recording and use mutable device-side routing tables and record flags. Post-mapping ID rewrites cannot use early recording.
- A separate TP2/EP2 ALLTOALL model audit found different mapping-stage and MoE destination counts. The source-rank input set and final distributed load aggregation need further investigation; this issue is preserved separately, not fixed by tiling or covered by standalone atomic/reduction equivalence.

## Origin and Differences

- **Origin**: The existing Ascend EPLB logical-to-physical mapping and downstream expert-load record. PR #17574 supplies the padding-aware valid-prefix semantics.
- **Differences**:
    - NPU adaptation for performance: one mapping pass also accumulates program-private expert counts, replacing the separate map and downstream record kernels on eligible paths. A second single-writer reduction removes global atomic contention.
    - Integration: the existing Router/CANN TopK output is unchanged. The helper is called at the Ascend EPLB mapping hook; the downstream record is skipped only after mapping-stage recording occurred.
    - Per-step state: `mapping_valid_tokens=None` selects the original map/downstream-record path; a non-`None` count enables map+record, including a zero count. `record_done_in_mapping` prevents duplicate downstream recording even when the device record flag is false. Both fields are cleared after the step.

## Test Cases

The NPU test checks exact physical IDs against the existing mapping op and exact cumulative load against an independent valid-token assignment count. It covers E=128/896, K=16, a nonzero local physical range, padding, disabled recording, runtime table/flag/count graph replay, and noncontiguous input. CPU tests cover grouped/ungrouped routing continuity and the real forward/mapping-hook call chain with a CPU operator reference, including downstream-record skip, state reset and a subsequent fallback step. Both integer outputs require exact equality.

```bash
pytest -sv tests/e2e/pull_request/one_card/test_eplb_map_record_npu.py
pytest -sv tests/ut/ops/test_eplb_map_record_routing.py tests/ut/ops/test_fused_moe.py tests/ut/patch/platform/test_patch_fused_moe.py
```
