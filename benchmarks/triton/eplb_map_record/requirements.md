# EPLB map and valid-token record contract

## Scope and execution path

The existing Router produces routing weights and logical TopK IDs without
changes to scoring, grouping, bias, renormalization, hash routing or scaling.
At the existing Ascend EPLB mapping hook, `eplb_map_and_record` maps those
IDs through the runtime replica table and counts only valid-token physical
assignments. The MoE operator consumes the returned physical IDs. When the
mapping hook records a step, the downstream MoE-count update is skipped.

## ABI

| Item | Contract |
|---|---|
| Logical IDs | int32/int64 `[T,K]`, from the unchanged Router; noncontiguous input is copied to a contiguous view |
| Routing table | Runtime int32 `[R,E_logical]`; lookup row `token % R` |
| Physical IDs | Same shape and dtype as logical IDs; exact mainline mapping |
| Valid count | Device scalar int32/int64 or host int; valid rows form a prefix |
| Record flag | Mutable device scalar; false leaves the cumulative load unchanged |
| Expert load | Contiguous int32 vector in the global physical-expert domain |
| Local range | `[local_expert_start, start + local_expert_count)`; independent of logical E |

Each valid token/TopK slot contributes one count to its final local physical
expert, regardless of routing weight. Padding rows are mapped as usual but
never recorded. The first kernel writes one grid-private histogram row; the
second kernel reduces rows into cumulative load. Neither uses a global atomic.
The comparison tensor obeys `BLOCK * next_power_of_2(local_count) <= 8192`.

## Acceptance

- Exact physical IDs against the current Ascend EPLB mapping op, including
  routing-table updates under NPU graph replay.
- Exact cumulative physical load against a padding-aware assignment count,
  including nonzero initial load, record-off, valid count 0/T/intermediate,
  EP-local range and `E_logical != local_count`.
- Compare the same assignments with PR #17574's atomic record semantics.
- Verify grouped/ungrouped and renorm variants reach the unchanged Router
  output and then the mapping hook.
- Distinguish model call-path evidence, short torch profiler diagnostics,
  standalone `msprof op` cost and sustained unprofiled serving performance.

## Current fallback boundary

The mapping-stage record is disabled if post-router code rewrites IDs
(`log2phy`, mixed placement, forced EPLB/load-balance), or when the prepared
rows cannot be expressed by one valid prefix (notably DP/PCP padded
AllGather). With DP=PCP=1, EP sequence-parallel sharding pads only the
global suffix and gathering restores rank order, so the device-side
unpacked-token count remains the valid prefix. The mainline mapping and
MoE-count record remain in unsupported paths.
The local physical expert count must fit the comparison resource budget.
These are not restrictions on scoring, grouping, TopK K, logical E or T.
