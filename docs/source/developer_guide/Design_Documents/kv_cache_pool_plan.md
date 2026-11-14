# Planning separate history and request-state pools

The [heterogeneous KV cache pool RFC](https://github.com/vllm-project/vllm-ascend/issues/21324)
separates token-growing history from fixed per-request state. For example,
DeepSeek V4.1 retains compressed history and sparse indexes by token block,
while its sliding-window and compressor states have bounded storage per request.
Assigning all of them the same physical page size couples their capacities and
can introduce padding.

`vllm_ascend.core.kv_cache_pool_plan` defines the CPU-only planning contract for
separate pools. It does not enable separate pools in the scheduler, model runner,
or KV connector. Existing models keep their current allocation behavior.

## Pool ownership and capacity

- `CachePoolSpec` identifies a pool, its cache groups, its allocation unit
  (`token_block` or `request_slot`), and its physical bytes per unit.
- `CachePoolPlan` adds a physical capacity. Reserved IDs are included explicitly:
  serving C requests with one null slot requires C + 1 slots.
- `RequestStatePoolPlan` holds one growing history pool and one or more fixed
  request-state pools. Each group has one owner; `validate_group_coverage` checks
  that the builder's complete group list is covered.

A pool describes a shared ID space, not necessarily one backing tensor. History
KV and sparse indexes may use separate tensors while sharing history page IDs.
`bytes_per_unit` must include all physical backing bytes for one ID, including
alignment and padding. Pools do not borrow memory from one another.

## Budget calculation

For one worker, let H be bytes per history page, F the total bytes of all fixed
state pools, W the workspace reservation, and M the available memory:

```text
fixed bytes       = F + W
history capacity  = floor((M - F - W) / H)
total plan bytes  = F + W + history capacity * H
```

`RequestStateMemoryBudget` exposes this accounting.
`plan_request_state_pools` resolves capacity and rejects plans that cannot fit
the caller's required minimum. The minimum includes the null history block;
the caller must derive it from admission requirements, rather than interpreting
one usable page as sufficient for the model's maximum sequence length.

A small layout example (bytes are illustrative, not model measurements):

```python
from vllm_ascend.core.kv_cache_pool_plan import (
    CachePoolPlan,
    CachePoolSpec,
    plan_request_state_pools,
)

history = CachePoolSpec("history", (0, 1), "token_block", 64)
states = (
    CachePoolPlan(CachePoolSpec("swa", (2,), "request_slot", 32), 5),
    CachePoolPlan(CachePoolSpec("compressor", (3,), "request_slot", 16), 5),
)
plan = plan_request_state_pools(
    history,
    states,
    available_memory=1000,
    workspace_bytes=128,
    min_history_blocks=2,
)
plan.validate_group_coverage(4)
assert plan.history.capacity == 9
assert plan.size_bytes == 944  # 240 state + 128 workspace + 9 * 64 history
```

The five state slots include one reserved slot and four request slots. Increasing
history capacity does not increase state capacity. Workspace must be counted
exactly once: pass only the portion not already deducted by memory profiling.

An explicit `num_history_blocks` is checked against the same full budget; it
cannot bypass fixed state or workspace reservations. For workers sharing history
IDs, callers can select the minimum feasible capacity, then build each worker's
plan again with that count. Each worker retains its own byte sizes and fixed
reservations. This helper performs no collective and establishes no topology
compatibility by itself.

## Integration boundary

The current vLLM `KVCacheConfigBuilder` materialization hook receives a
Core-resolved `num_blocks`. Its device-memory model is proportional to that
count. A fixed reservation must not be folded into `_get_pool_bytes_per_block`
or inferred from a one-block probe: `F + B * H` is not `B * (F + H)`.

These descriptors therefore remain separate from `KVCacheConfig` and are not
wired into `AscendKVCacheConfigBuilder` in this change. Runtime integration needs
Core planning to account for fixed reservations before cross-worker capacity
resolution and admission checks, followed by MRV2 materialization and independent
state-slot lifecycle management. P/D integration additionally needs per-pool
registration, local ID mapping, and state handoff. None of these runtime paths is
activated by constructing a plan.

The planner validates byte accounting, units, capacities, and group ownership.
It does not validate tensor offsets, dtype/layout compatibility, kernel access,
or transfer ordering; those checks belong to materialization and execution.
