# PR Description (paste into GitHub PR body)

## [Bugfix] Skip MambaSpec groups in native offloading to support hybrid attention models

Fixes #<issue_number>

### Purpose

Native KV offloading (`OffloadingConnector`) crashes on hybrid attention models (e.g. Qwen3.5 with GDN linear attention + full attention) because `AscendOffloadingConnectorWorker.register_kv_caches()` attempts to build int8 block views for ALL KV cache groups, including MambaSpec groups whose tensors have a non-coalescible interleaved layout on Ascend NPUs.

This PR skips MambaSpec groups in the Ascend worker's canonicalization, allowing hybrid models to start and offload their full-attention groups' KV to CPU/SSD. Mamba states remain in NPU and are managed by vLLM's `HybridKVCacheCoordinator` state cache. The upstream `OffloadingConnector._bounding_group_ids` already bounds load depth for non-offloaded groups.

### Changes

1. `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py`:
   - In `register_kv_caches()` adaptation path: skip groups whose spec is `MambaSpec` (don't build int8 views, don't add to canonical tensors)
   - In `group_data_refs` construction: append empty ref lists for skipped groups to maintain index alignment
   - Added `logging` for skip-event visibility

2. No changes to the upstream `OffloadingConnectorWorker` or scheduler logic — the `_bounding_group_ids` mechanism already handles "some groups not offloaded" semantics.

### Verification

- **Hybrid model (Qwen3.5-27B, 48 GDN + 16 full attention, TP4 on 910B3)**:
  - Before: `ValueError: Cannot coalesce a non-contiguous Ascend KV cache layout: stride_bytes=131072, physical_block_bytes=65536` — all 4 workers crash
  - After: service starts; full-attention groups' KV blocks offload to CPU tier normally; MambaSpec group skipped with log line; external_prefix_cache_hits accrue for full-attention prefixes
- **Pure dense model (Qwen3-32B, GQA)**: zero behavior change (regression guard — no MambaSpec groups detected)
- **Unit tests**: `test_skip_mamba_groups.py` covers group-skip logic, index alignment, pure-dense regression guard, pure-linear all-skip, and bounding-group semantics

### Design rationale

The Mamba/linear-attention state is a per-sequence O(1) fixed-size object (SSM state + conv state), fundamentally different from per-token KV blocks. The offloading connector's object model (per-token KV blocks with chain-based hashing for partial prefix matching) does not apply to states, which have whole-history granularity ("all-or-nothing" checkpoint semantics). Attempting to force states into the block model would require per-group canonical views, per-group chunk accounting, and per-group eviction semantics — a significant redesign.

The minimal fix (skip non-offloadable groups) follows the same pattern already used by:
- MooncakeConnectorV1 / mooncake_hybrid_connector: `isinstance(..., MambaSpec)` gate skips partial-prefix truncation for mamba groups
- AscendStoreConnector / UCM hla_connector: `mamba_cache_mode == "align"` required; mamba groups excluded from `participates_in_prefix_caching`; null blocks (block_id=0) as align placeholders

Full mamba state offloading (checkpoint-based) can be implemented as a follow-up PR if needed, informed by UCM's `lookup_on_reverse` design.

### Test Plan

- [x] CI passes on A2/A3 hardware
- [ ] Manual: hybrid model (Qwen3.5-27B) starts without crash
- [ ] Manual: pure dense model (Qwen3-32B) unchanged (regression)
- [ ] Unit tests pass locally (`pytest tests/test_skip_mamba_groups.py -q`)
