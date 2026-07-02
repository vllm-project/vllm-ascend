# Fix: Per-Group Progressive Alignment for Hybrid KV Cache

## Issue: #9247 — DeepSeek V4 MTP over-truncates prefix cache from 32K to 16K

## Root Cause

`AscendHybridKVCacheCoordinator.find_longest_cache_hit()` aligns all groups
to a single global `lcm_block_size`, then applies MTP/EAGLE "drop one block".
The global LCM is typically 16K tokens for DeepSeek V4. The EAGLE block drop
creates a small shortfall (e.g. 32K -> 31.8K), then the LCM alignment rounds
it down to the previous LCM boundary (16K), losing ~50% of the cached prefix.

## Fix: Progressive Per-Group Alignment with Backoff

Instead of one global `lcm_block_size` that forces every attention group
through the same coarse alignment, align each group to its own
`effective_block_size`. When groups disagree, progressively back off from
the largest block size downward, finding the largest hit length where ALL
groups have matching cached blocks.

### Algorithm

```text
1. Start with hit_length = max_cache_hit_length
2. Collect effective_block_sizes for all attention groups (sorted descending)
3. For each alignment step from largest to smallest:
   a. Align hit_length to current step
   b. Check all groups at this length
   c. If all hit: RETURN hit_length (early success)
   d. If any group misses: continue to next smaller alignment step
4. If no alignment works: RETURN 0 (all miss)
```

### Code Change Location

File: `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
Class: `AscendHybridKVCacheCoordinator`
Method: `find_longest_cache_hit()` (lines 231-339)

### Expected Result

| Scenario | Before (LCM) | After (Progressive) |
|----------|-------------|---------------------|
| 32K hit, MTP on | 16K (lost 50%) | ~31.7K (lost <2%) |
| 16K hit, MTP on | 8K | ~15.8K |
| No MTP | 32K | 32K (unchanged) |

## Backward Compatibility

- When all block sizes are equal (single attention type), the algorithm
  degenerates to the original behavior (one step, always matches).
- The `is_simple_hybrid` fast path is preserved for non-MTP, non-EAGLE cases.
- No changes to external API, no new configuration options.
