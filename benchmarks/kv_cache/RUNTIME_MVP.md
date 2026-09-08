# Runtime MVP design: typed KV cache pages on Ascend

> **Status note (2026-09-08):** This is a design-history document.  The first
> checkpoint used exact-LCM superpages; the current runtime mode uses a shared
> byte arena with group-specific address tables.  Framework-side allocation
> and translation tests pass.  Two GDN stride-aware implementations are now in
> the source tree, but formal serving performance remains gated on rebuilt NPU
> operator tests and exact-token comparisons described at the end.

The end-to-end experiment is gated by
`VLLM_ASCEND_ENABLE_TYPED_KV_CACHE=1` and remains disabled by default. The
sizing stage attaches a `TypedKVCachePlan` to `KVCacheConfig`; the scheduler
selects `TypedKVCacheCoordinatorNoPrefixCache`, while the worker uses
group-local block IDs, a group-specific page address table, and page-strided
Attention/Mamba views. New physical pages are cleared before reuse.

On a dedicated idle NPU die, run `run_typed_e2e.sh` from the matching CANN
container. It performs a uniform/typed deterministic greedy-output comparison,
a fixed-QPS TTFT/TPOT benchmark, and a concurrency sweep. Both arms use the
same seed, model, prompt distribution, memory fraction, and engine flags; only
the typed allocator switch differs.

## Goal and baseline

The baseline is vLLM-Ascend PR #14340, not the older fully contiguous layout.
PR #14340 provides the framework-side layout prerequisite for non-contiguous
Attention and Mamba views. Every device operator must still honor that layout;
the original GDN decode kernel was the exception that exposed the blocker. The
allocator experiment asks whether the scheduler can stop charging every group
the same physical page size once that kernel contract is validated on device.

The target result is a reproducible improvement in useful KV cache capacity or
maximum concurrency without a material latency/throughput regression.

## Scope of the first runtime experiment

Start with Qwen3.5-2B on one NPU and deliberately disable:

- prefix caching;
- KV offload/connectors;
- DCP/PCP;
- speculative decoding;
- CUDA/NPU graph capture.

This isolates allocation, block-table translation, slot mapping, and kernel
consumption. Prefix-cache hashing and eviction should only be added after this
path is correct and beneficial.

## Proposed representation

```text
                    global LargePagePool
                 fixed-size reusable superpages
                              |
              +---------------+---------------+
              |                               |
       Attention SmallPagePool          Mamba SmallPagePool
       page = one kernel block           page = one state object
              |                               |
       typed block IDs                    typed block IDs
              |                               |
       Attention BlockTable              Mamba BlockTable
              |                               |
       K/V strided views                 Conv/SSM strided views
```

A typed block ID is `(cache_group_id, large_page_id, slot_in_large_page)`.
IDs from different groups must never be compared or inserted into another
group's block table. A large page is assigned to one group while it contains
live small pages and returns to the global pool when its last small page is
freed.

Use the exact LCM only as one experiment. Also sweep bounded 2/8/32/64 MiB
superpages: exact LCM removes layout-tail waste but can make reclamation too
coarse when object sizes have an unfriendly greatest common divisor.

## Required runtime changes

### Configuration

Extend the experimental `KVCacheConfig` data carried between scheduler and
worker with:

- backing large-page size and count;
- per-group small-page size and small pages per large page;
- per-group allocation block size (Attention kernel block vs Mamba state);
- a mapping from typed small-page ID to byte offset.

Keep the existing fields intact for the baseline and put the new path behind a
benchmark-only constructor argument until the design is proven.

### Scheduler and coordinator

Replace the single uniform `BlockPool` only in the no-prefix-cache coordinator:

1. Compute every group's additional small-page demand.
2. Perform an atomic admission check across all groups.
3. Prefer free slots in large pages associated with the same request.
4. Allocate an empty large page from the global pool if needed.
5. Fall back to a compatible free slot owned by another request.
6. On request completion, return a large page globally only when all of its
   typed slots are free.

Do not initially implement eviction. A failed atomic allocation leaves every
group unchanged and the request remains waiting.

### Block tables and slot mapping

Maintain one block table per KV cache group. Translate a typed small-page ID to
the group-local dense ID consumed by the existing NPU kernel metadata:

```text
group_local_id = large_page_id * small_pages_per_large_page + slot
slot_mapping   = group_local_id * kernel_block_size + offset_in_block
```

Attention uses `offset_in_block = token_position % kernel_block_size`. Mamba
state operations use the typed page directly and continue to follow the align
mode's running/previous-state lifecycle.

### Worker views

Allocate one raw backing buffer for the global large-page pool. Construct each
group's logical cache view with a group-specific stride:

```text
large-page stride -> large_page_size_bytes
small-page stride -> group_small_page_size_bytes
component stride  -> current K/V or Conv/SSM layout
```

PR #14340's `as_strided` work is the kernel-facing prerequisite. The new code
changes which byte range a block ID selects. The original assumption that no
math kernel changes were needed proved false for GDN decode: that custom
operator bypasses the tensor's leading stride when indexing recurrent state.

## Validation matrix

### Correctness

- allocator unit tests: reuse, cross-type reassignment, atomic rollback,
  partial-page sharing, and whole-page release;
- block-table/slot-mapping tests at empty, first, last, and overflow slots;
- deterministic generation: identical token IDs for at least 100 prompts;
- prefill/decode boundary lengths around every block size;
- memory canaries around each small page to detect cross-group overwrite.

### Capacity

Run two trace families using the same available KV cache bytes:

- short prompts (512-4K), high arrival rate, high concurrency;
- long context (8K-49,152), mixed lifetimes and request completion order.

Report useful/reserved bytes, current-layout padding, superpage tail waste,
stranded free slots, accepted/rejected requests, and maximum live sequences.

### Performance

After a warm-up, run at least three repetitions and report median plus spread:

- maximum concurrency at 49,152 tokens;
- offline throughput;
- serving throughput at fixed QPS;
- TTFT and TPOT/ITL (mean, p50, p99);
- allocator CPU time per scheduler step;
- NPU memory and any additional address-translation/kernel launches.

## Go/no-go criteria

Proceed to prefix-cache integration only if all correctness tests pass and one
of the following is repeatable on the NPU:

- at least 10% more admitted/live sequences at the same memory budget; or
- at least 20% less active fragmentation on a representative production trace.

The initial performance guardrail is no more than 2% throughput/TPOT regression
and no more than 3% TTFT regression. If bounded superpages beat exact LCM, keep
the bounded design; matching the paper's allocator literally is not the goal.

## Later phases (original roadmap, updated status)

1. Prefix-cache hash ownership and type-aware eviction now have an
   exact-LCM whole-page experimental implementation; partial-block COW and
   production validation remain.
2. Fixed-width serial Qwen3-Next/Qwen3.5 MTP now has a no-prefix validation
   path; broader speculative decoding and Mamba external-cache loading remain.
3. Add KV offload/connector address metadata.
4. Validate DCP/PCP and multi-rank deterministic allocation.
5. Decide whether the abstraction belongs upstream in vLLM or remains an
   Ascend platform backend.

## Implemented checkpoint (2026-09-01)

The exact-LCM, no-prefix-cache checkpoint is implemented in:

- `vllm_ascend/core/typed_kv_cache.py`: typed plan, block IDs, global
  superpage allocator, per-group BlockPool facade, admission facade, and dense
  worker byte views;
- `vllm_ascend/core/typed_kv_cache_coordinator.py`: real vLLM single-type
  managers backed by the typed pool and multi-group admission measured in
  superpages;
- `benchmarks/kv_cache/run_typed_runtime_mvp.py`: NPU address canaries, Mamba
  view checks, slot mapping, and actual cache-scatter kernel comparison.

Qwen3.5-0.8B on one A3 die produced a 69,468,160-byte exact-LCM superpage. It
holds 265 Attention pages (262,144 bytes each) or 64 Mamba state pages
(1,085,440 bytes each). Seven runtime/coordinator tests passed in the A3 image.
The 1,000-iteration NPU kernel run passed correctness and measured a typed over
uniform median-latency ratio of 1.0177. This is within the initial 2% kernel
guardrail, but end-to-end serving evidence is still required before enabling
the path by default.

## End-to-end checkpoint (2026-09-01)

The typed plan is now propagated through the real startup path behind the
default-off `VLLM_ASCEND_ENABLE_TYPED_KV_CACHE` switch. The sizing stage
attaches the plan to `KVCacheConfig`, the scheduler selects the typed
coordinator, and the worker consumes group-local IDs when clearing and binding
the cache. The plan survives deepcopy, pickle, and scheduler config generation.
Nine typed sizing/coordinator tests pass in the A3 CANN container.

The first real Qwen3.5 dummy-weight serving smoke test found and fixed two
runtime integration defects:

- `aclnnInplaceIndexFill` cannot clear the int8 cache backing tensor, so the
  worker now coalesces adjacent page IDs and clears dense ranges with `zero_`;
- `KVCacheManager` polls `BlockPool.take_events()` even with events disabled,
  so the typed admission facade now implements the empty event interface.

After those fixes, both uniform and typed modes complete deterministic
generation, fixed-QPS requests, and a small concurrency run without request
failures. The result is not yet semantically valid: the uniform output starts
`..!!`, while the typed output starts `....`; divergence begins at the third
generated token, after the first decode state has been written and consumed.
The typed pages are cleared only once, and the per-group allocation observed by
the worker is Attention `[265]` and Mamba `[128]`, `[192]`, `[256]`, so repeated
zeroing is not the cause.

Do not report the smoke-test TTFT/TPOT numbers as an optimization result. The
available Qwen3.8 A3 image contains a prebuilt custom-op bundle and source from
2026-08-14, while the local PR #14340 snapshot dates from 2026-08-17. More importantly, #14340
supports non-contiguous components inside an equal-size HMA page; it does not
prove that every kernel bind consumes different physical page strides for
Attention and Mamba groups. The next correctness gate is therefore a matching
post-#14340 custom-op build plus per-decode state canaries at the Attention
slot mapping and all three Mamba state indices. Full QPS and maximum-concurrency
runs remain blocked until deterministic token IDs match.

## Address-table checkpoint and GDN stride remediation (2026-09-08)

The Qwen3.5-27B cache geometry makes an exact-LCM superpage impractical:
410,517,504 bytes (391.5 MiB) exceeds the 392,691,712-byte (374.5 MiB) managed
budget per shared layer slice. The current MVP therefore maps each group's
logical page IDs through an explicit address table into one shared byte arena.
`static_partition` uses the same worker translation path with fixed,
non-overlapping group regions and serves as the comparison mode.

A controlled deterministic probe found that static and address-table modes
first diverge at the layer-0 GDN decode output. The recurrent state has
`stride[0] = 801,792` float elements, while one dense state contains 786,432
elements; the 15,360-element (61,440-byte) difference is page padding. The
custom kernel receives no leading-stride parameter and advances by the dense
size, so nonzero physical state indices access the wrong bytes.

The implementation now passes the state view's first three element strides
from the Torch adapter through ACLNN/L0 and tiling, and both generic and arch35
kernels use them for state reads and writes.  A separate `triton-strided`
backend provides a correctness-first path: packed recurrent decode for ordinary
decode and the general fused recurrent operator for MTP/speculative decode.
Typed-cache MTP is deliberately limited to matching target/draft families for
fixed-width serial Qwen3-Next and Qwen3.5 MTP
(`1 <= num_speculative_tokens <= 15`) in no-prefix
`address_table`/`static_partition` mode with `mamba_cache_mode=none`;
Prefix/Jenga, aligned-state copying, dynamic widths,
and other speculative methods remain rejected.
Until padded-stride kernel tests and cross-mode exact-token checks pass on NPU,
all end-to-end throughput, TTFT, TPOT, E2EL, and accuracy observations remain
diagnostic rather than performance claims.
