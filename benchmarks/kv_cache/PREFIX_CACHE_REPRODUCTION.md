# Jenga attention/recurrent-state prefix-policy reproduction

This document records the evidence boundary for the experimental Jenga-style
prefix cache in this fork. It is based on the public
[Jenga paper](https://arxiv.org/abs/2503.18292) and the artifact's
[`vllm-v0-jenga` branch](https://github.com/heheda12345/Jenga-SOSP25-AE/tree/vllm-v0-jenga)
at reference commit
[`bb5cee614`](https://github.com/heheda12345/Jenga-SOSP25-AE/commit/bb5cee614b054e54cd2a4791cb214051cde46fda).
The code is a clean implementation against the vLLM-Ascend interfaces in this
branch; it does not copy model data, logs, prompts, or benchmark results from
the artifact.

## Policy subset reproduced

The dependency-free reference implementation reproduces the following
cache-group/layer-type policy semantics for attention and recurrent state:

- independent prefix legality for each cache group/layer type;
- dense, contiguous dependencies for full attention;
- tail-window dependencies for sliding-window attention;
- periodic state checkpoints (512 tokens by default) for recurrent-state
  caches;
- scheduler chunk clipping at the next configured checkpoint boundary; this is
  a required precondition for state materialization, not proof that the current
  NPU kernel writes the correct state there;
- intersection of all groups' legal prefixes before a request resumes;
- small-page states `EMPTY`, `EVICTABLE`, and `USED`;
- request-associated placement and the five allocation tiers, in order;
- small-page LRU ordered by `(last_access, -prefix_length, block_id)`;
- large-page LRU using the most recent child access time;
- large-page eviction only when every child is evictable;
- removal of every stale hash when an evicted large page is rebound;
- atomic multi-group admission and generation-safe page handles;
- consistency rollback for a cache-hit touch or staged allocation that fails
  partway through multiple cache groups.

The policy calls Jenga's large-page operation an **ownership rebind**. It
invalidates cached metadata and reuses the same physical byte range under a
new cache type. It is not a CPU/NVMe/NPU payload transfer and reports
`bytes_copied=0`.

The experimental runtime adapter additionally maps the policy to vLLM v0.27.1
`KVCacheBlock` metadata. It keeps block IDs group-local, protects all prefix
hits before allocation, commits one multi-group decision, routes flattened
free lists by object identity, and removes upstream hash entries whenever the
policy evicts a page. The scheduler path clips a prefill at each effective
recurrent checkpoint boundary; without that stop, metadata at an intermediate
slot could falsely imply a state checkpoint. Correct NPU state materialization
is still unverified because of the GDN stride blocker described below. Failed
multi-group operations release touched/consumed references and restore manager
bookkeeping; cache entries already evicted while preparing the failed operation
remain evicted. The mode is default-off and uses an exact-LCM physical large
page so every cache type divides it without a tail.

## Reproduce the deterministic policy checks

Run the pure-Python tests from the repository root:

```shell
python -m pytest --confcutdir=tests/ut/core \
  tests/ut/core/test_jenga_prefix_cache.py \
  tests/ut/core/test_jenga_prefix_policy.py \
  tests/ut/core/test_jenga_prefix_runtime.py \
  tests/ut/core/test_jenga_prefix_replay.py -q
```

Run the standalone replay:

```shell
python -m benchmarks.kv_cache.run_jenga_prefix_replay --iterations 10000
```

The replay must report all five tier IDs exactly once, one large-page
eviction/rebind, stale-hash misses for all evicted children, no invariant
errors, and zero copied bytes. Its timed section repeatedly constructs tiny
host-side policy objects. `verified_tier_decisions_per_second` is therefore a
scheduler-policy microbenchmark, not model throughput, NPU throughput, TTFT,
TPOT, or a production-scale allocator result.

## Experimental runtime mode

The normal allocator is unchanged. The runtime path is selected explicitly:

```shell
export VLLM_ASCEND_ENABLE_TYPED_KV_CACHE=1
export VLLM_ASCEND_TYPED_KV_CACHE_MODE=jenga_lcm_prefix
# Also enable vLLM prefix caching through the normal CLI/config option.
# Set SchedulerConfig.watermark to 0; this mode rejects a nonzero watermark.
```

The implementation intentionally supports whole-page/common-alignment prefix
hits only. Prefix hashes are namespaced by cache-group ID. Fine-grained
partial-block copy-on-write, cache events, connector-directed eviction,
KV transfer/offload, speculative decoding, DCP, PCP, PP, packed KV tensors,
and cross attention are rejected or kept outside this experimental path. A
uniform-block scheduler watermark is also rejected because the coordinator's
atomic heterogeneous-page admission is exposed through a boolean scalar
sentinel; a sound heterogeneous headroom unit has not yet been implemented.

## Scope and known blockers

"Policy reproduced" does not mean that the paper's performance result has
been reproduced. In particular:

- the paper evaluates Mamba/linear-state checkpoints; applying the configurable
  checkpoint policy to GDN is this project's engineering extension, not a
  result reported by Jenga;
- the configured 512-token state interval is coarsened to
  `lcm(512, scheduler_block_size)` when the runtime's legal whole-page boundary
  is larger; every crossed effective boundary becomes a separate prefill step;
- page sizes from a local Qwen3.5-27B cache-profile capture produce a 391.50
  MiB exact-LCM large page, larger than its 374.50 MiB managed layer-slice
  budget, so this layout cannot start for that geometry; the raw profile is
  not published because it contains environment-derived metadata;
- the address-table allocator remains the practical 27B experiment, but it
  does not yet carry this exact all-children large-page eviction policy;
- the current custom GDN decode operator does not consume the leading state
  stride used by the heterogeneous view, so end-to-end NPU correctness and
  serving performance remain blocked;
- no Jenga paper utilization or throughput figure is reused as a result of
  this project.

Consequently, the publishable result of this branch is the tested policy
subset plus an experimental whole-page runtime and scheduler integration. It
is not a reproduction of every model path or every feature in the Jenga
artifact. The dependency-free policy/runtime tests do not import torch or
vLLM; the real coordinator integration test requires the repository's pinned
vLLM/torch environment. A formal NPU result requires a stride-aware GDN
kernel, a geometry that can hold the exact-LCM layout (or a separately
validated address-table adaptation), and a fresh correctness-first serving
benchmark.
