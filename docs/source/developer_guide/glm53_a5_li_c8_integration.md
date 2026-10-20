# GLM-5.3-Flash FP8 CANN indexer

On Ascend A5 with matching `cann_ops_transformer` and CANN operator packages,
select FP8 pooled indexing with:

```bash
--attention-config '{"indexer_kv_dtype":"fp8"}'
```

The CANN `KeyPool` operator computes pooled keys. Its cache adapter quantizes
and writes FP8 E4M3FN keys with separate FP32 per-pool scales; CANN
`PoolKeyIndexer` performs sparse selection. The existing Triton cache-write
and index-packing adapters remain part of that CANN integration.

The `auto` indexer dtype retains main's BF16 CANN implementation. INT8 KPool
indexing and a Triton FP8 compression/scoring backend are outside this change.
KPool LI C4 is also rejected. Main attention KV-cache dtype remains a separate
setting. P/D peers must agree on indexer dtype and physical cache layout.

## Context parallelism and model runners

DSA context parallelism can be requested with:

```bash
--additional-config '{"enable_dsa_cp":true}'
```

The current main branch defaults to ModelRunner V2, while `enable_dsa_cp`
selects V1 when no runner override is set. This change preserves that policy.
The model/backend and FP8 cache views are shared; CPU tests cover both V1 and
V2 cache allocation/reshape interfaces. They do not establish full V2 model
execution or V2 DSA-CP support.

FP8 CP gathers global hidden rows for pooling and cache updates. Query
projection and scoring remain local. Rank-local causal lengths and partial
pool tails prevent reading future pools. Empty CP ranks still update global
keys and the rollback ring; outside FULL capture they skip empty scoring.
FULL graphs retain scoring for changing replay inputs.

## Cache layout and validation scope

FP8 keys and FP32 scale planes preserve their separate padded-page layout.
KPool runtime quantization does not depend on LightningIndexer calibration
tags, so FLOAT checkpoint weights do not disable FP8 pooled indexing.

This change builds on the merged CANN integration
[#16730](https://github.com/vllm-project/vllm-ascend/pull/16730) and adapters
[#18295](https://github.com/vllm-project/vllm-ascend/pull/18295), contributed by
@lijiahang226. The required FP8 page alignment is shared with
[#18648](https://github.com/vllm-project/vllm-ascend/pull/18648).

CPU coverage includes routing, unsupported configuration, causal boundaries,
stable metadata storage, global cache writes, scale planes and runner policy.
The standalone A5 tests exercise CANN FP8 CP shard selection and the cache
writer's padded pages and graph replay. Multi-rank model output/acceptance,
complete model graph replay, P/D transfer and performance acceptance remain
separate checks. No throughput improvement is claimed.
