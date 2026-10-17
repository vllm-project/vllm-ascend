# GLM-5.3-Flash LI C8 with DSA context parallelism on A5

A5 uses CANN FP8 KeyPool/PoolKeyIndexer when the hardware and installed
operator package support it. Merge these fields into the existing options:

```bash
--attention-config '{"indexer_kv_dtype":"fp8"}' \
--additional-config '{"enable_dsa_cp":true}'
```

The main KV cache and indexer cache have separate dtype settings. P and D must
agree on indexer dtype and physical layout. PCP/DCP are not enabled by this
option. INT8 remains an explicit compatibility option; KPool LI C4 is rejected.

## Data flow and cache layout

FP8 CP gathers global hidden rows for pooling and cache updates, while query
projection and scoring remain local. Rank-local causal lengths and partial-pool
tails prevent reading future pools. Empty CP ranks still update global keys
and the rollback ring; outside FULL capture they skip empty scoring and return
invalid rows. FULL graphs retain scoring for changing replay inputs.

FP8 E4M3FN keys and FP32 scale planes preserve their separate page layout.
INT8 uses packed key/scale rows. Both honor padded scheduler-page strides.
KPool runtime quantization does not depend on LightningIndexer calibration
tags, so FLOAT checkpoint weights do not disable pooled-cache C8. The existing
Triton FP8 fallback implementation and byte-decoding regressions are retained.

## Prerequisites and validation

The branch inherits the merged CANN integration from
[#16730](https://github.com/vllm-project/vllm-ascend/pull/16730),
merged as `1847b4eaf`, and preserves its BF16 path and fused metadata builders.
The quantized paths add an FP8 CANN adapter and retain the INT8 fallback.
The adapters from
[#18295](https://github.com/vllm-project/vllm-ascend/pull/18295),
merged as `218024e42`, are inherited from main; this branch adds the optional
FP8 scale plane without replacing their newer launch and visibility guards.
INT8 kernels derive from
[#18534](https://github.com/vllm-project/vllm-ascend/pull/18534),
`d0b63227cbc10aa896b16e276de5ac8e19e511a1`.

CPU tests cover local causal boundaries, split requests/pools, empty ranks,
stable per-step metadata storage, global writes, FP32 scale contracts and FP8
byte decoding. NPU kernel tests are provided. Multi-rank FP8 CP output accuracy,
graph replay, P/D cache transfer and performance acceptance remain pending for
this extracted branch. TP=1 D-side MTP profiling does not validate FP8 CP.

Compare BF16 LI + CP, FP8 LI without CP and FP8 LI + CP using identical model,
seed, requests, concurrency and MTP count. Validate outputs and acceptance
before comparing TTFT/TPOT/throughput. Gathering hidden rows transfers more
bytes than projected K/gates; no performance gain is claimed for this path.
