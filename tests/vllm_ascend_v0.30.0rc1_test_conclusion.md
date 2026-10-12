# vLLM-Ascend v0.30.0rc1 Test Conclusion

## 1. Model Validation

Targeted validation was conducted on the following 4 models, which are the only models supported in this release:

| Model | Hardware / Quantization / Deployment | Validation Scope |
|-------|--------------------------------------|------------------|
| DeepSeek V4.1 Flash | A3, 2 nodes, large EP, 1P1D; A2, 4 nodes, large EP, 1P1D | Performance validation; GPQA and MMMU-pro accuracy validation |
| GLM 5.3 | A3 w4a8, 4 nodes, large EP, 1P1D; A2, 8 nodes, large EP, 1P1D | Performance validation; GPQA accuracy validation |
| GLM 5.3 Flash | A3 w4a8, 2 nodes, large EP, 1P1D; A2, 4 nodes, large EP, 1P1D | Performance validation; GPQA and MMMU-pro accuracy validation |
| Qwen3.8 27B | A3 single-node co-located; A2 single-node co-located | Performance validation; GPQA and MMMU-pro accuracy validation |

### Performance Scenarios

| Hardware | Model | Scenario | Topology | Total Cards | Context Length | Avg Input | Avg Output | Prefix Cache Hit Rate |
|----------|-------|----------|----------|-------------|----------------|-----------|------------|------------------------|
| A3 (or A2) | DeepSeek V4.1 Flash | Pure P | 1P1D | 16 (A2: 32) | 1M | 16k | 1 | 0% |
| A3 (or A2) | DeepSeek V4.1 Flash | Pure P | 1P1D | 16 (A2: 32) | 1M | 128k | 1 | 0% |
| A3 | DeepSeek V4.1 Flash | Pure P | 1P1D | 16 (A2: 32) | 1M | 40K-80K | 1 | 0% |
| A3 (or A2) | DeepSeek V4.1 Flash | Pure D | 1P1D | 16 (A2: 32) | 1M | 16k | 1k | 100% |
| A3 (or A2) | DeepSeek V4.1 Flash | Pure D | 1P1D | 16 (A2: 32) | 1M | 128k | 1k | 100% |
| A3 (or A2) | DeepSeek V4.1 Flash | Full PD | 1P1D | 32 | 1M | 16k | 1k | 0% |
| A3 | DeepSeek V4.1 Flash | Full PD | 1P1D | 16 | 1M | 16k | 1k | 0% |
| A3 | DeepSeek V4.1 Flash | Full PD | 1P1D | 16 | 1M | 128k | 1K | Prefix Cache 90%, 32 prefixes |
| A3 | DeepSeek V4.1 Flash | Full PD | 1P1D | 16 | 1M | 40K-80K | 2.5k | 0% |
| A3 (or A2) | GLM 5.3 | Pure P | 1P1D | 32 (A2: 64) | 200000 | 16k | 1 | 0% |
| A3 (or A2) | GLM 5.3 | Pure P | 1P1D | 32 (A2: 64) | 200000 | 128k | 1 | 0% |
| A3 (or A2) | GLM 5.3 | Pure D | 1P1D | 32 (A2: 64) | 200000 | 16k | 1k | 100% |
| A3 (or A2) | GLM 5.3 | Pure D | 1P1D | 32 (A2: 64) | 200000 | 128k | 1k | 100% |
| A3 (or A2) | GLM 5.3 | Full PD | 1P1D | 32 (A2: 64) | 200000 | 16k | 1k | 0% |
| A3 | GLM 5.3 | Full PD | 1P1D | 32 | 200000 | 128k | 1K | Prefix Cache 90%, 32 prefixes |
| A3 | GLM 5.3 | Full PD | 1P1D | 32 | 200000 | 40K-80K | 2.5k | 0% |
| A3 (or A2) | GLM 5.3 Flash | Pure P | 1P1D | 16 (A2: 32) | 1M | 16k | 1 | 0% |
| A3 (or A2) | GLM 5.3 Flash | Pure P | 1P1D | 16 (A2: 32) | 1M | 128k | 1 | 0% |
| A3 (or A2) | GLM 5.3 Flash | Pure D | 1P1D | 16 (A2: 32) | 1M | 16k | 1k | 100% |
| A3 (or A2) | GLM 5.3 Flash | Pure D | 1P1D | 16 (A2: 32) | 1M | 128k | 1k | 100% |
| A3 (or A2) | GLM 5.3 Flash | Full PD | 1P1D | 16 (A2: 32) | 1M | 16k | 1k | 0% |
| A3 | GLM 5.3 Flash | Full PD | 1P1D | 16 (A2: 32) | 1M | 40K-80K | 2.5k | 0% |
| A3 | GLM 5.3 Flash | Full PD | 1P1D | 16 (A2: 32) | 1M | 128k | 1K | Prefix Cache 90%, 32 prefixes |
| A3 (or A2) | Qwen3.8 27B | Full PD | PD co-located TP2 | 2 | 133120 | 16k | 1k | 0% |
| A3 (or A2) | Qwen3.8 27B | Full PD | PD co-located TP2 | 2 | 133120 | 128k | 1k | 0% |

### Accuracy Scenarios

| Model | Device | Dataset | Accuracy (Average) |
|-------|--------|---------|--------------------|
| DeepSeek V4.1 Flash | A3 PD disaggregation | GPQA | 90.40 |
| DeepSeek V4.1 Flash | A3 PD disaggregation | MMMU-pro | 75.70 |
| DeepSeek V4.1 Flash | A2 PD disaggregation | GPQA | 91.25 |
| DeepSeek V4.1 Flash | A2 PD disaggregation | MMMU-pro | 76.49 |
| GLM 5.3 | A3 PD disaggregation | GPQA | 91.92 |
| GLM 5.3 | A2 PD disaggregation | GPQA | 90.00 |
| GLM 5.3 Flash | A3 PD disaggregation | GPQA | 89.90 |
| GLM 5.3 Flash | A3 PD disaggregation | MMMU-pro | 78.36 |
| GLM 5.3 Flash | A2 PD disaggregation | GPQA | 90.90 |
| GLM 5.3 Flash | A2 PD disaggregation | MMMU-pro | 77.11 |
| Qwen3.8 27B | A3 co-located | GPQA | 88.89 |
| Qwen3.8 27B | A3 co-located | MMMU-pro | 75.78 |
| Qwen3.8 27B | A2 co-located | GPQA | 91.08 |
| Qwen3.8 27B | A2 co-located | MMMU-pro | 75.34 |

### Test Configuration Links

- DeepSeek-V4.1-Flash configuration archive: [DeepSeek-V4.1-Flash#5.2 1P1D PD Disaggregated Deployment](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/DeepSeek-V4.1-Flash.html#52-1p1d-pd-disaggregated-deployment)
- GLM-5.3 configuration archive: [GLM-5.3#5.2 Prefill-Decode Disaggregation](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/GLM5.3.html#52-prefill-decode-disaggregation)
- GLM-5.3-Flash configuration archive: [GLM-5.3-Flash#5.2 Multi-Node PD Separation Deployment](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/GLM5.3-Flash.html#52-1p1d-pd-disaggregated-deployment)
- Qwen3.8-27B configuration archive: [Qwen3.8-27B#5.1 Single-Node Online Deployment](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/Qwen3.8-27B.html#51-single-node-online-deployment)

---

## 2. Reliability Validation

Stress tests were conducted on a subset of models as follows:

| No. | Model | Validation Method | Running Duration |
|-----|-------|-------------------|------------------|
| 1 | DeepSeek V4.1 Flash | High-concurrency stress test | More than 3 hours |
| 2 | GLM 5.3 | High-concurrency stress test | More than 3 hours |
| 3 | GLM 5.3 Flash | High-concurrency stress test | More than 3 hours |
| 4 | Qwen3.8 27B | High-concurrency stress test | More than 3 hours |

## 3. Known Issues

Some issues found during validation remain unresolved in this release, summarized as follows:

| No. | Issue | Description | Impact |
|-----|-------|-------------|--------|
| 1 | [#17661](https://github.com/vllm-project/vllm-ascend/pull/17661) | Reverts #16848, which had made Model Runner V2 the default when `VLLM_USE_V2_MODEL_RUNNER` is unset (falling back to V1 only via the mrv2_utils.py blacklist). The revert restores the previous rule: unset selects V1, `=1` forces V2, `=0` forces V1 | Configurations that do not set `VLLM_USE_V2_MODEL_RUNNER` use Model Runner V1 again; MRV2 becomes opt-in per configuration — set `VLLM_USE_V2_MODEL_RUNNER=1` to keep V2 |
| 2 | [#17470](https://github.com/vllm-project/vllm-ascend/pull/17470) | Reverts #17066 (MRV1 dirty block-table optimization) on the release line, restoring the previous full block-table copy path and removing the `block_table_no_commit_optimize` option along with its kernel, documentation and tests. The revert is used to test whether the optimization contributes to the DeepSeek-V4-Pro-w4a8-prefix-cache-PD throughput regression (observed ~703.61 / 719.66 tokens/s) | Restores pre-#17066 MRV1 block-table behavior; an experimental regression-isolation branch, not yet confirmed as the final fix (the observations do not establish #17066 as the root cause) |
| 3 | [#18029](https://github.com/vllm-project/vllm-ascend/pull/18029) | Reverts #17480, which made the Ascend-owned KV cache paths (KVPoolScheduler, KVPoolWorker._build_cache_coordinator, AscendHybridKVCacheCoordinator, the GLM-Next custom KVCacheConfig planner) consume the resolved `CacheConfig` / `KVCacheConfig.prefix_cache_retention_interval` field instead of reading `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` from the environment | Restores pre-#17480 behavior: these paths read the `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` environment variable from `vllm.envs` again |
| 4 | [#17729](https://github.com/vllm-project/vllm-ascend/issues/17729) | DeepSeek V4.1 Flash with DSpark disabled, DSA context parallelism (DSA-CP) enabled, and `FULL_DECODE_ONLY` graph mode: a cudagraph replay may consume metadata / sparse indices from the capture batch instead of the current batch, causing accuracy regression, unstable answer quality, empty / reasoning-only responses, and occasional max-token truncation | Accuracy regression on the non-DSpark + DSA-CP + FULL_DECODE_ONLY path; enabling DSpark changes the result and can mask the regression but is not a correctness fix for that path; the accuracy investigation is ongoing (see the issue tracker) |
| 5 | [#17541](https://github.com/vllm-project/vllm-ascend/issues/17541) | GLM-5.3 W8A8C8 on A3 six-node P/D deployment, two accuracy failures: (1) with Prefill `--max-num-batched-tokens=8192`, a 6816-token two-turn prompt produces repeated / garbled output (RCA: the fused-MoE receive capacity `mega_moe_max_tokens=8192` was set too small, silently clipping expert routing beyond the limit; with 65536 the original prompt and a ~54K control case pass); (2) at Prefill 4096, a 919,528-token prompt still fails to retrieve an event near the middle of the context (near-end events are retrieved correctly), and the SFA C8 switch has no effect | (1) Traced to the undersized `mega_moe_max_tokens` setting (the running build defaults to 65536); increasing it works around the issue; (2) the root cause of the 919K middle-retrieval failure remains open, investigation ongoing (see the issue tracker) |
