# vLLM-Ascend V0.26.0RC1 Test Conclusion

## 1. Model Validation

Targeted validation was conducted on the following 4 models, which are the only models supported in this release:

| Model | Quantization | Hardware / Deployment | Validation Scope |
|-------|--------------|------------------------|------------------|
| GLM-5.2 | w8a8c8 | A3, 4 nodes, large EP, 1P1D | Performance validation; GPQA accuracy validation |
| Kimi K3 | w4a8 | A3, 8 nodes, large EP, 1P1D; Ascend 950, large EP, 4 nodes, 1P1D | Performance validation; GPQA and OCRBench accuracy validation |
| DeepSeek-V4-Flash | w8a8c8 | A3, 4 nodes, large DP, 1P1D | Performance validation; GPQA accuracy validation |
| DeepSeek-V4-Pro | w8a8c8 | A3, 2 nodes, large EP, 1P1D | Performance validation; GPQA accuracy validation |

### Performance Scenarios

| Scenario Type | Input Length | Output Length | Prefix Cache |
|---------------|--------------|---------------|--------------|
| Prefill-only (Pure P) | 16k | 1k | — |
| Prefill-only (Pure P) | 128k | 1k | — |
| Prefill-only (Pure P) | Variable 40k–80k | 2.5k | — |
| Decode-only (Pure D) | 16k | 1k | — |
| Decode-only (Pure D) | 128k | 1k | — |
| Decode-only (Pure D) | Variable 40k–80k | 2.5k | — |
| End-to-End | 16k | 1k | 0% |
| End-to-End | 128k | 1k | 90% |
| End-to-End | 1M | 1k | — |

### Accuracy

| Dataset | Model | Score |
|---------|-------|-------|
| GPQA | GLM-5.2 | 92.165 |
| GPQA | Kimi K3 | 92.42 |
| GPQA | DeepSeek-V4-Flash | 91.67 |
| GPQA | DeepSeek-V4-Pro | 91.92 |
| OCRBench | Kimi K3 | 0.88 |

---

## 2. Reliability Validation

Small-scale online real-business validation was conducted on a subset of models, each running for more than 3 days:

| No. | Model | Validation Method | Running Duration |
|-----|-------|-------------------|------------------|
| 1 | Qwen3.5-122B | Online real-business validation (small scale) | More than 3 days |
| 2 | Qwen3.6-35B | Online real-business validation (small scale) | More than 3 days |
| 3 | Qwen3.5-397B | Online real-business validation (small scale) | More than 3 days |
| 4 | DeepSeek-V4-Flash | Online real-business validation (small scale) | More than 3 days |
| 5 | GLM-5.2 | Online real-business validation (small scale) | More than 3 days |
| 6 | Kimi K3 | Online real-business validation (small scale) | More than 3 days |

## 3. Known Issues

Some issues found during validation remain unresolved in this release. The table below is compiled from the release known-issues list; the community-side records are also available in the Known Issues section of the version [release note](../docs/source/user_guide/release_notes.md).

| No. | Issue | Description | Impact |
|-----|-------|-------------|--------|
| 1 | [#15237](https://github.com/vllm-project/vllm-ascend/issues/15237) | Kimi K3 128k-1k case performance below expectations: DSpark weights are weak, with generally low acceptance rate | Performance only: the DSpark draft acceptance rate is very low for long inputs (~20k–40k tokens), so speculative decoding provides no effective benefit; fix planned in v0.27 |
| 2 | [#15649](https://github.com/vllm-project/vllm-ascend/issues/15649) | DSV4-Flash with pooling enabled: 128k-1k, prefix 90 scenario shows performance regression vs. no pooling (>10%, pool misses) | After enabling pooling, the 128k-1k case shows lower performance than without pooling — a single-prefix regression (>10%); no confirmed root cause or workaround yet; improvement planned for a future Q3 release |
| 3 | [#15649](https://github.com/vllm-project/vllm-ascend/issues/15649) | Qwen3-32B w8a8 A2 PD co-location pooling test: TPS 2313 with pooling vs. 2443 without, >3% regression (measured >5%) | Same pooling single-prefix regression: TPS drops from 2443 to 2313 (~5.3%) with pooling enabled; planned for a future Q3 release |
| 4 | [#15648](https://github.com/vllm-project/vllm-ascend/issues/15648) | DeepSeek-V4 with sleep_mode_extra_cleanup enabled: service fails to start after sleep then wakeup | After the sleep/wakeup sequence with `sleep_mode_extra_cleanup` enabled, the service fails to start or serve requests; sleep mode is mainly intended for training scenarios; upstream fix planned for v0.27 |
| 5 | [#14911](https://github.com/vllm-project/vllm-ascend/issues/14911) | PD disaggregation: DeepSeek-V3.1_w8a8 service starts, then the D node hangs on inference | DeepSeek-V3.1 is being sunset per the model sunsetting process; only the four export models are supported in this release (see release note); retest ongoing on main |
| 6 | [#14911](https://github.com/vllm-project/vllm-ascend/issues/14911) | DeepSeek-V3.1-w4a8 PD co-location: multi-concurrency inference causes service hang | model being sunset, outside the supported export-model scope; retest ongoing on main |
| 7 | [#15678](https://github.com/vllm-project/vllm-ascend/issues/15678) | A3 dual-node co-located GLM-5.2-W4A8C8: server-side error on inference after service start | The service starts successfully, but inference requests fail on the server side in the A3 double-node co-located scenario; this model/weight is sunset and outside the supported export-model scope |
| 8 | [#15677](https://github.com/vllm-project/vllm-ascend/issues/15677) | PD disaggregation: DeepSeek-V3.1 layerwise gain below target in 64k+1k (expected 2%) | The measured layerwise gain is below the expected 2% target in the 64k-1k case; the layerwise RP3 path is intended for KV-offload use only and is not a standalone export feature — recorded as a feature-scope limitation rather than a general model performance regression |
| 9 | [#15296](https://github.com/vllm-project/vllm-ascend/issues/15296) | Qwen3-235B-A22B-w8a8 PD co-location, A3 single-node graph mode aclgraph_32768_4096_TPOT50ms performance regression (W8A8 ~10%) | Some PD separation/disaggregation tests show lower accuracy or performance than the baseline (W8A8 ~10% regression); tracked in the follow-up sunset plan |
| 10 | [#15296](https://github.com/vllm-project/vllm-ascend/issues/15296) | Qwen3-235B W4A8 quantization, 800IA3 PD co-location aclgraph performance regression (W4A8 ~4%) |lower performance than baseline in PD tests (W4A8 ~4%); tracked in the follow-up sunset plan |
| 11 | [#14911](https://github.com/vllm-project/vllm-ascend/issues/14911) | A3 single-node DeepSeek-V3.1-w8a8: server-side error on inference after service start | Model being sunset, outside the supported export-model scope; retest ongoing on main |
| 12 | [#15268](https://github.com/vllm-project/vllm-ascend/issues/15268) | Ascend 950 GLM-5.1-W4A4C8 dual-node co-located performance regression of 40% (baseline 133.85, measured 84.91/77.16/76.75) | Load imbalance across Ascend 950 cards after vLLM v0.24.0: uneven card execution speed, uneven work distribution, some cards hostbound; short-term workaround via msboost (not independently verified) |
| 13 | [#15296](https://github.com/vllm-project/vllm-ascend/issues/15296) | Qwen3-32B-QuaRot v0.26.0rc performance regression | Performance regression caused by the aclnnscatterpakvcache operator (reshape_and_cache switched to scatter_nd_update) plus CPU-binding impact; not an export model; planned for 0.27 |
| 14 | [#15650](https://github.com/vllm-project/vllm-ascend/issues/15650) | Ascend 950 Qwen3.6-27B with d-flash: service start fails with NotImplementedError (DFlash drafters require the V2 model runner) | DFlash drafters with mixed sliding/full attention require the V2 model runner: `NotImplementedError: ... relaunch with VLLM_USE_V2_MODEL_RUNNER=1`; documented workaround is relaunching with `VLLM_USE_V2_MODEL_RUNNER=1`; support expected in 0.27/0.28 |
| 15 | [#15268](https://github.com/vllm-project/vllm-ascend/issues/15268) | Ascend 950 GLM-5.1-W4A4C8 co-located service start hangs then errors (NPUEvent 507034) | C16 combined with dsa_cp hangs on Ascend 950; root cause still under investigation (suspected impact of the new operator package in the Ascend 950 image); not an export model |
| 16 | [#14911](https://github.com/vllm-project/vllm-ascend/issues/14911) | A3 four-node DeepSeek_V3.1T_MTP1_PD: D node hangs on inference (AclrtSynchronizeStreamWithTimeout 507001) | Model being sunset, outside the supported export-model scope; retest ongoing on main |
| 17 | [#15651](https://github.com/vllm-project/vllm-ascend/issues/15651) | A3 DeepSeek-V3.2-w8a8 single-node co-located case: partial performance regression | Partial performance regression vs. the baseline in the A3 single-node mixed deployment scenario; CPU binding identified as one possible contributor, an operator issue cannot be ruled out; not an export model; 0.27 expected to include improvements |
| 18 | [#15658](https://github.com/vllm-project/vllm-ascend/issues/15658) | Minimax_m2.7_w8a8_A3 main-branch nightly smoke performance regression | Single-node W8A8 case shows a clear regression below the 3% thresholds (3609.53 < 4302.74; 1836.65 < 2162.26); the double-node case (669.56 < 698.50) may be variation — more samples needed; not an export model; planned for 0.27 |
