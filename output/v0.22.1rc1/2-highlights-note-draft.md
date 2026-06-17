## v0.22.1rc1 - 2026.06.18

We're excited to announce the release of v0.22.1rc1 for vLLM Ascend. This is the first release candidate for the v0.22.1 release line, building on v0.21.0rc1 and aligning the plugin with upstream vLLM v0.22.1. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.

### Highlights

- **Mooncake Connector for DeepSeek V4 / Hybrid KV Cache**: Mooncake connector now supports DeepSeek V4 and hybrid KV cache disaggregated prefill scenarios with correct block stride handling, compressed KV transfer calculation, and hybrid Mamba token alignment. [#10342](https://github.com/vllm-project/vllm-ascend/pull/10342)
- **HCCL Weight Transfer for RL Workloads**: Added an HCCL-based weight transfer backend for Ascend NPU so trainer and inference workers can synchronize weights in RL pipelines without a CUDA/NCCL dependency. [#9152](https://github.com/vllm-project/vllm-ascend/pull/9152)
- **Ascend 950 Expansion**: Extended Ascend 950 support with W8A8FP8 dynamic quantization and platform-specific CPU binding support. [#10236](https://github.com/vllm-project/vllm-ascend/pull/10236) [#10483](https://github.com/vllm-project/vllm-ascend/pull/10483)

### Features

- Added file logging with daily rotation and a configurable output path for vLLM Ascend services. [#10306](https://github.com/vllm-project/vllm-ascend/pull/10306)
- Added multimodal input support for DFlash workloads. [#9340](https://github.com/vllm-project/vllm-ascend/pull/9340)
- Added KV consumer partial-group caching for hybrid Mamba models. [#10009](https://github.com/vllm-project/vllm-ascend/pull/10009)
- Added MiniMax M2 C8 cache-scale support in GQA `load_weights`. [#10461](https://github.com/vllm-project/vllm-ascend/pull/10461)
- Added SSD support for multiple DP ranks on the same machine to avoid local-rank path collisions in Mooncake offload directories. [#10477](https://github.com/vllm-project/vllm-ascend/pull/10477)
- Added PCP + DP validation interception for unsupported configuration combinations. [#10178](https://github.com/vllm-project/vllm-ascend/pull/10178)

### Hardware and Operator Support

- Added W8A8FP8 dynamic quantization support for Ascend 950. [#10236](https://github.com/vllm-project/vllm-ascend/pull/10236)
- Added Ascend 950 CPU binding support for A5 server topology and process layout. [#10483](https://github.com/vllm-project/vllm-ascend/pull/10483)
- Added an HCCL-based weight transfer engine for Ascend NPU weight synchronization. [#9152](https://github.com/vllm-project/vllm-ascend/pull/9152)

### Performance

- Optimized `split_qkv_tp_rmsnorm_rope` with grid-stride loading and host-side reciprocal precomputation; the PR reports about a 5x kernel speedup on the tested MiniMax-M2.5 W8A8 QuaRot prefill workload. [#9830](https://github.com/vllm-project/vllm-ascend/pull/9830)
- Reused prebuilt chunk host metadata for Ascend chunk ops to reduce host-device synchronization overhead on Qwen3.5 workloads. [#9310](https://github.com/vllm-project/vllm-ascend/pull/9310)
- Skipped `compute_slot_mapping` for Mamba groups to reduce unnecessary work in hybrid cache paths. [#10492](https://github.com/vllm-project/vllm-ascend/pull/10492)
- Enabled multistream DSV4 DSA overlap and removed redundant DSA v1 code paths. [#10518](https://github.com/vllm-project/vllm-ascend/pull/10518)

### Dependencies

- Upgraded the mainline integration baseline to upstream vLLM v0.22.1. [#10476](https://github.com/vllm-project/vllm-ascend/pull/10476)
- Added release metadata so image and wheel workflows can publish `v0.22.1rc1` artifacts directly. [#10578](https://github.com/vllm-project/vllm-ascend/pull/10578)

### Documentation

- Refreshed the context parallel, EPLB, and speculative decoding documentation. [#10332](https://github.com/vllm-project/vllm-ascend/pull/10332)
- Added Kimi 2.6 and GLM5.2 documentation. [#9969](https://github.com/vllm-project/vllm-ascend/pull/9969) [#10544](https://github.com/vllm-project/vllm-ascend/pull/10544)

### Others

- Improved CI routing, test selection, and nightly/image workflow robustness for the v0.22.1 release line. [#10445](https://github.com/vllm-project/vllm-ascend/pull/10445) [#10573](https://github.com/vllm-project/vllm-ascend/pull/10573) [#10576](https://github.com/vllm-project/vllm-ascend/pull/10576)

### Known Issues

- MiniMax 2.7 dual-node 16-card deployments may hang or crash after 10-20 minutes under load. [#10591](https://github.com/vllm-project/vllm-ascend/issues/10591)
- Llama LoRA can still hit an einsum tensor-dimension mismatch on Ascend. [#10577](https://github.com/vllm-project/vllm-ascend/issues/10577)
