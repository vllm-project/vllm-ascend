## v0.14.0rc1 - 2026.01.26

This is the first release candidate of v0.14.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started. This release includes all the changes in v0.13.0rc2. So We just list the differences from v0.13.0rc2. If you are upgrading from v0.13.0rc1, please read both v0.14.0rc1 and v0.13.0rc2 release notes.

### Highlights

- 310P support is back now. In this release, only basic dense and vl models are supported with eager mode. We'll keep improving and maintaining the support for 310P. [#5776](https://github.com/vllm-project/vllm-ascend/pull/5776)
- Support compressed tensors moe w8a8-int8 quantization. [#5718](https://github.com/vllm-project/vllm-ascend/pull/5718)
- Support Medusa speculative decoding. [#5668](https://github.com/vllm-project/vllm-ascend/pull/5668)
- Support Eagle3 speculative decoding for Qwen3vl. [#4848](https://github.com/vllm-project/vllm-ascend/pull/4848)

### Features

- Xlite Backend supports Qwen3 MoE now. [#5951](https://github.com/vllm-project/vllm-ascend/pull/5951)
- Support DSA-CP for PD-mix deployment case. [#5702](https://github.com/vllm-project/vllm-ascend/pull/5702)
- Add support of new W4A4_LAOS_DYNAMIC quantization method. [#5143](https://github.com/vllm-project/vllm-ascend/pull/5143)

### Performance

- The performance of Qwen3-next has been improved. [#5664](https://github.com/vllm-project/vllm-ascend/pull/5664) [#5984](https://github.com/vllm-project/vllm-ascend/pull/5984) [#5765](https://github.com/vllm-project/vllm-ascend/pull/5765)
- The CPU bind logic and performance has been improved. [#5555](https://github.com/vllm-project/vllm-ascend/pull/5555)
- Merge Q/K split to simplify AscendApplyRotaryEmb for better performance. [#5799](https://github.com/vllm-project/vllm-ascend/pull/5799)
- Add Matmul Allreduce Rmsnorm fusion Pass. It's disabled by default. Set `fuse_allreduce_rms=True` in `--additional_config` to enable it. [#5034](https://github.com/vllm-project/vllm-ascend/pull/5034)
- Optimize rope embedding with triton kernel for huge performance gain. [#5918](https://github.com/vllm-project/vllm-ascend/pull/5918)
- support advanced apply_top_k_top_p without top_k constraint. [#6098](https://github.com/vllm-project/vllm-ascend/pull/6098)
- Parallelize Q/K/V padding in AscendMMEncoderAttention for better performance. [#6204](https://github.com/vllm-project/vllm-ascend/pull/6204)

### Others

- model runner v2 support triton of penalty. [#5854](https://github.com/vllm-project/vllm-ascend/pull/5854)
- model runner v2 support eagle spec decoding. [#5840](https://github.com/vllm-project/vllm-ascend/pull/5840)
- Fix multi-modal inference OOM issues by setting `expandable_segments:True` by default. [#5855](https://github.com/vllm-project/vllm-ascend/pull/5855)
- `VLLM_ASCEND_ENABLE_MLAPO` is set to `True` by default. It's enabled automatically on decode node in PD deployment case. Please note that this feature will cost more memory. If you are memory sensitive, please set it to False. [#5952](https://github.com/vllm-project/vllm-ascend/pull/5952)
- SSL config can be set to kv_extra_config for PD deployment with mooncake layerwise connector. [#5875](https://github.com/vllm-project/vllm-ascend/pull/5875)
- support `--max_model_len=auto`. [#6193](https://github.com/vllm-project/vllm-ascend/pull/6193)

### Dependencies

- torch-npu is upgraded to 2.9.0 [#6112](https://github.com/vllm-project/vllm-ascend/pull/6112)

### Deprecation & Breaking Changes

- EPLB config options is moved to `eplb_config` in [additional config](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html). The old ones are removed in this release.
- The profiler envs, such as `VLLM_TORCH_PROFILER_DIR` and `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY` do not work with vLLM Ascend now. Please use vLLM `--profiler-config` parameters instead. [#5928](https://github.com/vllm-project/vllm-ascend/pull/5928)

### Known Issues

- If you hit the pickle error from `EngineCore` process sometimes, please cherry-pick the [PR](https://github.com/vllm-project/vllm/pull/32022) into your local vLLM code. This known issue will be fixed in vLLM in the next release.

## v0.13.0rc2 - 2026.01.24

This is the second release candidate of v0.13.0 for vLLM Ascend. In this rc release, we fixed lots of bugs and improved the performance of many models. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to get started. Any feedback is welcome to help us to improve the final version of v0.13.0.

### Highlights

We mainly focus on quality and performance improvement in this release. The spec decode, graph mode, context parallel and EPLB have been improved significantly. A lot of bugs have been fixed and the performance has been improved for DeepSeek3.1/3.2, Qwen3 Dense/MOE models.

### Features

- implement basic framework for batch invariant [#5517](https://github.com/vllm-project/vllm-ascend/pull/5517)
- Eagle spec decode feature now works with full graph mode. [#5118](https://github.com/vllm-project/vllm-ascend/pull/5118)
- Context Parallel(PCP&DCP) feature is more stable now. And it works for most case. Please try it out.
- MTP and eagle spec decode feature now works in most cases. And it's suggested to use them in most cases.
- EPLB feature more stable now. Many bugs have been fixed. Mix placement works now [#6086](https://github.com/vllm-project/vllm-ascend/pull/6086)
- Support kv nz feature for DeepSeek decode node in disagg-prefill scenario [#3072](https://github.com/vllm-project/vllm-ascend/pull/3072)

### Model Support

- LongCat-Flash is supported now.[#3833](https://github.com/vllm-project/vllm-ascend/pull/3833)
- minimax_m2 is supported now. [#5624](https://github.com/vllm-project/vllm-ascend/pull/5624)
- Support for cross-attention and whisper models [#5592](https://github.com/vllm-project/vllm-ascend/pull/5592)

### Performance

- Many custom ops and triton kernels are added in this release to speed up the performance of models. Such as `RejectSampler`, `MoeInitRoutingCustom`, `DispatchFFNCombine` and so on.
- Improved the performance of Layerwise Connector [#5303](https://github.com/vllm-project/vllm-ascend/pull/5303)

### Others

- Basic support Model Runner v2. Model Runner V2 is the next generation of vLLM. It will be used by default in the future release. [#5210](https://github.com/vllm-project/vllm-ascend/pull/5210)
- Fixed a bug that the zmq send/receive may failed [#5503](https://github.com/vllm-project/vllm-ascend/pull/5503)
- Supported to use full-graph with Qwen3-Next-MTP [#5477](https://github.com/vllm-project/vllm-ascend/pull/5477)
- Fix weight transpose in RL scenarios [#5567](https://github.com/vllm-project/vllm-ascend/pull/5567)
- Adapted SP to eagle3 [#5562](https://github.com/vllm-project/vllm-ascend/pull/5562)
- Context Parallel(PCP&DCP) support mlapo [#5672](https://github.com/vllm-project/vllm-ascend/pull/5672)
- GLM4.6 support mtp with fullgraph [#5460](https://github.com/vllm-project/vllm-ascend/pull/5460)
- Flashcomm2 now works with oshard generalized feature [#4723](https://github.com/vllm-project/vllm-ascend/pull/4723)
- Support setting tp=1 for the Eagle draft model [#5804](https://github.com/vllm-project/vllm-ascend/pull/5804)
- Flashcomm1 feature now works with qwen3-vl [#5848](https://github.com/vllm-project/vllm-ascend/pull/5848)
- Support fine-grained shared expert overlap [#5962](https://github.com/vllm-project/vllm-ascend/pull/5962)

### Dependencies

- CANN is upgraded to 8.5.0
- torch-npu is upgraded to 2.8.0.post1. Please note that the post version will not be installed by default. Please install it by hand from [pypi mirror](https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/).
- triton-ascend is upgraded to 3.2.0

### Deprecation & Breaking Changes

- `CPUOffloadingConnector` is deprecated. We'll remove it in the next release. It'll be replaced by CPUOffload feature from vLLM in the future.
- eplb config options is moved to `eplb_config` in [additional config](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html). The old ones will be removed in the next release.
- `ProfileExecuteDuration` [feature](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/profile_execute_duration.html) is deprecated. It's replaced by `ObservabilityConfig` from vLLM.
- The value of `VLLM_ASCEND_ENABLE_MLAPO` env will be set to True by default in the next release. It'll be enabled in decode node by default. Please note that this feature will cost more memory. If you are memory sensitive, please set it to False.

## v0.13.0rc1 - 2025.12.27

This is the first release candidate of v0.13.0 for vLLM Ascend. We landed lots of bug fix, performance improvement and feature support in this release. Any feedback is welcome to help us to improve vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.

### Highlights

- Improved the performance of DeepSeek V3.2, please refer to [tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/DeepSeek-V3.2.html)
- Qwen3-Next MTP with chunked prefill is supported now [#4770](https://github.com/vllm-project/vllm-ascend/pull/4770), please refer to [tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/Qwen3-Next.html)
- [Experimental] Prefill Context Parallel and Decode Context Parallel are supported, but notice that it is an experimental feature now, welcome any feedback. please refer to [context parallel feature guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html)

### Features

- Support openPangu Ultra MoE [4615](https://github.com/vllm-project/vllm-ascend/pull/4615)
- A new quantization method W8A16 is supported now. [#4541](https://github.com/vllm-project/vllm-ascend/pull/4541)
- Cross-machine Disaggregated Prefill is supported now. [#5008](https://github.com/vllm-project/vllm-ascend/pull/5008)
- Add UCMConnector for KV Cache Offloading. [#4411](https://github.com/vllm-project/vllm-ascend/pull/4411)
- Support async_scheduler and disable_padded_drafter_batch in eagle. [#4893](https://github.com/vllm-project/vllm-ascend/pull/4893)
- Support pcp + mtp in full graph mode. [#4572](https://github.com/vllm-project/vllm-ascend/pull/4572)
- Enhance all-reduce skipping logic for MoE models in NPUModelRunner [#5329](https://github.com/vllm-project/vllm-ascend/pull/5329)

### Performance

Some general performance improvement:

- Add l2norm triton kernel [#4595](https://github.com/vllm-project/vllm-ascend/pull/4595)
- Add new pattern for AddRmsnormQuant with SP, which could only take effect in graph mode. [#5077](https://github.com/vllm-project/vllm-ascend/pull/5077)
- Add async exponential while model executing. [#4501](https://github.com/vllm-project/vllm-ascend/pull/4501)
- Remove the transpose step after attention and switch to transpose_batchmatmul [#5390](https://github.com/vllm-project/vllm-ascend/pull/5390)
- To optimize the performance in small batch size scenario, an attention operator with flash decoding function is offered, please refer to item 22 in [FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) to enable it.

### Other

- OOM error on VL models is fixed now. We're keeping observing it, if you hit OOM problem again, please submit an issue. [#5136](https://github.com/vllm-project/vllm-ascend/pull/5136)
- Fixed an accuracy bug of Qwen3-Next-MTP when batched inferring. [#4932](https://github.com/vllm-project/vllm-ascend/pull/4932)
- Fix npu-cpu offloading interface change bug. [#5290](https://github.com/vllm-project/vllm-ascend/pull/5290)
- Fix MHA model runtime error in aclgraph mode [#5397](https://github.com/vllm-project/vllm-ascend/pull/5397)
- Fix unsuitable moe_comm_type under ep=1 scenario [#5388](https://github.com/vllm-project/vllm-ascend/pull/5388)

### Deprecation & Breaking Changes

- `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` is removed and `VLLM_ASCEND_ENABLE_PREFETCH_MLP` is recommend to replace as they always be enabled together. [#5272](https://github.com/vllm-project/vllm-ascend/pull/5272)
- `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` is dropped now. [#5270](https://github.com/vllm-project/vllm-ascend/pull/5270)
- `VLLM_ASCEND_ENABLE_NZ` is disabled for float weight case, since we notice that the performance is not good in some float case. Feel free to set it to 2 if you make sure it works for your case. [#4878](https://github.com/vllm-project/vllm-ascend/pull/4878)
- `chunked_prefill_for_mla` in `additional_config` is dropped now. [#5296](https://github.com/vllm-project/vllm-ascend/pull/5296)
- `dump_config` in `additional_config` is renamed to `dump_config_path` and the type is change from `dict` to `string`. [#5296](https://github.com/vllm-project/vllm-ascend/pull/5296)

### Dependencies

- vLLM version has been upgraded to 0.13.0 and drop 0.12.0 support. [#5146](https://github.com/vllm-project/vllm-ascend/pull/5146)
- Transformer version has been upgraded >= 4.57.3 [#5250](https://github.com/vllm-project/vllm-ascend/pull/5250)

### Known Issues

- Qwen3-Next doesn't support long sequence scenario, and we should limit `gpu-memory-utilization` according to the doc to run Qwen3-Next. We'll improve it in the next release
- The functional break on Qwen3-Next when the input/output is around 3.5k/1.5k is fixed, but it introduces a regression on performance. We'll fix it in next release. [#5357](https://github.com/vllm-project/vllm-ascend/issues/5357)
- There is a precision issue with curl on ultra-short sequences in DeepSeek-V3.2. We'll fix it in next release. [#5370](https://github.com/vllm-project/vllm-ascend/issues/5370)

## v0.11.0 - 2025.12.16

We're excited to announce the release of v0.11.0 for vLLM Ascend. This is the official release for v0.11.0. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.11.0) to get started. We'll consider to release post version in the future if needed. This release note will only contain the important change and note from v0.11.0rc3.

### Highlights

- Improved the performance for deepseek 3/3.1. [#3995](https://github.com/vllm-project/vllm-ascend/pull/3995)
- Fixed the accuracy bug for qwen3-vl. [#4811](https://github.com/vllm-project/vllm-ascend/pull/4811)
- Improved the performance of sample. [#4153](https://github.com/vllm-project/vllm-ascend/pull/4153)
- Eagle3 is back now. [#4721](https://github.com/vllm-project/vllm-ascend/pull/4721)

### Other

- Improved the performance for kimi-k2.  [#4555](https://github.com/vllm-project/vllm-ascend/pull/4555)
- Fixed a quantization bug for deepseek3.2-exp. [#4797](https://github.com/vllm-project/vllm-ascend/pull/4797)
- Fixed qwen3-vl-moe bug under high concurrency. [#4658](https://github.com/vllm-project/vllm-ascend/pull/4658)
- Fixed an accuracy bug for Prefill Decode disaggregation case. [#4437](https://github.com/vllm-project/vllm-ascend/pull/4437)
- Fixed some bugs for EPLB [#4576](https://github.com/vllm-project/vllm-ascend/pull/4576) [#4777](https://github.com/vllm-project/vllm-ascend/pull/4777)
- Fixed the version incompatibility issue for openEuler docker image. [#4745](https://github.com/vllm-project/vllm-ascend/pull/4745)

### Deprecation announcement

- LLMdatadist connector has been deprecated, it'll be removed in v0.12.0rc1
- Torchair graph has been deprecated, it'll be removed in v0.12.0rc1
- Ascend scheduler has been deprecated, it'll be removed in v0.12.0rc1

### Upgrade notice

- torch-npu is upgraded to 2.7.1.post1. Please note that the package is pushed to [pypi mirror](https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/). So it's hard to add it to auto dependence. Please install it by yourself.
- CANN is upgraded to 8.3.rc2.

### Known Issues

- Qwen3-Next doesn't support expert parallel and MTP features in this release. And it'll be oom if the input is too long. We'll improve it in the next release
- Deepseek 3.2 only work with torchair graph mode in this release. We'll make it work with aclgraph mode in the next release.
- Qwen2-audio doesn't work by default. Temporary solution is to set `--gpu-memory-utilization` to a suitable value, such as 0.8.
- CPU bind feature doesn't work if more than one vLLM instance is running on the same node.
