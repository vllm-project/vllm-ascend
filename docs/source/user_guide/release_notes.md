# Release note

## v0.9.2rc1 - 2025.07.11

This is the 1st release candidate of v0.9.2 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to get started. From this release, V1 engine will be enabled by default, there is no need to set `VLLM_USE_V1=1` any more. And this release is the last version to support V0 engine, V0 code will be clean up in the future.

### Highlights
- Pooling model works with V1 engine now. You can take a try with Qwen3 embedding model [#1359](https://github.com/vllm-project/vllm-ascend/pull/1359).
- The performance on Atlas 300I series has been improved. [#1591](https://github.com/vllm-project/vllm-ascend/pull/1591)
- aclgraph mode works with Moe models now. Currently, only Qwen3 Moe is well tested. [#1381](https://github.com/vllm-project/vllm-ascend/pull/1381)

### Core
- Ascend PyTorch adapter (torch_npu) has been upgraded to `2.5.1.post1.dev20250619`. Don’t forget to update it in your environment. [#1347](https://github.com/vllm-project/vllm-ascend/pull/1347)
- The **GatherV3** error has been fixed with **aclgraph** mode. [#1416](https://github.com/vllm-project/vllm-ascend/pull/1416)
- W8A8 quantization works on Atlas 300I series now. [#1560](https://github.com/vllm-project/vllm-ascend/pull/1560)
- Fix the accuracy problem with deploy models with parallel parameters. [#1678](https://github.com/vllm-project/vllm-ascend/pull/1678)
- The pre-built wheel package now requires lower version of glibc. Users can use it by `pip install vllm-ascend` directly. [#1582](https://github.com/vllm-project/vllm-ascend/pull/1582)

## Other
- Official doc has been updated for better read experience. For example, more deployment tutorials are added, user/developer docs are updated. More guide will coming soon.
- Fix accuracy problem for deepseek V3/R1 models with torchair graph in long sequence predictions. [#1331](https://github.com/vllm-project/vllm-ascend/pull/1331)
- A new env variable `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` has been added. It enables the fused allgather-experts kernel for Deepseek V3/R1 models. The default value is `0`. [#1335](https://github.com/vllm-project/vllm-ascend/pull/1335)
- A new env variable `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` has been added to improve the performance of topk-topp sampling. The default value is 0, we'll consider to enable it by default in the future[#1732](https://github.com/vllm-project/vllm-ascend/pull/1732)
- A batch of bugs have been fixed for Data Parallelism case [#1273](https://github.com/vllm-project/vllm-ascend/pull/1273) [#1322](https://github.com/vllm-project/vllm-ascend/pull/1322) [#1275](https://github.com/vllm-project/vllm-ascend/pull/1275) [#1478](https://github.com/vllm-project/vllm-ascend/pull/1478)
- The DeepSeek performance has been improved. [#1194](https://github.com/vllm-project/vllm-ascend/pull/1194) [#1395](https://github.com/vllm-project/vllm-ascend/pull/1395) [#1380](https://github.com/vllm-project/vllm-ascend/pull/1380)
- Ascend scheduler works with prefix cache now. [#1446](https://github.com/vllm-project/vllm-ascend/pull/1446)
- DeepSeek now works with prefix cache now. [#1498](https://github.com/vllm-project/vllm-ascend/pull/1498)
- Support prompt logprobs to recover ceval accuracy in V1 [#1483](https://github.com/vllm-project/vllm-ascend/pull/1483)

## v0.9.1rc1 - 2025.06.22

This is the 1st release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to get started.

### Highlights

- Atlas 300I series is experimental supported in this release. [#1333](https://github.com/vllm-project/vllm-ascend/pull/1333) After careful consideration, this feature **will NOT be included in v0.9.1-dev branch** taking into account the v0.9.1 release quality and the feature rapid iteration to improve performance on Atlas 300I series. We will improve this from 0.9.2rc1 and later.
- Support EAGLE-3 for speculative decoding. [#1032](https://github.com/vllm-project/vllm-ascend/pull/1032)

### Core
- Ascend PyTorch adapter (torch_npu) has been upgraded to `2.5.1.post1.dev20250528`. Don’t forget to update it in your environment. [#1235](https://github.com/vllm-project/vllm-ascend/pull/1235)
- Support Atlas 300I series container image. You can get it from [quay.io](https://quay.io/repository/vllm/vllm-ascend)
- Fix token-wise padding mechanism to make multi-card graph mode work. [#1300](https://github.com/vllm-project/vllm-ascend/pull/1300)
- Upgrade vllm to 0.9.1 [#1165]https://github.com/vllm-project/vllm-ascend/pull/1165

### Other Improvements
- Initial support Chunked Prefill for MLA. [#1172](https://github.com/vllm-project/vllm-ascend/pull/1172)
- An example of best practices to run DeepSeek with ETP has been added. [#1101](https://github.com/vllm-project/vllm-ascend/pull/1101)
- Performance improvements for DeepSeek using the TorchAir graph. [#1098](https://github.com/vllm-project/vllm-ascend/pull/1098), [#1131](https://github.com/vllm-project/vllm-ascend/pull/1131)
- Supports the speculative decoding feature with AscendScheduler. [#943](https://github.com/vllm-project/vllm-ascend/pull/943)
- Improve `VocabParallelEmbedding` custom op performance. It will be enabled in the next release. [#796](https://github.com/vllm-project/vllm-ascend/pull/796)
- Fixed a device discovery and setup bug when running vLLM Ascend on Ray [#884](https://github.com/vllm-project/vllm-ascend/pull/884)
- DeepSeek with [MC2](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0043.html) (Merged Compute and Communication) now works properly. [#1268](https://github.com/vllm-project/vllm-ascend/pull/1268)
- Fixed log2phy NoneType bug with static EPLB feature. [#1186](https://github.com/vllm-project/vllm-ascend/pull/1186)
- Improved performance for DeepSeek with DBO enabled. [#997](https://github.com/vllm-project/vllm-ascend/pull/997), [#1135](https://github.com/vllm-project/vllm-ascend/pull/1135)
- Refactoring AscendFusedMoE [#1229](https://github.com/vllm-project/vllm-ascend/pull/1229)
- Add initial user stories page (include LLaMA-Factory/TRL/verl/MindIE Turbo/GPUStack) [#1224](https://github.com/vllm-project/vllm-ascend/pull/1224)
- Add unit test framework [#1201](https://github.com/vllm-project/vllm-ascend/pull/1201)

### Known Issues
- In some cases, the vLLM process may crash with a **GatherV3** error when **aclgraph** is enabled. We are working on this issue and will fix it in the next release. [#1038](https://github.com/vllm-project/vllm-ascend/issues/1038)
- Prefix cache feature does not work with the Ascend Scheduler but without chunked prefill enabled. This will be fixed in the next release. [#1350](https://github.com/vllm-project/vllm-ascend/issues/1350)

### Full Changelog
https://github.com/vllm-project/vllm-ascend/compare/v0.9.0rc2...v0.9.1rc1

## v0.9.0rc2 - 2025.06.10

This release contains some quick fixes for v0.9.0rc1. Please use this release instead of v0.9.0rc1.

### Highlights

- Fix the import error when vllm-ascend is installed without editable way. [#1152](https://github.com/vllm-project/vllm-ascend/pull/1152)

## v0.9.0rc1 - 2025.06.09

This is the 1st release candidate of v0.9.0 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. From this release, V1 Engine is recommended to use. The code of V0 Engine is frozen and will not be maintained any more. Please set environment `VLLM_USE_V1=1` to enable V1 Engine.

### Highlights

- DeepSeek works with graph mode now. Follow the [official doc](https://vllm-ascend.readthedocs.io/en/latest/user_guide/graph_mode.html) to take a try. [#789](https://github.com/vllm-project/vllm-ascend/pull/789)
- Qwen series models works with graph mode now. It works by default with V1 Engine. Please note that in this release, only Qwen series models are well tested with graph mode. We'll make it stable and generalize in the next release. If you hit any issues, please feel free to open an issue on GitHub and fallback to eager mode temporarily by set `enforce_eager=True` when initializing the model.

### Core

- The performance of multi-step scheduler has been improved. Thanks for the contribution from China Merchants Bank. [#814](https://github.com/vllm-project/vllm-ascend/pull/814)
- LoRA、Multi-LoRA And Dynamic Serving is supported for V1 Engine now. Thanks for the contribution from China Merchants Bank. [#893](https://github.com/vllm-project/vllm-ascend/pull/893)
- Prefix cache and chunked prefill feature works now [#782](https://github.com/vllm-project/vllm-ascend/pull/782) [#844](https://github.com/vllm-project/vllm-ascend/pull/844)
- Spec decode and MTP features work with V1 Engine now. [#874](https://github.com/vllm-project/vllm-ascend/pull/874) [#890](https://github.com/vllm-project/vllm-ascend/pull/890)
- DP feature works with DeepSeek now. [#1012](https://github.com/vllm-project/vllm-ascend/pull/1012)
- Input embedding feature works with V0 Engine now. [#916](https://github.com/vllm-project/vllm-ascend/pull/916)
- Sleep mode feature works with V1 Engine now. [#1084](https://github.com/vllm-project/vllm-ascend/pull/1084)

### Model

- Qwen2.5 VL works with V1 Engine now. [#736](https://github.com/vllm-project/vllm-ascend/pull/736)
- LLama4 works now. [#740](https://github.com/vllm-project/vllm-ascend/pull/740)
- A new kind of DeepSeek model called dual-batch overlap(DBO) is added. Please set `VLLM_ASCEND_ENABLE_DBO=1` to use it. [#941](https://github.com/vllm-project/vllm-ascend/pull/941)

### Other

- online serve with ascend quantization works now. [#877](https://github.com/vllm-project/vllm-ascend/pull/877)
- A batch of bugs for graph mode and moe model have been fixed. [#773](https://github.com/vllm-project/vllm-ascend/pull/773) [#771](https://github.com/vllm-project/vllm-ascend/pull/771) [#774](https://github.com/vllm-project/vllm-ascend/pull/774) [#816](https://github.com/vllm-project/vllm-ascend/pull/816) [#817](https://github.com/vllm-project/vllm-ascend/pull/817) [#819](https://github.com/vllm-project/vllm-ascend/pull/819) [#912](https://github.com/vllm-project/vllm-ascend/pull/912) [#897](https://github.com/vllm-project/vllm-ascend/pull/897) [#961](https://github.com/vllm-project/vllm-ascend/pull/961) [#958](https://github.com/vllm-project/vllm-ascend/pull/958) [#913](https://github.com/vllm-project/vllm-ascend/pull/913) [#905](https://github.com/vllm-project/vllm-ascend/pull/905)
- A batch of performance improvement PRs have been merged. [#784](https://github.com/vllm-project/vllm-ascend/pull/784) [#803](https://github.com/vllm-project/vllm-ascend/pull/803) [#966](https://github.com/vllm-project/vllm-ascend/pull/966) [#839](https://github.com/vllm-project/vllm-ascend/pull/839) [#970](https://github.com/vllm-project/vllm-ascend/pull/970) [#947](https://github.com/vllm-project/vllm-ascend/pull/947) [#987](https://github.com/vllm-project/vllm-ascend/pull/987) [#1085](https://github.com/vllm-project/vllm-ascend/pull/1085)
- From this release, binary wheel package will be released as well. [#775](https://github.com/vllm-project/vllm-ascend/pull/775)
- The contributor doc site is [added](https://vllm-ascend.readthedocs.io/en/latest/community/contributors.html)

### Known Issue

- In some case, vLLM process may be crashed with aclgraph enabled. We're working this issue and it'll be fixed in the next release.
- Multi node data-parallel doesn't work with this release. This is a known issue in vllm and has been fixed on main branch. [#18981](https://github.com/vllm-project/vllm/pull/18981)

## v0.7.3.post1 - 2025.05.29

This is the first post release of 0.7.3. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey. It includes the following changes:

### Highlights

- Qwen3 and Qwen3MOE is supported now. The performance and accuracy of Qwen3 is well tested. You can try it now. Mindie Turbo is recomanded to improve the performance of Qwen3. [#903](https://github.com/vllm-project/vllm-ascend/pull/903) [#915](https://github.com/vllm-project/vllm-ascend/pull/915)
- Added a new performance guide. The guide aims to help users to improve vllm-ascend performance on system level. It includes OS configuration, library optimization, deploy guide and so on. [#878](https://github.com/vllm-project/vllm-ascend/pull/878) [Doc Link](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/developer_guide/performance/optimization_and_tuning.html)

### Bug Fix

- Qwen2.5-VL  works for RLHF scenarios now. [#928](https://github.com/vllm-project/vllm-ascend/pull/928)
- Users can launch the model from online weights now. e.g. from huggingface or modelscope directly [#858](https://github.com/vllm-project/vllm-ascend/pull/858) [#918](https://github.com/vllm-project/vllm-ascend/pull/918)
- The meaningless log info `UserWorkspaceSize0` has been cleaned. [#911](https://github.com/vllm-project/vllm-ascend/pull/911)
- The log level for `Failed to import vllm_ascend_C` has been changed to `warning` instead of `error`. [#956](https://github.com/vllm-project/vllm-ascend/pull/956)
- DeepSeek MLA now works with chunked prefill in V1 Engine. Please note that V1 engine in 0.7.3 is just expermential and only for test usage. [#849](https://github.com/vllm-project/vllm-ascend/pull/849) [#936](https://github.com/vllm-project/vllm-ascend/pull/936)

### Docs

- The benchmark doc is updated for Qwen2.5 and Qwen2.5-VL [#792](https://github.com/vllm-project/vllm-ascend/pull/792)
- Add the note to clear that only "modelscope<1.23.0" works with 0.7.3. [#954](https://github.com/vllm-project/vllm-ascend/pull/954)

## v0.7.3 - 2025.05.08

🎉 Hello, World!

We are excited to announce the release of 0.7.3 for vllm-ascend. This is the first official release. The functionality, performance, and stability of this release are fully tested and verified. We encourage you to try it out and provide feedback. We'll post bug fix versions in the future if needed. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.

### Highlights
- This release includes all features landed in the previous release candidates ([v0.7.1rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.1rc1), [v0.7.3rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc1), [v0.7.3rc2](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc2)). And all the features are fully tested and verified. Visit the official doc the get the detail [feature](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/user_guide/suppoted_features.html) and [model](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/user_guide/supported_models.html) support matrix.
- Upgrade CANN to 8.1.RC1 to enable chunked prefill and automatic prefix caching features. You can now enable them now.
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automatically. [#662](https://github.com/vllm-project/vllm-ascend/pull/662)
- Integrate MindIE Turbo into vLLM Ascend to improve DeepSeek V3/R1, Qwen 2 series performance. [#708](https://github.com/vllm-project/vllm-ascend/pull/708)

### Core
- LoRA、Multi-LoRA And Dynamic Serving is supported now. The performance will be improved in the next release. Please follow the official doc for more usage information. Thanks for the contribution from China Merchants Bank. [#700](https://github.com/vllm-project/vllm-ascend/pull/700)

### Model
- The performance of Qwen2 vl and Qwen2.5 vl is improved. [#702](https://github.com/vllm-project/vllm-ascend/pull/702)
- The performance of `apply_penalties` and `topKtopP` ops are improved. [#525](https://github.com/vllm-project/vllm-ascend/pull/525)

### Other
- Fixed a issue that may lead CPU memory leak. [#691](https://github.com/vllm-project/vllm-ascend/pull/691) [#712](https://github.com/vllm-project/vllm-ascend/pull/712)
- A new environment `SOC_VERSION` is added. If you hit any soc detection error when building with custom ops enabled, please set `SOC_VERSION` to a suitable value. [#606](https://github.com/vllm-project/vllm-ascend/pull/606)
- openEuler container image supported with v0.7.3-openeuler tag. [#665](https://github.com/vllm-project/vllm-ascend/pull/665)
- Prefix cache feature works on V1 engine now. [#559](https://github.com/vllm-project/vllm-ascend/pull/559)

## v0.8.5rc1 - 2025.05.06

This is the 1st release candidate of v0.8.5 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. Now you can enable V1 egnine by setting the environment variable `VLLM_USE_V1=1`, see the feature support status of vLLM Ascend in [here](https://vllm-ascend.readthedocs.io/en/latest/user_guide/suppoted_features.html).

### Highlights
- Upgrade CANN version to 8.1.RC1 to support chunked prefill and automatic prefix caching (`--enable_prefix_caching`) when V1 is enabled [#747](https://github.com/vllm-project/vllm-ascend/pull/747)
- Optimize Qwen2 VL and Qwen 2.5 VL [#701](https://github.com/vllm-project/vllm-ascend/pull/701)
- Improve Deepseek V3 eager mode and graph mode performance, now you can use --additional_config={'enable_graph_mode': True} to enable graph mode. [#598](https://github.com/vllm-project/vllm-ascend/pull/598) [#719](https://github.com/vllm-project/vllm-ascend/pull/719)

### Core
- Upgrade vLLM to 0.8.5.post1 [#715](https://github.com/vllm-project/vllm-ascend/pull/715)
- Fix early return in CustomDeepseekV2MoE.forward during profile_run [#682](https://github.com/vllm-project/vllm-ascend/pull/682)
- Adapts for new quant model generated by modelslim [#719](https://github.com/vllm-project/vllm-ascend/pull/719)
- Initial support on P2P Disaggregated Prefill based on llm_datadist [#694](https://github.com/vllm-project/vllm-ascend/pull/694)
- Use `/vllm-workspace` as code path and include `.git` in container image to fix issue when start vllm under `/workspace` [#726](https://github.com/vllm-project/vllm-ascend/pull/726)
- Optimize NPU memory usage to make DeepSeek R1 W8A8 32K model len work. [#728](https://github.com/vllm-project/vllm-ascend/pull/728)
- Fix `PYTHON_INCLUDE_PATH` typo in setup.py [#762](https://github.com/vllm-project/vllm-ascend/pull/762)

### Other
- Add Qwen3-0.6B test [#717](https://github.com/vllm-project/vllm-ascend/pull/717)
- Add nightly CI [#668](https://github.com/vllm-project/vllm-ascend/pull/668)
- Add accuracy test report [#542](https://github.com/vllm-project/vllm-ascend/pull/542)

## v0.8.4rc2 - 2025.04.29

This is the second release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. Some experimental features are included in this version, such as W8A8 quantization and EP/DP support. We'll make them stable enough in the next release.

### Highlights
- Qwen3 and Qwen3MOE is supported now. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/latest/tutorials/single_npu.html) to run the quick demo. [#709](https://github.com/vllm-project/vllm-ascend/pull/709)
- Ascend W8A8 quantization method is supported now. Please take the [official doc](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_npu_quantization.html) for example. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/619) is welcome. [#580](https://github.com/vllm-project/vllm-ascend/pull/580)
- DeepSeek V3/R1 works with DP, TP and MTP now. Please note that it's still in experimental status. Let us know if you hit any problem. [#429](https://github.com/vllm-project/vllm-ascend/pull/429) [#585](https://github.com/vllm-project/vllm-ascend/pull/585)  [#626](https://github.com/vllm-project/vllm-ascend/pull/626) [#636](https://github.com/vllm-project/vllm-ascend/pull/636) [#671](https://github.com/vllm-project/vllm-ascend/pull/671)

### Core
- ACLGraph feature is supported with V1 engine now. It's disabled by default because this feature rely on CANN 8.1 release. We'll make it available by default in the next release [#426](https://github.com/vllm-project/vllm-ascend/pull/426)
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automatically. [#661](https://github.com/vllm-project/vllm-ascend/pull/661)

### Other
- MiniCPM model works now. [#645](https://github.com/vllm-project/vllm-ascend/pull/645)
- openEuler container image supported with `v0.8.4-openeuler` tag and customs Ops build is enabled by default for openEuler OS. [#689](https://github.com/vllm-project/vllm-ascend/pull/689)
- Fix ModuleNotFoundError bug to make Lora work [#600](https://github.com/vllm-project/vllm-ascend/pull/600)
- Add "Using EvalScope evaluation" doc [#611](https://github.com/vllm-project/vllm-ascend/pull/611)
- Add a `VLLM_VERSION` environment to make vLLM version configurable to help developer set correct vLLM version if the code of vLLM is changed by hand locally. [#651](https://github.com/vllm-project/vllm-ascend/pull/651)

## v0.8.4rc1 - 2025.04.18

This is the first release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. From this version, vllm-ascend will follow the newest version of vllm and release every two weeks. For example, if vllm releases v0.8.5 in the next two weeks, vllm-ascend will release v0.8.5rc1 instead of v0.8.4rc2. Please find the detail from the [official documentation](https://vllm-ascend.readthedocs.io/en/latest/developer_guide/versioning_policy.html#release-window).

### Highlights

- vLLM V1 engine experimental support is included in this version. You can visit [official guide](https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html) to get more detail. By default, vLLM will fallback to V0 if V1 doesn't work, please set `VLLM_USE_V1=1` environment if you want to use V1 forcely. 
- LoRA、Multi-LoRA And Dynamic Serving is supported now. The performance will be improved in the next release. Please follow the [official doc](https://docs.vllm.ai/en/latest/features/lora.html) for more usage information. Thanks for the contribution from China Merchants Bank. [#521](https://github.com/vllm-project/vllm-ascend/pull/521).
- Sleep Mode feature is supported. Currently it's only work on V0 engine. V1 engine support will come soon. [#513](https://github.com/vllm-project/vllm-ascend/pull/513)

### Core

- The Ascend scheduler is added for V1 engine. This scheduler is more affinity with Ascend hardware. More scheduler policy will be added in the future. [#543](https://github.com/vllm-project/vllm-ascend/pull/543)
- Disaggregated Prefill feature is supported. Currently only 1P1D works. NPND is under design by vllm team. vllm-ascend will support it once it's ready from vLLM. Follow the [official guide](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) to use. [#432](https://github.com/vllm-project/vllm-ascend/pull/432)
- Spec decode feature works now. Currently it's only work on V0 engine. V1 engine support will come soon. [#500](https://github.com/vllm-project/vllm-ascend/pull/500)
- Structured output feature works now on V1 Engine. Currently it only supports xgrammar backend while using guidance backend may get some errors. [#555](https://github.com/vllm-project/vllm-ascend/pull/555)

### Other

- A new communicator `pyhccl` is added. It's used for call CANN HCCL library directly instead of using `torch.distribute`. More usage of it will be added in the next release [#503](https://github.com/vllm-project/vllm-ascend/pull/503)
- The custom ops build is enabled by default. You should install the packages like `gcc`, `cmake` first to build `vllm-ascend` from source. Set `COMPILE_CUSTOM_KERNELS=0` environment to disable the compilation if you don't need it. [#466](https://github.com/vllm-project/vllm-ascend/pull/466)
- The custom op `rotay embedding` is enabled by default now to improve the performance. [#555](https://github.com/vllm-project/vllm-ascend/pull/555)

## v0.7.3rc2 - 2025.03.29

This is 2nd release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.
- Quickstart with container: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/quick_start.html
- Installation: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html

### Highlights
- Add Ascend Custom Ops framewrok. Developers now can write customs ops using AscendC. An example ops `rotary_embedding` is added. More tutorials will come soon. The Custom Ops compilation is disabled by default when installing vllm-ascend. Set `COMPILE_CUSTOM_KERNELS=1` to enable it.  [#371](https://github.com/vllm-project/vllm-ascend/pull/371)
- V1 engine is basic supported in this release. The full support will be done in 0.8.X release. If you hit any issue or have any requirement of V1 engine. Please tell us [here](https://github.com/vllm-project/vllm-ascend/issues/414). [#376](https://github.com/vllm-project/vllm-ascend/pull/376)
- Prefix cache feature works now. You can set `enable_prefix_caching=True` to enable it. [#282](https://github.com/vllm-project/vllm-ascend/pull/282)

### Core
- Bump torch_npu version to dev20250320.3 to improve accuracy to fix `!!!` output problem. [#406](https://github.com/vllm-project/vllm-ascend/pull/406)

### Model
- The performance of Qwen2-vl is improved by optimizing patch embedding (Conv3D). [#398](https://github.com/vllm-project/vllm-ascend/pull/398)

### Other

- Fixed a bug to make sure multi step scheduler feature work. [#349](https://github.com/vllm-project/vllm-ascend/pull/349)
- Fixed a bug to make prefix cache feature works with correct accuracy. [#424](https://github.com/vllm-project/vllm-ascend/pull/424)

## v0.7.3rc1 - 2025.03.14

🎉 Hello, World! This is the first release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.
- Quickstart with container: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/quick_start.html
- Installation: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html

### Highlights
- DeepSeek V3/R1 works well now. Read the [official guide](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/tutorials/multi_node.html) to start! [#242](https://github.com/vllm-project/vllm-ascend/pull/242)
- Speculative decoding feature is supported. [#252](https://github.com/vllm-project/vllm-ascend/pull/252)
- Multi step scheduler feature is supported. [#300](https://github.com/vllm-project/vllm-ascend/pull/300)

### Core
- Bump torch_npu version to dev20250308.3 to improve `_exponential` accuracy
- Added initial support for pooling models. Bert based model, such as `BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-v2-m3` works now. [#229](https://github.com/vllm-project/vllm-ascend/pull/229)

### Model
- The performance of Qwen2-VL is improved. [#241](https://github.com/vllm-project/vllm-ascend/pull/241)
- MiniCPM is now supported [#164](https://github.com/vllm-project/vllm-ascend/pull/164)

### Other
- Support MTP(Multi-Token Prediction) for DeepSeek V3/R1 [#236](https://github.com/vllm-project/vllm-ascend/pull/236)
- [Docs] Added more model tutorials, include DeepSeek, QwQ, Qwen and Qwen 2.5VL. See the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/tutorials/index.html) for detail
- Pin modelscope<1.23.0 on vLLM v0.7.3 to resolve: https://github.com/vllm-project/vllm/pull/13807

### Known issues
- In [some cases](https://github.com/vllm-project/vllm-ascend/issues/324), especially when the input/output is very long, the accuracy of output may be incorrect. We are working on it. It'll be fixed in the next release.
- Improved and reduced the garbled code in model output. But if you still hit the issue, try to change the generation config value, such as `temperature`, and try again. There is also a knonwn issue shown below. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/267) is welcome. [#277](https://github.com/vllm-project/vllm-ascend/pull/277)

## v0.7.1rc1 - 2025.02.19

🎉 Hello, World!

We are excited to announce the first release candidate of v0.7.1 for vllm-ascend.

vLLM Ascend Plugin (vllm-ascend) is a community maintained hardware plugin for running vLLM on the Ascend NPU. With this release, users can now enjoy the latest features and improvements of vLLM on the Ascend NPU.

Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.1-dev) to start the journey. Note that this is a release candidate, and there may be some bugs or issues. We appreciate your feedback and suggestions [here](https://github.com/vllm-project/vllm-ascend/issues/19)

### Highlights

- Initial supports for Ascend NPU on vLLM. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- DeepSeek is now supported. [#88](https://github.com/vllm-project/vllm-ascend/pull/88) [#68](https://github.com/vllm-project/vllm-ascend/pull/68)
- Qwen, Llama series and other popular models are also supported, you can see more details in [here](https://vllm-ascend.readthedocs.io/en/latest/user_guide/supported_models.html).

### Core

- Added the Ascend quantization config option, the implementation will coming soon. [#7](https://github.com/vllm-project/vllm-ascend/pull/7) [#73](https://github.com/vllm-project/vllm-ascend/pull/73)
- Add silu_and_mul and rope ops and add mix ops into attention layer. [#18](https://github.com/vllm-project/vllm-ascend/pull/18)

### Other

- [CI] Enable Ascend CI to actively monitor and improve quality for vLLM on Ascend. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- [Docker] Add vllm-ascend container image [#64](https://github.com/vllm-project/vllm-ascend/pull/64)
- [Docs] Add a [live doc](https://vllm-ascend.readthedocs.org) [#55](https://github.com/vllm-project/vllm-ascend/pull/55)

### Known issues

- This release relies on an unreleased torch_npu version. It has been installed within official container image already. Please [install](https://vllm-ascend.readthedocs.io/en/v0.7.1rc1/installation.html) it manually if you are using non-container environment.
- There are logs like `No platform detected, vLLM is running on UnspecifiedPlatform` or `Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")` shown when running vllm-ascend. It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/12432) which will be included in v0.7.3 soon.
- There are logs like `# CPU blocks: 35064, # CPU blocks: 2730` shown when running vllm-ascend which should be `# NPU blocks:` . It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/13378) which will be included in v0.7.3 soon.
