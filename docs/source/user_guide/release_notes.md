# Release note

## v0.8.4rc2

This is the second release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. Some experimental features are included in this version, such as W8A8 quantization and EP/DP support. We'll make them stable enough in the next release.

### Highlights
- Qwen3 and Qwen3MOE is supported now. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/latest/tutorials/single_npu.html) to run the quick demo. [#709](https://github.com/vllm-project/vllm-ascend/pull/709)
- Ascend W8A8 quantization method is supported now. Please take the [official doc](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_npu_quantization.html) for example. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/619) is welcome. [#580](https://github.com/vllm-project/vllm-ascend/pull/580)
- DeepSeek V3/R1 works with DP, TP and MTP now. Please note that it's still in experimental status. Let us know if you hit any problem. [#429](https://github.com/vllm-project/vllm-ascend/pull/429) [#585](https://github.com/vllm-project/vllm-ascend/pull/585)  [#626](https://github.com/vllm-project/vllm-ascend/pull/626) [#636](https://github.com/vllm-project/vllm-ascend/pull/636) [#671](https://github.com/vllm-project/vllm-ascend/pull/671)

### Core
- ACLGraph feature is supported with V1 engine now. It's disabled by default because this feature rely on CANN 8.1 release. We'll make it avaiable by default in the next release [#426](https://github.com/vllm-project/vllm-ascend/pull/426)
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automaticlly. [#661](https://github.com/vllm-project/vllm-ascend/pull/661)

### Other
- MiniCPM model works now. [#645](https://github.com/vllm-project/vllm-ascend/pull/645)
- openEuler container image supported with `v0.8.4-openeuler` tag and customs Ops build is enabled by default for openEuler OS. [#689](https://github.com/vllm-project/vllm-ascend/pull/689)
- Fix ModuleNotFoundError bug to make Lora work [#600](https://github.com/vllm-project/vllm-ascend/pull/600)
- Add "Using EvalScope evaluation" doc [#611](https://github.com/vllm-project/vllm-ascend/pull/611)
- Add a `VLLM_VERSION` environment to make vLLM version configurable to help developer set correct vLLM version if the code of vLLM is changed by hand locally. [#651](https://github.com/vllm-project/vllm-ascend/pull/651)

## v0.8.4rc1

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

## v0.7.3rc2

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

## v0.7.3rc1

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

## v0.7.1rc1

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
