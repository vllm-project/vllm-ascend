# Release note

## v0.7.3rc2

This is 2nd release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.
- Quickstart with container: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/quick_start.html
- Installation: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html

### Highlights
- Add Ascend Custom Ops framewrok. Developers now can write customs ops using AscendC. An example ops `rotary_embedding` is added. More tutorials will come soon. The Custome Ops complation is disabled by default when installing vllm-ascend. Set `COMPILE_CUSTOM_KERNELS=1` to enable it.  [#371](https://github.com/vllm-project/vllm-ascend/pull/371)
- V1 engine is basic supported in this release. The full support will be done in 0.8.X release. If you hit any issue or have any requirement of V1 engine. Please tell us [here](https://github.com/vllm-project/vllm-ascend/issues/414). [#376](https://github.com/vllm-project/vllm-ascend/pull/376)
- Prefix cache feature works now. You can set `enable_prefix_caching=True` to enable it. [#282](https://github.com/vllm-project/vllm-ascend/pull/282)

### Core
- Bump torch_npu version to dev20250320.3 to improve accuracy to fix `!!!` output problem. [#406](https://github.com/vllm-project/vllm-ascend/pull/406)

### Model
- The performance of Qwen2-vl is improved by optimizing patch embedding (Conv3D). [#398](https://github.com/vllm-project/vllm-ascend/pull/398)

### Other

- Fixed a bug to make sure multi step scheduler feature work. [#349](https://github.com/vllm-project/vllm-ascend/pull/349)
- Fixed a bug to make prefix cache feature works with correct accuracy. [#424](https://github.com/vllm-project/vllm-ascend/pull/424)

### Known issues
- There is a error in the case of long prefix input when set `enable_prefix_caching=True` as [issue](https://github.com/vllm-project/vllm-ascend/issues/447) show, which rely on CANN 8.1 NNAL package release.

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
- In some cases, expecially when the input/output is very long with VL model, the accuracy of output may be incorrect. You may see many `!` or some other unreadable code in the output. We are working on it. It'll be fixed in the next release.
- Improved and reduced the garbled code in model output. But if you still hit the issue, try to change the gerneration config value, such as `temperature` and try again. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/267) is welcome. [#277](https://github.com/vllm-project/vllm-ascend/pull/277)

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

- Added the Ascend quantization config option, the implementation will comming soon. [#7](https://github.com/vllm-project/vllm-ascend/pull/7) [#73](https://github.com/vllm-project/vllm-ascend/pull/73)
- Add silu_and_mul and rope ops and add mix ops into attention layer. [#18](https://github.com/vllm-project/vllm-ascend/pull/18)

### Other

- [CI] Enable Ascend CI to actively monitor and improve quality for vLLM on Ascend. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- [Docker] Add vllm-ascend container image [#64](https://github.com/vllm-project/vllm-ascend/pull/64)
- [Docs] Add a [live doc](https://vllm-ascend.readthedocs.org) [#55](https://github.com/vllm-project/vllm-ascend/pull/55)

### Known issues

- This release relies on an unreleased torch_npu version. It has been installed within official container image already. Please [install](https://vllm-ascend.readthedocs.io/en/v0.7.1rc1/installation.html) it manually if you are using non-container environment.
- There are logs like `No platform deteced, vLLM is running on UnspecifiedPlatform` or `Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")` shown when runing vllm-ascend. It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/12432) which will be included in v0.7.3 soon.
- There are logs like `# CPU blocks: 35064, # CPU blocks: 2730` shown when runing vllm-ascend which should be `# NPU blocks:` . It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/13378) which will be included in v0.7.3 soon.
