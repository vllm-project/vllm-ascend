# Release note

## v0.7.1.rc1

We are excited to announce the release candidate of v0.7.1 for vllm-ascend. vllm-ascend is a community maintained hardware plugin for running vLLM on the Ascend NPU. With this release, users can now enjoy the latest features and improvements of vLLM on the Ascend NPU.

Note that this is a release candidate, and there may be some bugs or issues. We appreciate your feedback and suggestions [here](https://github.com/vllm-project/vllm-ascend/issues/19)

### Highlights

- The first release which official supports the Ascend NPU on vLLM originally. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/latest/) to start the journey.

### Other changes

- Added the Ascend quantization config option, the implementation will comming soon.

### Known issues

- This release relies on an unreleased torch_npu version. Please [install](https://vllm-ascend.readthedocs.io/en/latest/installation.html) it manually.
- There are logs like `No platform deteced, vLLM is running on UnspecifiedPlatform` or `Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")` shown when runing vllm-ascend. It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/12432) which will be included in v0.7.3 soon.
