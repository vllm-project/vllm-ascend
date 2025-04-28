# FAQs

## Version Specific FAQs

- [[v0.7.1rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/19)
- [[v0.7.3rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/267)
- [[v0.7.3rc2] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/418)
- [[v0.8.4rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/546)

## General FAQs

### 1. What devices are currently supported?

Currently, **ONLY Atlas A2 series**  (Ascend-cann-kernels-910b) are supported:

- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)

Below series are NOT supported yet:
- Atlas 300I Duo、Atlas 300I Pro (Ascend-cann-kernels-310p) might be supported on 2025.Q2
- Atlas 200I A2 (Ascend-cann-kernels-310b) unplanned yet
- Ascend 910, Ascend 910 Pro B (Ascend-cann-kernels-910) unplanned yet

From a technical view, vllm-ascend support would be possible if the torch-npu is supported. Otherwise, we have to implement it by using custom ops. We are also welcome to join us to improve together.

### 2. How to get our docker containers?

You can get our containers at `Quay.io`, e.g., [<u>vllm-ascend</u>](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and [<u>cann</u>](https://quay.io/repository/ascend/cann?tab=tags).

If you are in China, you can use `daocloud` to accelerate your downloading:

```bash
# Replace with tag you want to pull
TAG=v0.7.3rc2
docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:$TAG
```

### 3. What models does vllm-ascend supports?

Find more details [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/supported_models.html).

### 4. How to get in touch with our community?

There are many channels that you can communicate with our community developers / users:

- Submit a GitHub [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues?page=1).
- Join our [<u>weekly meeting</u>](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z) and share your ideas.
- Join our [<u>WeChat</u>](https://github.com/vllm-project/vllm-ascend/issues/227) group and ask your quenstions.
- Join our ascend channel in [<u>vLLM forums</u>](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support/6) and publish your topics.

### 5. What features does vllm-ascend V1 supports?

Find more details [<u>here</u>](https://github.com/vllm-project/vllm-ascend/issues/414).

### 6. How to solve the problem of "Failed to infer device type" or "libatb.so: cannot open shared object file"?

Basically, the reason is that the NPU environment is not configured correctly. You can:
1. try `source /usr/local/Ascend/nnal/atb/set_env.sh` to enable NNAL package.
2. try `source /usr/local/Ascend/ascend-toolkit/set_env.sh` to enable CANN package.
3. try `npu-smi info` to check whether the NPU is working.

If all above steps are not working, you can try the following code with python to check whether there is any error:

```
import torch
import torch_npu
import vllm
```

If all above steps are not working, feel free to submit a GitHub issue.

### 7. How does vllm-ascend perform?

Currently, only some models are improved. Such as `Qwen2 VL`, `Deepseek  V3`. Others are not good enough. In the future, we will support graph mode and custom ops to improve the performance of vllm-ascend. And when the official release of vllm-ascend is released, you can install `mindie-turbo` with `vllm-ascend` to speed up the inference as well.

### 8. How vllm-ascend work with vllm?
vllm-ascend is a plugin for vllm. Basically, the version of vllm-ascend is the same as the version of vllm. For example, if you use vllm 0.7.3, you should use vllm-ascend 0.7.3 as well. For main branch, we will make sure `vllm-ascend` and `vllm` are compatible by each commit.

### 9. Does vllm-ascend support Prefill Disaggregation feature?

Currently, only 1P1D is supported by vllm. For vllm-ascend, it'll be done by [this PR](https://github.com/vllm-project/vllm-ascend/pull/432). For NPND, vllm is not stable and fully supported yet. We will make it stable and supported by vllm-ascend in the future.

### 10. Does vllm-ascend support quantization method?

Currently, w8a8 quantization is already supported by vllm-ascend originally on v0.8.4rc2 or heigher, If you're using vllm 0.7.3 version, w8a8 quantization is supporeted with the integration of vllm-ascend and mindie-turbo, please use `pip install vllm-ascend[mindie-turbo]`.

### 11. How to run w8a8 DeepSeek model?

Currently, w8a8 DeepSeek is working in process: [support AscendW8A8 quantization](https://github.com/vllm-project/vllm-ascend/pull/511)

Please run DeepSeek with BF16 now, follwing the [Multi-Node DeepSeek inferencing tutorail](https://vllm-ascend.readthedocs.io/en/main/tutorials/multi_node.html)

### 12. There is not output in log when loading models using vllm-ascend, How to solve it?

If you're using vllm 0.7.3 version, this is a known progress bar display issue in VLLM, which has been resolved in [this PR](https://github.com/vllm-project/vllm/pull/12428), please cherry-pick it locally by yourself. Otherwise, please fill up an issue.

### 13. How vllm-ascend is tested

vllm-ascend is tested by functional test, performance test and accuracy test.

- **Functional test**: we added CI, includes portion of vllm's native unit tests and vllm-ascend's own unit tests，on vllm-ascend's test, we test basic functionality、popular models availability and [supported features](https://vllm-ascend.readthedocs.io/en/latest/user_guide/suppoted_features.html) via e2e test

- **Performance test**: we provide [benchmark](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks) tools for end-to-end performance benchmark which can easily to re-route locally, we'll publish a perf website like [vllm](https://simon-mo-workspace.observablehq.cloud/vllm-dashboard-v0/perf) does to show the performance test results for each pull request

- **Accuracy test**: we're working on adding accuracy test to CI as well.

Finnall, for each release, we'll publish the performance test and accuracy test report in the future.

### 14. How to fix the error "InvalidVersion" when using vllm-ascend?
It's usually because you have installed an dev/editable version of vLLM package. In this case, we provide the env variable `VLLM_VERSION` to let users specify the version of vLLM package to use. Please set the env variable `VLLM_VERSION` to the version of vLLM package you have installed. The format of `VLLM_VERSION` should be `X.Y.Z`.
