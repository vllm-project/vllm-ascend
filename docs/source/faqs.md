# FAQs

## Version Specific FAQs

- [[v0.9.1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/2643)
- [[v0.10.1rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/2630)

## General FAQs

### 1. What devices are currently supported?

Currently, **ONLY** Atlas A2 series(Ascend-cann-kernels-910b)，Atlas A3 series(Atlas-A3-cann-kernels) and Atlas 300I(Ascend-cann-kernels-310p) series are supported:

- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)
- Atlas A3 Training series (Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD)
- Atlas 800I A3 Inference series (Atlas 800I A3)
- [Experimental] Atlas 300I Inference series (Atlas 300I Duo)

Below series are NOT supported yet:
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

#### Load Docker Images for offline environment
If you want to use container image for offline environments (no internet connection), you need to download container image in a environment with internet access:

**Exporting Docker images:**

```{code-block} bash
   :substitutions:
# Pull the image on a machine with internet access
TAG=|vllm_ascend_version|
docker pull quay.io/ascend/vllm-ascend:$TAG

# Export the image to a tar file and compress to tar.gz
docker save quay.io/ascend/vllm-ascend:$TAG | gzip > vllm-ascend-$TAG.tar.gz
```

**Importing Docker images in environment without internet access:**

```{code-block} bash
   :substitutions:
# Transfer the tar/tar.gz file to the offline environment and load it
TAG=|vllm_ascend_version|
docker load -i vllm-ascend-$TAG.tar.gz

# Verify the image is loaded
docker images | grep vllm-ascend
```

### 3. What models does vllm-ascend supports?

Find more details [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_models.html).

### 4. How to get in touch with our community?

There are many channels that you can communicate with our community developers / users:

- Submit a GitHub [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues?page=1).
- Join our [<u>weekly meeting</u>](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z) and share your ideas.
- Join our [<u>WeChat</u>](https://github.com/vllm-project/vllm-ascend/issues/227) group and ask your quenstions.
- Join our ascend channel in [<u>vLLM forums</u>](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support/6) and publish your topics.

### 5. What features does vllm-ascend V1 supports?

Find more details [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_features.html).

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

Currently, only some models are improved. Such as `Qwen2.5 VL`, `Qwen3`, `Deepseek  V3`. Others are not good enough. From 0.9.0rc2, Qwen and Deepseek works with graph mode to play a good performance. What's more, you can install `mindie-turbo` with `vllm-ascend v0.7.3` to speed up the inference as well.

### 8. How vllm-ascend work with vllm?
vllm-ascend is a plugin for vllm. Basically, the version of vllm-ascend is the same as the version of vllm. For example, if you use vllm 0.7.3, you should use vllm-ascend 0.7.3 as well. For main branch, we will make sure `vllm-ascend` and `vllm` are compatible by each commit.

### 9. Does vllm-ascend support Prefill Disaggregation feature?

Currently, only 1P1D is supported on V0 Engine. For V1 Engine or NPND support, We will make it stable and supported by vllm-ascend in the future.

### 10. Does vllm-ascend support quantization method?

Currently, w8a8 quantization is already supported by vllm-ascend originally on v0.8.4rc2 or higher, If you're using vllm 0.7.3 version, w8a8 quantization is supporeted with the integration of vllm-ascend and mindie-turbo, please use `pip install vllm-ascend[mindie-turbo]`.

### 11. How to run w8a8 DeepSeek model?

Please following the [inferencing tutorail](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node.html) and replace model to DeepSeek.

### 12. There is no output in log when loading models using vllm-ascend, How to solve it?

If you're using vllm 0.7.3 version, this is a known progress bar display issue in VLLM, which has been resolved in [this PR](https://github.com/vllm-project/vllm/pull/12428), please cherry-pick it locally by yourself. Otherwise, please fill up an issue.

### 13. How vllm-ascend is tested

vllm-ascend is tested by functional test, performance test and accuracy test.

- **Functional test**: we added CI, includes portion of vllm's native unit tests and vllm-ascend's own unit tests，on vllm-ascend's test, we test basic functionality、popular models availability and [supported features](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_features.html) via e2e test

- **Performance test**: we provide [benchmark](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks) tools for end-to-end performance benchmark which can easily to re-route locally, we'll publish a perf website to show the performance test results for each pull request

- **Accuracy test**: we're working on adding accuracy test to CI as well.

Finnall, for each release, we'll publish the performance test and accuracy test report in the future.

### 14. How to fix the error "InvalidVersion" when using vllm-ascend?
It's usually because you have installed an dev/editable version of vLLM package. In this case, we provide the env variable `VLLM_VERSION` to let users specify the version of vLLM package to use. Please set the env variable `VLLM_VERSION` to the version of vLLM package you have installed. The format of `VLLM_VERSION` should be `X.Y.Z`.

### 15. How to handle Out Of Memory?
OOM errors typically occur when the model exceeds the memory capacity of a single NPU. For general guidance, you can refer to [vLLM's OOM troubleshooting documentation](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#out-of-memory).

In scenarios where NPUs have limited HBM (High Bandwidth Memory) capacity, dynamic memory allocation/deallocation during inference can exacerbate memory fragmentation, leading to OOM. To address this:

- **Adjust `--gpu-memory-utilization`**: If unspecified, will use the default value of `0.9`. You can decrease this param to reserve more memory to reduce fragmentation risks. See more note in: [vLLM - Inference and Serving - Engine Arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html#vllm.engine.arg_utils-_engine_args_parser-cacheconfig).

- **Configure `PYTORCH_NPU_ALLOC_CONF`**: Set this environment variable to optimize NPU memory management. For example, you can `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` to enable virtual memory feature to mitigate memory fragmentation caused by frequent dynamic memory size adjustments during runtime, see more note in: [PYTORCH_NPU_ALLOC_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/700/comref/Envvariables/Envir_012.html).

### 16. Failed to enable NPU graph mode when running DeepSeek?
You may encounter the following error if running DeepSeek with NPU graph mode enabled. The allowed number of queries per kv when enabling both MLA and Graph mode only support {32, 64, 128}, **Thus this is not supported for DeepSeek-V2-Lite**, as it only has 16 attention heads. The NPU graph mode support on DeepSeek-V2-Lite will be done in the future.

And if you're using DeepSeek-V3 or DeepSeek-R1, please make sure after the tensor parallel split, num_heads / num_kv_heads in {32, 64, 128}.

```bash
[rank0]: RuntimeError: EZ9999: Inner Error!
[rank0]: EZ9999: [PID: 62938] 2025-05-27-06:52:12.455.807 numHeads / numKvHeads = 8, MLA only support {32, 64, 128}.[FUNC:CheckMlaAttrs][FILE:incre_flash_attention_tiling_check.cc][LINE:1218]
```

### 17. Failed to reinstall vllm-ascend from source after uninstalling vllm-ascend?
You may encounter the problem of C compilation failure when reinstalling vllm-ascend from source using pip. If the installation fails, it is recommended to use `python setup.py install` to install, or use `python setup.py clean` to clear the cache.

### 18. How to generate determinitic results when using vllm-ascend?
There are several factors that affect output certainty:

1. Sampler Method: using **Greedy sample** by setting `temperature=0` in `SamplingParams`, e.g.:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)
# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

2. Set the following enveriments parameters:

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
```

### 19. How to fix the error "ImportError: Please install vllm[audio] for audio support" for Qwen2.5-Omni model？
The `Qwen2.5-Omni` model requires the `librosa` package to be installed, you need to install the `qwen-omni-utils` package to ensure all dependencies are met `pip install qwen-omni-utils`,
this package will install `librosa` and its related dependencies, resolving the `ImportError: No module named 'librosa'` issue and ensuring audio processing functionality works correctly.
