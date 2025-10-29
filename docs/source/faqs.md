# FAQs

## Version Specific FAQs

- [[v0.9.1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/2643)
- [[v0.11.0rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/3222)

## General FAQs

### 1. What devices are currently supported?

Currently, **ONLY** Atlas A2 series (Ascend-cann-kernels-910b)，Atlas A3 series (Atlas-A3-cann-kernels) and Atlas 300I (Ascend-cann-kernels-310p) series are supported:

- Atlas A2 training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 inference series (Atlas 800I A2)
- Atlas A3 training series (Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD)
- Atlas 800I A3 inference series (Atlas 800I A3)
- [Experimental] Atlas 300I inference series (Atlas 300I Duo). Currently for 310I Duo, the stable version is vllm-ascend v0.10.0rc1.

Below series are NOT supported yet:
- Atlas 200I A2 (Ascend-cann-kernels-310b) unplanned yet
- Ascend 910, Ascend 910 Pro B (Ascend-cann-kernels-910) unplanned yet

From a technical view, vllm-ascend support would be possible if the torch-npu is supported. Otherwise, we have to implement it by using custom operators. You are also welcome to join us to improve together.

### 2. How to get our Docker containers?

You can get our containers at `Quay.io`, e.g., [<u>vllm-ascend</u>](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and [<u>cann</u>](https://quay.io/repository/ascend/cann?tab=tags).

If you are in China, you can use `daocloud` to accelerate your downloading:

```bash
# Replace with tag you want to pull
TAG=v0.7.3rc2
docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:$TAG
```

#### Load Docker images for the offline environment
If you want to use container images for offline environments (without Internet connection), you need to download the container image in an environment with Internet access:

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

### 3. What models does vllm-ascend support?

Find more details [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_models.html).

### 4. How to get in touch with our community?

There are many channels that you can communicate with our community developers and users:

- Submit a GitHub [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues?page=1).
- Join our [<u>weekly meeting</u>](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z) and share your ideas.
- Join our [<u>WeChat</u>](https://github.com/vllm-project/vllm-ascend/issues/227) group and ask your questions.
- Join the Ascend channel in [<u>vLLM forums</u>](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support/6) and publish your topics.

### 5. What features does vllm-ascend V1 support?

Find more details [<u>here</u>](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_features.html).

### 6. How to solve the problem of "Failed to infer device type" or "libatb.so: cannot open shared object file"?

Basically, the reason is that the NPU environment is not configured correctly. You can:
1. try `source /usr/local/Ascend/nnal/atb/set_env.sh` to enable NNAL package.
2. try `source /usr/local/Ascend/ascend-toolkit/set_env.sh` to enable CANN package.
3. try `npu-smi info` to check whether the NPU is working.

If all above steps are not working, you can try the following code with Python to check whether there is any error:

```
import torch
import torch_npu
import vllm
```

If the problem still persists, feel free to submit a GitHub issue.

### 7. How does vllm-ascend perform?

Currently, the performance is improved on some models, such as `Qwen2.5 VL`, `Qwen3`, and `Deepseek  V3`. From 0.9.0rc2, Qwen and DeepSeek work with graph mode to deliver good performance. What's more, you can install `mindie-turbo` with `vllm-ascend v0.7.3` to speed up the inference as well.

### 8. How does vllm-ascend work with vllm?
vllm-ascend is a plugin for vllm. Basically, the version of vllm-ascend is the same as the version of vllm. For example, if you use vllm 0.7.3, you should use vllm-ascend 0.7.3 as well. For the main branch, we will make sure `vllm-ascend` and `vllm` are compatible by each commit.

### 9. Does vllm-ascend support the prefill-decode disaggregation feature?

Currently, only 1P1D is supported on V0 Engine. For V1 Engine or NPND support, we will make it stable and supported by vllm-ascend in the future.

### 10. Does vllm-ascend support quantization methods?

Currently, W8A8 quantization is already supported by vllm-ascend originally on v0.8.4rc2 or higher. If you're using vllm 0.7.3, W8A8 quantization is supported with the integration of vllm-ascend and mindie-turbo, please use `pip install vllm-ascend[mindie-turbo]`.

### 11. How to run a W8A8 DeepSeek model?

Follow the [inference tutorial](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node.html) and replace the model with DeepSeek.

### 12. How to solve the problem that there is no output in the log when loading models using vllm-ascend?

If you're using vllm 0.7.3, this is a known progress bar display issue in vLLM, which has been resolved in [this PR](https://github.com/vllm-project/vllm/pull/12428), please cherry-pick it locally by yourself. Otherwise, please fill up an issue.

### 13. How is vllm-ascend tested?

vllm-ascend is tested in three aspects, functions, performance, and accuracy.

- **Functional test**: We added CI, including part of vllm's native unit tests and vllm-ascend's own unit tests. On vllm-ascend's test, we test basic functionalities, popular model availability, and [supported features](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_features.html) through E2E test.

- **Performance test**: We provide [benchmark](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks) tools for E2E performance benchmark, which can be easily re-routed locally. We will publish a perf website to show the performance test results for each pull request.

- **Accuracy test**: We are working on adding accuracy test to the CI as well.

Finally, for each release, we will publish the performance test and accuracy test report in the future.

### 14. How to fix the error "InvalidVersion" when using vllm-ascend?
The problem is usually caused by the installation of a dev or editable version of the vLLM package. In this case, we provide the environment variable `VLLM_VERSION` to let users specify the version of vLLM package to use. Please set the environment variable `VLLM_VERSION` to the version of the vLLM package you have installed. The format of `VLLM_VERSION` should be `X.Y.Z`.

### 15. How to handle the out-of-memory issue?
OOM errors typically occur when the model exceeds the memory capacity of a single NPU. For general guidance, you can refer to [vLLM OOM troubleshooting documentation](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#out-of-memory).

In scenarios where NPUs have limited high bandwidth memory (HBM) capacity, dynamic memory allocation/deallocation during inference can exacerbate memory fragmentation, leading to OOM. To address this:

- **Adjust `--gpu-memory-utilization`**: If unspecified, the default value is `0.9`. You can decrease this value to reserve more memory to reduce fragmentation risks. See details in: [vLLM - Inference and Serving - Engine Arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html#vllm.engine.arg_utils-_engine_args_parser-cacheconfig).

- **Configure `PYTORCH_NPU_ALLOC_CONF`**: Set this environment variable to optimize NPU memory management. For example, you can use `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` to enable virtual memory feature to mitigate memory fragmentation caused by frequent dynamic memory size adjustments during runtime. See details in: [PYTORCH_NPU_ALLOC_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/700/comref/Envvariables/Envir_012.html).

### 16. Failed to enable NPU graph mode when running DeepSeek.
You may encounter the following error if running DeepSeek with NPU graph mode is enabled. The allowed number of queries per KV when enabling both MLA and Graph mode is {32, 64, 128}. **Thus this is not supported for DeepSeek-V2-Lite**, as it only has 16 attention heads. The NPU graph mode support on DeepSeek-V2-Lite will be implemented in the future.

And if you're using DeepSeek-V3 or DeepSeek-R1, please make sure after the tensor parallel split, num_heads/num_kv_heads is {32, 64, 128}.

```bash
[rank0]: RuntimeError: EZ9999: Inner Error!
[rank0]: EZ9999: [PID: 62938] 2025-05-27-06:52:12.455.807 numHeads / numKvHeads = 8, MLA only support {32, 64, 128}.[FUNC:CheckMlaAttrs][FILE:incre_flash_attention_tiling_check.cc][LINE:1218]
```

### 17. Failed to reinstall vllm-ascend from source after uninstalling vllm-ascend.
You may encounter the problem of C compilation failure when reinstalling vllm-ascend from source using pip. If the installation fails, use `python setup.py install` (recommended) to install, or use `python setup.py clean` to clear the cache.

### 18. How to generate deterministic results when using vllm-ascend?
There are several factors that affect output certainty:

1. Sampler method: using **Greedy sample** by setting `temperature=0` in `SamplingParams`, e.g.:

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

2. Set the following environment parameters:

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
```

### 19. How to fix the error "ImportError: Please install vllm[audio] for audio support" for the Qwen2.5-Omni model？
The `Qwen2.5-Omni` model requires the `librosa` package to be installed, you need to install the `qwen-omni-utils` package to ensure all dependencies are met `pip install qwen-omni-utils`.
This package will install `librosa` and its related dependencies, resolving the `ImportError: No module named 'librosa'` issue and ensure that the audio processing functionality works correctly.

### 20. How to troubleshoot and resolve size capture failures resulting from stream resource exhaustion, and what are the underlying causes?

```
error example in detail: 
ERROR 09-26 10:48:07 [model_runner_v1.py:3029] ACLgraph sizes capture fail: RuntimeError:
ERROR 09-26 10:48:07 [model_runner_v1.py:3029] ACLgraph has insufficient available streams to capture the configured number of sizes.Please verify both the availability of adequate streams and the appropriateness of the configured size count.
```

Recommended mitigation strategies:
1. Manually configure the compilation_config parameter with a reduced size set: '{"cudagraph_capture_sizes":[size1, size2, size3, ...]}'.
2. Employ ACLGraph's full graph mode as an alternative to the piece-wise approach.

Root cause analysis:
The current stream requirement calculation for size captures only accounts for measurable factors including: data parallel size, tensor parallel size, expert parallel configuration, piece graph count, multistream overlap shared expert settings, and HCCL communication mode (AIV/AICPU). However, numerous unquantifiable elements, such as operator characteristics and specific hardware features, consume additional streams outside of this calculation framework, resulting in stream resource exhaustion during size capture operations.

### 21. Installing vllm-ascend will overwrite the existing torch-npu package.
Installing vllm-ascend will overwrite the existing torch-npu package. If you need to install a specific version of torch-npu, you can manually install the specified version of torch-npu after installing vllm-ascend.
