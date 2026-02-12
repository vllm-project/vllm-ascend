# FAQs

## Version Specific FAQs

- [[v0.11.0] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/4808)
- [[v0.13.0rc1] FAQ & Feedback](https://github.com/vllm-project/vllm-ascend/issues/5333)

## General FAQs

### 1. What devices are currently supported?

Currently, **ONLY** Atlas A2 series(Ascend-cann-kernels-910b)，Atlas A3 series(Atlas-A3-cann-kernels) and Atlas 300I(Ascend-cann-kernels-310p) series are supported:

- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)
- Atlas A3 Training series (Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD)
- Atlas 800I A3 Inference series (Atlas 800I A3)
- [Experimental] Atlas 300I Inference series (Atlas 300I Duo).
- [Experimental] Currently for 310I Duo the stable version is vllm-ascend v0.10.0rc1.

Below series are NOT supported yet:
- Atlas 200I A2 (Ascend-cann-kernels-310b) unplanned yet
- Ascend 910, Ascend 910 Pro B (Ascend-cann-kernels-910) unplanned yet

From a technical view, vllm-ascend support would be possible if the torch-npu is supported. Otherwise, we have to implement it by using custom ops. We also welcome you to join us to improve together.

### 2. How to get our docker containers?

You can get our containers at `Quay.io`, e.g., [<u>vllm-ascend</u>](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and [<u>cann</u>](https://quay.io/repository/ascend/cann?tab=tags).

If you are in China, you can use `daocloud` or some other mirror sites to accelerate your downloading:

```bash
# Replace with tag you want to pull
TAG=v0.9.1
docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:$TAG
# or
docker pull quay.nju.edu.cn/ascend/vllm-ascend:$TAG
```

#### Load Docker Images for offline environment
If you want to use container image for offline environments (no internet connection), you need to download container image in an environment with internet access:

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

Find more details [<u>here</u>](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html).

### 4. How to get in touch with our community?

There are many channels that you can communicate with our community developers / users:

- Submit a GitHub [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues?page=1).
- Join our [<u>weekly meeting</u>](https://docs.google.com/document/d/1hCSzRTMZhIB8vRq1_qOOjx4c9uYUxvdQvDsMV2JcSrw/edit?tab=t.0#heading=h.911qu8j8h35z) and share your ideas.
- Join our [<u>WeChat</u>](https://github.com/vllm-project/vllm-ascend/issues/227) group and ask your questions.
- Join our ascend channel in [<u>vLLM forums</u>](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support/6) and publish your topics.

### 5. What features does vllm-ascend V1 supports?

Find more details [<u>here</u>](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html).

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

### 7. How vllm-ascend work with vLLM?
vllm-ascend is a hardware plugin for vLLM. Basically, the version of vllm-ascend is the same as the version of vllm. For example, if you use vllm 0.9.1, you should use vllm-ascend 0.9.1 as well. For main branch, we will make sure `vllm-ascend` and `vllm` are compatible by each commit.

### 8. Does vllm-ascend support Prefill Disaggregation feature?

Yes, vllm-ascend supports Prefill Disaggregation feature with Mooncake backend. Take [official tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/pd_disaggregation_mooncake_multi_node.html) for example.

### 9. Does vllm-ascend support quantization method?

Currently, w8a8, w4a8 and w4a4 quantization methods are already supported by vllm-ascend.

### 10. How to run a W8A8 DeepSeek model?

Follow the [inference tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/multi_node.html) and replace the model with DeepSeek.

### 11. How is vllm-ascend tested?

vllm-ascend is tested in three aspects, functions, performance, and accuracy.

- **Functional test**: We added CI, including part of vllm's native unit tests and vllm-ascend's own unit tests. On vllm-ascend's test, we test basic functionalities, popular model availability, and [supported features](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html) through E2E test.

- **Performance test**: We provide [benchmark](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks) tools for E2E performance benchmark, which can be easily re-routed locally. We will publish a perf website to show the performance test results for each pull request.

- **Accuracy test**: We are working on adding accuracy test to the CI as well.

- **Nightly test**: we'll run full test every night to make sure the code is working.

Finnall, for each release, we'll publish the performance test and accuracy test report in the future.

### 12. How to fix the error "InvalidVersion" when using vllm-ascend?
The problem is usually caused by the installation of a dev or editable version of the vLLM package. In this case, we provide the environment variable `VLLM_VERSION` to let users specify the version of vLLM package to use. Please set the environment variable `VLLM_VERSION` to the version of the vLLM package you have installed. The format of `VLLM_VERSION` should be `X.Y.Z`.

### 13. How to handle the out-of-memory issue?
OOM errors typically occur when the model exceeds the memory capacity of a single NPU. For general guidance, you can refer to [vLLM OOM troubleshooting documentation](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#out-of-memory).

In scenarios where NPUs have limited high bandwidth memory (HBM) capacity, dynamic memory allocation/deallocation during inference can exacerbate memory fragmentation, leading to OOM. To address this:

- **Limit `--max-model-len`**:  It can save the HBM usage for kv cache initialization step.

- **Adjust `--gpu-memory-utilization`**: If unspecified, the default value is `0.9`. You can decrease this value to reserve more memory to reduce fragmentation risks. See details in: [vLLM - Inference and Serving - Engine Arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html#vllm.engine.arg_utils-_engine_args_parser-cacheconfig).

- **Configure `PYTORCH_NPU_ALLOC_CONF`**: Set this environment variable to optimize NPU memory management. For example, you can use `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` to enable virtual memory feature to mitigate memory fragmentation caused by frequent dynamic memory size adjustments during runtime. See details in: [PYTORCH_NPU_ALLOC_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/700/comref/Envvariables/Envir_012.html).

### 14. Failed to enable NPU graph mode when running DeepSeek.
Enabling NPU graph mode for DeepSeek may trigger an error. This is because when both MLA and NPU graph mode are active, the number of queries per KV head must be 32, 64, or 128. However, DeepSeek-V2-Lite has only 16 attention heads, which results in 16 queries per KV—a value outside the supported range. Support for NPU graph mode on DeepSeek-V2-Lite will be added in a future update.

And if you're using DeepSeek-V3 or DeepSeek-R1, please make sure after the tensor parallel split, num_heads/num_kv_heads is {32, 64, 128}.

```bash
[rank0]: RuntimeError: EZ9999: Inner Error!
[rank0]: EZ9999: [PID: 62938] 2025-05-27-06:52:12.455.807 numHeads / numKvHeads = 8, MLA only support {32, 64, 128}.[FUNC:CheckMlaAttrs][FILE:incre_flash_attention_tiling_check.cc][LINE:1218]
```

### 15. Failed to reinstall vllm-ascend from source after uninstalling vllm-ascend.
You may encounter the problem of C compilation failure when reinstalling vllm-ascend from source using pip. If the installation fails, use `python setup.py install` (recommended) to install, or use `python setup.py clean` to clear the cache.

### 16. How to generate deterministic results when using vllm-ascend?
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

### 17. How to fix the error "ImportError: Please install vllm[audio] for audio support" for the Qwen2.5-Omni model？
The `Qwen2.5-Omni` model requires the `librosa` package to be installed, you need to install the `qwen-omni-utils` package to ensure all dependencies are met `pip install qwen-omni-utils`.
This package will install `librosa` and its related dependencies, resolving the `ImportError: No module named 'librosa'` issue and ensure that the audio processing functionality works correctly.

### 18. How to troubleshoot and resolve size capture failures resulting from stream resource exhaustion, and what are the underlying causes?

```
error example in detail: 
ERROR 09-26 10:48:07 [model_runner_v1.py:3029] ACLgraph sizes capture fail: RuntimeError:
ERROR 09-26 10:48:07 [model_runner_v1.py:3029] ACLgraph has insufficient available streams to capture the configured number of sizes.Please verify both the availability of adequate streams and the appropriateness of the configured size count.
```

Recommended mitigation strategies:
1. Manually configure the compilation_config parameter with a reduced size set: '{"cudagraph_capture_sizes":[size1, size2, size3, ...]}'.
2. Employ ACLgraph's full graph mode as an alternative to the piece-wise approach.

Root cause analysis:
The current stream requirement calculation for size captures only accounts for measurable factors including: data parallel size, tensor parallel size, expert parallel configuration, piece graph count, multistream overlap shared expert settings, and HCCL communication mode (AIV/AICPU). However, numerous unquantifiable elements, such as operator characteristics and specific hardware features, consume additional streams outside of this calculation framework, resulting in stream resource exhaustion during size capture operations.

### 19. How to install custom version of torch_npu?
torch-npu will be overridden  when installing vllm-ascend. If you need to install a specific version of torch-npu, you can manually install the specified version of torch-npu after vllm-ascend is installed.

### 20. On certain systems (e.g., Kylin OS), `docker pull` may fail with an `invalid tar header` error

On certain operating systems, such as Kylin OS , you may encounter an `invalid tar header` error during the `docker pull` process:

```text
failed to register layer: ApplyLayer exit status 1 stdout: stderr: archive/tar: invalid tar header
```

This is often due to system compatibility issues. You can resolve this by using an offline loading method with a second machine.

1. On a separate host machine (e.g., a standard Ubuntu server), pull the image for the target ARM64 architecture and package it into a `.tar` file.

   ```bash
   export IMAGE_TAG=v0.10.0rc1-310p
   export IMAGE_NAME="quay.io/ascend/vllm-ascend:${IMAGE_TAG}"
   # If in China region, uncomment to use a mirror:
   # export IMAGE_NAME="m.daocloud.io/quay.io/ascend/vllm-ascend:${IMAGE_TAG}"
   
   # Pull the image for the ARM64 platform and save it
   docker pull --platform linux/arm64 "${IMAGE_NAME}"
   docker save -o "vllm_ascend_${IMAGE_TAG}.tar" "${IMAGE_NAME}"
   ```

2. Transfer the image archive

Copy the `vllm_ascend_<tag>.tar` file (where `<tag>` is the image tag you used) to your target machine

### 21. Why am I getting an error when executing the script to start a Docker container? The error message is: "operation not permitted".
When using `--shm-size`, you may need to add the `--privileged=true` flag to your `docker run` command to grant the container necessary permissions. Please be aware that using `--privileged=true` grants the container extensive privileges on the host system, which can be a security risk. Only use this option if you understand the implications and trust the container's source.

### 22. How to achieve low latency in a small batch scenario?
The performance of `torch_npu.npu_fused_infer_attention_score` in small batch scenario is not satisfactory, mainly due to the lack of flash decoding function. We offer an alternative operator in `tools/install_flash_infer_attention_score_ops_a2.sh` and `tools/install_flash_infer_attention_score_ops_a3.sh`, you can install it by the following instruction:

```bash
bash tools/install_flash_infer_attention_score_ops_a2.sh
## change to run the following instruction if you're using A3 machine
# bash tools/install_flash_infer_attention_score_ops_a3.sh
```

**NOTE**: Don't set `additional_config.pa_shape_list` when using this method, otherwise it will lead to another attention operator.
**Important**: Please make sure you're using the **official image** of vllm-ascend, otherwise you **must change** the directory `/vllm-workspace` in `tools/install_flash_infer_attention_score_ops_a2.sh` or `tools/install_flash_infer_attention_score_ops_a3.sh` to your own or create one. If you're not in root user, you need `sudo` permission to run this script.
