# GLM-4.1V-9B-Thinking

## Introduction

GLM-4.1V is an advanced multimodal model based on the GLM architecture specifically designed for agent applications, with strong reasoning and vision-language capabilities.

This document will show the main verification steps of the model on Ascend NPUs, including environment preparation, single-node deployment, offline inference, online serving, and end-to-end testing.

## Environment Preparation

### Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

### Model Weight

- `GLM-4.1V-9B-Thinking`: [Download model weight](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking).

It is recommended to use `hf-transfer` to speed up the download process and download the model weight to a shared directory like `/root/.cache/`.

### Installation

You can use the official docker image to run `GLM-4.1V-9B-Thinking` directly.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on your node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
:::::

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Offline Inference

You can run a minimal offline inference script to verify the basic model loading and multimodal capabilities without starting a full server.

Create a Python script (e.g., `test_glm4v_minimal.py`):

```python
from vllm import LLM, SamplingParams
from PIL import Image
import numpy as np

def test_minimal():
    # Replace with your downloaded model path
    model_path = "zai-org/GLM-4.1V-9B-Thinking"
    
    # Initialize the LLM engine with eager mode enabled to avoid compilation timeouts
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.7,
        enforce_eager=True 
    )

    # Create a dummy pure black image for testing without network dependency
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    # Construct the prompt strictly following the GLM-4V chat template structure
    prompt = "<|user|>\n<|begin_of_image|><|image|><|end_of_image|>What is in this image?<|assistant|>\n"
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=64)

    print("Generating...")
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params
    )
    
    for output in outputs:
        print(f"Output: {output.outputs[0].text}")

if __name__ == "__main__":
    test_minimal()
```

## Online Serving

### Single-node Deployment

GLM-4.1V-9B-Thinking can be deployed on a single Atlas A2 (e.g., Ascend-910b) NPU.

Run the following script to start the vLLM API server:

```{test} bash
:sync-yaml: tests/e2e/models/configs/GLM-4.1V-9B-Thinking.yaml
:sync-target: test_cases[0].model test_cases[0].server_cmd
:sync-class: cmd

vllm serve "zai-org/GLM-4.1V-9B-Thinking" \
  --served-model-name glm4v \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager
```

**Notice on Parameters:**

- `--max-model-len`: Set to `8192` to accommodate long multimodal prompts and images.
- `--gpu-memory-utilization`: Due to the memory footprint of vision tensors and caching, `0.7` is a stable baseline for Ascend-910b (64GB HBM). If you encounter `ValueError: Free memory on device...`, ensure no zombie `VLLMEngineCor` processes exist via `npu-smi info`.
- `--enforce-eager`: It is **highly recommended** to enforce eager execution for this model to bypass potential NPU graph compilation timeouts and stabilize the 2D RoPE fallback computations.

## Functional Verification

Once your server is started successfully, you will see `Application startup complete.`

You can query the multimodal endpoint using `curl`:

```shell
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm4v", 
        "messages": [{ 
            "role": "user", 
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
            ]
        }], 
        "stream": false, 
        "ignore_eos": false, 
        "temperature": 0.1, 
        "max_tokens": 200 
    }' http://localhost:8000/v1/chat/completions
```

## Accuracy Evaluation

You can run automated end-to-end multimodal testing via the built-in `pytest` evaluation suite to verify accuracy against standard datasets like MMMU.

```shell
# Export HF_ENDPOINT to prevent dataset download hangups
export HF_ENDPOINT=https://hf-mirror.com

# Run the correctness test with the provided configuration
pytest -s tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/GLM-4.1V-9B-Thinking.yaml
```

After execution, you can get the result. Here is the evaluated result of `GLM-4.1V-9B-Thinking` on `vllm-ascend` for reference only. Note that because of the intelligent eager-mode fallback for 2D RoPE, there might be slight numerical deviations.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| --- | --- | --- | --- | --- | --- |
| mmmu_val | - | acc,none | gen | 40.33 | 1 Atlas 800 A2 (64G) with eager mode |

## Performance

For multimodal models like GLM-4.1V, the standard text-only throughput performance benchmarks (like `vllm bench` or `AISBench`) may not accurately reflect the real-world processing speed of vision tensors. The Vision-Language Model benchmarking is still being optimized on Ascend. We will update the detailed multimodal TTFT and TPOT performance metrics in future releases.

## Special Notes for GLM-4.1V on Ascend

### Vision Tensor Memory Continuity

To resolve the crash of underlying NPU operators caused by `pixel_values` (which typically become non-contiguous in memory due to multidimensional Slice and Concat operations during image preprocessing), an implicit `.contiguous()` patch (`patch_glm4v.py`) is applied to ensure the memory continuity of vision tensors before they are passed into the model's forward pass.
