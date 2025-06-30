# Multi-NPU (Pangu Pro MoE 72B)

## Run vllm-ascend on Multi-NPU

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Setup environment variables:

```bash
# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

```bash
vllm serve /path/to/pangu-pro-moe-model \
--tensor-parallel-size 4 \
--trust-remote-code \
--enforce-eager
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/pangu-pro-moe-model",
        "prompt": "The future of AI is",
        "max_tokens": 128,
        "temperature": 0
    }'
```

If you run this successfully, you can see the info shown below:

```json
{"id":"cmpl-013558085d774d66bf30c704decb762a","object":"text_completion","created":1750472788,"model":"/path/to/pangu-pro-moe-model","choices":[{"index":0,"text":" not just about creating smarter machines but about fostering collaboration between humans and AI systems. This partnership can lead to more efficient problem-solving, innovative solutions, and a better quality of life for people around the globe.\n\nHowever, achieving this future requires addressing several challenges. Ethical considerations, such as bias in AI algorithms and privacy concerns, must be prioritized. Additionally, ensuring that AI technologies are accessible to all and do not exacerbate existing inequalities is crucial.\n\nIn conclusion, AI stands at the forefront of technological advancement, with vast potential to transform industries and everyday life. By embracing its opportunities while responsibly managing its risks, we can harn","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":134,"completion_tokens":128,"prompt_tokens_details":null},"kv_transfer_params":null}
```

### Offline Inference on Multi-NPU

Run the following script to execute offline inference on multi-NPU:

```python
import gc

import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":

    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

    llm = LLM(model="/path/to/pangu-pro-moe-model",
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            max_model_len=1024,
            trust_remote_code=True,
            enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: ' Daniel and I am an 8th grade student at York Middle School. I'
Prompt: 'The future of AI is', Generated text: ' following you. As the technology advances, a new report from the Institute for the'
```
