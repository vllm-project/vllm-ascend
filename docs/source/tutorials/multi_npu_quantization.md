# Multi-NPU (QwQ 32B W8A8)

## Run docker container
:::{note}
w8a8 quantization feature is supported by v0.8.4rc2 or higher
:::

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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

## Install modelslim

```bash
git clone https://gitee.com/ascend/msit

cd msit/msmodelslim
# Install by run this script
bash install.sh
pip install accelerate
```

## Convert models

### QwQ-32B
:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded,
see https://www.modelscope.cn/models/vllm-ascend/QwQ-32B-W8A8
:::

```bash
cd example/Qwen
# Original weight path, Replace with your local model path
MODEL_PATH=/home/models/QwQ-32B
# Path to save converted weight, Replace with your local path
SAVE_PATH=/home/models/QwQ-32B-w8a8

# In this conversion process, the npu device is not must, you can also set --device_type cpu to have a conversion
python3 quant_qwen.py --model_path $MODEL_PATH --save_directory $SAVE_PATH --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m1 --trust_remote_code True
```

### Kimi-K2-Instruct
:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded,
see https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8
This conversion process will require a larger CPU memory, please ensure that the RAM size is greater than 2TB
:::

#### Adapts and change
1. Ascend does not support the `flash_attn` library. To run the model, you need to follow the [guide](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/DeepSeek/README.md#deepseek-v3r1) and comment out certain parts of the code in `modeling_deepseek.py` located in the weights folder.
2. The current version of transformers does not support loading weights in FP8 quantization format. you need to follow the [guide](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/DeepSeek/README.md#deepseek-v3r1) and delete the quantization related fields from `config.json` in the weights folder

#### Generate the w8a8 weights

```bash
cd example/DeepSeek

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export MODEL_PATH="/root/.cache/Kimi-K2-Instruct"
export SAVE_PATH="/root/.cache/Kimi-K2-Instruct-W8A8"

python3 quant_deepseek_w8a8.py --model_path $MODEL_PATH --save_path $SAVE_PATH --batch_size 4
```

## Verify the quantized model
The converted model files looks like:

```bash
.
|-- config.json
|-- configuration.json
|-- generation_config.json
|-- quant_model_description.json
|-- quant_model_weight_w8a8.safetensors
|-- README.md
|-- tokenizer.json
`-- tokenizer_config.json
```

Run the following script to start the vLLM server with quantized model:

:::{note}
The value "ascend" for "--quantization" argument will be supported after [a specific PR](https://github.com/vllm-project/vllm-ascend/pull/877) is merged and released, you can cherry-pick this commit for now.
:::

```bash
vllm serve /home/models/QwQ-32B-w8a8  --tensor-parallel-size 4 --served-model-name "qwq-32b-w8a8" --max-model-len 4096 --quantization ascend
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwq-32b-w8a8",
        "prompt": "what is large language model?",
        "max_tokens": "128",
        "top_p": "0.95",
        "top_k": "40",
        "temperature": "0.0"
    }'
```

Run the following script to execute offline inference on multi-NPU with quantized model:

:::{note}
To enable quantization for ascend, quantization method must be "ascend"
:::

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

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/home/models/QwQ-32B-w8a8",
          tensor_parallel_size=4,
          distributed_executor_backend="mp",
          max_model_len=4096,
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```
