# Single-NPU (Qwen3 32B W4A4)

## Introduction

W4A4 Flat Quantization is for better model compression and inference efficiency on Ascend devices.
And W4A4 is supported since `v0.11.0rc1`. For modelslim, W4A4 is supported since `tag_MindStudio_8.2.RC1.B120_002`.

The following steps will show how to quantize Qwen3 32B to W4A4.

## Environment Preparation

### Run Docker Container

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
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

### Install modelslim and Convert Model

:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded,
see https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4
:::

```bash
git clone -b tag_MindStudio_8.2.RC1.B120_002 https://gitcode.com/Ascend/msit
cd msit/msmodelslim

# Install by run this script
bash install.sh
pip install accelerate
# transformers 4.51.0 is required for Qwen3 series model
# see https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/Qwen/README.md#%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE
pip install transformers==4.51.0

cd example/Qwen
# Original weight path, Replace with your local model path
MODEL_PATH=/home/models/Qwen3-32B
# Path to save converted weight, Replace with your local path
SAVE_PATH=/home/models/Qwen3-32B-w4a4

python3 w4a4.py --model_path $MODEL_PATH \
                --save_directory $SAVE_PATH \
                --calib_file ../common/qwen_qwen3_cot_w4a4.json \
                --trust_remote_code True \
                --batch_size 1
```

### Verify the Quantized Model

The converted model files look like:

```bash
.
|-- config.json
|-- configuration.json
|-- generation_config.json
|-- quant_model_description.json
|-- quant_model_weight_w4a4_flatquant_dynamic-00001-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00002-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00003-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00004-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00005-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00006-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00007-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00008-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00009-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00010-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00011-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic.safetensors.index.json
|-- tokenizer.json
|-- tokenizer_config.json
`-- vocab.json
```

## Deployment

### Online Serving on Single NPU

```bash
vllm serve /home/models/Qwen3-32B-w4a4 --served-model-name "qwen3-32b-w4a4" --max-model-len 4096 --quantization ascend
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-32b-w4a4",
        "prompt": "what is large language model?",
        "max_tokens": "128",
        "top_p": "0.95",
        "top_k": "40",
        "temperature": "0.0"
    }'
```

### Offline Inference on Single NPU

:::{note}
To enable quantization for ascend, quantization method must be "ascend".
:::

```python

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/home/models/Qwen3-32B-w4a4",
          max_model_len=4096,
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
