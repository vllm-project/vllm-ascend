# Single-NPU (Qwen3 8B W4A8)

## Run docker container
:::{note}
w4a8 quantization feature is supported by v0.9.1rc2 or higher
:::

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
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

## Install modelslim and convert model
:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded,
see https://www.modelscope.cn/models/vllm-ascend/Qwen3-8B-W4A8
:::

```bash
# The branch(br_release_MindStudio_8.1.RC2_TR5_20260624) has been verified
git clone -b br_release_MindStudio_8.1.RC2_TR5_20260624 https://gitee.com/ascend/msit

cd msit/msmodelslim

# Install by run this script
bash install.sh
pip install accelerate

cd example/Qwen
# Original weight path, Replace with your local model path
MODEL_PATH=/home/models/Qwen3-8B
# Path to save converted weight, Replace with your local path
SAVE_PATH=/home/models/Qwen3-8B-w4a8

python quant_qwen.py \
          --model_path $MODEL_PATH \
          --save_directory $SAVE_PATH \
          --device_type npu \
          --model_type qwen3 \
          --calib_file None \
          --anti_method m6 \
          --anti_calib_file ./calib_data/mix_dataset.json \
          --w_bit 4 \
          --a_bit 8 \
          --is_lowbit True \
          --open_outlier False \
          --group_size 256 \
          --is_dynamic True \
          --trust_remote_code True \
          --w_method HQQ
```

## Verify the quantized model
The converted model files looks like:

```bash
.
|-- config.json
|-- configuration.json
|-- generation_config.json
|-- merges.txt
|-- quant_model_description.json
|-- quant_model_weight_w4a8_dynamic-00001-of-00003.safetensors
|-- quant_model_weight_w4a8_dynamic-00002-of-00003.safetensors
|-- quant_model_weight_w4a8_dynamic-00003-of-00003.safetensors
|-- quant_model_weight_w4a8_dynamic.safetensors.index.json
|-- README.md
|-- tokenizer.json
`-- tokenizer_config.json
```

Run the following script to start the vLLM server with quantized model:

```bash
vllm serve /home/models/Qwen3-8B-w4a8 --served-model-name "qwen3-8b-w4a8" --max-model-len 4096 --quantization ascend
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-8b-w4a8",
        "prompt": "what is large language model?",
        "max_tokens": "128",
        "top_p": "0.95",
        "top_k": "40",
        "temperature": "0.0"
    }'
```

Run the following script to execute offline inference on Single-NPU with quantized model:

:::{note}
To enable quantization for ascend, quantization method must be "ascend"
:::

```python

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/home/models/Qwen3-8B-w4a8",
          max_model_len=4096,
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
