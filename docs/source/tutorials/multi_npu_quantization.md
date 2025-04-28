# Multi-NPU (deepseek-v2-lite-w8a8)

## Run docker container:
:::{note}
w8a8 quantization feature is supported by v0.8.4rc2 or highter
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

## Install modelslim and convert model
:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded, 
see https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V2-Lite-w8a8
:::

```bash
git clone https://gitee.com/ascend/msit

# (Optional)This commit has been verified
git checkout a396750f930e3bd2b8aa13730401dcbb4bc684ca
cd msit/msmodelslim
# Install by run this script
bash install.sh
pip install accelerate

cd /msit/msmodelslim/example/DeepSeek
# Original weight path, Replace with your local model path
MODEL_PATH=/home/weight/DeepSeek-V2-Lite
# Path to save converted weight, Replace with your local path
SAVE_PATH=/home/weight/DeepSeek-V2-Lite-w8a8
mkdir -p $SAVE_PATH
# In this conversion process, the npu device is not must, you can also set --device_type cpu to have a conversion
python3 quant_deepseek.py --model_path $MODEL_PATH --save_directory $SAVE_PATH --device_type npu --act_method 2 --w_bit 8 --a_bit 8  --is_dynamic True
```

## Verify the quantized model
The converted model files looks like:
```bash
.
|-- config.json
|-- configuration_deepseek.py
|-- fusion_result.json
|-- generation_config.json
|-- quant_model_description_w8a8_dynamic.json
|-- quant_model_weight_w8a8_dynamic-00001-of-00004.safetensors
|-- quant_model_weight_w8a8_dynamic-00002-of-00004.safetensors
|-- quant_model_weight_w8a8_dynamic-00003-of-00004.safetensors
|-- quant_model_weight_w8a8_dynamic-00004-of-00004.safetensors
|-- quant_model_weight_w8a8_dynamic.safetensors.index.json
|-- tokenization_deepseek_fast.py
|-- tokenizer.json
`-- tokenizer_config.json
```

Run the following script to start the vLLM server with quantize model:
```bash
vllm serve /home/weight/DeepSeek-V2-Lite-w8a8  --tensor-parallel-size 4 --trust-remote-code --served-model-name "dpsk-w8a8" --max-model-len 4096
```

Once your server is started, you can query the model with input prompts
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dpsk-w8a8",
        "prompt": "what is deepseek？",
        "max_tokens": "128",
        "top_p": "0.95",
        "top_k": "40",
        "temperature": "0.0"
    }'
```
