# w8a8 quantization(deepseek-v2-lite)

## Run docker container:
```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/cann:8.0.0-910b-ubuntu22.04-py3.10
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

## Install vllm and vllm-ascend
Install system dependencies
```bash
apt update  -y
apt install -y wget git gcc g++ cmake libnuma-dev
```

Install pta

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
mkdir pta
cd pta
wget https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.5.1/20250320.3/pytorch_v2.5.1_py310.tar.gz
tar -xvf pytorch_v2.5.1_py310.tar.gz
pip install ./torch_npu-2.5.1.dev20250320-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
cd ..
rm -rf pta
```

Install vllm

```bash
git clone --depth 1 --branch v0.8.4 https://ghfast.top/https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install . --extra-index https://download.pytorch.org/whl/cpu/
```

Install vllm-ascend

```bash
git clone  --depth 1 --branch main https://ghfast.top/https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
```

## Install modelslim and convert model
```bash
# You can choose to convert the model yourself or use the quantized model we uploaded, 
# see https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V2-Lite-w8a8
git clone https://gitee.com/ascend/msit
cd msit/msmodelslim
# Install by run this script
bash install.sh
pip install accelerate

cd /msit/msmodelslim/example/DeepSeek
# Original weight path, Replace with your local model path
MODEL_PATH=/home/weight/DeepSeek-V2-Lite
# Path to save converted weight, Replace with your local path
mkdir -p /home/weight/DeepSeek-V2-Lite-w8a8
SAVE_PATH=/home/weight/DeepSeek-V2-Lite-w8a8
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
