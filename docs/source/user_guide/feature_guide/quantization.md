# Quantization Guide

Model quantization is a technique that reduces the size and computational requirements of a model by lowering the data precision of the weights and activation values in the model, thereby saving the memory and improving the inference speed.

Since 0.9.0rc2 version, quantization feature is experimentally supported in vLLM Ascend. Users can enable quantization feature by specifying `--quantization ascend`. Currently, only Qwen, DeepSeek series models are well tested. We’ll support more quantization algorithm and models in the future.

## Install modelslim

To quantize a model, users should install [ModelSlim](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md) which is the Ascend compression and acceleration tool. It is an affinity-based compression tool designed for acceleration, using compression as its core technology and built upon the Ascend platform.

Currently, only the specific tag [modelslim-VLLM-8.1.RC1.b020_001](https://gitee.com/ascend/msit/blob/modelslim-VLLM-8.1.RC1.b020_001/msmodelslim/README.md) of modelslim works with vLLM Ascend. Please do not install other version until modelslim master version is available for vLLM Ascend in the future.

Install modelslim:

```bash
git clone https://gitee.com/ascend/msit -b modelslim-VLLM-8.1.RC1.b020_001
cd msit/msmodelslim
bash install.sh
pip install accelerate
```

## Quantize model

Take [DeepSeek-V2-Lite](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite) as an example, you just need to download the model, and then execute the convert command. The command is shown below. More info can be found in modelslim doc [deepseek w8a8 dynamic quantization docs](https://gitee.com/ascend/msit/blob/modelslim-VLLM-8.1.RC1.b020_001/msmodelslim/example/DeepSeek/README.md#deepseek-v2-w8a8-dynamic%E9%87%8F%E5%8C%96).

```bash
cd example/DeepSeek
python3 quant_deepseek.py --model_path {original_model_path} --save_directory {quantized_model_save_path} --device_type cpu --act_method 2 --w_bit 8 --a_bit 8  --is_dynamic True
```

:::{note}
You can also download the quantized model that we uploaded. Please note that these weights should be used for test only. For example, https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V2-Lite-W8A8
:::

Once convert action is done, there are two important files generated.

1. [config.json](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V2-Lite-W8A8/file/view/master/config.json?status=1). Please make sure that there is no `quantization_config` field in it.

2. [quant_model_description.json](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V2-Lite-W8A8/file/view/master/quant_model_description.json?status=1). All the converted weights info are recorded in this file.

Here is the full converted model files:

```bash
.
├── config.json
├── configuration_deepseek.py
├── configuration.json
├── generation_config.json
├── quant_model_description.json
├── quant_model_weight_w8a8_dynamic-00001-of-00004.safetensors
├── quant_model_weight_w8a8_dynamic-00002-of-00004.safetensors
├── quant_model_weight_w8a8_dynamic-00003-of-00004.safetensors
├── quant_model_weight_w8a8_dynamic-00004-of-00004.safetensors
├── quant_model_weight_w8a8_dynamic.safetensors.index.json
├── README.md
├── tokenization_deepseek_fast.py
├── tokenizer_config.json
└── tokenizer.json
```

## Run the model

Now, you can run the quantized models with vLLM Ascend. Here is the example for online and offline inference.

### Offline inference

```python
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="{quantized_model_save_path}",
          max_model_len=2048,
          trust_remote_code=True,
          # Enable quantization by specifying `quantization="ascend"`
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Online inference

```bash
# Enable quantization by specifying `--quantization ascend`
vllm serve {quantized_model_save_path} --served-model-name "deepseek-v2-lite-w8a8" --max-model-len 2048 --quantization ascend --trust-remote-code
```

## FAQs

### 1. How to solve the KeyError: 'xxx.layers.0.self_attn.q_proj.weight' problem?

First, make sure you specify `ascend` quantization method. Second, check if your model is converted by this `modelslim-VLLM-8.1.RC1.b020_001` modelslim version. Finally, if it still doesn't work, please
submit a issue, maybe some new models need to be adapted.

### 2. How to solve the error "Could not locate the configuration_deepseek.py"?

Please convert DeepSeek series models using `modelslim-VLLM-8.1.RC1.b020_001` modelslim, this version has fixed the missing configuration_deepseek.py error.
