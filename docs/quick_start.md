# Quickstart

## 1. Prerequisites

### Supported Devices
- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)

### Prepare Environment

You can use the container image directly with one line command:

```bash
DEVICE=/dev/davinci7
IMAGE=quay.io/ascend/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10
docker run \
    --name $NAME --device vllm-ascend-env \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it --rm $IMAGE bash
```

You can verify by running below commands in above container shell:

```bash
npu-smi info
```

You will see following message:

```
+-------------------------------------------------------------------------------------------+
| npu-smi 23.0.2              Version: 23.0.2                                               |
+----------------------+---------------+----------------------------------------------------+
| NPU   Name           | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                 | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+======================+===============+====================================================+
| 0     xxx            | OK            | 0.0         40                0    / 0             |
| 0                    | 0000:C1:00.0  | 0           882  / 15169      0    / 32768         |
+======================+===============+====================================================+
```


## 2. Installation

Prepare:

```bash
apt update
apt install git curl vim -y
# Config pypi mirror to speedup
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Create your venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You can install vLLM and vllm-ascend plugin by using:

```bash
# Install vLLM main branch (About 5 mins)
git clone --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install .
cd ..

# Install vLLM Ascend Plugin:
git clone --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
cd ..
```


## 3. Usage

After vLLM and vLLM Ascend plugin installation, you can start to
try [vLLM QuickStart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html).

You have two ways to start vLLM on Ascend NPU:

### Offline Batched Inference with vLLM

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing).

```bash
# Use Modelscope mirror to speed up download
pip install modelscope
export VLLM_USE_MODELSCOPE=true
$ python3
>>>
```

Try to execute below script to generate texts:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# The first run will take about 1.5 mins (10 MB/s) to download models
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### OpenAI Completions API with vLLM

vLLM can also be deployed as a server that implements the OpenAI API protocol. Run
the following command to start the vLLM server with the
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model:

```bash
# Use Modelscope mirror to speed up download
pip install modelscope
export VLLM_USE_MODELSCOPE=true
# Deploy vLLM server (The first run will take about 1.5 mins (10 MB/s) to download models)
vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

Once your server is started, you can query in a new terminal:
```bash
docker exec -it vllm-ascend-env bash
```

You can query the list the models:
```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

You can also query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Beijing is a",
        "max_tokens": 5,
        "temperature": 0
    }' | python3 -m json.tool
```
