# QwQ-32B

## 1 Introduction

QwQ-32B is a 32B-parameter reasoning language model developed by Qwen Team, based on the Qwen2 architecture. It is designed for complex reasoning, dialogue, and instruction-following tasks, supporting long context up to 32,768 tokens.

This document will demonstrate the main validation steps for QwQ-32B in the vLLM-Ascend environment, including supported features, environment preparation, single-node deployment, as well as accuracy and performance evaluation.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

| Model | Hardware Requirement | Download |
|-------|--------------------|----------|
| QwQ-32B (BF16) | Atlas 800I A2 (64G, 4 cards) | [Download](https://huggingface.co/Qwen/QwQ-32B) |

It is recommended to download the model weight to a shared directory accessible to all nodes, such as `/data/models/`.

## 4 Installation

### 4.1 Docker Image Installation

::::{tab-set}
:sync-group: hardware

:::{tab-item} Atlas 800I A2
:sync: a2

**Docker Pull:**

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run \
    --name vllm-ascend-env \
    --shm-size=128g \
    --net=host \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /data:/data \
    -v /tmp:/tmp \
    -v /mnt:/mnt \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /root:/host_root \
    -it -d $IMAGE bash
```

:::

::::

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

**Installation Verification:**

```bash
docker ps | grep vllm-ascend-env
```

### 4.2 Source Code Installation

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install vLLM-Ascend:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600

vllm serve /path/to/your/model \
    --served-model-name qwq-32b \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --async-scheduling \
    --enforce-eager \
    --port 8000
```

:::{note}

- `ASCEND_RT_VISIBLE_DEVICES`: must be set to the NPU chip IDs allocated to your environment (e.g., `0,1,2,3` for 4 chips).
- `--port`: adjust to avoid conflicts with other services running on the same machine.

:::

**Service Verification:**

After the service is started, verify it is running by sending a prompt. Refer to [Section 6](#6-functional-verification) for a usage example.

## 6 Functional Verification

**Chat Completions API:**

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwq-32b",
        "messages": [
            {"role": "user", "content": "Explain what is quantum computing"}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024
    }'
```

:::{note}
Adjust the following fields based on your deployment:

- **URL** (`http://localhost:8000`): Replace `localhost` and `8000` with your server IP and the `--port` value from the `vllm serve` command.
- **`model`**: Must match the `--served-model-name` value from the `vllm serve` command (e.g., `qwq-32b`).
:::

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

**Python OpenAI Client:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwq-32b",
    messages=[
        {"role": "user", "content": "Write a Python function for quick sort"}
    ],
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## 7 Accuracy Evaluation

### Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

The following is an example configuration for the accuracy evaluation config file, demonstrated using the GSM8K dataset:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="your_model_path",
        model="qwq-32b",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8000,
        max_out_len=32768,
        batch_size=32,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.7,
        ),
    )
]
```

```shell
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_str --mode all --dump-eval-details --debug
```

| Dataset       | `--datasets` Parameter               |
| ------------- | ------------------------------------ |
| GSM8K         | `gsm8k_gen_4_shot_cot_str`           |
| GPQA-Diamond  | `gpqa_gen_0_shot_cot_chat_prompt`    |
| AIME 2024     | `aime2024_gen_0_shot_str`            |
| LiveCodeBench | `livecodebench_0_shot_chat_v4_v5_v6` |

**Accuracy Results (Atlas 800I A2, vLLM-Ascend v0.21.0rc1, BF16):**

| Dataset | Metric | Score |
| ------- | ------ | ----- |
| GSM8K   | accuracy (5-shot) | 88.40% |

:::{note}
vLLM-Ascend also supports the following evaluation tools:

- [lm_eval](../../developer_guide/evaluation/using_lm_eval.md)
- [OpenCompass](../../developer_guide/evaluation/using_opencompass.md)
- [EvalScope](../../developer_guide/evaluation/using_evalscope.md)
:::

## 8 Performance Evaluation

### Using AISBench

For setup details, please refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

Configure the model for streaming performance testing:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-stream-chat',
        path="your_model_path",
        model="qwq-32b",
        stream=True,
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8000,
        max_out_len=1500,
        batch_size=32,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=True,
        ),
    )
]
```

### Using vLLM Benchmark

```shell
vllm bench serve \
    --model /path/to/your/model \
    --served-model-name qwq-32b \
    --port 8000 \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 256 \
    --num-prompts 50 \
    --request-rate inf \
    --endpoint /v1/completions \
    --save-result \
    --result-dir ./
```

**Benchmark Results (Atlas 800I A2, 4×910B3, TP=4, vLLM-Ascend v0.21.0rc1):**

| Metric | Value |
|--------|-------|
| Request throughput | 0.44 req/s |
| Output token throughput | 113.03 tok/s |
| Total token throughput | 1017.31 tok/s |
| Mean TTFT | 2961.50 ms |
| Mean TPOT | 160.14 ms |
| Mean ITL | 160.14 ms |

## 9 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
