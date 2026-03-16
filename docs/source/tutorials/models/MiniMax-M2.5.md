# MiniMax-M2.5

## Introduction

MiniMax‑M2.5 is MiniMax’s flagship large language model, reinforced for high‑value scenarios such as code generation, agentic tool calling/search, and complex office workflows, with an emphasis on reasoning efficiency and end‑to‑end speed on challenging tasks.

This document describes the main verification and deployment steps for `MiniMax-M2.5` on vLLM Ascend, including environment preparation, single-node (multi-NPU) startup, and functional verification.

## Environment Preparation

### Model Weights

- `MiniMax-M2.5` (fp8 checkpoint): recommended to use **1× Atlas 800 A3 (64G × 16)** node. Download the model weights from [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5).

It is recommended to download the model weights to a shared directory, such as `/mnt/sfs_turbo/.cache/`. The current release automatically detects the MiniMax-M2 fp8 checkpoint, disables fp8 quantization kernels on NPU, and loads the weights by dequantizing to bf16. This behavior may be removed once public bf16 weights are available.

### Installation

You can use the official docker image to run `MiniMax-M2.5` directly.

Select an image based on your machine type and start the container on your node. See [using docker](../../installation.md#set-up-using-docker).

## Run with Docker

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /mnt/sfs_turbo/.cache:/home/cache \
-it $IMAGE bash
```

## Online Inference on Multi-NPU

Below is a recommended startup configuration (default performance profile: full context + Tool Calling + Reasoning).

Notes:

- By default, `--max-model-len` is not explicitly set. The server reads the model config (M2.5 uses `196608`) and enables verified performance parameters.
- If you only care about short-context low latency, you can explicitly set `--max-model-len 32768`.

```{code-block} bash
cd /workspace
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /models/MiniMax-M2.5 \
  --served-model-name MiniMax-M2.5 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --port 8000 \
  > /tmp/minimax-m25-serve.log 2>&1 &

tail -f /tmp/minimax-m25-serve.log
```

Remarks:

- `minimax_m2_append_think` keeps `<think>...</think>` inside `content`.
- If you mainly rely on the reasoning semantics of `/v1/responses`, it is recommended to use `--reasoning-parser minimax_m2` instead.

## Verify the Service

Test with an OpenAI-compatible client:

```{code-block} python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="na")

resp = client.chat.completions.create(
    model="MiniMax-M2.5",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己，并展示一次工具调用的参数格式。"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

Or send a request using curl:

```{code-block} bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5",
    "messages": [{"role": "user", "content": "请查询上海的天气。"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get weather by city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 512
  }'
```

## Performance Reference (Ascend A3 single node, tp=16, 4k/1k@bs16)

**Baseline** (`4k/1k@bs=16`):

- Success/Failure: `16/0`
- Mean TTFT: `616.20 ms`
- Mean TPOT: `31.92 ms`
- Mean ITL: `31.92 ms`
- Output tok/s: `492.39`
- Total tok/s: `2461.95`

**Long-context reference** (`190k/1k@bs=4`):

- Output tok/s: `37.12`
- Mean TTFT: `2002.37 ms`
- Mean TPOT: `105.54 ms`
- Mean ITL: `105.54 ms`

## FAQ

- **Q: What should I do if the output is garbled in EP mode?**

  A: It is recommended to keep `--enable-expert-parallel` and `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.

- **Q: Why is the `reasoning` field often empty after using `minimax_m2_append_think`?**

  A: This is expected. The parser keeps `<think>...</think>` inside `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, use `--reasoning-parser minimax_m2` instead.

- **Q: Startup fails with HCCL port conflicts (address already bound). What should I do?**

  A: Clean up old processes and restart: `pkill -f "vllm serve /models/MiniMax-M2.5"`.

- **Q: How to handle OOM or unstable startup?**

  A: Reduce `--max-num-seqs` and `--max-num-batched-tokens` first. If needed, reduce concurrency and load-testing pressure (e.g., `max-concurrency` / `num-prompts`).
