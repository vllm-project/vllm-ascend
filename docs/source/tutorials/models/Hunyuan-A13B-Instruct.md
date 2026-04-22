# Hunyuan-A13B-Instruct

## Introduction

Hunyuan-A13B-Instruct is a fine-grained hybrid expert model (MoE) developed by Tencent. This model has a total of 80 billion parameters, 13 billion activation parameters, supports 256K ultra-long contexts, and possesses native thought chain (CoT) reasoning capabilities.

## Environment Preparation

### Model Weight

- `Hunyuan-A13B-Instruct`(BF16 version): [Download model weight](https://www.modelscope.cn/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
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

Build from source:

```{code-block} bash
   :substitutions:
# Install vLLM.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch |vllm_ascend_version| https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e .
cd ..
```

### Software Stack Version Verification

The environment is based on CANN built into the GiteeAI platform, and successfully runs vLLM |vllm_ascend_version|, and vLLM-Ascend:|vllm_ascend_version| through the Python 3.11.6 Conda environment.

## Deployment

### Single-node Deployment (4-NPU)

```bash
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data
export MODEL_PATH="Hunyuan-A13B-Instruct"

vllm serve ${MODEL_PATH} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Hunyuan \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```

### Key Performance Indicators

Based on verified CANN 8.5.1 test logs:

- Memory usage for weights: each NPU has a static memory usage of approximately 37.46 GB.
- Graph compilation (ACL Graph): with PIECEWISE mode enabled, the system automatically captures the graph in approximately 18 seconds, which can significantly accelerate subsequent inference.
- KV cache capacity: the remaining NPU memory can provide concurrent cache space for approximately 529,152 tokens.

## Functional Verification

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hunyuan",
        "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

Expected output:

```json
{"id":"chatcmpl-9a60df2b23bb539f","object":"chat.completion","created":1774751760,"model":"Hunyuan","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, I need to write a short introduction to large language models. Let me start by recalling what I know. First, what are LLMs? They're machine learning models trained on vast amounts of text data. The key here is \"large\"—so they have a huge number of parameters. Maybe mention the scale, like billions or trillions of parameters.\n\nThen, how are they trained? They're trained on diverse text sources—books, websites, articles, etc. The","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":12,"total_tokens":112,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

## Accuracy Evaluation

On the GiteeAI platform, the model was tested and verified using the AISBench tool on the GSM8K benchmark set: Under the 7cd45e version configuration, the model achieved an accuracy of 94.77% in the accuracy generation mode.

```bash
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --summarizer example --debug
```

output:

```bash
03/29 03:20:03 - AISBench - INFO - Running 1-th replica of evaluation
03/29 03:20:03 - AISBench - INFO - Task [vllm-api-general-chat/gsm8k]: {'accuracy': 94.76876421531463}
03/29 03:20:03 - AISBench - INFO - time elapsed: 2.15s
03/29 03:20:04 - AISBench - INFO - Evaluation tasks completed.
03/29 03:20:04 - AISBench - INFO - Summarizing evaluation results...
dataset    version    metric    mode      vllm-api-general-chat
---------  ---------  --------  ------  -----------------------
gsm8k      7cd45e     accuracy  gen                       94.77
03/29 03:20:04 - AISBench - INFO - write summary to /data/outputs/default/20260329_025345/summary/summary_20260329_025345.txt
03/29 03:20:04 - AISBench - INFO - write csv to /data/outputs/default/20260329_025345/summary/summary_20260329_025345.csv
```

The markdown formatted result is as follows:

| dataset | version | metric | mode | vllm-api-general-chat |
| --- | --- | --- | --- | --- |
| gsm8k | 7cd45e | accuracy | gen | 94.77 |

## Performance

### Using AISBench

```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer default_perf --mode perf
```

output:

```bash
[2026-04-08 05:27:40,180] [ais_bench] [INFO] Performance Results of task [vllm-api-stream-chat/demo_gsm8k]: 
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75             │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 29982.6 ms      │ 16472.9 ms      │ 41147.2 ms      │ 30919.1 ms      │ 33514.9 ms      │ 39413.8 ms      │ 40973.9 ms      │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 238.6 ms        │ 107.9 ms        │ 276.7 ms        │ 254.0 ms        │ 265.6 ms        │ 272.4 ms        │ 276.3 ms        │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TPOT                     │ total   │ 60.1 ms         │ 57.7 ms         │ 61.3 ms         │ 60.4 ms         │ 60.8 ms         │ 61.2 ms         │ 61.3 ms         │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ ITL                      │ total   │ 59.7 ms         │ 0.0 ms          │ 219.7 ms        │ 51.7 ms         │ 64.1 ms         │ 81.9 ms         │ 146.2 ms        │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ InputTokens              │ total   │ 1457.5          │ 1426.0          │ 1511.0          │ 1456.5          │ 1465.25         │ 1481.6          │ 1508.06         │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ OutputTokens             │ total   │ 497.5           │ 268.0           │ 710.0           │ 508.5           │ 555.75          │ 666.6           │ 705.66          │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ OutputTokenThroughput    │ total   │ 16.5261 token/s │ 16.2402 token/s │ 17.2551 token/s │ 16.4461 token/s │ 16.5728 token/s │ 16.9063 token/s │ 17.2202 token/s │  8  │
╘══════════════════════════╧═════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════╛
╒══════════════════════════╤═════════╤═══════════════════╕
│ Common Metric            │ Stage   │ Value             │
╞══════════════════════════╪═════════╪═══════════════════╡
│ Benchmark Duration       │ total   │ 41161.2934 ms     │
├──────────────────────────┼─────────┼───────────────────┤
│ Total Requests           │ total   │ 8                 │
├──────────────────────────┼─────────┼───────────────────┤
│ Failed Requests          │ total   │ 0                 │
├──────────────────────────┼─────────┼───────────────────┤
│ Success Requests         │ total   │ 8                 │
├──────────────────────────┼─────────┼───────────────────┤
│ Concurrency              │ total   │ 5.8273            │
├──────────────────────────┼─────────┼───────────────────┤
│ Max Concurrency          │ total   │ 16                │
├──────────────────────────┼─────────┼───────────────────┤
│ Request Throughput       │ total   │ 0.1944 req/s      │
├──────────────────────────┼─────────┼───────────────────┤
│ Total Input Tokens       │ total   │ 11660             │
├──────────────────────────┼─────────┼───────────────────┤
│ Prefill Token Throughput │ total   │ 6108.0184 token/s │
├──────────────────────────┼─────────┼───────────────────┤
│ Total Generated Tokens   │ total   │ 3980              │
├──────────────────────────┼─────────┼───────────────────┤
│ Input Token Throughput   │ total   │ 283.2758 token/s  │
├──────────────────────────┼─────────┼───────────────────┤
│ Output Token Throughput  │ total   │ 96.6928 token/s   │
├──────────────────────────┼─────────┼───────────────────┤
│ Total Token Throughput   │ total   │ 379.9686 token/s  │
╘══════════════════════════╧═════════╧═══════════════════╛
```

### Using vLLM Benchmark

Run performance evaluation of `Hunyuan-A13B-Instruct` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve \
    --model ./Hunyuan-A13B-Instruct/ \
    --port 8000 \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./perf_results/ \
    --trust-remote-code
```

After about several minutes, you can get the performance evaluation result.
