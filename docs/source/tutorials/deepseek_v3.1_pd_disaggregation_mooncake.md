# DeepSeek-V3.1

## Introduction

This document outlines the main verification steps of the deployment of disaggregated prefill with `DeepSeek-V3.1` on Atlas A3 nodes.

## Environment Preparation

### Model Weight

- `DeepSeek-V3.1-BF16`(BF16 version): require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/unsloth/DeepSeek-V3.1-BF16)
- `DeepSeek-V3.1-W8A8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.1-W8A8)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

- Using docker
  Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

- Build from source code
  Referring to [installation](../installation.md) section to build from source code

## Deployment

take `vllm-ascend/DeepSeek-V3.1-W8A8`, assume you have 2 Atlas A3 node, the following section shows how to run the model with disaggregated prefill.

:::::{tab-set}

::::{tab-item} Prefiller node
:sync: prefill node

```shell
export VLLM_USE_MODELSCOPE=true
# The local ip
export LOCAL_IP=10.0.0.221
# nic name corresponding to the local_ip
export NIC_NAME="eth0"
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve vllm-ascend/DeepSeek-V3.1-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address $LOCAL_IP \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --enforce-eager \
  --quantization ascend \
  --distributed-executor-backend mp \
  --served-model-name deepseek_v3.1 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

::::

::::{tab-item} Decoder node
:sync: decode node

```shell
export VLLM_USE_MODELSCOPE=true
# The local ip
export LOCAL_IP=10.0.0.200
# nic name corresponding to the local_ip
export NIC_NAME="eth0"
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

vllm serve vllm-ascend/DeepSeek-V3.1-W8A8 \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address $LOCAL_IP \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name deepseek_v3.1 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --additional-config '{"torchair_graph_config":{"enabled":true}}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
            },
            "decode": {
                    "dp_size": 2,
                    "tp_size": 8
            }
      }
  }'
```

::::

:::::

## Example Proxy for Deployment

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_layerwise\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)

```shell
python load_balance_proxy_layerwise_server_example.py \
    --host 10.0.0.221 \
    --port 8080 \
    --prefiller-hosts 10.0.0.221 \
    --prefiller-port 8004 \
    --decoder-hosts 10.0.0.200 \
    --decoder-ports 8004
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://10.0.0.221:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3.1",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-V3.1-W8A8` for reference only.

dataset    version    metric    mode      vllm-api-general-chat
---------  ---------  --------  ------  -----------------------
gsm8k      7cd45e     accuracy  gen                       96.88

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-V3.1-W8A8` in online mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.1,base_url=http://127.0.0.1:8080/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.1-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model vllm-ascend/DeepSeek-V3.1-W8A8 --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
