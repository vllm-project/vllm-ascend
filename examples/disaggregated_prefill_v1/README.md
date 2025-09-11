# Disaggregated Prefill-Decode Deployment Guide

## Overview
This demo document provides instructions for running a disaggregated vLLM-ascend service with separate prefill and decode stages across 4 nodes, uses 16 Ascend NPUs for two prefill nodes (P1/P2) and 16 Ascend NPUS for two decode nodes (D1/D2).

## Prerequisites
- Ascend NPU environment with vLLM 0.9.1 installed
- Network interfaces configured for distributed communication (eg: eth0)
- Model weights located at `/models/deepseek_r1_w8a8`

## Rank table generation
The rank table is a JSON file that specifies the mapping of Ascend NPU ranks to nodes. The following command generates a rank table for all nodes with 16 cards prefill and 16 cards decode:

Run the following command on every node to generate the rank table:
```shell
cd /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/
bash gen_ranktable.sh --ips 172.19.32.175 172.19.241.49 172.19.123.51 172.19.190.36 \
  --npus-per-node 8 --network-card-name eth0 --prefill-device-cnt 16 --decode-device-cnt 16
```
Rank table will generated at `/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json`

## Start disaggregated vLLM-ascend service
For demonstration purposes, we will utilize the quantized version of Deepseek-R1. Recommended Parallelization Strategies:
- P-node: DP2-TP8-EP16 (Data Parallelism 2, Tensor Parallelism 8, Expert Parallelism 16)
- D-node: DP4-TP4-EP16 (Data Parallelism 4, Tensor Parallelism 4, Expert Parallelism 16)

Execution Sequence
- 4 configured node ip are: 172.19.32.175 172.19.241.49 172.19.123.51 172.19.190.36
- Start Prefill on Node 1 (P1)
- Start Prefill on Node 2 (P2)
- Start Decode on Node 1 (D1)
- Start Decode on Node 2 (D2)
- Start proxy server on Node1

Run prefill server P1 on first node:
```shell
export HCCL_IF_IP=172.19.32.175  # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_ASCEND_LLMDD_RPC_PORT=5559

vllm serve /models/deepseek_r1_w8a8 \
  --host 0.0.0.0 \
  --port 20002 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --api-server-count 2 \
  --data-parallel-address 172.19.32.175 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 32768  \
  --max-num-batched-tokens 32768  \
  --max-num-seqs 256 \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"chunked_prefill_for_mla":true}' 
```

Run prefill server P2 on second node:
```shell
export HCCL_IF_IP=172.19.241.49
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_ASCEND_LLMDD_RPC_PORT=5659

vllm serve /models/deepseek_r1_w8a8 \
  --host 0.0.0.0 \
  --port 20002 \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-start-rank 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address 172.19.32.175 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 32768  \
  --max-num-batched-tokens 32768  \
  --max-num-seqs 256 \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"chunked_prefill_for_mla":true}'
```

Run decode server d1 on third node:

* In the D node, the `max-num-batched-tokens` parameter can be set to a smaller value since the D node processes at most `max-num-seqs` batches concurrently. As the `profile_run` only needs to handle `max-num-seqs` sequences at a time, we can safely set `max-num-batched-tokens` equal to `max-num-seqs`. This optimization will help reduce activation memory consumption.
```shell
export HCCL_IF_IP=172.19.123.51
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_ASCEND_LLMDD_RPC_PORT=5759

vllm serve /models/deepseek_r1_w8a8 \
  --host 0.0.0.0 \
  --port 20002 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --api-server-count 2 \
  --data-parallel-address 172.19.123.51 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 32768  \
  --max-num-batched-tokens 256  \
  --max-num-seqs 256 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enabled":true}}' 
```

Run decode server d2 on last node:
```shell
export HCCL_IF_IP=172.19.190.36
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_ASCEND_LLMDD_RPC_PORT=5859

vllm serve /models/deepseek_r1_w8a8 \
  --host 0.0.0.0 \
  --port 20002 \
  --headless \
  --data-parallel-size 4 \
  --data-parallel-start-rank 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address 172.19.123.51 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 32768  \
  --max-num-batched-tokens 256  \
  --max-num-seqs 256 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enabled":true}}' 
```

Run proxy server on the first node:
```shell
cd /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1
python toy_proxy_server.py --host 172.19.32.175 --port 1025 --prefiller-hosts 172.19.241.49 --prefiller-port 20002 --decoder-hosts 172.19.123.51 --decoder-ports 20002
```

Verification
Check service health using the proxy server endpoint:
```shell
curl http://localhost:1025/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek",
        "prompt": "Who are you?",
        "max_tokens": 100,
        "temperature": 0
    }'
```

Performance
Test performance with vllm benchmark:
```shell
cd /vllm-workspace/vllm/benchmarks
python3 benchmark_serving.py \
    --backend vllm \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 1536 \
    --num-prompts 256 \
    --ignore-eos \
    --model deepseek \
    --tokenizer /models/deepseek_r1_w8a8 \
    --host localhost \
    --port 1025 \
    --endpoint /v1/completions \
    --max-concurrency 4 \
    --request-rate 4
```
