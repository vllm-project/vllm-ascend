# Disaggregated Prefill-Decode Deployment Guide

## Overview
This demo document provides instructions for running a disaggregated vLLM-ascend service with separate prefill and decode stages across 4 nodes, uses 16 Ascend NPUs for two prefill nodes (P1/P2) and 16 Ascend NPUS for two decode nodes (D1/D2).

## Prerequisites
- Ascend NPU environment with vLLM 0.9.1 installed
- Network interfaces configured for distributed communication (eg: eth0)
- Model weights located at `/data01/deepseek_r1_w8a8_zhw`

## Rank table generation
The rank table is a JSON file that specifies the mapping of Ascend NPU ranks to nodes. The following command generates a rank table for all nodes with 16 cards prefill and 16 cards decode:

Run the following command on every node to generate the rank table:
```bash
cd vllm-ascend/examples/disaggregate_prefill_v1/
bash generate_ranktable.sh 16 16
```
Rank table will generated at `/vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json`

## Start disaggregated vLLM-ascend service 
Execution Sequence
- 4 configured node ip are: 172.19.32.175 172.19.241.49 172.19.123.51 172.19.190.36
- Start Prefill on Node 1 (P1)
- Start Prefill on Node 2 (P2)
- Start Decode on Node 1 (D1)
- Start Decode on Node 2 (D2)
- Start proxy server on Node1

* Run prefill server P1 on first node
```bash
export HCCL_IF_IP=`hostname -I|awk -F " " '{print$1}'`
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1
vllm serve /data01/deepseek_r1_w8a8_zhw \
  --host 0.0.0.0 \
  --port 20002 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --api-server-count 2 \
  --data-parallel-address 172.19.32.175 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistConnectorA3",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_connector_v1_a3"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enable": false, "enable_multistream_shared_expert": false}, "expert_tensor_parallel_size": 1}'
```

* Run prefill server P2 on second node
```bash
export HCCL_IF_IP=`hostname -I|awk -F " " '{print$1}'`
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1
vllm serve /data01/deepseek_r1_w8a8_zhw \
  --host 0.0.0.0 \
  --port 20002 \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-start-rank 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address 172.19.32.175 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistConnectorA3",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_connector_v1_a3"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enable": false, "enable_multistream_shared_expert": false}, "expert_tensor_parallel_size": 1}' 
```

* Run decode server d1 on third node
```bash
export HCCL_IF_IP=`hostname -I|awk -F " " '{print$1}'`
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1
vllm serve /data01/deepseek_r1_w8a8_zhw \
  --host 0.0.0.0 \
  --port 20002 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --api-server-count 2 \
  --data-parallel-address 172.19.123.51 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistConnectorA3",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_connector_v1_a3"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enable": false, "enable_multistream_shared_expert": false}, "expert_tensor_parallel_size": 1}'
```

* Run decode server d2 on last node
```bash
export HCCL_IF_IP=`hostname -I|awk -F " " '{print$1}'`
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1
vllm serve /data01/deepseek_r1_w8a8_zhw \
  --host 0.0.0.0 \
  --port 20002 \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-start-rank 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address 172.19.123.51 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 6144  \
  --max-num-batched-tokens 6144  \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistConnectorA3",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_connector_v1_a3"
  }'  \
  --additional-config \
  '{"torchair_graph_config": {"enable": false, "enable_multistream_shared_expert": false}, "expert_tensor_parallel_size": 1}' 
```

* Run proxy server on the first node
```bash
cd /vllm-workspace/vllm-ascend/examples/disaggregate_prefill_v1
python toy_proxy_server.py --host 172.19.32.175 --port 1025 --prefiller-hosts 172.19.241.49 --prefiller-port 20002 --decoder-hosts 172.19.123.51 --decoder-ports 20002
```

* Verification
Check service health using the proxy server endpoint:
```bash
curl http://localhost:1025/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek",
        "prompt": "你是谁？",
        "max_tokens": 100,
        "temperature": 0
    }'
```
