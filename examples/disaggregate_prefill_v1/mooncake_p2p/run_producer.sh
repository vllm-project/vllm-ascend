set +x
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=xx.xx.xx.xx
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCAKE_CONFIG_PATH="/home/xxx/json/mooncake_barebone.json"
export VLLM_USE_V1=1
export VLLM_BASE_PORT=9700

vllm serve "model_path" \
  --host xx.xx.xx.xx \
  --port 8100 \
  --tensor-parallel-size 2\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000  \
  --trust-remote-code \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address xx.xx.xx.xx \
  --data-parallel-rpc-port 9100 \
  --data-parallel-start-rank 0 \
  --gpu-memory-utilization 0.8  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1_barebone",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_rank": 0,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector_v1_barebone",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 2
             }
      }
  }'  \
  --additional-config \
  '{}'\
