export HCCL_IF_IP=0.0.0.0
export GLOO_SOCKET_IFNAME="enp48s3u1u1"
export TP_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_SOCKET_IFNAME="enp48s3u1u1"
export DISAGGREGATED_RPEFILL_RANK_TABLE_PATH="rank_table_path"
export ASCEND_RT_VISIBLE_DEVICES=3
export VLLM_LLMDD_CHANNEL_PORT=6657
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCKAE_CONFIG_PATH="mooncake.json"
export VLLM_USE_V1=1

vllm serve "/model/path" \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 1\
  --seed 1024 \
  --max-model-len 2000  \
  ---max-num-batched-tokens 2000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1_barebone",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": 0,
  "kv_rank": 0,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector_v1_barebone"
  }'  \
  --additional-config \
  '{"enable_graph_mode": "True"}'\