export HCCL_IF_IP=141.61.39.117
export GLOO_SOCKET_IFNAME="enp48s3u1u1"
export TP_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_SOCKET_IFNAME="enp48s3u1u1"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=path-to-rank-table

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=1

vllm serve model_path \
  --host 0.0.0.0 \
  --port 20002 \
  --tensor-parallel-size 1\
  --seed 1024 \
  --served-model-name dsv3 \
  --max-model-len 2000  \
  ---max-num-batched-tokens 2000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistConnectorA3",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": 0,
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_connector_v1_a3"
  }'  \
  --additional-config \
  '{"enable_graph_mode": "True"}'\
