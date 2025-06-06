export HCCL_IF_IP=xx.xx.xx.xx
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
export DISAGGREGATED_RPEFILL_RANK_TABLE_PATH="/your_path/ranktable_118.json"
export ASCEND_RT_VISIBLE_DEVICES=0
#export VLLM_LLMDD_CHANNEL_PORT=6657
#export VLLM_LLMDD_CHANNEL_PORT=14012
export VLLM_LLMDD_CHANNEL_PORT=15272
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCAKE_CONFIG_PATH="/your_path/vllm/mooncake_barebone.json"
export VLLM_USE_V1=1

vllm serve "/your_path/model/Qwen2.5-7B-Instruct"
--host xx.xx.xx.xx
--port 8100
--tensor-parallel-size 1
--seed 1024
--max-model-len 2000
--max-num-batched-tokens 2000
--trust-remote-code
--gpu-memory-utilization 0.9
--kv-transfer-config
'{"kv_connector": "MooncakeConnectorV1_barebone",
"kv_buffer_device": "npu",
"kv_role": "kv_producer",
"kv_parallel_size": 1,
"kv_port": "20001",
"engine_id": 0,
"kv_rank": 0,
"kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector_v1_barebone"
}'
--additional-config
'{"enable_graph_mode": "True"}'\
