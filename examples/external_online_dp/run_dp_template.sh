export HCCL_IF_IP=your_ip_here
export GLOO_SOCKET_IFNAME="enp48s3u1u1"
export TP_SOCKET_IFNAME="enp48s3u1u1"
export HCCL_SOCKET_IFNAME="enp48s3u1u1"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=your_rank_table_path_here
export VLLM_LOGGING_LEVEL="info"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7
export HCCL_DETERMINISTIC=True
export HCCL_BUFFER_SIZE=1024
export TASK_QUEUE_ENABLE=1
# Spawn the process inside the vllm maybe cause the circular import issue, using fork here is necessary
export VLLM_WORKER_MULTIPROC_METHOD="fork"


export VLLM_USE_V1=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve model_path \
    --host 0.0.0.0 \
    --port $6 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv3 \
    --max-model-len 5200 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 28 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --kv-transfer-config \
    '{"kv_connector": "LLMDataDistCMgrConnector",
      "kv_buffer_device": "npu",
      "kv_role": "kv_consumer",
      "kv_parallel_size": "1",
      "kv_port": "20001",
      "engine_id": "0",
      "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
    }' \
    --additional-config \
    '{"ascend_scheduler_config": {"enabled": true}, "torchair_graph_config":{"enabled":true,"enable_kv_nz":false, "enable_multistream_moe":false, "graph_batch_size":[28]}, "enable_weight_nz_layout":true}`
