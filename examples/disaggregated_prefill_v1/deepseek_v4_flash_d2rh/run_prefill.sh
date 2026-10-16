#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

set -euo pipefail

# Run inside a prepared 950PR/950DT container; see the model tutorial.
: "${LOCAL_IP:?Set LOCAL_IP to the reachable service IP of this node}"
: "${NIC_NAME:?Set NIC_NAME to the corresponding network interface}"
: "${MODEL_PATH:?Set MODEL_PATH to the local DSpark W4A4C8 checkpoint}"

unset ftp_proxy https_proxy http_proxy FTP_PROXY HTTPS_PROXY HTTP_PROXY
unset HCCL_INTRA_ROCE_ENABLE

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_ALGO=level0:fullmesh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=512
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

vllm serve "$MODEL_PATH" \
    --host "$LOCAL_IP" \
    --port 8100 \
    --served-model-name dsv4 \
    --data-parallel-size 4 \
    --data-parallel-address "$LOCAL_IP" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --block-size 32 \
    --enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --api-server-count 1 \
    --async-scheduling \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code \
    --quantization ascend \
    --attention_config.indexer_kv_dtype fp8 \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "MooncakeD2RHConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "36010",
        "engine_id": "d2rh-prefill",
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 4, "tp_size": 1},
            "decode": {"dp_size": 4, "tp_size": 1}
        }
    }' \
    --additional-config '{
        "enable_cpu_binding": true,
        "enable_shared_expert_dp": true,
        "multistream_overlap_shared_expert": true,
        "enable_dsa_cp": false,
        "enable_flashcomm1": false
    }' \
    2>&1 | tee "${LOG_FILE:-d2rh_prefill.log}"
