#!/bin/bash
# =============================================================================
# SFA PD-disaggregated CPU-offload — Prefill (P) node launcher.
# Connector: SFAPDCpuOffloadConnector, kv_role = kv_producer.
#
# P computes prefill KV layer-wise and pushes it to D via Mooncake RDMA:
#   - indexer KV  -> D HBM
#   - main MLA KV -> D CPU pool
# (the split destination is metadata-driven on P; no sender-side branching.)
#
# Model: use an MLA + sparse model such as DeepSeek-V3.2 — SFA backend is
# selected automatically when (use_mla, use_sparse) == (True, True)
# (see platform.py:get_attn_backend_cls). No extra enable flag needed.
#
# Bring-up notes for this connector (still pending hardware verification):
#   - 2 MiB alignment of the D-side CPU pool (worker.py NOTE)
#   - per-layer 5-tuple -> LayerMetadata packing correctness (watch PDDBG log)
#   - P-side buffer-reuse gating (wait_for_layer_send) wiring
# =============================================================================
set -euo pipefail

# ---------------------------- CONFIG (edit me) -------------------------------
MODEL_PATH="/mnt/share/GLM-5.2-W4A8-0628"     # MLA + sparse (SFA) model
#MODEL_PATH="/home/l00948936/weights/GLM-5.2-W4A8-0628"     # MUST match theP node
#MODEL_PATH="/mnt/weight/DeepSeek-V2-Lite-W8A8"
SERVE_HOST="141.61.81.142"                    # external HTTP listen addr (proxy connects here)
SERVE_PORT=7120                         # external HTTP port
TP_SIZE=16                               # tensor parallel size
VISIBLE_DEVICES=12,13,14,15                       # NPU cards for the P node (e.g. "0" or "0,1")
NET_IFACE="enp48s3u1u2"                          # NIC for gloo/tp/hccl; multi-host -> real iface

KV_PORT=20080                           # Mooncake side-channel base port
KV_RANK=0                               # P node kv_rank (P=0, D=1)
export VLLM_VERSION=0.25.1
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
export LOCAL_WORLD_SIZE=16
# export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
# P MUST run with use_offload=false: the producer worker inherits mooncake's
# register_kv_caches, which expects standard paged KV tensors (not the 5-tuple
# that only exists when use_offload=true). Default is false; set explicitly as a
# guard against misconfiguration.
ADDITIONAL_CONFIG='{"use_offload": false, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_sparse_li_c8": true}'
export VLLM_ASCEND_KV_TRANSFER_BACKEND="memfabric"
export VLLM_ASCEND_MF_VERIFY="0"
export VLLM_ASCEND_SFA_DEBUG="0"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
# ----------------------------------------------------------------------------

export HCCL_IF_IP="141.61.81.142"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
#export VLLM_ASCEND_PD_REUSE_DEBUG="1"
unset VLLM_ASCEND_PD_REUSE_DEBUG
#export VLLM_ASCEND_PD_REUSE_SYNCFIX="sendfull"
unset VLLM_ASCEND_PD_REUSE_SY
export VLLM_ASCEND_LAYER_REUSE_DEBUG="1"
export  MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.1.2/aarch64-linux/lib64
export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf

#  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "enforce_eager": true}' \

#NCFIX
# export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
# export PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-$VISIBLE_DEVICES}"

exec vllm serve "$MODEL_PATH" \
  --host "$SERVE_HOST" \
  --port "$SERVE_PORT" \
  --served-model-name glm \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --max-model-len 1048576 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 65536 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "enforce_eager": true}' \
  --safetensors-load-strategy 'prefetch' \
  --hf-overrides '{"use_index_cache": true}' \
  --additional-config "$ADDITIONAL_CONFIG" \
  --profiler_config '{"profiler":"torch", "torch_profiler_dir":"/home/l00948936/profiling", "torch_profiler_with_stack":false, "torch_profiler_with_memory":false, "torch_profiler_record_shapes":true}' \
  --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "layerwise_num_shared_buffers":"3",
        "layerwise_prefetch_layers":"3",
        "layerwise_independent_layers":0,
        "connectors": [
            {
                "kv_connector": "SFAPDCpuOffloadConnector",
                "kv_buffer_device": "npu",
                "kv_role": "kv_producer",
                "kv_parallel_size": "1",
                "kv_port": "20020",
                "kv_rank": "0",
                "kv_connector_extra_config": {"use_layerwise": "true"}
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0", "layerwise_num_shared_buffers":"3", "layerwise_prefetch_layers":"3", "layerwise_independent_layers":0}
            }
        ]
    }
    }'









#  --kv-transfer-config '{"kv_connector": "AscendStoreConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0", "layerwise_num_shared_buffers":"2", "layerwise_prefetch_layers":"2"}}'

  # --kv-transfer-config "{
  #   \"kv_connector\": \"SFAPDCpuOffloadConnector\",
  #   \"kv_buffer_device\": \"npu\",
  #   \"kv_role\": \"kv_producer\",
  #   \"kv_parallel_size\": 1,
  #   \"kv_port\": ${KV_PORT},
  #   \"kv_rank\": ${KV_RANK},
  #   \"kv_connector_extra_config\": {\"use_layerwise\": true}
  # }"