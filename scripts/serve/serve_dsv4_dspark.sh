#!/usr/bin/env bash
# DeepSeek V4 + DSpark 普通启动脚本（单进程，不用 AFD）。
#
# 起： nohup bash serve_dsv4_dspark.sh > /tmp/serve_dsv4.log 2>&1 &
# 停： pkill -TERM -f 'vllm serve'        # SIGTERM，别用 -9
#
# 关键三点：
#   1. DSpark 的 num_speculative_tokens 必须等于 checkpoint 的 dspark_block_size（这里是 5）；
#   2. 一步 decode 每序列带 1+5=6 个 token，所以 cudagraph_capture_sizes 必须都是 6 的倍数，
#      且最大值 = max_num_seqs × 6 = 192；
#   3. --kv-cache-dtype fp8 决定走 FP8 KV 计划；写 auto 会落到 BF16 计划，是另一条代码路径。

set -euo pipefail

MODEL_PATH="/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8"
SERVED_MODEL_NAME="deepseek-v4"

DEVICES="0,1,2,3"          # DP4 用 4 张卡，另外 4 张空着
DP_SIZE="4"
TP_SIZE="1"
PORT="18000"
MAX_MODEL_LEN="8192"
MAX_NUM_SEQS="32"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_PLUGINS="ascend"
export PYTHONUNBUFFERED="1"
export VLLM_USE_V1="1"
export HCCL_BUFFSIZE="800"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false,"draft_sample_method":"greedy"}' \
    --kv-cache-dtype fp8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6,12,24,48,96,144,192]}' \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --trust-remote-code
