#!/bin/bash

: <<'COMMENT'
运行fuxi_alpha_kuairand_demo.py脚本，可直接修改模型参数，如embedding_dim、num_heads、attention_dim等，可控制是否使用随机模型or加载训练好的模型pth（use_ramdon）
参数列表：
    --embedding_dim     embedding维度（use_random为True时才生效）
    --num_heads         注意力头数（use_random为True时才生效）
    --dim               Attention维度（use_random为True时才生效）
    --use_random        是否采用随机模型
    --max_seq_len       最大生成测试序列长度（需比max_model_len // 2小）
    --max_batch_size    批处理数（待支持）
    --aclgraph          是否采用aclgraph成图
    --candidate_num     候选集数（use_random为True时才生效）
    --has_ffn           是否开启FFN（fuxi_alpha）
    --max_vocab_size    最大embedding表维度
    --concat_batch      是否拼接batch，当前使用kuairand每一个batch的样本数较少，因此为增大序列长度，一次性读了多个batch的数据（PS.最好把该值设置为1）
    --profiler          是否采集profiler
    --max_model_len     模型最大长度（请输入2的幂次）
    --range             跑的推理轮次
    --graph_step        成图的最大挡位差

e.g.
0.1B模型：
bash run.sh   \
  --embedding_dim 1024        \
  --num_heads 8               \
  --dim 128                   \
  --max_seq_len 4096          \
  --max_batch_size 1          \
  --use_random 1              \
  --aclgraph 1                \
  --candidate_num 512         \
  --has_ffn 1                 \
  --max_vocab_size 1000000    \
  --concat_batch 1            \
  --profiler 0                \
  --max_model_len 8832        \
  --range 2                   \
  --graph_step 500            \

0.01B模型
bash run.sh   \
  --embedding_dim 256         \
  --num_heads 4               \
  --dim 64                    \
  --max_seq_len 4096          \
  --max_batch_size 1          \
  --use_random 1              \
  --aclgraph 1                \
  --candidate_num 512         \
  --has_ffn 1                 \
  --max_vocab_size 1000000    \
  --concat_batch 1            \
  --profiler 0                \
  --max_model_len 8832        \
  --range 2                   \
  --graph_step 500            \

# 1B model
bash run.sh   \
     --embedding_dim 4096        \
     --num_heads 16               \
     --dim 256                   \
     --max_seq_len 2048          \
     --max_batch_size 1          \
     --use_random 1              \
     --aclgraph 1                \
     --candidate_num 256         \
     --has_ffn 0                 \
     --max_vocab_size 1000000    \
     --concat_batch 1            \
     --profiler 1                \
     --max_model_len 8192        \
     --range 2                   \
     --graph_step 512            \
     --block_size 128

COMMENT

DEMO="KUAIRAND"
# DEMO="FUXI-MUSIC"

if [ "${DEMO}" == "KUAIRAND" ]; then
    PYTHON_FILE="fuxi_alpha_kuairand_demo.py"
elif [ "${DEMO}" == "FUXI-MUSIC" ]; then
    export MODEL_PATH="${MODEL_PATH:-/path/to/fuxi_music_model}"
    PYTHON_FILE="fuxi_alpha_music_demo.py"
fi


#----------------------------------------
# vllm related
#----------------------------------------
export VLLM_VERSION="0.11.0"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

#----------------------------------------
# speedup related
#----------------------------------------
# cpu-binding
NPU_NUM=8
CPU_CORES=$(nproc --all)
CORES_PER_NPU=$((CPU_CORES / NPU_NUM))
CPU_AFFINITY_CONF_TMP=1
if [ "$NPU_NUM" -gt 0 ]; then
    for (( i=0; i<NPU_NUM; i++ )); do
        start_core=$(( i * CORES_PER_NPU ))
        end_core=$(( start_core + CORES_PER_NPU - 1 ))
        CPU_AFFINITY_CONF_TMP+=",npu${i}:${start_core}-${end_core}"
    done
fi
export CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF_TMP
echo "CPU_AFFINITY_CONF="$CPU_AFFINITY_CONF

# multi-threads for CPU ops
export OMP_NUM_THREADS=12

#----------------------------------------
# prof related
# ----------------------------------------
# export VLLM_TORCH_PROFILER_DIR=/path/to/profiler/output
# export VLLM_TORCH_PROFILER_WITH_MODULES=0 # 是否采集调用栈
# export VLLM_TORCH_PROFILER_WITH_STACK=0

#----------------------------------------
# debug related
#----------------------------------------
#export ASCEND_LAUNCH_BLOCKING=1 # 使用aclgraph时不能启用

#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_GLOBAL_EVENT_ENABLE=0
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#export ASDOPS_LOG_LEVEL=INFO
#export ASDOPS_LOG_TO_STDOUT=1

#----------------------------------------
# fbgemm related
#----------------------------------------
# install custom ops
# bash /path/to/mxrec_ops/install.sh
# install torch_plugin
# bash /path/to/torch_plugin/torch_library/2.6.0/common/build_ops.sh
export LIB_FBGEMM_NPU_API_SO_PATH="${LIB_FBGEMM_NPU_API_SO_PATH:-/usr/local/python3.11.13/lib/python3.11/site-packages/libfbgemm_npu_api.so}"
export NEED_PATCH_TOKENIZER=1
export PROMPT_LOGPROBS_USE_TENSOR=1

echo "========================================"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "========================================"

DEFAULT_EMBEDDING_DIM=256
DEFAULT_NUM_HEADS=4
DEFAULT_DIM=64
DEFAULT_MAX_SEQ_LEN=1000
DEFAULT_BATCH_SIZE=1
DEFAULT_USE_RANDOM=0
DEFAULT_ACLGRAPH=1
DEFAULT_CANDIDATE_NUM=400
DEFAULT_HAS_FFN=0
DEFAULT_MAX_VOCAB_SIZE=10000000
DEFAULT_CONCAT_BATCH=0
DEFAULT_START_PROFILER=0
DEFAULT_MAX_MODEL_LEN=8192
DEFAULT_RANGE=1
DEFAULT_GRAPH_STEP=500
DEFAULT_ASYNC=0
DEFAULT_BLOCK_SIZE=128

EMBEDDING_DIM=""
NUM_HEADS=""
DIM=""
MAX_SEQ_LEN=""
BATCH_SIZE=""
USE_RANDOM=""
ACLGRAPH=""
CANDIDATE_NUM=""
HAS_FFN=""
VOCAB_SIZE=""
CONCAT_BATCH=""
START_PROFILER=""
MAX_MODEL_LEN=""
RANGE=""
GRAPH_STEP=""
ASYNC=""
BLOCK_SIZE=""

TEMP=$(getopt -o e:n:d:m:b:u:a:c:f:v:t:p:d:r:g:y: --long embedding_dim:,num_heads:,dim:,max_seq_len:,max_batch_size:,use_random:,aclgraph:,candidate_num:,has_ffn:,max_vocab_size:,concat_batch:,profiler:,max_model_len:,range:,graph_step:,async:,block_size: -n "$0" -- "$@")
if [ $? != 0 ]; then
  echo "参数解析失败。" >&2
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -e|--embedding_dim)
      EMBEDDING_DIM="$2"
      shift 2
      ;;
    -n|--num_heads)
      NUM_HEADS="$2"
      shift 2
      ;;
    -d|--dim)
      DIM="$2"
      shift 2
      ;;
    -m|--max_seq_len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    -b|--max_batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -u|--use_random)
      USE_RANDOM="$2"
      shift 2
      ;;
    -a|--aclgraph)
      ACLGRAPH="$2"
      shift 2
      ;;
    -c|--candidate_num)
      CANDIDATE_NUM="$2"
      shift 2
      ;;
    -f|--has_ffn)
      HAS_FFN="$2"
      shift 2
      ;;
    -v|--max_vocab_size)
      VOCAB_SIZE="$2"
      shift 2
      ;;
    -t|--concat_batch)
      CONCAT_BATCH="$2"
      shift 2
      ;;
    -p|--profiler)
      START_PROFILER="$2"
      shift 2
      ;;
    -d|--max_model_len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    -r|--range)
      RANGE="$2"
      shift 2
      ;;
    -g|--graph_step)
      GRAPH_STEP="$2"
      shift 2
      ;;
    -y|--async)
      ASYNC="$2"
      shift 2
      ;;
    --block_size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "内部错误" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MAX_SEQ_LEN" ]]; then
    MAX_SEQ_LEN=$DEFAULT_MAX_SEQ_LEN
fi

if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE=$DEFAULT_BATCH_SIZE
fi

if [[ -z "$USE_RANDOM" ]]; then
    USE_RANDOM=$DEFAULT_USE_RANDOM
fi

if [[ -z "$ACLGRAPH" ]]; then
    ACLGRAPH=$DEFAULT_ACLGRAPH
fi

if [[ -z "$CANDIDATE_NUM" ]]; then
    CANDIDATE_NUM=$DEFAULT_CANDIDATE_NUM
fi

if [[ -z "$HAS_FFN" ]]; then
    HAS_FFN=$DEFAULT_HAS_FFN
fi

if [[ -z "$VOCAB_SIZE" ]]; then
    VOCAB_SIZE=$DEFAULT_MAX_VOCAB_SIZE
fi

if [[ -z "$CONCAT_BATCH" ]]; then
    CONCAT_BATCH=$DEFAULT_CONCAT_BATCH
fi

if [[ -z "$START_PROFILER" ]]; then
    START_PROFILER=$DEFAULT_START_PROFILER
fi

if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN=$DEFAULT_MAX_MODEL_LEN
fi

if [[ -z "$RANGE" ]]; then
    RANGE=$DEFAULT_RANGE
fi

if [[ -z "$GRAPH_STEP" ]]; then
    GRAPH_STEP=$DEFAULT_GRAPH_STEP
fi

if [[ -z "$ASYNC" ]]; then
    IS_ASYNC=$DEFAULT_ASYNC
fi

if [[ -z "$BLOCK_SIZE" ]]; then
    BLOCK_SIZE=$DEFAULT_BLOCK_SIZE
fi

python3 ${PYTHON_FILE} \
    --embedding_dim $EMBEDDING_DIM \
    --num_heads $NUM_HEADS \
    --dim "$DIM" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --max_batch_size "$BATCH_SIZE" \
    --use_random "$USE_RANDOM" \
    --aclgraph "$ACLGRAPH" \
    --candidate_num "$CANDIDATE_NUM" \
    --has_ffn "$HAS_FFN" \
    --max_vocab_size "$VOCAB_SIZE" \
    --concat_batch "$CONCAT_BATCH" \
    --profiler "$START_PROFILER" \
    --max_model_len "$MAX_MODEL_LEN" \
    --range "$RANGE" \
    --graph_step "$GRAPH_STEP" \
    --is_async "$IS_ASYNC" \
    --block_size "$BLOCK_SIZE"

# export COLLECT_LOGS_PATH=/path/to/AscendLog
# export ASCEND_SLOG_PRINT_TO_STDOUT=0  # 1/0 Plog是否打屏（推荐为0）
# export ASCEND_GLOBAL_LOG_LEVEL=3  # 日志等级 0: debug 1: info 2: warning 3: error
# export ASCEND_PROCESS_LOG_PATH="$COLLECT_LOGS_PATH"  # 设置Plog存储路径

# python3 -m debugpy --listen 5678 --wait-for-client ${PYTHON_FILE} \
#     --embedding_dim $EMBEDDING_DIM \
#     --num_heads $NUM_HEADS \
#     --dim "$DIM" \
#     --max_seq_len "$MAX_SEQ_LEN" \
#     --max_batch_size "$BATCH_SIZE" \
#     --use_random "$USE_RANDOM" \
#     --aclgraph "$ACLGRAPH" \
#     --candidate_num "$CANDIDATE_NUM" \
#     --has_ffn "$HAS_FFN" \
#     --max_vocab_size "$VOCAB_SIZE" \
#     --concat_batch "$CONCAT_BATCH" \
#     --profiler "$START_PROFILER" \
#     --max_model_len "$MAX_MODEL_LEN" \
#     --range "$RANGE" \
#     --graph_step "$GRAPH_STEP" \
#     --is_async "$IS_ASYNC" \
#     --block_size "$BLOCK_SIZE"
