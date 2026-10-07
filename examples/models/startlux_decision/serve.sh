#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a local StartLux checkpoint}"
: "${VLLM_SOURCE:?Set VLLM_SOURCE to the paired vLLM checkout}"
: "${STARTLUX_SOURCE:?Set STARTLUX_SOURCE to the official prompt/evaluation checkout}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select the assigned NPU devices}"
python_bin=${PYTHON_BIN:-.venv/bin/python}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
port=${PORT:-18190}
ascend_source=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$VLLM_SOURCE:$ascend_source:$STARTLUX_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "$python_bin" "$VLLM_SOURCE/examples/pooling/classify/serve_startlux_decision.py" \
 --model "$MODEL_PATH" --tensor-parallel-size "$tensor_parallel_size" \
 --max-model-len "${MAX_MODEL_LEN:-8192}" --port "$port" "$@"
