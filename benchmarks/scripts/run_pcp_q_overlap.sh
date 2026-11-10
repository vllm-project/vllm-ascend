#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
set -eo pipefail

# Run in the same container/software stack as the target prefill service.
# Example: bash benchmarks/scripts/run_pcp_q_overlap.sh --model /data/models --output /bench/results
# Eight idle NPUs are required. No model server is started or modified.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
ascend_package_dir=$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm_ascend").origin).parent)')
export ASCEND_CUSTOM_OPP_PATH="${ascend_package_dir}/_cann_ops_custom/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=2048
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_EXEC_TIMEOUT=120
export HCCL_CONNECT_TIMEOUT=120
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
timeout 1800 torchrun --standalone --nproc_per_node=8 "${benchmark_dir}/benchmark_pcp_q_overlap.py" "$@"
