# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=1
export HCCL_OP_EXPANSION_MODE=AIV
export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
export VLLM_USE_V1=1
export VLLM_DP_MASTER_IP=7.150.15.29
export VLLM_WORKER_MULTIPROC_METHOD=fork
export HCCL_CONNECT_TIMEOUT=3600
export VLLM_DP_SIZE_PER_NODE=1

export RANDOM_MODE=RANDOM
export MOCK_CAPTURE_DIR=/home/kc/capture/  # saving folder for logs of inputs and outputs, ensure this exists
export MOCK_CAPTURE_FILE=cache_test
export MOCK_CAPTURE_FILE_LOCK=.lock

ray stop --force

ray start --head --num-gpus=8 --port=8361

sleep 10

ray status

python3 /home/kc/vllm_mock/scripts/vllm_serve_multinode.py