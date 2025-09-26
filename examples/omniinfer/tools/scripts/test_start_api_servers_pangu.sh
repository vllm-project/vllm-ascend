# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
export ASCEND_LAUNCH_BLOCKING=0

export OMNI_USE_QWEN=1

python start_api_servers.py \
        --num-servers 1 \
        --model-path /your-model-path \
        --master-ip 8.8.8.8 \
        --tp 1 \
        --master-port 35678 \
        --served-model-name qwen \
        --log-dir apiserverlog \
        --extra-args "--enforce-eager " \
        --gpu-util 0.6 \
        --base-api-port 9555 \
        --no-enable-prefix-caching \
        --additional-config '{ "enable_hybrid_graph_mode": true}' # 混部模式开启enable_hybrid_graph_mode
