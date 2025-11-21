# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import runpy
import sys

from vllm_serve import run_vllm_serve

if __name__ == "__main__":
    # Set environment variables
    os.environ['VLLM_ENABLE_MC2'] = '0'
    os.environ['VLLM_USE_V1'] = '1'
    os.environ[
        "ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
    os.environ['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = "1"
    os.environ['HCCL_CONNECT_TIMEOUT'] = "3600"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    os.environ["USING_LCCL_COM"] = "0"

    os.environ["NO_NPU_MOCK"] = "1"
    os.environ["RANDOM_MODE"] = "1"

    run_vllm_serve(tp=1, dp=24, model="/home/kc/models/Qwen2.5-7B-Instruct")
