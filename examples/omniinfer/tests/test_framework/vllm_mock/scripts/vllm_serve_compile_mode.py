# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import runpy
import sys


def run_vllm_serve(tp=1, dp=1, model="/home/kc/models/DeepSeek-V2-Lite"):
    # Define the parameters you want to pass to vllm serve
    params = [
        "vllm.entrypoints.openai.api_server",
        "--port", "8089",
        "--model", model,
        "--enable-expert-parallel",
        "--max_num_seqs", "128",
        "--max_model_len", "8000",
        "--tensor_parallel_size", f"{tp}",
        "--data_parallel_size", f"{dp}",
        "--gpu_memory_utilization", "0.9",
        "--trust_remote_code",
        "--served-model-name", "deepseek",
        "--dtype", "bfloat16",
        "--distributed-executor-backend", "mp",
        "--block_size", "128",
        "--additional-config", '{"enable_graph_mode": true}',
    ]

    # Set sys.argv to include the script name and all parameters
    sys.argv = params

    # Run the vllm serve command using runpy
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')

if __name__ == "__main__":
    # Set environment variables
    os.environ['VLLM_ENABLE_MC2'] = '0'
    os.environ['VLLM_USE_V1'] = '1'
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = "1"
    os.environ['HCCL_CONNECT_TIMEOUT'] = "3600"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    os.environ["USING_LCCL_COM"] = "0"

    os.environ["RANDOM_MODE"] = "1"
    os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"
    os.environ["MOCK_CAPTURE_FILE"] = ".mock_cache2"
    os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"
    
    run_vllm_serve(tp=2, dp=1, model="/home/kc/models/DeepSeek-V2-Lite")