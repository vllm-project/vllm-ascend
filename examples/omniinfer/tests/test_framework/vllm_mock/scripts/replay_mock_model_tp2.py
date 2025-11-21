# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

import torch
import torch_npu
from random_mock_model_tp2 import run_offline_prompts
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    # Comment out the ones not in use. IMPORTANT: Make sure you use temperature=0!
    os.environ["REPLAY_MODE"] = "1"  # replay inputs and outputs from the cache
    os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"
    os.environ["MOCK_CAPTURE_FILE"] = ".mock_cache"
    os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"

    run_offline_prompts()