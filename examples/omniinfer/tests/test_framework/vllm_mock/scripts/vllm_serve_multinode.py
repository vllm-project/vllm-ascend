# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import runpy
import sys

from vllm_serve import run_vllm_serve


if __name__ == "__main__":
    run_vllm_serve(tp=1, dp=16, model="/home/kc/models/DeepSeek-V2-Lite")
