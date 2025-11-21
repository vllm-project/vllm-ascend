# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

from random_mock_model_tp2 import run_offline_prompts

if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    os.environ[
        "CAPTURE_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
    os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"
    os.environ["MOCK_CAPTURE_FILE"] = ".mock_cache"
    os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"

    run_offline_prompts()
