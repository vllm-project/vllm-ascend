#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import os
from datetime import datetime

DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")

_log_dir = DEFAULT_LOG_DIR
_base_name = f"vllm_ascend_{os.getpid()}"


def get_log_dir_and_basename():
    return _log_dir, _base_name


def setup_log_dir_and_basename(target_dir: str | None):
    global _log_dir, _base_name
    if target_dir is None:
        target_dir = DEFAULT_LOG_DIR
    os.makedirs(target_dir, exist_ok=True)
    _log_dir = target_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _base_name = f"vllm_ascend_{timestamp}_{os.getpid()}"
    return get_log_dir_and_basename()
