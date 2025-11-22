#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

current_dir = os.path.dirname(os.path.abspath(__file__))
opp_path = os.path.join(current_dir, "CANN", "vendors", "hwcomputing")
lib_path = os.path.join(current_dir, "CANN", "vendors", "hwcomputing",
                        "op_api", "lib")
# Set environment variables related to custom operators
os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
    f"{opp_path}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}")
os.environ[
    "LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_model():
    from .models import register_model
    register_model()


def register_connector():
    from vllm_ascend.distributed import register_connector
    register_connector()


def register_model_loader():
    from .model_loader.netloader import register_netloader
    register_netloader()


def register_service_profiling():
    from .profiling_config import generate_service_profiling_config
    generate_service_profiling_config()
