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

import logging

logger = logging.getLogger(__name__)

# Supported SOC_VERSION codes
ASCEND_A2_SOC_VERSION = range(220, 226)
ASCEND_A3_SOC_VERSION = range(250, 256)
ASCEND_310P_SOC_VERSION = [202]


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_model():
    try:
        import torch_npu  # type: ignore
    except ImportError:
        logger.warning("torch_npu not found. Proceeding with CPU testing.")
    else:
        from vllm_ascend import _build_info  # type: ignore
        ascend_soc_version = getattr(_build_info, "__ascend_soc_version__",
                                     None)
        raw = torch_npu.npu.get_soc_version()
        soc_version = (
            "ASCEND910B1" if raw in ASCEND_A2_SOC_VERSION else
            "ASCEND910_9392" if raw in ASCEND_A3_SOC_VERSION else
            "ASCEND310P3" if raw in ASCEND_310P_SOC_VERSION else "UNDEFINED")

        if soc_version == "UNDEFINED":
            raise RuntimeError("Unsupported or undefined Ascend SOC version.")
        elif soc_version != _build_info.__soc_version__:
            raise RuntimeError(
                f"Built for SOC version {_build_info.__soc_version__}, "
                f"but running on SOC version {soc_version}. "
                f"Please reinstall vllm-ascend with 'export SOC_VERSION={soc_version}' ."
            )
        elif ascend_soc_version is None:
            raise RuntimeError(
                "vllm_ascend._build_info is missing '__ascend_soc_version__'. "
                f"Please reinstall vllm-ascend with 'export SOC_VERSION={soc_version}' ."
            )

    from .models import register_model
    register_model()
