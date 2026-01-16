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

"""
Communication utilities for vLLM Ascend.

This module provides functionality for:
- HCCL library detection and configuration
- HCCL process group options
- Buffer size calculation for data parallelism
"""

import math
from typing import Optional

import torch
import torch_npu  # noqa: F401

from vllm_ascend import envs_ascend


def find_hccl_library() -> str:
    """
    We either use the library file specified by the `HCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libhccl.so` can be
    found by `ctypes` automatically.
    """
    so_file = envs_ascend.HCCL_SO_PATH

    # manually load the hccl library
    if so_file:
        from vllm.logger import logger
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s",
                    so_file)
    else:
        if torch.version.cann is not None:
            so_file = "libhccl.so"
        else:
            raise ValueError("HCCL only supports Ascend NPU backends.")
        from vllm.logger import logger
        logger.info("Found hccl from library %s", so_file)
    return so_file


def get_default_buffer_config() -> dict:
    return {"hccl_buffer_size": 200}


def calculate_dp_buffer_size() -> int:
    """
    formula of dp buffer size:
    dp_size + 1 (flags: with_prefill)
    """
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    dp_size = vllm_config.parallel_config.data_parallel_size
    int32_size = torch.iinfo(torch.int32).bits // 8
    dp_buffer_size = math.ceil((dp_size + 1) * int32_size / (1024 * 1024))
    return max(dp_buffer_size, 50)


def get_hccl_config_for_pg_options(group_name: str) -> Optional[dict]:
    """
    Get HCCL process group options for the given communication group name.

    Args:
        group_name: Name of the communication group

    Returns:
        HCCL pg_options or None for mc2 group
    """
    # FIXME: Current mc2 operators only perform communication space partitioning
    # based on HCCL_BUFFSIZE configuration. Using pg_options with mc2 group would
    # result in memory misalignment problems.
    if group_name and "mc2" in group_name:
        return None
    hccl_config_map = {
        "dp": {
            "hccl_buffer_size": calculate_dp_buffer_size()
        },
    }
    return hccl_config_map.get(group_name, get_default_buffer_config())


# Currently, when in A2, setting the environment variables HCCL_INTRA_PCIE_ENABLE=1
# and HCCL_INTRA_ROCE_ENABLE=0 can reduce cross-machine communication traffic and
# significantly improve communication performance of MC2 ops dispatch/combine.
def is_hierarchical_communication_enabled():
    import os
    return (os.getenv("HCCL_INTRA_ROCE_ENABLE", "") == "0"
            and os.getenv("HCCL_INTRA_PCIE_ENABLE", "") == "1")
