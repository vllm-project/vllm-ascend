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
Device type management utilities for vLLM Ascend.

This module provides functionality for:
- Detecting and managing Ascend device types
- Checking device compatibility
"""

from enum import Enum

import torch_npu  # noqa: F401


class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


_ascend_device_type = None


def _init_ascend_device_type():
    global _ascend_device_type
    from vllm_ascend import _build_info  # type: ignore
    _ascend_device_type = AscendDeviceType[_build_info.__device_type__]


def check_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()

    soc_version = torch_npu.npu.get_soc_version()
    if 220 <= soc_version <= 225:
        cur_device_type = AscendDeviceType.A2
    elif 250 <= soc_version <= 255:
        cur_device_type = AscendDeviceType.A3
    elif 200 <= soc_version <= 205:
        cur_device_type = AscendDeviceType._310P
    elif soc_version == 260:
        cur_device_type = AscendDeviceType.A5
    else:
        raise RuntimeError(f"Can not support soc_version: {soc_version}.")

    assert _ascend_device_type == cur_device_type, f"Current device type: {cur_device_type} does not match the installed version's device type: {_ascend_device_type}, please check your installation package."


def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type
