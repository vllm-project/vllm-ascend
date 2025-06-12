#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/examples/offline_inference/basic.py
#
"""
 This file provides a function to obtain ips of all NPU Devices in current machine.
"""

import os
import re
import subprocess

import vllm_ascend.envs as envs

# Get all device ips using hccn_tool
HCCN_TOOL_PATH = envs.HCCN_PATH


def get_device_ips():
    npu_info = subprocess.run(['npu-smi', 'info', '-m'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
    if npu_info.returncode != 0 or not os.path.exists(HCCN_TOOL_PATH):
        raise RuntimeError("No npu-smi/hccn_tool tools provided for NPU.")

    # ‌Extract NPU IDs for all Ascend devices (excluding Mcu rows)
    device_ids = []
    for line in npu_info.stdout.strip().split('\n'):
        match = re.match(r'^\s*(\d+)\s+\d+\s+\d+\s+Ascend', line)
        if match:
            device_ids.append(int(match.group(1)))

    if not device_ids:
        raise RuntimeError(
            "Cannot parse any valid device ID from npu-smi output.")

    device_ip_list = []
    for device_id in device_ids:
        cmd = [HCCN_TOOL_PATH, '-i', str(device_id), '-ip', '-g']
        device_ip_info = subprocess.run(cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
        ip_match = re.search(r'ipaddr:(.*)', device_ip_info.stdout)
        if not ip_match:
            raise RuntimeError(
                f"Cannot parse IP from hccn_tool for device {device_id}")
        device_ip = ip_match.group(1).strip()
        device_ip_list.append(device_ip)

    return device_ip_list


print(get_device_ips())
