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

"""Frozen pattern-repeat alert reasons, in native rule evaluation order."""

from dataclasses import dataclass

REASONS = (
    "vllm[32,500]x5",
    "vllm[32,2000]x3",
    "run>=500",
    "A3_n4_W1024>=0.6 S8 N_n4_W512<=0.05",
    "A5_n4_W512>=0.7 S4 N_n4_W512<=0.1",
    "A5_n4_W512>=0.5 S4 sep<=0.5 N_n4_W512<=0.05",
    "P_n8_W1024>=0.6 S24 N_n4_W512<=0.02",
    "wrep_W2048>=0.95 S16 & mark_W1024>=0.05",
    "wrep_W2048>=0.99 S24",
    "wrep_W1024>=0.99 S8 & mark_W1024>=0.04",
    "wrep_W1024>=0.98 S8 & mark_W1024>=0.05",
    "sdup_N16>=0.7 S24 & wrep_W2048>=0.9 & mark_W1024>=0.03",
    "sdup_N16>=0.6 S40 & mark_W1024>=0.03",
    "wrep_W2048>=0.95 S48 & mark_W1024>=0.03",
)


@dataclass
class Result:
    is_ill: bool = False
    fired_at: int = -1  # raw-token boundary at which the alert is raised
    reason: str = ""
