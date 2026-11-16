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
"""Validate the number of explicitly configured execution participants."""

from tools.bisect.execution.targets import ExecutionTargets
from tools.bisect.orchestration.config import SCENE_SINGLE


def resolve_node_count(scene: str, num_nodes: int | None, targets: ExecutionTargets | None = None) -> int | None:
    if num_nodes is not None and (isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1):
        raise ValueError("--num-nodes must be a positive integer")
    count = len(targets.nodes) if targets is not None else num_nodes
    if targets is not None and num_nodes is not None and num_nodes != count:
        raise ValueError("--num-nodes disagrees with explicit target count")
    if scene == SCENE_SINGLE:
        if count is not None and count != 1:
            raise ValueError("single_node requires exactly one participant")
        return 1
    if count == 0:
        raise ValueError("At least one execution participant is required")
    return count
