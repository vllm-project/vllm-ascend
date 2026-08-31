# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from vllm.v1.engine.core import EngineCore


def _wake_up_without_early_scheduler_resume(self: EngineCore, tags: list[str] | None = None) -> None:
    """Keep scheduling paused until all sleep-mode allocations are awake.

    vLLM supports waking weights and KV cache in separate stages. The generic
    EngineCore implementation resumes scheduling after every wake_up call,
    including a weights-only wake. CaMem has not remapped the KV cache at that
    point, so a queued partial-rollout request can execute against unmapped NPU
    memory.

    Preserve the explicit ``scheduling`` wake behavior, but otherwise resume
    only after the executor reports that no allocation tags remain asleep.
    """
    resume_scheduling = tags is not None and "scheduling" in tags
    if resume_scheduling:
        tags = [tag for tag in tags if tag != "scheduling"]

    if tags is None or tags:
        self.model_executor.wake_up(tags)

    if resume_scheduling or not self.model_executor.is_sleeping:
        self.resume_scheduler()


EngineCore.wake_up = _wake_up_without_early_scheduler_resume
