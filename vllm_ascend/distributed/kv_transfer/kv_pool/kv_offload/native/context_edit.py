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
"""Ascend context-edit backend for Agent Hint M2.

M2 is the control-plane slice: vLLM calls this backend with the validated
management request, the backend selects a tier and records the job. The actual
NPU/CPU/disk transfer is wired in M3/M4; returning ``tier=cpu`` here does not
mean blocks were moved.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.agent_hint import AgentHint, AgentManagementResult

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.tier_policy import (
    TierPolicy,
)

logger = init_logger(__name__)


class NPUContextEditBackend:
    """Control-plane context-edit backend for Ascend."""

    def __init__(self, policy: TierPolicy | None = None) -> None:
        self._policy = policy or TierPolicy(disk_enabled=True)

    def select_tier(self, expected_resume_s: int | None) -> str:
        return self._policy.decide(expected_resume_s).tier

    def execute(
        self,
        action: str,
        hint: AgentHint,
        job_id: str,
        request_id: str,
    ) -> AgentManagementResult:
        expected_resume_s = (
            hint.prediction.expected_resume_s if hint.prediction is not None else None
        )
        decision = self._policy.decide(expected_resume_s)
        block_count = len(hint.block_hashes or [])
        logger.info(
            "kv_context_edit action=%s session_id=%s expected_resume_s=%s tier=%s "
            "fallback=%s blocks=%s job=%s",
            action,
            hint.session_id,
            expected_resume_s,
            decision.tier,
            decision.fallback,
            block_count,
            job_id,
        )
        return AgentManagementResult(
            job_id=job_id,
            session_id=hint.session_id or "",
            status="accepted",
            detail=(
                f"M4 tier-policy: action={action} tier={decision.tier} "
                f"({decision.reason}) blocks={block_count} "
                "(data-plane owner=offloading connector)"
            ),
            tier=decision.tier,
            blocks=block_count,
        )
