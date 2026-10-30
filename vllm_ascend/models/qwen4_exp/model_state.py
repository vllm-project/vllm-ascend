# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Ascend V2 model-runner state for Qwen4Exp."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)

if TYPE_CHECKING:
    from vllm_ascend.models.qwen4_exp.decode_queue import QwenDecodeQueueCapability


class AscendQwen4ExpModelState(Qwen4ExpModelState, AscendMambaHybridModelState):
    """Qwen4Exp state with Ascend attention metadata construction.

    Method resolution order keeps the upstream PLE n-gram buffers and input
    preparation from :class:`Qwen4ExpModelState`, while ``prepare_attn``
    resolves to :class:`AscendMambaHybridModelState`, which builds Ascend
    attention metadata and stores it on the state for FULL aclgraph replay
    (``update_full_graph_params`` reads ``model_state.attn_metadata``).
    """

    def get_decode_queue_capability(self, runner: Any) -> QwenDecodeQueueCapability | None:
        """Offer model-consumer knowledge to the shared queue controller."""
        if (
            type(self) is not AscendQwen4ExpModelState
            or getattr(runner, "model_state", None) is not self
            or getattr(getattr(runner, "model_config", None), "architecture", None)
            not in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
        ):
            return None
        from vllm_ascend.models.qwen4_exp.decode_queue import QwenDecodeQueueCapability

        return QwenDecodeQueueCapability()


__all__ = ["AscendQwen4ExpModelState"]
