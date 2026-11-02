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

from typing import Any

from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)


class AscendQwen4ExpModelState(Qwen4ExpModelState, AscendMambaHybridModelState):
    """Qwen4Exp state with Ascend attention metadata construction.

    Method resolution order keeps the upstream PLE n-gram buffers and input
    preparation from :class:`Qwen4ExpModelState`, while ``prepare_attn``
    resolves to :class:`AscendMambaHybridModelState`, which builds Ascend
    attention metadata and stores it on the state for FULL aclgraph replay
    (``update_full_graph_params`` reads ``model_state.attn_metadata``).
    """

    _skip_spec_cpu_length_sync: bool

    def initialize_spec_cpu_length_sync(self, runner: Any) -> None:
        """Bind length consumers once, after target and draft KV initialization."""
        from vllm_ascend.models.qwen4_exp.cpu_length_sync import can_skip_qwen_cpu_length_sync

        self._skip_spec_cpu_length_sync = False
        if not can_skip_qwen_cpu_length_sync(runner):
            return
        stream = getattr(runner, "num_computed_tokens_stream", None)
        if stream is None or not callable(getattr(stream, "synchronize", None)):
            return
        # Retire any profiling/previous-initialization copy before dropping
        # both its producer and consumer. This runs once, never per token.
        stream.synchronize()
        self._skip_spec_cpu_length_sync = True

    def can_skip_spec_cpu_length_sync(self) -> bool:
        """Keep the D2H producer and consumer on the same stable decision."""
        return getattr(self, "_skip_spec_cpu_length_sync", False) is True


__all__ = ["AscendQwen4ExpModelState"]
