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
    from vllm_ascend.models.qwen4_exp.decode_submission import DecodeQueueController


class AscendQwen4ExpModelState(Qwen4ExpModelState, AscendMambaHybridModelState):
    """Qwen4Exp state with Ascend attention metadata construction.

    Method resolution order keeps the upstream PLE n-gram buffers and input
    preparation from :class:`Qwen4ExpModelState`, while ``prepare_attn``
    resolves to :class:`AscendMambaHybridModelState`, which builds Ascend
    attention metadata and stores it on the state for FULL aclgraph replay
    (``update_full_graph_params`` reads ``model_state.attn_metadata``).
    """

    _decode_submission: DecodeQueueController | None

    def initialize_decode_submission(self, runner: Any) -> None:
        """Create model-owned scheduling proofs after each KV initialization."""
        previous = getattr(self, "_decode_submission", None)
        if previous is not None:
            previous.invalidate()
        self._decode_submission = None
        connector = getattr(runner, "kv_connector", None)
        if connector is not None:
            from vllm_ascend.models.qwen4_exp.decode_queue_connector import unwrap_pd_decode_queue_connector

            # Even an opt-out or a fresh profiling model state must detach an
            # earlier queue observer before returning to the baseline path.
            runner.kv_connector = unwrap_pd_decode_queue_connector(connector)
        if self.get_decode_queue_capability(runner) is None:
            return
        config = getattr(runner, "ascend_config", None)
        if getattr(config, "enable_decode_early_queue", False) is not True:
            return
        from vllm.config.compilation import CUDAGraphMode

        from vllm_ascend.models.qwen4_exp.decode_submission import DecodeQueueController

        controller = DecodeQueueController(
            enabled=True,
            multimodal=getattr(config, "enable_mm_decode_early_queue", False) is True,
            full_graph_mode=CUDAGraphMode.FULL,
        )
        if controller.initialize(runner):
            self._decode_submission = controller

    def before_decode_requests(self, runner: Any, output: Any) -> None:
        controller = getattr(self, "_decode_submission", None)
        if controller is not None:
            controller.before_requests_for_runner(runner, output)

    def prepare_decode_seq_lens(self, runner: Any, output: Any, batch: Any, desc: Any) -> bool:
        controller = getattr(self, "_decode_submission", None)
        return controller is not None and controller.prepare_seq_lens(runner, output, batch, desc)

    def record_decode_snapshot(self, runner: Any, source_stream: Any) -> None:
        controller = getattr(self, "_decode_submission", None)
        if controller is not None:
            controller.record_runner_snapshot(runner, source_stream)

    def get_decode_queue_capability(self, runner: Any) -> QwenDecodeQueueCapability | None:
        """Bind the Qwen submission policy to audited model consumers."""
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
