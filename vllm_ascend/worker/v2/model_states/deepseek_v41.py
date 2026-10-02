# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend attention state and device history for DeepSeek V4.1 Engram."""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.models.deepseek_v41.nvidia.model_state import _gather_lookback_kernel
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import ring_state_update_skipped
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class AscendDeepseekV41ModelState(AscendModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids = (
            torch.full((self.max_num_reqs, depth), -1, dtype=torch.int32, device=device) if depth > 0 else None
        )
        self._engram_inputs: dict[str, Any] | None = None
        if depth > 0:
            # Eager/piecewise calls enter here after set_forward_context. FULL
            # replay bypasses Python forward and calls prepare_engram explicitly.
            model.register_forward_pre_hook(self._prepare_engram_before_forward)

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states: RequestState) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        inputs: dict[str, Any] = {
            "input_ids": input_batch.input_ids[: input_batch.num_tokens],
            "positions": input_batch.positions[: input_batch.num_tokens],
            "padded_tokens": input_batch.num_tokens_after_padding,
        }
        if self.kvpp_is_dummy_run or ring_state_update_skipped():
            # Idle DP ranks must still look up dummy hashes with their peers.
            # They have no request history, even when make_dummy supplies rows.
            window.fill_(-1)
        else:
            all_token_ids = req_states.all_token_ids.gpu
            depth = window.shape[1]
            _gather_lookback_kernel[(window.shape[0],)](
                window,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                input_batch.num_reqs,
                DEPTH=depth,
                BLOCK_DEPTH=triton.next_power_of_2(depth),
            )
            inputs["lookback_token_ids"] = window
            inputs["query_start_loc"] = input_batch.query_start_loc[: input_batch.num_reqs + 1]
        self._engram_inputs = inputs
        model_inputs["lookback_token_ids"] = window
        model_inputs.update(self.model.prepare_engram_graph_inputs(input_batch.num_tokens_after_padding))
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        self._engram_inputs = None
        if self.lookback_token_ids is not None:
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        # Capture only the fixed row/mask buffers, never HOST_UVA lookup or DP
        # exchange. Each real/idle execution refreshes them before replay.
        model_inputs.update(self.model.prepare_engram_graph_inputs(num_tokens))
        return model_inputs

    def _prepare_engram_before_forward(self, model, args) -> None:
        self.prepare_engram()

    def prepare_engram(self) -> None:
        """Refresh fixed buffers once per step, with valid DP forward metadata."""
        if self._engram_inputs is not None:
            self.model.prepare_engram_inputs(**self._engram_inputs)
            self._engram_inputs = None
