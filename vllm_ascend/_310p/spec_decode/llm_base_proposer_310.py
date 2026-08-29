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

from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.dflash_full_decode_only import (
    is_310p_dflash_full_decode_only,
)
from vllm_ascend._310p.dflash_piecewise import is_310p_dflash_piecewise
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendRotaryEmbedding310,
    clear_full_decode_draft_rope_310,
    get_full_decode_draft_rope_buffers_310,
    prepare_full_decode_draft_rope_310,
)
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer overrides for NPU-specific spec-decode workarounds."""

    def _prepare_full_decode_draft_rope(
        self,
        *,
        query_positions: torch.Tensor,
        query_actual_tokens: int,
        descriptor_tokens: int,
        runtime_mode: CUDAGraphMode,
    ) -> bool:
        """Refresh stable query/context RoPE inputs outside a compiled graph."""
        runner = getattr(self, "runner", None)
        scope_config = getattr(runner, "vllm_config", self.vllm_config)
        uses_graph_external_rope = is_310p_dflash_full_decode_only(
            scope_config,
        ) or is_310p_dflash_piecewise(scope_config)
        if getattr(self, "method", None) != "dflash" or not uses_graph_external_rope:
            return False
        # Both graph modes compile the rotary branch while precomputed buffers
        # are selected. The callable keeps those stable addresses, so every
        # draft call must refresh them from its current positions.
        if runtime_mode != CUDAGraphMode.FULL:
            query_positions = self._get_positions(descriptor_tokens)
        if query_positions.ndim != 1 or query_positions.shape[0] != descriptor_tokens:
            raise RuntimeError(
                "310P DFlash graph draft RoPE requires one descriptor-sized "
                f"query position vector, got shape={tuple(query_positions.shape)}, "
                f"descriptor={descriptor_tokens}"
            )

        context_positions_buffer = self._context_positions_buffer
        context_actual_tokens = int(getattr(self, "_dflash_num_context", 0))
        context_descriptor_tokens = max(
            descriptor_tokens,
            context_actual_tokens,
        )
        if context_positions_buffer.shape[0] < context_descriptor_tokens:
            raise RuntimeError(
                "310P DFlash graph context-position buffer is smaller than the "
                f"descriptor: capacity={context_positions_buffer.shape[0]}, "
                f"descriptor={context_descriptor_tokens}"
            )
        context_positions = context_positions_buffer[:context_descriptor_tokens]
        if not 0 <= context_actual_tokens <= context_descriptor_tokens:
            raise RuntimeError(
                "310P DFlash graph context extent is outside the descriptor: "
                f"actual={context_actual_tokens}, "
                f"descriptor={context_descriptor_tokens}"
            )

        draft_rotary = getattr(self, "_full_decode_draft_rotary_310", None)
        if draft_rotary is None:
            draft_rotary = next(
                (module for module in self.model.modules() if isinstance(module, AscendRotaryEmbedding310)),
                None,
            )
            if draft_rotary is None:
                raise RuntimeError("310P DFlash graph draft model has no Ascend rotary embedding")
            self._full_decode_draft_rotary_310 = draft_rotary

        capacity_tokens = max(
            int(getattr(runner, "max_num_tokens", descriptor_tokens)),
            descriptor_tokens,
        )
        prepare_full_decode_draft_rope_310(
            draft_rotary.cos_sin_cache,
            query_positions=query_positions,
            query_actual_tokens=query_actual_tokens,
            context_positions=context_positions,
            context_actual_tokens=context_actual_tokens,
            capacity_tokens=capacity_tokens,
        )
        (
            query_cos,
            query_sin,
            context_cos,
            context_sin,
        ) = get_full_decode_draft_rope_buffers_310()
        assert query_cos is not None and query_sin is not None
        assert context_cos is not None and context_sin is not None
        self._full_decode_draft_query_rope_cos_310 = query_cos
        self._full_decode_draft_query_rope_sin_310 = query_sin
        self._full_decode_draft_context_rope_cos_310 = context_cos
        self._full_decode_draft_context_rope_sin_310 = context_sin
        logger.debug(
            "[310p-dflash-graph/rope-precompute] "
            "component=draft actual_runtime=%s descriptor_tokens=%d "
            "query_actual=%d "
            "context_actual=%d query_positions_ptr=%d "
            "context_positions_ptr=%d query_cos_ptr=%d context_cos_ptr=%d",
            runtime_mode.name,
            descriptor_tokens,
            query_actual_tokens,
            context_actual_tokens,
            query_positions.data_ptr(),
            context_positions.data_ptr(),
            query_cos.data_ptr(),
            context_cos.data_ptr(),
        )
        return True

    def _finish_full_decode_draft_rope(self, prepared: bool) -> None:
        if prepared:
            clear_full_decode_draft_rope_310()

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            result = _original_run_merged_draft(
                self,
                num_input_tokens,
                batch_size,
                token_indices_to_sample,
                target_positions,
                inputs_embeds,
                multi_steps_attn_metadata,
                num_tokens,
                is_prefill,
            )
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
        return result

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # 310P workaround for MTP:
            # The NPU implementation of the slice assign
            #   self.input_ids[:num_tokens-1] = target_token_ids[1:]
            # can corrupt the tail element (index num_tokens-1) of the
            # persistent drafter input_ids buffer. We save/restore it to
            # avoid feeding garbage to the draft model or later GatherV2.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]

            # Protected shift (310P specific)
            tail_save = self.input_ids[num_tokens - 1].clone()
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            self.input_ids[num_tokens - 1] = tail_save

            # Replace the last token with the next token.
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None

            # 310P does not support PCP/DCP, so we skip all PCP handling.
            ori_token_indices_to_sample = None
            query_lens_d = None

            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        return super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
