# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.spec_decode.dflash2_proposer import greedy_select_path
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


class AscendDFlash2Speculator(AscendDFlashSpeculator):
    """DFlash2 V2 speculator: selector walk instead of the DFlash1 argmax path.

    Mirrors upstream ``DFlash2Speculator`` but reuses the Ascend greedy selector
    walk kernel; probabilistic (Gumbel) draft sampling is not supported on NPU,
    matching the V1 ``AscendDflash2Proposer``.
    """

    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DFlash2 probabilistic draft sampling is not supported on NPU; use greedy (the default) instead."
            )
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device) * self.num_query_per_req
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(hidden_states.flatten(0, 1))
        candidate_ids = candidate_ids.view(num_reqs, self.num_speculative_steps, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self.draft_tokens[:num_reqs] = greedy_select_path(candidate_ids, scores)
