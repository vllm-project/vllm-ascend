# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
# AscendMTPSpeculator layers the shared Ascend draft loop
# (AscendAutoRegressiveSpeculator) on upstream MTPSpeculator. All MLA-specific
# handling lives in the base, gated by is_mla.
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft). All MLA handling is in the base
    (AscendAutoRegressiveSpeculator)"""

    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the MTP draft backbone and publish the draft LM-head capacity.

        ``num_tokens_across_dp`` is synchronized with one token per request, so
        its max is the group-agreed request count. Each MTP layer emits one
        logits row per request, so the draft-head row capacity every rank must
        feed is ``max_reqs_across_dp * num_mtp_layers``. Publish it on every
        layer's head before the layer's compute_logits runs in this same step.
        """
        hidden_states = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )
        if num_tokens_across_dp is not None and num_tokens_across_dp.numel() > 0:
            max_reqs_across_dp = int(num_tokens_across_dp.max().item())
            num_mtp_layers = self.model.model.num_mtp_layers
            draft_capacity = max_reqs_across_dp * num_mtp_layers
            for layer in self.model.model.layers.values():
                layer.shared_head.head._lmhead_tp_dynamic_capacity = draft_capacity
        else:
            # No DP sync this round: drop the previous round's dynamic value so
            # the static lmhead_tp_capacity stays authoritative.
            for layer in self.model.model.layers.values():
                if hasattr(layer.shared_head.head, "_lmhead_tp_dynamic_capacity"):
                    del layer.shared_head.head._lmhead_tp_dynamic_capacity
        return hidden_states
