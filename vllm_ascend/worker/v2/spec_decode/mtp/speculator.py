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
import copy
import logging
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.utils import lmhead_tp_enable, lmhead_tp_max_num_logits
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator

logger = logging.getLogger(__name__)


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft). All MLA handling is in the base
    (AscendAutoRegressiveSpeculator)"""

    # Set by load_draft_model: True when the loaded draft's MTP layers mount
    # the target's lm_head object (upstream load_eagle_model ties every
    # layer's shared_head.head — and a top-level lm_head alias — to the
    # target's lm_head when the MTP checkpoint ships no own head weights;
    # DeepSeek MTP doesn't). In that case load_draft_model DECOUPLES each
    # shared head onto a shallow copy: same weight tensors, separate
    # object, so the per-layer capacity publish in _run_model lands on the
    # copies and the target's head object stays clean — the target's
    # sample() must never resolve the draft-scale dyn (rows > dyn ->
    # ValueError, the Q6/DFlash crash). Default False ("not shared / not
    # yet loaded"): _run_model can never run before load_draft_model.
    _lmhead_tp_head_is_shared = False

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # Resolve the per-layer heads the same way _run_model's publish
        # does (self.model.model.layers for DeepSeek-V4, self.model.layers
        # for the other MTP families).
        mtp_model = getattr(model, "model", model)
        layers = getattr(mtp_model, "layers", None) or {}
        layer_heads = []
        for layer in layers.values():
            shared_head = getattr(layer, "shared_head", None)
            if shared_head is None:
                continue
            head = getattr(shared_head, "head", None)
            if head is not None:
                layer_heads.append((shared_head, head))
        # A draft head is shared when it is literally one of the target
        # model's submodules (identity, not weight equality — mirrors the
        # DFlash detection; the loader ties the objects, not the tensors).
        target_module_ids = {id(module) for module in target_model.modules()}
        self._lmhead_tp_head_is_shared = any(id(head) in target_module_ids for _, head in layer_heads)
        if self._lmhead_tp_head_is_shared:
            # Decouple each shared head OBJECT while sharing the weight
            # tensors: the per-layer capacity publish lands on the copies,
            # and the target's head object never carries draft capacity.
            # copy.copy on an nn.Module shares the parameter/buffer tensors
            # by reference; the plain capacity attributes written later
            # live in the copy's own __dict__. The loader replaced the
            # constructor's head (which carried the static capacity), so
            # re-attach the static bound on every decoupled copy.
            static_capacity = lmhead_tp_max_num_logits(self.max_num_reqs, self.num_speculative_steps + 1)
            for shared_head, head in layer_heads:
                if id(head) in target_module_ids:
                    shared_head.head = copy.copy(head)
                    shared_head.head.lmhead_tp_capacity = static_capacity
            # The loader also leaves a top-level lm_head alias pointing at
            # the target's head; decouple it so no future draft-side
            # head-level writer can poison the target's head object.
            draft_head = getattr(model, "lm_head", None)
            if draft_head is not None and id(draft_head) in target_module_ids:
                model.lm_head = copy.copy(draft_head)
            if lmhead_tp_enable():
                logger.info(
                    "lmhead TP: the MTP draft layers share the target LM head "
                    "weights; decoupled head objects carry the draft-side "
                    "capacity so the target's head object stays clean."
                )
        return model

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
        The MTP model may expose its layer module directly (``self.model.layers``)
        or through an inner ``model`` (DeepSeek-V4: ``self.model.model.layers``),
        so resolve it tolerantly to cover every MTP family.
        """
        hidden_states = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )
        mtp_model = getattr(self.model, "model", self.model)
        layers = getattr(mtp_model, "layers", None)
        if layers is None or not layers:
            return hidden_states
        assert mtp_model is not None
        if num_tokens_across_dp is not None and num_tokens_across_dp.numel() > 0:
            max_reqs_across_dp = int(num_tokens_across_dp.max().item())
            num_mtp_layers = mtp_model.num_mtp_layers
            draft_capacity = max_reqs_across_dp * num_mtp_layers
            for layer in layers.values():
                layer.shared_head.head._lmhead_tp_dynamic_capacity = draft_capacity
        else:
            # No DP sync this round: drop the previous round's dynamic value so
            # the static lmhead_tp_capacity stays authoritative.
            for layer in layers.values():
                if hasattr(layer.shared_head.head, "_lmhead_tp_dynamic_capacity"):
                    del layer.shared_head.head._lmhead_tp_dynamic_capacity
        return hidden_states
