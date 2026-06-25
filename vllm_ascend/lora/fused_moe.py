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
#
"""Ascend MoE-LoRA wrapper (v1).

Design (see plan in conversation history):

  - Inherits weight allocation / set_lora / slice helpers from upstream
    FusedMoEWithLoRA. Only the injection mechanism differs: upstream wraps
    Triton modular kernel internals (`TritonExperts.activation` / `moe_sum`),
    which do not exist on Ascend. We instead wrap the per-layer
    `quant_method.apply` and, inside it, temporarily swap the active
    `MoECommMethod._apply_mlp` so the LoRA delta is added on permuted
    activations between the grouped GMMs.

  - Per-layer ownership is critical: `_MoECommMethods` is a module-level
    singleton shared by all 48 MoE layers. If we wrapped `_apply_mlp` at
    init time, layer N+1 would compose on top of layer N's wrapper and
    every forward would stack all layers' LoRA deltas. We bracket the swap
    inside `apply_wrapper` so only the active layer is in effect.

  - v1 deliberately limits scope to: unquant + AllGather + TP-only +
    no shared experts + no FusedMC2 + no dynamic EPLB. These are the exact
    conditions under which `Qwen3-30B-A3B-Thinking-2507` runs cleanly with
    TP=4 EP=1 on 4×64GB. Other paths assert early so users get a clear
    error rather than silently wrong outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE


def _assert_ascend_moe_lora_supported(base_layer: AscendFusedMoE) -> None:
    """Centralized v1 capability checks. Asserts up-front for clarity."""
    if getattr(base_layer, "use_ep", False):
        raise AssertionError(
            "Ascend MoE LoRA v1 does not support expert parallelism. "
            "Launch with `--enable-expert-parallel=false` and use TP only "
            "(e.g. TP=4 for Qwen3-30B-A3B on 4x64GB)."
        )
    if getattr(base_layer, "dynamic_eplb", False):
        raise AssertionError(
            "Ascend MoE LoRA v1 is incompatible with dynamic EPLB "
            "(expert migration would break the per-expert LoRA layout)."
        )
    if int(envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2) != 0:
        raise AssertionError(
            "Ascend MoE LoRA v1 cannot patch FusedMC2 path "
            "(dispatch_ffn_combine is a single fused C++ op). "
            "Set VLLM_ASCEND_ENABLE_FUSED_MC2=0."
        )
    if getattr(base_layer, "_shared_experts", None) is not None:
        raise AssertionError(
            "Ascend MoE LoRA v1 does not wrap the shared_experts path "
            "(it runs outside quant_method.apply). The target model "
            "Qwen3-30B-A3B-Thinking-2507 has no shared experts; models "
            "like DeepSeek-V3 are not yet supported."
        )
    if getattr(base_layer, "multistream_overlap_gate", False):
        raise AssertionError(
            "multistream_overlap_gate=True interleaves quant_method.apply "
            "calls on multiple streams; the MoE LoRA path has not been "
            "validated under this overlap. Disable it for MoE LoRA."
        )


def _recover_moe_lora_routing(lora_context, expanded_row_idx, topk_ids):
    """Recover per-permuted-row (expert_id, lora_slot) for the dispatched rows.

    npu_moe_init_routing semantics (verified empirically): ``expanded_row_idx``
    is indexed by the ORIGINAL flat (token, k) position and gives where that
    pair landed in the expert-sorted array -- not the reverse. So recovering
    "which (token, k) pair does sorted row i hold" needs the inverse permutation
    of ``expanded``, not a direct gather by it. ``argsort`` output shape ==
    input shape (value-independent), so this stays graph-capturable -- no
    ``.item()``/data-dependent host sync.
    """
    top_k = lora_context.top_k
    expanded = torch.abs(expanded_row_idx)
    inv_perm = torch.argsort(expanded)
    expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.long)

    # token_lora_indices is a 1D LongTensor sized to max_num_batched_tokens
    # (host-known constant). Clamping defensively to the last index is a no-op
    # in normal operation but keeps the gather graph-safe.
    orig_token = inv_perm // top_k
    token_lora_indices = lora_context.punica_wrapper.token_lora_indices
    orig_token = orig_token.clamp_(max=token_lora_indices.numel() - 1)
    lora_per_row = token_lora_indices[orig_token]
    return expert_per_row, lora_per_row


def moe_lora_apply_w13(lora_context, *, gate_up_out, hidden_states, expanded_row_idx, topk_ids):
    """Add the w13 LoRA delta into ``gate_up_out`` (in place), before activation.

    Called from ``unquant_apply_mlp`` right after the base gate_up GMM. Returns
    the recovered per-row routing so the w2 delta can reuse it.
    """
    routing = _recover_moe_lora_routing(lora_context, expanded_row_idx, topk_ids)
    expert_per_row, lora_per_row = routing
    lora_context.punica_wrapper.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        token_lora_mapping=lora_per_row,
    )
    return routing


def moe_lora_apply_w2(lora_context, *, down_out, silu_out, lora_routing):
    """Add the w2 LoRA delta into ``down_out`` (in place), after the down GMM.

    Reuses the per-row routing computed by ``moe_lora_apply_w13``; ``silu_out``
    is the activation output that fed the base down GMM.
    """
    expert_per_row, lora_per_row = lora_routing
    lora_context.punica_wrapper.add_lora_fused_moe(
        y=down_out,
        x=silu_out,
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        token_lora_mapping=lora_per_row,
    )


class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-native MoE-LoRA wrapper.

    Reuses upstream weight allocation, set_lora, reset_lora, and slicing.
    Instead of the GPU modular-kernel injection, it publishes a per-layer
    ``MoELoRAContext`` onto the base layer (``_ascend_moe_lora_context``).
    The Ascend unquant MoE path threads that context through
    ``MoEFusedExpertsInput`` -> ``MoEMlpComputeInput`` and applies the LoRA
    delta natively inside ``unquant_apply_mlp`` (see
    ``moe_lora_apply_w13`` / ``moe_lora_apply_w2`` below) -- no runtime
    monkey-patch of ``comm._apply_mlp``.
    """

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        # Skip FusedMoEWithLoRA.__init__: it immediately asserts Triton
        # internals and calls _inject_lora_into_fused_moe which is GPU-only.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        _assert_ascend_moe_lora_supported(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = _get_lora_device(base_layer)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def set_mapping(self, punica_wrapper):
        # Upstream FusedMoEWithLoRA.set_mapping (vllm v0.22.0+) chains into
        # ``self._moe_kernel.fused_experts.set_lora_context(...)``, but
        # ``_moe_kernel`` is only set by the GPU modular-kernel path that we
        # deliberately skip in __init__. We instead build the per-layer
        # MoELoRAContext (now that punica_wrapper is available) and publish it
        # on the base layer; AscendUnquantizedFusedMoEMethod.apply reads it via
        # ``getattr(layer, "_ascend_moe_lora_context", None)`` and threads it
        # down to unquant_apply_mlp. The context holds stable references (the
        # in-place-updated LoRA stacks, adapter_enabled and the punica wrapper),
        # so building it once here is sufficient.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
        self.base_layer._ascend_moe_lora_context = self._build_lora_context()

    # ------------------------------------------------------------------
    # Layer-replacement registration
    # ------------------------------------------------------------------
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        del lora_config, model_config
        # AscendSharedFusedMoE inherits from AscendFusedMoE so this isinstance
        # check matches both. _assert_ascend_moe_lora_supported in __init__
        # rejects layers that actually carry shared experts.
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 2


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        AscendFusedMoEWithLoRA.__init__(self, base_layer)
        # Override: 3D MoE LoRA uses a single w13 slice.
        self._w13_slices = 1

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        del lora_config, model_config
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 1


# ----------------------------------------------------------------------
# Upstream compatibility shim: vllm/lora/model_manager.py:create_dummy_lora
# branches on `module.__class__.__name__ == "FusedMoEWithLoRA"` (and the
# 3D variant). Without this override, our subclasses would skip the
# pack_moe path and hit the generic pack() fallback, which produces a
# flat list of N_experts * 3 sub-LoRAs -- `set_lora` then fails with
# "too many values to unpack (expected 3)".
#
# Overriding only __name__ keeps the actual class object distinct (so
# isinstance / type identity / debugging are unaffected) but lets the
# upstream string compare hit our objects.
# ----------------------------------------------------------------------
AscendFusedMoEWithLoRA.__name__ = "FusedMoEWithLoRA"
AscendFusedMoE3DWithLoRA.__name__ = "FusedMoE3DWithLoRA"