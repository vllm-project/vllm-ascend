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

  - Inherits weight allocation / set_lora / TP slicing from vLLM 0.23.0's
    FusedMoEWithLoRA. Ascend publishes one context on the base MoE layer and
    threads it explicitly through dispatch and MLP inputs; LoRA deltas are
    injected at the BF16 boundaries between the grouped matmuls.

  - Quantized v1 deliberately limits scope to W8A8_DYNAMIC + AllGather +
    TP-only + no FusedMC2 + no dynamic EPLB. Shared experts stay on their
    dense LoRA wrappers while routed experts use this MoE path. Other
    combinations fail early rather than silently producing wrong outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm import envs
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend


def _moe_lora_projection_enabled(
    lora_b: list[torch.Tensor],
    w13_num_slices: int,
) -> tuple[bool, bool]:
    """Return whether routed w13 and w2 contain a non-zero LoRA B."""
    if len(lora_b) == 2:
        w13_lora_b, w2_lora_b = lora_b
        return (
            bool(torch.count_nonzero(w13_lora_b).item()),
            bool(torch.count_nonzero(w2_lora_b).item()),
        )

    if len(lora_b) != 3:
        raise ValueError(f"Expected 2 or 3 routed-expert LoRA B tensors, got {len(lora_b)}")

    w1_lora_b, w2_lora_b, w3_lora_b = lora_b
    w13_enabled = bool(torch.count_nonzero(w1_lora_b).item())
    if w13_num_slices == 2:
        w13_enabled = w13_enabled or bool(torch.count_nonzero(w3_lora_b).item())
    return w13_enabled, bool(torch.count_nonzero(w2_lora_b).item())


def is_moe_lora_active(lora_context) -> bool:
    """Return whether the current batch contains at least one LoRA token."""
    return lora_context is not None and not lora_context.punica_wrapper.no_lora


def _assert_ascend_moe_lora_supported(base_layer: nn.Module) -> None:
    if getattr(base_layer, "use_ep", False):
        raise AssertionError(
            "Ascend MoE LoRA v1 does not support expert parallelism. "
            "Launch with `--enable-expert-parallel=false` and use TP only."
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
        logger.warning_once(
            "Ascend MoE LoRA: shared_experts detected. Routed-expert LoRA "
            "uses the MoE path; shared-expert LoRA uses dense wrappers with "
            "the compatible NPU expand-slice implementation."
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
        adapter_enabled=getattr(
            lora_context,
            "w13_adapter_enabled",
            lora_context.adapter_enabled,
        ),
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
        adapter_enabled=getattr(
            lora_context,
            "w2_adapter_enabled",
            lora_context.adapter_enabled,
        ),
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

    def __init__(self, base_layer: nn.Module) -> None:
        # Skip FusedMoEWithLoRA.__init__: it immediately asserts Triton
        # internals and calls _inject_lora_into_fused_moe which is GPU-only.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        _assert_ascend_moe_lora_supported(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = _get_lora_device(base_layer)
        self._enable_aux_cuda_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
        # vLLM 0.23.0's _build_lora_context reads these attributes. Ascend
        # does not use the CUDA auxiliary-stream implementation.
        self._lora_stream = None
        self._events = None
        self.moe_config = base_layer.moe_config
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        # FusedMoEWithLoRA.__init__ normally creates this field, but Ascend
        # intentionally skips that GPU-kernel constructor. The v0.23 model
        # manager uses it when constructing dummy/warmup MoE adapters.
        self.n_slices = base_layer.local_num_experts * (self._w13_slices + 1)
        # Keep the original module path visible after the FusedMoE layer is
        # replaced by this wrapper. vLLM's model manager continues walking
        # ``<moe>._shared_experts.*`` and wraps those dense projections in
        # place; both references point to the same shared-expert module.
        shared_experts = getattr(base_layer, "_shared_experts", None)
        if shared_experts is not None:
            self._shared_experts = shared_experts

    def create_lora_weights(
        self,
        max_loras,
        lora_config,
        model_config=None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        self.w13_adapter_enabled = torch.zeros_like(self.adapter_enabled)
        self.w2_adapter_enabled = torch.zeros_like(self.adapter_enabled)

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        self.w13_adapter_enabled[index] = 0
        self.w2_adapter_enabled[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        assert isinstance(lora_b, list)
        w13_enabled, w2_enabled = _moe_lora_projection_enabled(
            lora_b,
            self._w13_slices,
        )
        super().set_lora(index, lora_a, lora_b)
        self.w13_adapter_enabled[index] = int(w13_enabled)
        self.w2_adapter_enabled[index] = int(w2_enabled)

    def _build_lora_context(self):
        lora_context = super()._build_lora_context()
        lora_context.w13_adapter_enabled = self.w13_adapter_enabled
        lora_context.w2_adapter_enabled = self.w2_adapter_enabled
        return lora_context

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def set_mapping(self, punica_wrapper):
        # Upstream FusedMoEWithLoRA.set_mapping (vllm v0.22.0+) chains into
        # ``self._moe_kernel.fused_experts.set_lora_context(...)``, but
        # ``_moe_kernel`` is only set by the GPU modular-kernel path that we
        # deliberately skip in __init__. We instead build the per-layer
        # MoELoRAContext (now that punica_wrapper is available) and publish it
        # on the module that ``AscendUnquantizedFusedMoEMethod.apply`` reads via
        # ``getattr(layer, "_ascend_moe_lora_context", None)`` on v0.23.0's
        # FusedMoE layer.
        # The context holds stable references (the in-place-updated LoRA stacks,
        # adapter_enabled and the punica wrapper), so building it once here is
        # sufficient.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
        if getattr(self.base_layer, "_shared_experts", None) is not None:
            punica_wrapper.enable_compatible_lora_bmm_expand_slice()
        self.base_layer.set_lora_context(self._build_lora_context())


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: nn.Module) -> None:
        AscendFusedMoEWithLoRA.__init__(self, base_layer)
        # Override: 3D MoE LoRA uses a single w13 slice.
        self._w13_slices = 1


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
