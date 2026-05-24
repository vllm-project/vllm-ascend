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
from vllm.logger import init_logger
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpComputeInput

logger = init_logger(__name__)


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
            "calls on multiple streams, which breaks the bracketed "
            "comm._apply_mlp swap. Disable it for MoE LoRA."
        )


class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-native MoE-LoRA wrapper.

    Reuses upstream weight allocation, set_lora, reset_lora, and slicing.
    Overrides only the injection mechanism (`_inject_lora_into_fused_moe`
    is bypassed; we wrap `quant_method.apply` instead).
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
        # Per-layer scratch for state captured at apply-time and consumed
        # inside _apply_mlp_with_lora.
        self._moe_state: dict = {}
        self._inject_lora_into_ascend_fused_moe()

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def _inject_lora_into_ascend_fused_moe(self) -> None:
        """Patch this layer's quant_method.apply to bracket-swap _apply_mlp.

        Bound-method idiom: we replace `quant_method.apply` with a bound
        method that captures `self` (the LoRA wrapper) so each of the 48
        MoE layers has its own wrapper carrying its own stacked LoRA
        weights. The wrapped function does:

            comm = _EXTRA_CTX.moe_comm_method  # picked per-forward
            orig_mlp = comm._apply_mlp
            try:
                comm._apply_mlp = our LoRA-aware version
                return orig_apply(...)         # base path runs as usual,
                                               # _apply_mlp call goes through us
            finally:
                comm._apply_mlp = orig_mlp     # always restore

        This guarantees the swap is strictly bracketed within a single
        layer's forward pass.
        """
        quant_method = self.base_layer.quant_method
        orig_apply = quant_method.apply
        self_ref = self

        def apply_wrapper(qm_self, layer, x, *args, **kwargs):
            comm = _EXTRA_CTX.moe_comm_method
            if comm is None:
                # Without a comm method we cannot reach _apply_mlp; let the
                # base apply run and skip LoRA. This shouldn't happen in
                # practice because ascend_forward_context sets it per fwd.
                return orig_apply(layer, x, *args, **kwargs)
            orig_mlp = comm._apply_mlp
            self_ref._moe_state["expert_map"] = kwargs.get("expert_map")
            try:
                comm._apply_mlp = (
                    lambda mlp_input: self_ref._apply_mlp_with_lora(
                        orig_mlp, mlp_input
                    )
                )
                return orig_apply(layer, x, *args, **kwargs)
            finally:
                comm._apply_mlp = orig_mlp

        # Bind as instance attribute on quant_method so each layer has its own.
        # We cannot use __get__ because orig_apply already is a bound method;
        # storing the function directly works because Python looks up instance
        # attrs before class attrs.
        quant_method.apply = apply_wrapper.__get__(
            quant_method, type(quant_method)
        )

    # ------------------------------------------------------------------
    # LoRA-aware MLP
    # ------------------------------------------------------------------
    def _apply_mlp_with_lora(self, orig_mlp, mlp_input: MoEMlpComputeInput):
        """LoRA-aware replacement for MoECommMethod._apply_mlp.

        v1 supports only the unquant + AllGather (expanded_row_idx present)
        path. Any other path falls back to the base implementation so the
        forward still produces (non-LoRA-augmented) output.
        """
        if mlp_input.quant.is_quant:
            logger.warning_once(
                "Ascend MoE LoRA on quantized path is not implemented; "
                "running base path only (LoRA delta will be skipped)."
            )
            return orig_mlp(mlp_input)
        if mlp_input.expanded_row_idx is None:
            logger.warning_once(
                "Ascend MoE LoRA requires AllGather comm method "
                "(combine_metadata.expanded_row_idx); current comm method "
                "does not provide it. Skipping LoRA delta."
            )
            return orig_mlp(mlp_input)
        if mlp_input.topk_ids is None:
            logger.warning_once(
                "Ascend MoE LoRA: topk_ids unavailable in MoEMlpComputeInput; "
                "skipping LoRA delta."
            )
            return orig_mlp(mlp_input)

        # Local imports keep the GPU-only test environment importable.
        import torch_npu

        h = mlp_input.hidden_states  # [N_perm, hidden_in]
        gl = mlp_input.group_list
        glt = mlp_input.group_list_type
        w1 = mlp_input.weights.w1
        w2 = mlp_input.weights.w2
        need_trans = mlp_input.need_trans
        if need_trans:
            # process_weights_after_loading stores w1/w2 already transposed
            # to [num_experts, in, out]; only the legacy unquant path with
            # need_trans=True flips back.
            w1 = w1.transpose(1, 2)
            w2 = w2.transpose(1, 2)

        # ---- per-permuted-row expert_id (1D, length N_perm) ----
        # gl semantics:
        #   glt == 1: counts per expert (length local_E)
        #   glt == 0: cumulative counts (length local_E)
        if glt == 1:
            expert_per_row = torch.repeat_interleave(
                torch.arange(gl.numel(), device=gl.device, dtype=torch.long),
                gl.to(torch.long),
            )
        else:
            counts = torch.cat([gl[:1], gl[1:] - gl[:-1]])
            expert_per_row = torch.repeat_interleave(
                torch.arange(counts.numel(), device=gl.device, dtype=torch.long),
                counts.to(torch.long),
            )

        # ---- per-permuted-row lora slot (1D, length N_perm) ----
        # expanded_row_idx[i] encodes orig_token*top_k + k; orig_token = //top_k.
        # token_lora_indices is a 1D LongTensor on device.
        top_k = self.base_layer.top_k
        orig_token = torch.abs(mlp_input.expanded_row_idx) // top_k
        token_lora_indices = self.punica_wrapper.token_lora_indices
        # Guard against truncation (token_lora_indices is shaped to num_tokens
        # at update_metadata time; orig_token must be in range).
        if orig_token.numel() > 0 and int(orig_token.max().item()) >= \
                token_lora_indices.numel():
            logger.warning_once(
                "orig_token index %d exceeds token_lora_indices size %d; "
                "falling back to base path.",
                int(orig_token.max().item()), token_lora_indices.numel(),
            )
            return orig_mlp(mlp_input)
        lora_per_row = token_lora_indices[orig_token]

        # === Stage 1: gate_up GMM (base) ===
        gate_up = torch_npu.npu_grouped_matmul(
            x=[h],
            weight=[w1],
            split_item=2,
            group_list_type=glt,
            group_type=0,
            group_list=gl,
        )[0]  # [N_perm, 2*inter] (or [N_perm, inter] when _w13_slices==1)

        # === Stage 2: LoRA delta for w13 ===
        self.punica_wrapper.add_lora_fused_moe(
            y=gate_up,
            x=h,
            lora_a_stacked=self.w13_lora_a_stacked,
            lora_b_stacked=self.w13_lora_b_stacked,
            topk_weights=None,
            sorted_token_ids=None,
            expert_ids=expert_per_row,
            num_tokens_post_padded=None,
            max_lora_rank=self.w13_lora_a_stacked[0].shape[-2],
            top_k_num=1,
            shrink_config={},
            expand_config={},
            adapter_enabled=self.adapter_enabled,
            mul_routed_weight=False,
            fully_sharded=self.fully_sharded,
            offset=0,
            token_lora_mapping=lora_per_row,
        )

        # === Stage 3: activation (SiLU / SwiGLU) ===
        if mlp_input.activation == "swigluoai":
            silu_out = AscendSwigluOAIAndMul.swiglu_oai_forward(
                gate_up.view(-1, gate_up.shape[-1])
            )
        else:
            silu_out = torch_npu.npu_swiglu(gate_up)
        if mlp_input.topk_scales is not None:
            silu_out = silu_out * mlp_input.topk_scales

        # === Stage 4: down GMM (base) ===
        out = torch_npu.npu_grouped_matmul(
            x=[silu_out],
            weight=[w2],
            split_item=2,
            group_list_type=glt,
            group_type=0,
            group_list=gl,
        )[0]  # [N_perm, hidden_out]

        # === Stage 5: LoRA delta for w2 ===
        self.punica_wrapper.add_lora_fused_moe(
            y=out,
            x=silu_out,
            lora_a_stacked=self.w2_lora_a_stacked,
            lora_b_stacked=self.w2_lora_b_stacked,
            topk_weights=None,
            sorted_token_ids=None,
            expert_ids=expert_per_row,
            num_tokens_post_padded=None,
            max_lora_rank=self.w2_lora_a_stacked[0].shape[-2],
            top_k_num=1,
            shrink_config={},
            expand_config={},
            adapter_enabled=self.adapter_enabled,
            mul_routed_weight=False,
            fully_sharded=self.fully_sharded,
            offset=0,
            token_lora_mapping=lora_per_row,
        )
        return out

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
        return (
            isinstance(source_layer, AscendFusedMoE)
            and len(packed_modules_list) == 2
        )


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
        return (
            isinstance(source_layer, AscendFusedMoE)
            and len(packed_modules_list) == 1
        )


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
