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
from vllm import envs
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all

_MOE_LORA_INDEX_FIELDS = (
    "split_lora_indices",
    "permuted_lora_indices",
    "exchanged_lora_indices",
)


def reset_lora_indices(lora_context) -> None:
    for field in _MOE_LORA_INDEX_FIELDS:
        if hasattr(lora_context, field):
            delattr(lora_context, field)


def prepare_lora_indices(
    lora_context,
    *,
    num_tokens: int,
    pad_size: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Truncate, pad, and TP-split the per-token LoRA index tensor,
    storing the result in ``lora_context.split_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None``.
    """
    token_indices = lora_context.punica_wrapper.token_lora_indices
    token_indices = token_indices[:num_tokens]
    if pad_size > 0:
        token_indices = torch.nn.functional.pad(token_indices, (0, pad_size), value=-1)
    if tp_size > 1:
        lora_context.split_lora_indices = torch.tensor_split(token_indices, tp_size, dim=0)[tp_rank]
    else:
        # use ep for dp without tp.
        lora_context.split_lora_indices = token_indices


def preprocess_lora_indices(
    lora_context,
    *,
    topk_ids: torch.Tensor,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Expand and permute LoRA token indices for the AlltoAll dispatch path.

    Reads ``lora_context.split_lora_indices``, repeats each entry by
    ``topk``, applies the token permutation to align with the dispatched
    hidden states, and stores the result in
    ``lora_context.permuted_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``split_lora_indices`` has been populated.
    """
    split_indices = getattr(lora_context, "split_lora_indices", None)
    if split_indices is None:
        return
    expanded = split_indices.repeat_interleave(topk_ids.shape[1])
    permutation = torch.argsort(reversed_permutation_mapping.reshape(-1).long())
    lora_context.permuted_lora_indices = expanded[permutation]


def postprocess_lora_indices(
    lora_context,
    *,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Re-permute exchanged LoRA indices to align with the global token
    ordering after ``npu_moe_token_permute`` in the AlltoAll dispatch
    postprocess.

    Reads ``lora_context.exchanged_lora_indices``, applies the
    global permutation, and writes the result back.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``exchanged_lora_indices`` has been populated.
    """
    exchanged = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged is None:
        return
    permutation = torch.argsort(reversed_permutation_mapping.reshape(-1).long())
    lora_context.exchanged_lora_indices = exchanged[permutation]


def all2all_lora_indices(
    lora_context,
    *,
    output_splits,
    input_splits,
    ep_group,
) -> None:
    """Exchange permuted LoRA indices across EP ranks via all_to_all.

    Reads ``lora_context.permuted_lora_indices``, performs the all_to_all
    exchange with the given splits and group, and stores the result in
    ``lora_context.exchanged_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``permuted_lora_indices`` has been populated.
    """
    permuted = getattr(lora_context, "permuted_lora_indices", None)
    if permuted is None:
        return
    lora_dtype = permuted.dtype
    _, exchanged, handle = async_all_to_all(permuted, output_splits, input_splits, ep_group)
    handle.wait()
    lora_context.exchanged_lora_indices = exchanged.to(lora_dtype)


def sync_lora_context(quant_method, lora_context):
    """Push ``lora_context`` onto MoE communication singletons, or clear
    them when ``lora_context`` is ``None``.

    Encapsulates the ``hasattr``/``set_lora_context`` pattern shared by
    setup and teardown so callers just pass the target value.
    """
    if hasattr(_EXTRA_CTX.moe_comm_method, "set_lora_context"):
        _EXTRA_CTX.moe_comm_method.set_lora_context(lora_context)
    if hasattr(quant_method, "set_lora_context"):
        quant_method.set_lora_context(lora_context)


def _assert_ascend_moe_lora_supported(base_layer: nn.Module) -> None:
    if getattr(base_layer, "dynamic_eplb", False):
        raise AssertionError(
            "Ascend MoE LoRA is incompatible with dynamic EPLB "
            "(expert migration would break the per-expert LoRA layout)."
        )
    if int(envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2) != 0:
        raise AssertionError(
            "Ascend MoE LoRA cannot patch FusedMC2 path "
            "(dispatch_ffn_combine is a single fused C++ op). "
            "Set VLLM_ASCEND_ENABLE_FUSED_MC2=0."
        )
    if getattr(base_layer, "_shared_experts", None) is not None:
        raise AssertionError(
            "Ascend MoE LoRA does not wrap the shared_experts path "
            "(it runs outside quant_method.apply). The target model "
            "Qwen3-30B-A3B-Thinking-2507 has no shared experts; models "
            "like DeepSeek-V3 are not yet supported."
        )


def _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids):
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


def _recover_moe_lora_routing_all2all(
    lora_context,
    group_list: torch.Tensor,
):
    """Recover per-row (expert_id, lora_id) for the AlltoAll dispatched tokens.

    In the AlltoAll + EP path, tokens have already been exchanged via
    all_to_all and sorted by local expert.  ``group_list`` tells us how
    many tokens belong to each local expert. The LoRA indices for those
    dispatched rows are stored on ``lora_context.exchanged_lora_indices``.

    Returns:
        expert_per_row: [num_dispatched_tokens] local expert id (0..E-1)
        lora_per_row:   [num_dispatched_tokens] lora adapter id (-1 = none)
    """
    num_local_experts = lora_context.local_num_experts
    exchanged_lora_indices = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged_lora_indices is None:
        raise AssertionError("AlltoAll MoE LoRA requires exchanged_lora_indices in lora_context.")

    # Build per-token expert IDs
    expert_per_row = torch.repeat_interleave(
        torch.arange(num_local_experts, device=group_list.device),
        group_list,
    )

    lora_per_row = exchanged_lora_indices.reshape(-1).to(torch.long)
    if expert_per_row.numel() != lora_per_row.numel():
        raise AssertionError(
            "AlltoAll MoE LoRA routing metadata is misaligned: "
            f"group_list describes {expert_per_row.numel()} rows, but "
            f"received {lora_per_row.numel()} LoRA indices."
        )

    return expert_per_row, lora_per_row


def moe_lora_apply_w13(lora_context, *, gate_up_out, hidden_states, lora_routing):
    """Add the w13 LoRA delta into ``gate_up_out`` (in place), before activation.

    Called from ``unquant_apply_mlp`` right after the base gate_up GMM.

    Args:
        lora_routing: (expert_per_row, lora_per_row) pre-computed by the
            caller via _recover_moe_lora_routing (AllGather) or
            _recover_moe_lora_routing_all2all (AlltoAll).
    """
    expert_per_row, lora_per_row = lora_routing
    # EP rank may receive 0 dispatched tokens when all tokens route to
    # experts on other ranks. Skip LoRA to avoid passing empty tensors
    # to add_lora_fused_moe (which can trigger NPU kernel crashes).
    if expert_per_row.numel() == 0:
        return
    lora_context.punica_wrapper.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        token_lora_mapping=lora_per_row,
    )


def moe_lora_apply_w2(lora_context, *, down_out, silu_out, lora_routing):
    """Add the w2 LoRA delta into ``down_out`` (in place), after the down GMM.

    Reuses the per-row routing computed by ``moe_lora_apply_w13``; ``silu_out``
    is the activation output that fed the base down GMM.
    """
    expert_per_row, lora_per_row = lora_routing
    # EP rank may receive 0 dispatched tokens; skip LoRA to avoid NPU
    # kernel crashes with empty tensors.
    if expert_per_row.numel() == 0:
        return
    lora_context.punica_wrapper.add_lora_fused_moe(
        y=down_out,
        x=silu_out,
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        token_lora_mapping=lora_per_row,
    )
    # Clear per-forward intermediate indices now that the LoRA delta
    # for this layer has been fully applied — they are not needed for
    # the remaining combine/finalize stages.
    reset_lora_indices(lora_context)


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
        self.moe_config = base_layer.moe_config
        # Match upstream FusedMoEWithLoRA: EP collapses the MoE TP dimension
        # to one and shards experts across the original TP group.  Using the
        # global TP rank/size here would incorrectly TP-slice every local
        # expert's LoRA weights a second time.
        moe_parallel_config = self.moe_config.moe_parallel_config
        self.tp_size = moe_parallel_config.tp_size
        self.tp_rank = moe_parallel_config.tp_rank
        self.device = _get_lora_device(base_layer)
        self._enable_aux_cuda_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        # Mirrors per-(lora_id) layout of `self.lora_a_stacked` (built in
        # `create_lora_weights`) so `create_dummy_lora`'s n_slices fallback
        # matches `lora_a_stacked` length under EP.
        self.n_slices = self.local_num_experts * (self._w13_slices + 1)

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
        # ``getattr(layer, "_ascend_moe_lora_context", None)``
        # Build the per-layer MoELoRAContext once punica_wrapper is available and
        # publish it through the Ascend MoE runner. The runner stores it on
        # routed_experts; batch-local LoRA indices are refreshed before each forward.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
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
