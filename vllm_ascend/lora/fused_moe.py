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
"""Ascend integration for vLLM's fused MoE LoRA layers."""

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


def _moe_lora_projection_enabled(lora_b: list[torch.Tensor], w13_num_slices: int) -> tuple[bool, bool]:
    """Return whether routed w13 and w2 have any non-zero B weights."""
    if len(lora_b) == 2:
        w13_lora_b, w2_lora_b = lora_b
        return bool(torch.count_nonzero(w13_lora_b).item()), bool(torch.count_nonzero(w2_lora_b).item())

    if len(lora_b) != 3:
        raise ValueError(f"Expected 2 or 3 routed-expert LoRA B tensors, got {len(lora_b)}")

    w1_lora_b, w2_lora_b, w3_lora_b = lora_b
    w13_enabled = bool(torch.count_nonzero(w1_lora_b).item())
    if w13_num_slices == 2:
        w13_enabled = w13_enabled or bool(torch.count_nonzero(w3_lora_b).item())
    return w13_enabled, bool(torch.count_nonzero(w2_lora_b).item())


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
    """Build the local per-token LoRA indices for MoE dispatch."""
    token_indices = lora_context.punica_wrapper.token_lora_indices
    token_indices = token_indices[:num_tokens]
    if pad_size > 0:
        token_indices = torch.nn.functional.pad(token_indices, (0, pad_size), value=-1)
    if tp_size > 1:
        lora_context.split_lora_indices = torch.tensor_split(token_indices, tp_size, dim=0)[tp_rank]
    else:
        lora_context.split_lora_indices = token_indices


def preprocess_lora_indices(
    lora_context,
    *,
    topk_ids: torch.Tensor,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Align LoRA indices with AlltoAll-dispatched token rows."""
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
    """Align exchanged LoRA indices with post-dispatch token rows."""
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
    """Exchange LoRA indices with the activation AlltoAll split sizes."""
    permuted = getattr(lora_context, "permuted_lora_indices", None)
    if permuted is None:
        return
    lora_dtype = permuted.dtype
    _, exchanged, handle = async_all_to_all(permuted, output_splits, input_splits, ep_group)
    handle.wait()
    lora_context.exchanged_lora_indices = exchanged.to(lora_dtype)


def sync_lora_context(quant_method, lora_context):
    """Update the active LoRA context on initialized MoE communicators."""
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
            "(dispatch_ffn_combine/mega_moe is a single fused C++ op). "
            "Set VLLM_ASCEND_ENABLE_FUSED_MC2=0."
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
    """Recover expert and LoRA slots for AlltoAll-dispatched rows."""
    num_local_experts = lora_context.local_num_experts
    exchanged_lora_indices = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged_lora_indices is None:
        raise AssertionError("AlltoAll MoE LoRA requires exchanged_lora_indices in lora_context.")

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
    """Add the routed-expert gate/up LoRA delta before activation."""
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
        adapter_enabled=getattr(lora_context, "w13_adapter_enabled", lora_context.adapter_enabled),
        token_lora_mapping=lora_per_row,
    )


def moe_lora_apply_w2(lora_context, *, down_out, silu_out, lora_routing):
    """Add the routed-expert down LoRA delta after the down GMM."""
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
        adapter_enabled=getattr(lora_context, "w2_adapter_enabled", lora_context.adapter_enabled),
        token_lora_mapping=lora_per_row,
    )
    reset_lora_indices(lora_context)


class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Fused MoE LoRA wrapper for the Ascend MoE runner."""

    def __init__(self, base_layer: nn.Module) -> None:
        # Skip FusedMoEWithLoRA.__init__: it immediately asserts Triton
        # internals and calls _inject_lora_into_fused_moe which is GPU-only.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        _assert_ascend_moe_lora_supported(base_layer)
        self.moe_config = base_layer.moe_config
        self._shared_experts = base_layer._shared_experts
        # Match upstream FusedMoEWithLoRA: EP collapses the MoE TP dimension
        # to one and shards experts across the original TP group.  Using the
        # global TP rank/size here would incorrectly TP-slice every local
        # expert's LoRA weights a second time.
        moe_parallel_config = self.moe_config.moe_parallel_config
        self.tp_size = moe_parallel_config.tp_size
        self.tp_rank = moe_parallel_config.tp_rank
        self.device = _get_lora_device(base_layer)
        self._enable_aux_cuda_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
        # State normally initialized by the skipped GPU-only constructor.
        self._lora_stream = None
        self._events = None
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        self.enable_moe_shared_loras = False
        # Mirrors per-(lora_id) layout of `self.lora_a_stacked` (built in
        # `create_lora_weights`) so `create_dummy_lora`'s n_slices fallback
        # matches `lora_a_stacked` length under EP.
        self.n_slices = self.local_num_experts * (self._w13_slices + 1)

    def create_lora_weights(self, max_loras, lora_config, model_config=None) -> None:
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
        w13_enabled, w2_enabled = _moe_lora_projection_enabled(lora_b, self._w13_slices)
        super().set_lora(index, lora_a, lora_b)
        self.w13_adapter_enabled[index] = int(w13_enabled)
        self.w2_adapter_enabled[index] = int(w2_enabled)

    def _build_lora_context(self):
        lora_context = super()._build_lora_context()
        lora_context.w13_adapter_enabled = self.w13_adapter_enabled
        lora_context.w2_adapter_enabled = self.w2_adapter_enabled
        return lora_context

    def set_mapping(self, punica_wrapper):
        # The upstream implementation publishes this context through its
        # Triton kernel. Ascend keeps it on the backend MoE layer instead.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
        self.base_layer.set_lora_context(self._build_lora_context())


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: nn.Module) -> None:
        AscendFusedMoEWithLoRA.__init__(self, base_layer)
        # Override: 3D MoE LoRA uses a single w13 slice.
        self._w13_slices = 1
        self.n_slices = self.local_num_experts * (self._w13_slices + 1)


# vLLM's dummy adapter packer selects fused MoE handling by class name.
AscendFusedMoEWithLoRA.__name__ = "FusedMoEWithLoRA"
AscendFusedMoE3DWithLoRA.__name__ = "FusedMoE3DWithLoRA"
