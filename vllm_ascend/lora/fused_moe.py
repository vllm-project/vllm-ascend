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
"""Ascend MoE LoRA wrapper and routed-token mapping helpers.

The wrapper reuses upstream weight allocation, loading, and TP/EP slicing but
publishes the resulting LoRA context through Ascend's MoERunner pipeline rather
than the GPU modular kernel. Unquantized LoRA keeps the existing AllGather and
AlltoAll implementations. Quantized backends inject deltas at their floating
point GMM boundaries; the first implementation supports W8A8_DYNAMIC with
AllGather TP and AlltoAll EP execution.

Shared experts remain ordinary dense LoRA layers. This module preserves their
module hierarchy and selects a compatible NPU dense expand implementation when
the MoE wrapper is mapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm import envs
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.moe_utils import async_all_to_all

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import LoRAConfig

_MOE_LORA_INDEX_FIELDS = (
    "split_lora_indices",
    "permuted_lora_indices",
    "exchanged_lora_indices",
    "gathered_lora_indices",
)


def _is_unsupported_quant_moe_lora_type(quant_type) -> bool:
    if quant_type is None:
        return False
    from vllm_ascend.lora.quant_moe import has_quant_moe_lora_impl
    from vllm_ascend.quantization.quant_type import QuantType

    return quant_type != QuantType.NONE and not has_quant_moe_lora_impl(quant_type)


def has_lora(lora_context) -> bool:
    """Return whether the current batch contains at least one LoRA token."""
    return lora_context is not None and not lora_context.punica_wrapper.no_lora


def assert_moe_lora_dp1_supported(
    dp_size: int,
    *,
    matched_keys: list[str] | None = None,
) -> None:
    if dp_size > 1:
        matched_hint = f" Matched expert adapter key(s): {matched_keys[:3]}." if matched_keys else ""
        raise NotImplementedError(
            "Ascend MoE LoRA does not support data_parallel_size > 1. "
            "vLLM loads and evicts LoRA adapters independently per DP rank from local "
            "requests, so cross-DP expert routing cannot guarantee that a remote rank's "
            "adapter is resident on the receiving expert rank. "
            "Set data_parallel_size=1 (tensor parallelism and expert parallelism remain supported)."
            f"{matched_hint}"
        )


def _can_wrap_ascend_moe_lora_layer(source_layer: nn.Module) -> bool:
    """Skip routed-expert LoRA wrapping under DP > 1 so attention-only adapters can start."""
    moe_config = getattr(source_layer, "moe_config", None)
    dp_size = getattr(moe_config, "dp_size", 1) if moe_config is not None else 1
    if dp_size > 1:
        logger.warning_once(
            "Ascend MoE LoRA does not support data_parallel_size > 1 (dp_size=%d); "
            "skipping routed-expert LoRA wrapping so attention and shared-expert adapters "
            "can run. Loading adapters with routed-expert weights will be rejected.",
            dp_size,
        )
        return False
    return True


def _assert_replicated_moe_lora_batch(lora_context) -> None:
    if getattr(lora_context, "replicated_batch", True) is False:
        assert_moe_lora_dp1_supported(2)


def _collective_lora_slots(lora_context):
    """Return EP-uniform loaded slot IDs when available, else layer-local slots."""
    ep_slots = getattr(lora_context, "ep_lora_slots", None)
    if ep_slots is not None:
        return ep_slots
    return getattr(lora_context, "layer_lora_slots", None)


def _has_moe_lora_for_slots(lora_context, slots) -> bool:
    if lora_context is None:
        return False
    _assert_replicated_moe_lora_batch(lora_context)
    if slots is not None and not slots:
        return False
    if not has_lora(lora_context):
        return False
    punica_wrapper = getattr(lora_context, "punica_wrapper", None)
    active = getattr(punica_wrapper, "active_lora_slots", None)
    return slots is None or active is None or bool(slots.intersection(active))


def has_moe_lora(lora_context) -> bool:
    """Keep the base kernel when this local rank has no active expert adapter.

    Ascend MoE LoRA requires ``data_parallel_size == 1`` (replicated batches
    across TP/EP ranks). Under partial-expert 2D adapters, an EP rank whose
    local experts have no loaded weights skips local LoRA GEMM deltas while
    still participating in EP-uniform collective decisions via ``ep_lora_slots``.
    """
    if lora_context is None:
        return False
    return _has_moe_lora_for_slots(lora_context, getattr(lora_context, "layer_lora_slots", None))


def has_collective_moe_lora(lora_context) -> bool:
    """Return whether any EP rank holds an active expert adapter for this batch."""
    if lora_context is None:
        return False
    return _has_moe_lora_for_slots(lora_context, _collective_lora_slots(lora_context))


def should_disable_all2all_quant_for_lora(lora_context, quant_type=None) -> bool:
    """Return whether AlltoAll dispatch must stay floating-point for MoE LoRA.

    Every EP rank must agree on whether to issue the extra activation-scale
    AlltoAll exchange. Under ``DP=1``, collective decisions read EP-uniform
    ``ep_lora_slots`` (or fall back to ``layer_lora_slots`` when unset) so
    partial-expert adapters never split quant/collectives across ranks while
    unsupported quant types (for example, MXFP) stay quantized on every rank.

    When ``quant_type`` is supplied and routes through the unfused LoRA
    boundary (``should_use_unfused_quant_moe_for_lora``), dispatch must also
    stay floating-point so base and adapter batches agree across dispatch,
    MLP, empty EP ranks, and combine.
    """
    if lora_context is None:
        return False
    _assert_replicated_moe_lora_batch(lora_context)
    if _is_unsupported_quant_moe_lora_type(quant_type):
        return False
    if quant_type is not None and should_use_unfused_quant_moe_for_lora(lora_context, quant_type):
        return True
    return has_collective_moe_lora(lora_context)


def should_exchange_all2all_lora_indices(lora_context, quant_type=None) -> bool:
    """Return whether AlltoAll dispatch must exchange per-token adapter slots.

    Avoids unnecessary index exchanges on ``DP=1`` base batches that only keep
    the quantized unfused boundary active, while keeping all EP ranks aligned
    via EP-uniform ``ep_lora_slots``.
    """
    if lora_context is None:
        return False
    _assert_replicated_moe_lora_batch(lora_context)
    if _is_unsupported_quant_moe_lora_type(quant_type):
        return False
    return has_collective_moe_lora(lora_context)


def should_gather_allgather_lora_indices(lora_context) -> bool:
    """Return whether AllGather prepare must gather per-token adapter slots.

    Serves ``DP=1`` paths that shard tokens before MoE (sequence parallelism or
    PCP) and ACL graph capture so every rank participates uniformly in the LoRA
    collective.
    """
    if lora_context is None:
        return False
    _assert_replicated_moe_lora_batch(lora_context)
    if _is_lora_graph_capturing():
        return True
    slots = _collective_lora_slots(lora_context)
    return slots is None or bool(slots)


def _is_lora_graph_capturing() -> bool:
    if is_forward_context_available():
        if getattr(get_forward_context(), "capturing", False) is True:
            return True
        return getattr(_EXTRA_CTX, "capturing", False) is True
    # Unit tests may patch _EXTRA_CTX with a plain namespace outside a
    # ForwardContext; reading the real proxy without an active context raises.
    if type(_EXTRA_CTX).__name__ != "_ExtraForwardContextProxy":
        return getattr(_EXTRA_CTX, "capturing", False) is True
    return False


def should_use_unfused_quant_moe_for_lora(lora_context, quant_type) -> bool:
    """Keep quantized MoE on its unfused BF16 boundary for zero-delta parity.

    When an expert adapter is loaded on any EP rank or ACL graph capture is
    running for a LoRA-capable quantized MoE type, all EP ranks stay on the
    unfused quant -> gmm1 -> activation -> quant -> gmm2 BF16 boundary sequence
    for dispatch/combine parity while skipping LoRA delta kernels where inactive.
    """
    if lora_context is None:
        return False
    _assert_replicated_moe_lora_batch(lora_context)
    from vllm_ascend.lora.quant_moe import has_quant_moe_lora_impl

    if not has_quant_moe_lora_impl(quant_type):
        return False
    slots = _collective_lora_slots(lora_context)
    has_loaded_slots = isinstance(slots, (set, frozenset, list, tuple)) and bool(slots)
    return has_collective_moe_lora(lora_context) or has_loaded_slots or _is_lora_graph_capturing()


def reset_lora_indices(lora_context) -> None:
    for field in _MOE_LORA_INDEX_FIELDS:
        if hasattr(lora_context, field):
            delattr(lora_context, field)


def slice_local_lora_indices(
    lora_context,
    *,
    num_tokens: int,
    pad_size: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Slice and right-pad local per-token adapter slots to match routed tokens.

    Pads inactive/padding positions with ``-1`` (never ``0``, which selects
    adapter slot 0).
    """
    _assert_replicated_moe_lora_batch(lora_context)
    punica_wrapper = getattr(lora_context, "punica_wrapper", None)
    token_indices = getattr(punica_wrapper, "token_lora_indices", None)
    slots = _collective_lora_slots(lora_context)
    total_tokens = max(num_tokens, 0) + max(pad_size, 0)
    target_device = (
        device
        if device is not None
        else (token_indices.device if isinstance(token_indices, torch.Tensor) else torch.device("cpu"))
    )
    if not _is_lora_graph_capturing() and (
        token_indices is None or getattr(punica_wrapper, "no_lora", False) is True or (slots is not None and not slots)
    ):
        return torch.full((total_tokens,), -1, dtype=torch.long, device=target_device)

    assert token_indices is not None
    if token_indices.numel() < num_tokens:
        token_indices = torch.nn.functional.pad(
            token_indices,
            (0, num_tokens - token_indices.numel()),
            value=-1,
        )
    else:
        token_indices = token_indices[:num_tokens]
    if pad_size > 0:
        token_indices = torch.nn.functional.pad(token_indices, (0, pad_size), value=-1)
    if token_indices.dtype != torch.long:
        token_indices = token_indices.to(torch.long)
    if device is not None and token_indices.device != device:
        token_indices = token_indices.to(device)
    return token_indices


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
    token_indices = slice_local_lora_indices(
        lora_context,
        num_tokens=num_tokens,
        pad_size=pad_size,
    )
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
    if split_indices.numel() == 0 or reversed_permutation_mapping.numel() == 0:
        lora_context.permuted_lora_indices = split_indices.new_empty((0,))
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
    if exchanged.numel() == 0 or reversed_permutation_mapping.numel() == 0:
        lora_context.exchanged_lora_indices = exchanged.reshape(-1)
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
    moe_config = getattr(base_layer, "moe_config", None)
    if moe_config is not None:
        assert_moe_lora_dp1_supported(getattr(moe_config, "dp_size", 1))
    if getattr(base_layer, "dynamic_eplb", False):
        raise AssertionError(
            "Ascend MoE LoRA is incompatible with dynamic EPLB "
            "(expert migration would break the per-expert LoRA layout)."
        )
    if get_ascend_config().enable_fused_mc2 != 0:
        raise AssertionError(
            "Ascend MoE LoRA cannot patch FusedMC2 path "
            "(dispatch_ffn_combine/mega_moe is a single fused C++ op). "
            "Set additional_config.enable_fused_mc2 to 0."
        )
    if getattr(base_layer, "_shared_experts", None) is not None:
        logger.warning_once(
            "Ascend MoE LoRA: shared_experts detected. Routed-expert LoRA "
            "uses the MoE path; shared-expert LoRA uses dense wrappers with "
            "the compatible NPU expand-slice implementation."
        )


def _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids):
    """Recover per-permuted-row (expert_id, lora_slot) for the dispatched rows.

    npu_moe_init_routing semantics (verified empirically): ``expanded_row_idx``
    is indexed by the ORIGINAL flat (token, k) position and gives where that
    pair landed in the expert-sorted array -- or ``-1`` when its expert falls
    outside the rank's ``active_expert_range`` under EP. Sending ``-1`` past
    all valid indices keeps inactive tail rows from colliding with sorted row 1
    while keeping ``argsort`` output shape static and graph-capturable. Under
    sequence-parallel or PCP prepare paths, looks up ``gathered_lora_indices``
    aligned with the gathered tokens rather than the rank-local Punica array.
    """
    _assert_replicated_moe_lora_batch(lora_context)
    top_k = lora_context.top_k
    flat_expanded = expanded_row_idx.reshape(-1)
    num_expanded = flat_expanded.numel()
    expanded = torch.where(
        flat_expanded >= 0,
        flat_expanded,
        torch.full_like(flat_expanded, num_expanded),
    ).to(torch.float32)
    inv_perm = torch.argsort(expanded)
    active_row = flat_expanded[inv_perm] >= 0
    expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.long)
    if getattr(lora_context, "use_ep", False):
        ep_rank = getattr(lora_context, "ep_rank", 0)
        local_num_experts = getattr(lora_context, "local_num_experts", 0)
        if ep_rank and local_num_experts:
            expert_per_row = expert_per_row - ep_rank * local_num_experts

    gathered_lora_indices = getattr(lora_context, "gathered_lora_indices", None)
    if gathered_lora_indices is not None:
        token_lora_indices = gathered_lora_indices.reshape(-1).to(torch.long)
    else:
        # token_lora_indices is a 1D LongTensor sized to max_num_batched_tokens
        # (host-known constant). Clamping defensively to the last index is a
        # no-op in normal operation and keeps the gather graph-safe.
        token_lora_indices = lora_context.punica_wrapper.token_lora_indices

    orig_token = inv_perm // top_k
    if token_lora_indices.numel() == 0:
        lora_per_row = torch.full_like(orig_token, -1, dtype=torch.long)
    else:
        orig_token = orig_token.clamp(max=token_lora_indices.numel() - 1)
        lora_per_row = token_lora_indices[orig_token]
    lora_per_row = torch.where(active_row, lora_per_row, torch.full_like(lora_per_row, -1))
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
    _assert_replicated_moe_lora_batch(lora_context)
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
    # kernel crashes with empty tensors, but still clear per-forward indices.
    try:
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
    finally:
        # Clear per-forward intermediate indices now that the LoRA delta
        # for this layer has been fully applied — they are not needed for
        # the remaining combine/finalize stages.
        reset_lora_indices(lora_context)


def _describe_moe_lora_tensors(tensors) -> str:
    if isinstance(tensors, torch.Tensor):
        return str(tuple(tensors.shape))
    if isinstance(tensors, (list, tuple)):
        return str([tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in tensors])
    return repr(type(tensors).__name__)


def _format_moe_lora_shape_error(layer: nn.Module, index: int, lora_a, lora_b, err: Exception) -> str:
    module_name = getattr(layer.base_layer, "prefix", "") or type(layer).__name__
    expected_w13_a = [tuple(t.shape[-3:]) for t in getattr(layer, "w13_lora_a_stacked", ())]
    expected_w13_b = [tuple(t.shape[-3:]) for t in getattr(layer, "w13_lora_b_stacked", ())]
    expected_w2_a = [tuple(t.shape[-3:]) for t in getattr(layer, "w2_lora_a_stacked", ())]
    expected_w2_b = [tuple(t.shape[-3:]) for t in getattr(layer, "w2_lora_b_stacked", ())]
    return (
        f"Failed to load MoE LoRA weights for module '{module_name}' at slot {index}: "
        f"got lora_a={_describe_moe_lora_tensors(lora_a)}, "
        f"lora_b={_describe_moe_lora_tensors(lora_b)}; "
        f"expected local expert slice shapes w13_a={expected_w13_a}, w13_b={expected_w13_b}, "
        f"w2_a={expected_w2_a}, w2_b={expected_w2_b}. "
        f"Verify that the adapter's hidden/intermediate dimensions match the base model. "
        f"Original error: {err}"
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
        # _build_lora_context is inherited from vLLM, whose GPU constructor
        # normally initializes these fields. Ascend deliberately skips it.
        self._lora_stream = None
        self._events = None
        self.enable_moe_shared_loras = False
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        # Mirrors per-(lora_id) layout of `self.lora_a_stacked` (built in
        # `create_lora_weights`) so `create_dummy_lora`'s n_slices fallback
        # matches `lora_a_stacked` length under EP.
        self.n_slices = self.local_num_experts * (self._w13_slices + 1)
        self._lora_slots: set[int] = set()
        self._ep_lora_slots: set[int] = set()
        self._in_set_lora: bool = False
        # Preserve the model-manager-visible module path used to discover and
        # wrap shared_experts.{gate_up,down}_proj as ordinary dense LoRA.
        shared_experts = getattr(base_layer, "_shared_experts", None)
        if shared_experts is not None:
            self._shared_experts = shared_experts

    def _sync_ep_lora_slot(self, index: int, local_has_slot: bool) -> None:
        """Align EP-wide loaded slot presence across ranks with partial experts."""
        ep_slots = getattr(self, "_ep_lora_slots", None)
        if ep_slots is None:
            ep_slots = set()
            self._ep_lora_slots = ep_slots
        any_has_slot = local_has_slot
        if getattr(self, "use_ep", False) and torch.distributed.is_initialized():
            try:
                ep_group = get_ep_group()
            except (AssertionError, RuntimeError):
                ep_group = None
            if ep_group is not None and getattr(ep_group, "world_size", 1) > 1:
                cpu_group = getattr(ep_group, "cpu_group", None)
                device_group = getattr(ep_group, "device_group", None)
                if cpu_group is not None:
                    flag = torch.tensor([1 if local_has_slot else 0], dtype=torch.int32, device="cpu")
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=cpu_group)
                    any_has_slot = bool(flag.item())
                elif device_group is not None:
                    flag = torch.tensor([1 if local_has_slot else 0], dtype=torch.int32, device=self.device)
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=device_group)
                    any_has_slot = bool(flag.item())
                elif callable(getattr(ep_group, "all_reduce", None)):
                    flag = torch.tensor([1 if local_has_slot else 0], dtype=torch.int32, device=self.device)
                    reduced = ep_group.all_reduce(flag)
                    if isinstance(reduced, torch.Tensor):
                        flag = reduced
                    any_has_slot = bool(flag.item())
        if any_has_slot:
            ep_slots.add(index)
        else:
            ep_slots.discard(index)

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        lora_slots = getattr(self, "_lora_slots", None)
        if lora_slots is not None:
            lora_slots.discard(index)
        # Upstream FusedMoEWithLoRA.set_lora internally invokes reset_lora(index);
        # suppress the inner sync so set_lora and reset_lora each enter once.
        if not getattr(self, "_in_set_lora", False):
            self._sync_ep_lora_slot(index, False)

    def set_lora(self, index: int, lora_a, lora_b) -> None:
        prev = getattr(self, "_in_set_lora", False)
        self._in_set_lora = True
        try:
            try:
                super().set_lora(index, lora_a, lora_b)
            except RuntimeError as err:
                raise ValueError(_format_moe_lora_shape_error(self, index, lora_a, lora_b, err)) from err
            lora_slots = getattr(self, "_lora_slots", None)
            if lora_slots is None:
                self._lora_slots = {index}
            else:
                lora_slots.add(index)
        finally:
            self._in_set_lora = prev
        if not prev:
            self._sync_ep_lora_slot(index, True)

    def _build_lora_context(self):
        lora_context = super()._build_lora_context()
        lora_context.use_ep = self.use_ep
        lora_context.ep_rank = self.moe_config.moe_parallel_config.ep_rank
        lora_context.layer_lora_slots = self._lora_slots
        lora_context.ep_lora_slots = self._ep_lora_slots
        lora_context.replicated_batch = self.moe_config.dp_size == 1
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
        # ``getattr(layer, "_ascend_moe_lora_context", None)``
        # Build the per-layer MoELoRAContext once punica_wrapper is available and
        # publish it through the Ascend MoE runner. The runner stores it on
        # routed_experts; batch-local LoRA indices are refreshed before each forward.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
        self.base_layer.set_lora_context(self._build_lora_context())

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        if not FusedMoEWithLoRA.can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            return False
        return _can_wrap_ascend_moe_lora_layer(source_layer)


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: nn.Module) -> None:
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
        if not FusedMoE3DWithLoRA.can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            return False
        return _can_wrap_ascend_moe_lora_layer(source_layer)


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
