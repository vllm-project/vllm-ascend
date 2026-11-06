# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External Mega-KDA decode operator wrapper for Kimi K3.

``mega_recurrent_kda`` (external ``ops`` package) fuses the KDA input
projections, causal conv1d update, recurrent KDA, gated output norm and
output projection into a single launch.  The 20-argument ABI reads BF16
``[out, in]`` weights pre-cast to NZ format and a
``[3, heads, conv_size, head_dim]`` conv weight; state indices and
accepted-token counts are int32 and ``ssm_state_indices`` is flattened to
``[batch * spec_len]``. The first Verify round after prefill uses ``conv=S``;
the recurrent count always comes from scheduler metadata, as in PR #6.
"""

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature

import torch
import torch_npu
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

MEGA_KDA_NZ_FORMAT = 29
MEGA_KDA_ABI_POSITIONAL_ARGS = 20
MEGA_KDA_CACHE_ALIGNMENT_BYTES = 16


class MegaKDAFirstVerifyTracker:
    """Request-local history indexed by the actual Mamba cache block.

    This belongs only to the Kimi Mega metadata group. Prefill resets include
    single-token chunks; repeated builds at the same context are idempotent.
    No cache payload or device scalar is copied back to the host.
    """

    def __init__(self, num_slots: int, device: torch.device) -> None:
        self.context = torch.full((num_slots,), -1, dtype=torch.int64, device=device)
        self.first = torch.ones(num_slots, dtype=torch.int32, device=device)

    def update(
        self,
        slots: torch.Tensor,
        context: torch.Tensor,
        is_prefilling: torch.Tensor,
        spec_mask: torch.Tensor,
    ) -> torch.Tensor:
        slots = slots.long()
        valid = (slots >= 0) & (slots != NULL_BLOCK_ID) & (slots < self.context.numel())
        safe_slots = torch.where(valid, slots, 0)
        previous = self.context[safe_slots]
        # A prefill always starts a new verification epoch, even when its
        # length is one or a slot previously belonged to another request.
        previous = torch.where(is_prefilling, -1, previous)
        first = torch.where(
            context == previous,
            self.first[safe_slots],
            ((previous < 0) | (context < previous)).to(torch.int32),
        )
        # A no-draft decode also advances the Conv history. Returning to
        # Verify afterwards must use the previous accepted count, not S.
        first = torch.where(spec_mask, first, 0)
        update = valid & ~is_prefilling
        next_context = torch.where(update, context, previous)
        next_first = torch.where(update, first, self.first[safe_slots])
        self.context.scatter_(0, safe_slots, next_context)
        self.first.scatter_(0, safe_slots, next_first)
        # Padding never becomes a real request. Use the target vLLM sentinel.
        if 0 <= NULL_BLOCK_ID < self.context.numel():
            self.context[NULL_BLOCK_ID : NULL_BLOCK_ID + 1].fill_(-1)
            self.first[NULL_BLOCK_ID : NULL_BLOCK_ID + 1].fill_(1)
        return torch.where(valid, first, 0)


class MegaKDACacheWorkspace:
    """Bounded staging storage, shared by sequential KDA layers on one worker.

    Allocate during model loading, before KV sizing. Hybrid pages remain owned by
    the scheduler; only the current batch is gathered into this dense ABI view.
    Row zero is reserved locally as well as in the scheduler pool.
    """

    def __init__(self, max_batch, spec_len, conv_shape, recurrent_shape, device):
        self.max_batch = max_batch
        self.spec_len = spec_len
        self.conv = torch.zeros((max_batch + 1, *conv_shape), dtype=torch.bfloat16, device=device)
        self.recurrent = torch.zeros((max_batch * spec_len + 1, *recurrent_shape), dtype=torch.float32, device=device)
        self.conv_indices = torch.arange(1, max_batch + 1, dtype=torch.int32, device=device)
        self.recurrent_indices = torch.arange(1, max_batch * spec_len + 1, dtype=torch.int32, device=device)
        self.copy_flags = torch.ones(max_batch * spec_len, dtype=torch.bool, device=device)


@dataclass(frozen=True)
class MegaKDAPagedBatch:
    conv: torch.Tensor
    recurrent: torch.Tensor
    conv_indices: torch.Tensor
    recurrent_indices: torch.Tensor
    global_conv_indices: torch.Tensor
    global_recurrent_indices: torch.Tensor


def _mega_cache_layout(state):
    return (
        state.device,
        state.dtype,
        tuple(state.shape),
        tuple(state.stride()),
        state.data_ptr() % MEGA_KDA_CACHE_ALIGNMENT_BYTES,
    )


class MegaKDAPagedCache:
    """Adapt page-strided V2 caches without changing physical block ownership."""

    def __init__(
        self, cache, workspace, conv_plan=None, recurrent_plan=None, *, copy_plan=None, gather_flags=(False, False)
    ):
        self.workspace = workspace
        self.conv_plan = conv_plan
        self.recurrent_plan = recurrent_plan
        self.copy_plan = copy_plan
        self.gather_flags = gather_flags
        self.layouts = tuple(_mega_cache_layout(state) for state in cache)

    def supports(self, cache, spec_len, batch):
        return (
            spec_len == self.workspace.spec_len
            and batch <= self.workspace.max_batch
            and tuple(_mega_cache_layout(state) for state in cache) == self.layouts
        )

    @staticmethod
    def _indices(indices, num_slots):
        # Graph padding must neither read nor overwrite the shared null page.
        valid = (indices >= 0) & (indices != NULL_BLOCK_ID) & (indices < num_slots)
        return torch.where(valid, indices, -1).to(torch.int32).contiguous()

    def gather(self, cache, conv_indices, recurrent_indices, batch, spec_len):
        if not self.supports(cache, spec_len, batch):
            raise RuntimeError("MegaKDA paged cache layout changed; reinitialize the V2 worker")
        workspace = self.workspace
        conv = workspace.conv[: batch + 1]
        recurrent = workspace.recurrent[: batch * spec_len + 1]
        if self.copy_plan is not None:
            # The fused copy accepts the block-table column's actual stride.
            # Bounds/NULL masking is part of its payload load/store, not a
            # sequence of per-layer comparison, Select and Cast launches.
            global_conv, global_recurrent = conv_indices, recurrent_indices
            self.copy_plan.copy(cache, conv[1:], recurrent[1:], global_conv, global_recurrent, to_cache=False)
        else:
            global_conv = self._indices(conv_indices, cache[0].shape[0])
            global_recurrent = self._indices(recurrent_indices, cache[1].shape[0])
            self.conv_plan.gather_into(
                cache[0].unsqueeze(1),
                conv[1:].unsqueeze(1),
                global_conv,
                workspace.copy_flags[:batch] if self.gather_flags[0] else None,
            )
            self.recurrent_plan.gather_into(
                cache[1],
                recurrent[1:],
                global_recurrent,
                workspace.copy_flags[: batch * spec_len] if self.gather_flags[1] else None,
            )
        return MegaKDAPagedBatch(
            conv,
            recurrent,
            workspace.conv_indices[:batch],
            workspace.recurrent_indices[: batch * spec_len],
            global_conv,
            global_recurrent,
        )

    def scatter(self, cache, batch):
        # Each live destination is unique; invalid / null rows are skipped by
        # the prepared kernels, including all graph-padding destinations.
        if self.copy_plan is not None:
            self.copy_plan.copy(
                cache,
                batch.conv[1:],
                batch.recurrent[1:],
                batch.global_conv_indices,
                batch.global_recurrent_indices,
                to_cache=True,
            )
        else:
            self.conv_plan.scatter(cache[0].unsqueeze(1), batch.conv[1:].unsqueeze(1), batch.global_conv_indices)
            self.recurrent_plan.scatter(cache[1], batch.recurrent[1:], batch.global_recurrent_indices)


def initialize_mega_kda_cache(static_forward_context):
    """Prepare V2's cache adapters after binding and before graph capture.

    Reuse the existing sealed KDA state-copy kernels; preparation touches only
    disposable rows. Disabled or zero-copy layers need no additional plans.
    """
    layers = [
        layer for layer in static_forward_context.values() if getattr(layer, "_mega_cache_workspace", None) is not None
    ]
    if not layers:
        return
    for layer in layers:
        layer._mega_cache_adapter = None
    # Lazy to keep normal GDN/backend imports free of copy-plan dependencies.
    from vllm_ascend.ops.kda_state_copy_plan import (
        KDAStateCopyPlan,
        StridedKDAFallbackPlan,
        cache_signature,
        supports_kda_state_copy,
    )
    from vllm_ascend.ops.triton.mega_kda_cache import MegaKDAPagedCopyPlan

    plans = {}
    bindings = []
    for layer in layers:
        cache = layer.kv_cache
        if len(cache) != 2:
            continue
        if all(state.is_contiguous() for state in cache):
            continue
        workspace = layer._mega_cache_workspace
        if tuple(cache[0].shape[1:]) != tuple(workspace.conv.shape[1:]) or tuple(cache[1].shape[1:]) != tuple(
            workspace.recurrent.shape[1:]
        ):
            continue
        if all(supports_kda_state_copy(state) for state in (cache[0].unsqueeze(1), cache[1])):
            key = (
                MegaKDAPagedCopyPlan,
                *(cache_signature(state) for state in cache),
                workspace.max_batch,
                workspace.spec_len,
            )
            if key not in plans:
                plans[key] = MegaKDAPagedCopyPlan.prepare(cache, workspace.max_batch, workspace.spec_len)
            bindings.append((layer, MegaKDAPagedCache(cache, workspace, copy_plan=plans[key])))
            continue
        layer_plans = []
        for state, limit in (
            (cache[0].unsqueeze(1), workspace.max_batch),
            (cache[1], workspace.max_batch * workspace.spec_len),
        ):
            plan_type = KDAStateCopyPlan if supports_kda_state_copy(state) else StridedKDAFallbackPlan
            key = (plan_type, cache_signature(state), limit)
            if key not in plans:
                plans[key] = plan_type.prepare(state, limit)
            layer_plans.append(plans[key])
        bindings.append(
            (
                layer,
                MegaKDAPagedCache(
                    cache,
                    workspace,
                    *layer_plans,
                    gather_flags=tuple(isinstance(plan, KDAStateCopyPlan) for plan in layer_plans),
                ),
            )
        )
    for plan in plans.values():
        plan.seal()
    for layer, adapter in bindings:
        layer._mega_cache_adapter = adapter


def _as_int32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return an operator-compatible int32 tensor without redundant copies."""
    if tensor.dtype == torch.int32 and tensor.is_contiguous():
        return tensor
    return tensor.to(torch.int32).contiguous()


@dataclass(frozen=True)
class MegaKDAWeights:
    """NZ-cast BF16 ``[out, in]`` matmul weights, FP32 o_norm, [3,H,W,D] conv."""

    qkv: torch.Tensor
    f_a: torch.Tensor
    f_b: torch.Tensor
    beta: torch.Tensor
    output_gate: torch.Tensor
    o_norm: torch.Tensor
    o_proj: torch.Tensor
    conv: torch.Tensor


def load_mega_kda_operator():
    """Load the PR #6 ABI only when explicitly enabled by the user.

    Import/extension failures retain their original exception as the cause.
    No trial execution is allowed: the operator mutates both state caches.
    """
    try:
        operator = import_module("ops.mega_recurrent_kda").mega_recurrent_kda
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "enable_mega_kda requires ops.mega_recurrent_kda.mega_recurrent_kda "
            "with the PR #6 20-argument BF16/NZ ABI (including RMS epsilon)."
        ) from error
    if not callable(operator):
        raise TypeError("ops.mega_recurrent_kda.mega_recurrent_kda must be callable")
    try:
        parameters = tuple(signature(operator).parameters.values())
    except (TypeError, ValueError):
        parameters = ()  # Some binary exports expose no Python signature.
    if parameters and not any(p.kind is Parameter.VAR_POSITIONAL for p in parameters):
        positional = [p for p in parameters if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)]
        if len(positional) != MEGA_KDA_ABI_POSITIONAL_ARGS:
            raise RuntimeError(
                "MegaKDA ABI mismatch: expected the PR #6 20-argument BF16/NZ export, "
                f"got {signature(operator)}. The 19-argument ND recipe ABI is incompatible."
            )
    return operator


def ensure_mega_kda_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Assert one operand is BF16; the operator binary is BF16-only."""
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(
            f"MegaKDA requires BF16 weights, got {tensor.dtype}; use a BF16 "
            "KDA projection checkpoint for the fused operator"
        )
    return tensor


def mega_kda_nz_operand(tensor: torch.Tensor) -> torch.Tensor:
    ensure_mega_kda_bf16(tensor)
    return torch_npu.npu_format_cast(tensor.clone(), MEGA_KDA_NZ_FORMAT)


def ensure_mega_kda_nd(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "npu":
        return tensor
    if int(torch_npu.get_npu_format(tensor)) != int(ACL_FORMAT_FRACTAL_ND):
        return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
    return tensor


def run_mega_recurrent_kda(
    hidden_states: torch.Tensor,
    weights: MegaKDAWeights,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    conv_state_block_ids: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    gate_lower_bound: float,
    o_norm_variance_epsilon: float,
    *,
    operator,
) -> torch.Tensor:
    """Run the fused Mega-KDA decode operator for one KDA layer.

    ``hidden_states`` is ``[batch, spec_len, hidden]``.  The return value is
    the partial (pre-TP-reduction) ``[batch * spec_len, hidden]`` output with
    the output projection already applied.
    """
    output = operator(
        hidden_states,
        weights.qkv,
        weights.f_a,
        weights.f_b,
        weights.beta,
        weights.output_gate,
        weights.o_norm,
        weights.o_proj,
        weights.conv,
        conv_state,
        recurrent_state,
        _as_int32_contiguous(conv_state_block_ids),
        _as_int32_contiguous(ssm_state_indices),
        _as_int32_contiguous(conv_num_accepted_tokens),
        _as_int32_contiguous(num_accepted_tokens),
        a_log,
        dt_bias,
        scale,
        gate_lower_bound,
        o_norm_variance_epsilon,
    )
    if output.numel() != hidden_states.numel():
        raise RuntimeError("MegaKDA returned an unexpected token count; the state cache may already be mutated")
    return output.reshape(-1, hidden_states.shape[-1])
