# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External Mega-KDA decode operator wrapper for Kimi K3.

``mega_recurrent_kda`` (external ``ops`` package) fuses the KDA input
projections, causal conv1d update, recurrent KDA, gated output norm and
output projection into one ReplaySSM launch. It uses checkpoint and replay
buffers through keyword arguments; state indices and accepted-token counts
are int32. The first Verify round after prefill uses the neutral Conv history
count prepared by the metadata builder; the recurrent count always comes from
scheduler metadata.
"""

from dataclasses import dataclass
from importlib import import_module
from inspect import signature

import torch
import torch_npu
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

MEGA_KDA_NZ_FORMAT = 29
MEGA_KDA_REPLAY_ABI_REQUIRED_ARGS = (
    "hidden_states",
    "qkv_projection_weight",
    "decay_projection_a_weight",
    "decay_projection_b_weight",
    "beta_projection_weight",
    "output_gate_projection_weight",
    "output_norm_weight",
    "output_projection_weight",
    "conv1d_weight",
    "conv_state",
    "checkpoint_state",
    "replay_u",
    "replay_k",
    "replay_decay",
    "state_indices",
    "conv_num_accepted_tokens",
    "A_log",
    "dt_bias",
    "scale",
    "lower_bound",
    "rms_norm_eps",
    "cu_seqlens",
)
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
    """Load and validate the ReplaySSM MegaKDA operator."""
    try:
        operator = import_module("ops.mega_recurrent_kda").mega_recurrent_kda
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "enable_mega_kda_replayssm requires ops.mega_recurrent_kda.mega_recurrent_kda "
            "with the ReplaySSM keyword ABI."
        ) from error
    if not callable(operator):
        raise TypeError("ops.mega_recurrent_kda.mega_recurrent_kda must be callable")
    try:
        parameters = tuple(signature(operator).parameters.values())
    except (TypeError, ValueError):
        parameters = ()  # Some binary exports expose no Python signature.
    if parameters:
        parameter_names = {p.name for p in parameters}
        missing = [name for name in MEGA_KDA_REPLAY_ABI_REQUIRED_ARGS if name not in parameter_names]
        if missing:
            raise RuntimeError(
                "MegaKDA ReplaySSM ABI mismatch: the operator is missing required parameters "
                f"{missing}; got {signature(operator)}."
            )
    return operator


def load_commit_recurrent_kda_operator():
    """Load the ReplaySSM checkpoint commit operator from the runtime package."""
    try:
        operator = import_module("ops.commit_recurrent_kda").commit_recurrent_kda
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "enable_mega_kda_replayssm requires ops.commit_recurrent_kda.commit_recurrent_kda"
        ) from error
    if not callable(operator):
        raise TypeError("ops.commit_recurrent_kda.commit_recurrent_kda must be callable")
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
    conv_num_accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    gate_lower_bound: float,
    o_norm_variance_epsilon: float,
    *,
    operator,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
) -> torch.Tensor:
    """Run the fused Mega-KDA decode operator for one KDA layer.

    ``hidden_states`` is ``[batch, spec_len, hidden]``.  The return value is
    the partial (pre-TP-reduction) ``[batch * spec_len, hidden]`` output with
    the output projection already applied.
    """
    output = operator(
        hidden_states=hidden_states,
        qkv_projection_weight=weights.qkv,
        decay_projection_a_weight=weights.f_a,
        decay_projection_b_weight=weights.f_b,
        beta_projection_weight=weights.beta,
        output_gate_projection_weight=weights.output_gate,
        output_norm_weight=weights.o_norm,
        output_projection_weight=weights.o_proj,
        conv1d_weight=weights.conv,
        conv_state=conv_state,
        checkpoint_state=recurrent_state,
        replay_u=replay_u,
        replay_k=replay_k,
        replay_decay=replay_decay,
        state_indices=_as_int32_contiguous(conv_state_block_ids),
        conv_num_accepted_tokens=_as_int32_contiguous(conv_num_accepted_tokens),
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        lower_bound=gate_lower_bound,
        rms_norm_eps=o_norm_variance_epsilon,
        cu_seqlens=None,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.numel() != hidden_states.numel():
        raise RuntimeError("MegaKDA returned an unexpected token count; the state cache may already be mutated")
    return output.reshape(-1, hidden_states.shape[-1])


def run_commit_recurrent_kda(
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    operator,
) -> None:
    """Apply accepted ReplaySSM records to the canonical checkpoint state."""
    operator(
        checkpoint_state=checkpoint_state,
        replay_u=replay_u,
        replay_k=replay_k,
        replay_decay=replay_decay,
        num_accepted_tokens=_as_int32_contiguous(num_accepted_tokens),
        state_indices=_as_int32_contiguous(state_indices),
    )
