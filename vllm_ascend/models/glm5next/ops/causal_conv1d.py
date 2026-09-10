# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend causal conv1d entry points for the GLM-5.3-Flash KDA layers.

The upstream CUDA kernels reference ``tl.extra.cuda.gdc_wait``, which Ascend
Triton does not provide -- the AST visitor raises even when ``launch_pdl`` is
False. Both entry points are therefore routed to Ascend implementations.

The decode/draft-verify update has to stay free of host syncs: reading a device
tensor with ``.item()`` is rejected outright while an ACL graph is being
captured, so a single such read aborts decode-FULL capture. It also has to stay
free of per-request Python loops, because it runs once per KDA layer on every
decode step. Two implementations satisfy that, picked by hardware:

* ``npu_causal_conv1d_custom``, the fused operator the Qwen3-Next GDN and Kimi
  KDA layers already use. It is unavailable wherever ``enable_custom_op()`` is
  off, which currently includes A5 -- its hardware profile withholds
  ``RUNTIME_CUSTOM_OPS`` (see vllm-ascend issue #7157).
* A batched torch expression, below, that folds the whole batch into a handful
  of elementwise kernels.

The varlen (prefill) path keeps the per-request PyTorch implementation: its
lengths are genuinely ragged and prefill is never graph-captured.

Both entry points take ``x`` token-major (``[num_tokens, dim]``), ``weight`` in
the fused operator's ``[width, dim]`` kernel layout, and ``conv_state`` exactly
as the Mamba cache allocates it.
"""

import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_fn as _torch_causal_conv1d_fn,
)

# Mode selectors understood by npu_causal_conv1d_custom.
_ACTIVATION_SILU = 1
_RUN_MODE_VARLEN = 0
_RUN_MODE_UPDATE = 1


_FUSED_CONV1D_AVAILABLE: bool | None = None


def has_fused_conv1d() -> bool:
    """Whether the fused Ascend operator is registered on this hardware.

    Resolved once: this sits on the decode hot path, once per KDA layer per
    step, and ``enable_custom_op`` is itself a one-shot that either imports the
    extension or reports that the hardware profile withholds it.
    """
    global _FUSED_CONV1D_AVAILABLE

    if _FUSED_CONV1D_AVAILABLE is None:
        from vllm_ascend.utils import enable_custom_op

        # The extension is imported lazily, so the op namespace only fills in
        # once custom ops have been enabled for the process.
        _FUSED_CONV1D_AVAILABLE = enable_custom_op() and hasattr(torch.ops._C_ascend, "npu_causal_conv1d_custom")
    return _FUSED_CONV1D_AVAILABLE


def causal_conv1d_fn(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    initial_state_mode: torch.Tensor | None,
) -> torch.Tensor:
    """Run the varlen (prefill) convolution and seed ``conv_state``."""
    if not has_fused_conv1d():
        return _torch_causal_conv1d_fn(
            x.transpose(0, 1),
            # vLLM keeps these weights in fp32, and the reference convolves in
            # the weight dtype, so upcasting holds the accumulation there.
            weight.transpose(0, 1).float(),
            bias,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=initial_state_mode,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        ).transpose(0, 1)

    output = torch.empty_like(x)
    # Consume the operator's declared output alias. Returning ``output``
    # independently would let graph functionalization treat the custom-op
    # result as dead and expose the uninitialized allocation instead.
    return torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=initial_state_mode,
        num_accepted_tokens_opt=None,
        activation_mode=_ACTIVATION_SILU,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=_RUN_MODE_VARLEN,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slide ``conv_state`` over the decode / draft-verify tokens.

    ``cache_indices`` is the recurrent state index tensor, which carries one
    column per draft slot on the speculative path; only its first column names
    the conv state.
    """
    if not has_fused_conv1d():
        return _batched_causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    output = torch.empty_like(x)
    return torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=_ACTIVATION_SILU,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=_RUN_MODE_UPDATE,
    )


def _batched_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """Advance every request's conv state at once, without any host sync.

    Each request contributes at most ``max_query_len`` tokens -- one on a plain
    decode step, one per draft slot on a draft-verify step -- which is a host
    side shape rather than a device value, so the ragged batch folds into a
    dense ``[num_requests, max_query_len]`` grid. Requests the metadata builder
    padded away fall out of that grid through ``keep`` instead of through a
    length read.

    Every row of a verify step is convolved, drafts included. Which of them the
    sampler keeps is not known while the model runs, so ``num_accepted_tokens``
    describes the step before, and it selects where in the widened cache row
    this step starts reading rather than how many tokens it may consume.
    """
    num_tokens, dim = x.shape
    width = weight.shape[0]
    state_len = width - 1
    num_requests = query_start_loc.shape[0] - 1
    if num_requests <= 0 or num_tokens == 0:
        return x.clone()

    max_query_len = cache_indices.shape[-1] if cache_indices.dim() > 1 else 1
    state_slots = (cache_indices[:, 0] if cache_indices.dim() > 1 else cache_indices).to(torch.int64)

    # Normalize the cache to [num_slots, state_len, dim]. The merged q|k|v
    # channel count dwarfs the conv width, so the two layouts the Mamba cache
    # can be allocated in are told apart by which axis matches it. The cache is
    # allocated wider than state_len by num_spec, which is the room a verify
    # step needs to keep every draft it convolved so that the step after can
    # rewind to whichever prefix the sampler accepted.
    states = conv_state if conv_state.shape[-1] == dim else conv_state.transpose(-1, -2)
    state_width = states.shape[1]

    # A padded request has query_start_loc[i] == query_start_loc[i + 1], which
    # lands on zero.
    lengths = (query_start_loc[1:] - query_start_loc[:-1]).clamp(0, max_query_len)

    # Where the history of a request ends inside its cache row. Mirrors
    # conv_state_token_offset in the upstream kernel: the step before accepted
    # a tokens, so the window this step reads starts a - 1 columns in.
    if num_accepted_tokens is not None:
        rewind = (num_accepted_tokens[:num_requests].clamp(min=1) - 1).clamp(max=state_width - state_len)
        rewind = rewind.to(torch.int64)
    else:
        rewind = torch.zeros(num_requests, dtype=torch.int64, device=x.device)
    read_at = rewind.unsqueeze(1) + torch.arange(state_len, device=x.device).unsqueeze(0)
    prior = states.index_select(0, state_slots).gather(1, read_at.unsqueeze(-1).expand(num_requests, state_len, dim))

    offsets = torch.arange(max_query_len, device=x.device, dtype=lengths.dtype)
    keep = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    token_ids = query_start_loc[:num_requests].unsqueeze(1) + offsets.unsqueeze(0)
    # Dropped slots would index past the batch, so park them on token 0 and let
    # `keep` discard both the tokens read here and the outputs written below.
    token_ids = torch.where(keep, token_ids, torch.zeros_like(token_ids)).to(torch.int64)

    tokens = x.index_select(0, token_ids.reshape(-1)).view(num_requests, max_query_len, dim)
    # The reference convolves in the cache dtype, so round-trip through it.
    tokens = tokens.to(states.dtype) * keep.unsqueeze(-1)
    history = torch.cat([prior, tokens], dim=1).float()

    # width is a small host-side constant, so unrolling the taps keeps this to a
    # few elementwise kernels instead of materializing a
    # [num_requests, max_query_len, dim, width] window.
    conv_out = history[:, :max_query_len] * weight[0].float()
    for tap in range(1, width):
        conv_out += history[:, tap : tap + max_query_len] * weight[tap].float()
    if bias is not None:
        conv_out += bias.float()
    conv_out = F.silu(conv_out)

    # Store the history shifted by one, so that reading at offset a - 1 next
    # step lands on the window ending with this step's a-th token. A plain
    # decode step brings one token, where this is the usual slide by one.
    write_len = min(state_width, state_len + max_query_len - 1)
    advanced = history[:, 1 : 1 + write_len].to(states.dtype)
    # A request with no tokens keeps its row as it was, which the slots the
    # metadata builder pointed at the null block rely on.
    held = states[state_slots, :write_len]
    states[state_slots, :write_len] = torch.where(keep[:, :1].unsqueeze(-1), advanced, held)

    # Positions past the accepted prefix keep the projection they came in with,
    # matching the reference. Dropped slots are aimed at a scratch row that is
    # sliced off, so their duplicate indices cannot clobber a real token.
    result = torch.cat([x, x.new_zeros(1, dim)], dim=0)
    destinations = torch.where(keep, token_ids, torch.full_like(token_ids, num_tokens))
    result[destinations.reshape(-1)] = conv_out.reshape(-1, dim).to(x.dtype)
    return result[:num_tokens]
