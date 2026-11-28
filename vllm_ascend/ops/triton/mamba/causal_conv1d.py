# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import _causal_conv1d_update_kernel
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "causal_conv1d_update_npu", "extract_last_width"]


def causal_conv1d_update_varlen(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    null_block_id: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update GDN's dim-last state using device query boundaries.

    The model input may contain graph-padding tokens beyond the final live
    boundary. Use the upstream strided Triton kernel, with CUDA PDL disabled,
    instead of the native convolution's padded-input path. Keep state/weight
    layouts as views: [blocks, history, dim] and [width, dim], respectively.
    """
    if x.ndim != 2 or weight.ndim != 2 or conv_state.ndim != 3:
        raise ValueError("GDN convolution requires 2D inputs/weights and 3D states")
    dim = x.shape[1]
    width = weight.shape[0]
    state_len = width - 1 + max_query_len - 1
    if weight.shape[1] != dim or conv_state.shape[2] != dim:
        raise ValueError("GDN convolution requires token-major inputs and dim-last weights/states")
    if not 2 <= width <= 6 or max_query_len < 1 or conv_state.shape[1] < state_len:
        raise ValueError("Unsupported GDN convolution width or insufficient speculative state history")
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported GDN convolution activation: {activation}")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("GDN convolution output must match the input shape, dtype and device")
    batch = conv_state_indices.numel()
    if query_start_loc.numel() != batch + 1 or num_accepted_tokens.numel() != batch:
        raise ValueError("GDN convolution request boundaries and accepted counts must match cache indices")
    _causal_conv1d_update_kernel[(batch, triton.cdiv(dim, 256))](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        out,
        batch,
        dim,
        max_query_len,
        state_len,
        conv_state.shape[0],
        0,
        x.stride(1),
        x.stride(0),
        weight.stride(1),
        weight.stride(0),
        conv_state.stride(0),
        conv_state.stride(2),
        conv_state.stride(1),
        conv_state_indices.stride(0),
        0,
        out.stride(1),
        out.stride(0),
        null_block_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        IS_VARLEN=True,
        IS_SPEC_DECODING=True,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        HAS_NULL_BLOCK=True,
        BLOCK_N=256,
        launch_pdl=False,
    )
    return out


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Host-sync-free ``causal_conv1d_update`` covering both NPU state layouts.

    Signature-compatible drop-in for
    ``vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update``:
    it serves the mamba layout ``(cache_lines, dim, state_len)`` used by
    short_conv / mamba_mixer2 AND the dim-last layout
    ``(cache_lines, state_len, dim)`` used by the GDN/KDA layers (the Triton
    kernel addresses state through strides only, so the layouts differ just in
    the order the strides are passed). Unlike the upstream wrapper this path
    never syncs with the host — no ``.item()``, no per-request loop — and
    ``launch_pdl`` is pinned to False instead of re-probing the platform
    architecture on every call.

    Speculative-decode kwargs (``num_accepted_tokens`` / ``query_start_loc`` /
    ``max_query_len``) are accepted in both layouts, so patch_triton can bind
    this function for every caller without dropping arguments.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    if out is None:
        out = x

    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
        out = out.unsqueeze(-1)

    if query_start_loc is None:
        if conv_state_indices is not None:
            batch = conv_state_indices.shape[0]
        else:
            batch = x.shape[0]
        seqlen = x.shape[2]
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        if conv_state_indices is None:
            raise ValueError("query_start_loc requires conv_state_indices")
        batch = conv_state_indices.shape[0]
        seqlen = max_query_len
        # x (num_tokens, dim): token-major
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    num_cache_lines = conv_state.shape[0]

    # Layout dispatch (must resolve dim/width BEFORE deriving state_len).
    # Mamba: state (cache_lines, dim, state_len) and weight (dim, width).
    # GDN/KDA: state (cache_lines, state_len, dim) and weight (width, dim).
    # The kernel's state view is (seq, dim, token) in that order, so only
    # the stride order differs between the two layouts.
    if conv_state.shape[2] == weight.shape[1]:
        # dim-last GDN/KDA
        dim, width = weight.shape[1], weight.shape[0]
        stride_w_dim, stride_w_width = weight.stride(1), weight.stride(0)
        stride_istate_seq, stride_istate_token, stride_istate_dim = conv_state.stride()
    elif conv_state.shape[1] == weight.shape[0]:
        # mamba (lines, dim, state_len)
        dim, width = weight.shape[0], weight.shape[1]
        stride_w_dim, stride_w_width = weight.stride(0), weight.stride(1)
        stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    else:
        raise ValueError(
            "causal_conv1d_update_npu: unrecognized conv_state layout "
            f"{tuple(conv_state.shape)} with weight {tuple(weight.shape)}"
        )

    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    grid = (batch, triton.cdiv(dim, 256))
    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        conv_state_indices.stride(0) if conv_state_indices is not None else 0,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        null_block_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        IS_VARLEN=query_start_loc is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        HAS_NULL_BLOCK=True,
        BLOCK_N=256,
        launch_pdl=False,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
