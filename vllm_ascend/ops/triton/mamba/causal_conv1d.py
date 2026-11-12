# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import _causal_conv1d_update_kernel
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "extract_last_width"]


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
        None,  # block_idx_last_scheduled_token (APC disabled)
        None,  # initial_state_idx (APC disabled)
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
        IS_APC_ENABLED=False,
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
