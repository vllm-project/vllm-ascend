#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# mypy: ignore-errors

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p
from vllm_ascend.utils import is_rc_device

CHUNK_SIZE = 64


def _expand_qk_to_v_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    """
    Expand q/k heads to match v heads for grouped-value-attention semantics.
    x: [L, Hqk, D] -> [L, Hv, D]
    """
    h_qk = x.shape[-2]
    if h_qk == num_v_heads:
        return x
    if num_v_heads % h_qk != 0:
        raise ValueError(f"Invalid grouped heads: Hqk={h_qk}, Hv={num_v_heads}.")
    group_size = num_v_heads // h_qk
    return x.repeat_interleave(group_size, dim=-2)


def _iter_seq_ranges(batch_size: int, seq_len: int, cu_seqlens: torch.Tensor | None) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        return [(i, 0, seq_len) for i in range(batch_size)]
    return [(i, int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())) for i in range(len(cu_seqlens) - 1)]


def _normalize_chunk_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    Normalize inputs to [B, T, H, D] / [B, T, H] while preserving TND support.
    Returns normalized tensors and a flag indicating whether input was TND.
    """
    input_was_tnd = False

    if q.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("TND inputs require `cu_seqlens` for variable-length layout.")
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("When q is TND, k and v must also be TND.")
        if g.ndim != 2 or beta.ndim != 2:
            raise ValueError("When q is TND, g and beta must be shape [T, H].")
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        input_was_tnd = True
    elif q.ndim == 4:
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("When q is 4D, k and v must also be 4D.")
        if g.ndim != 3 or beta.ndim != 3:
            raise ValueError("When q is 4D, g and beta must be shape [B, T, H].")
    else:
        raise ValueError(f"Unsupported q ndim={q.ndim}; expected 3D(TND) or 4D(BTHD).")

    return q, k, v, g, beta, input_was_tnd


def _torch_chunk_gated_delta_rule_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked torch implementation aligned with the Qwen3-Next torch path:
    transformers/models/qwen3_next/modular_qwen3_next.py::torch_chunk_gated_delta_rule

    Shapes:
    query/key: [B, T, H, K]
    value:     [B, T, H, V]
    g/beta:    [B, T, H]
    initial_state: [B, H, V, K]
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm_310p(query)
        key = l2norm_310p(key)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask_diag = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, v_head_dim, k_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_inter_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state.transpose(-1, -2)
        v_new = v_i - v_prime
        inter_state = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state.transpose(-1, -2)
        core_attn_out[:, :, i] = inter_state + attn_inter_chunk @ v_new
        last_recurrent_state = last_recurrent_state * g[:, :, i, -1, None, None].exp() + v_new.transpose(-1, -2) @ (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_ascend_chunk_ops(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if q.device.type != "npu" or ascend_ops is None:
        raise RuntimeError("310P chunk_gated_delta_rule requires NPU AscendC kernels.")
    if not (hasattr(ascend_ops, "chunk_gated_delta_rule_fwd_h") and hasattr(ascend_ops, "chunk_fwd_o")):
        raise RuntimeError("Missing AscendC chunk-gdr ops: chunk_gated_delta_rule_fwd_h/chunk_fwd_o.")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"q/k/v must share float16 dtype on 310P, got {q.dtype}, {k.dtype}, {v.dtype}.")
    if v.shape[-1] < 128 or v.shape[-1] % 128 != 0:
        raise ValueError(f"v head dim must be >=128 and a multiple of 128, got {v.shape[-1]}.")


def _maybe_l2norm(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return l2norm_310p(x)


def _pad_bthd_to_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int]], None]:
    batch_size, seq_len = q.shape[:2]
    padded_len = _ceil_div(seq_len, chunk_size) * chunk_size
    pad_len = padded_len - seq_len
    seq_ranges = [(batch_idx, 0, seq_len) for batch_idx in range(batch_size)]
    if pad_len == 0:
        return q, k, v, g, beta, seq_ranges, None

    q_pad = q.new_zeros((batch_size, pad_len, *q.shape[2:]))
    k_pad = k.new_zeros((batch_size, pad_len, *k.shape[2:]))
    v_pad = v.new_zeros((batch_size, pad_len, *v.shape[2:]))
    g_pad = g.new_zeros((batch_size, pad_len, g.shape[-1]))
    beta_pad = beta.new_zeros((batch_size, pad_len, beta.shape[-1]))
    return (
        torch.cat((q, q_pad), dim=1),
        torch.cat((k, k_pad), dim=1),
        torch.cat((v, v_pad), dim=1),
        torch.cat((g, g_pad), dim=1),
        torch.cat((beta, beta_pad), dim=1),
        seq_ranges,
        None,
    )


def _pad_varlen_to_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int]], torch.Tensor
]:
    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    g_parts: list[torch.Tensor] = []
    beta_parts: list[torch.Tensor] = []
    seq_ranges: list[tuple[int, int, int]] = []
    padded_cu = [0]
    out_cursor = 0

    for seq_idx in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = end - start
        padded_len = _ceil_div(seq_len, chunk_size) * chunk_size if seq_len > 0 else 0
        pad_len = padded_len - seq_len

        q_seq = q[:, start:end]
        k_seq = k[:, start:end]
        v_seq = v[:, start:end]
        g_seq = g[:, start:end]
        beta_seq = beta[:, start:end]
        if pad_len > 0:
            q_seq = torch.cat((q_seq, q.new_zeros((1, pad_len, *q.shape[2:]))), dim=1)
            k_seq = torch.cat((k_seq, k.new_zeros((1, pad_len, *k.shape[2:]))), dim=1)
            v_seq = torch.cat((v_seq, v.new_zeros((1, pad_len, *v.shape[2:]))), dim=1)
            g_seq = torch.cat((g_seq, g.new_zeros((1, pad_len, g.shape[-1]))), dim=1)
            beta_seq = torch.cat((beta_seq, beta.new_zeros((1, pad_len, beta.shape[-1]))), dim=1)

        q_parts.append(q_seq)
        k_parts.append(k_seq)
        v_parts.append(v_seq)
        g_parts.append(g_seq)
        beta_parts.append(beta_seq)
        seq_ranges.append((0, out_cursor, out_cursor + seq_len))
        out_cursor += seq_len
        padded_cu.append(padded_cu[-1] + padded_len)

    if q_parts:
        q_padded = torch.cat(q_parts, dim=1)
        k_padded = torch.cat(k_parts, dim=1)
        v_padded = torch.cat(v_parts, dim=1)
        g_padded = torch.cat(g_parts, dim=1)
        beta_padded = torch.cat(beta_parts, dim=1)
    else:
        q_padded = q[:, :0]
        k_padded = k[:, :0]
        v_padded = v[:, :0]
        g_padded = g[:, :0]
        beta_padded = beta[:, :0]

    cu_padded = torch.tensor(padded_cu, dtype=torch.int64, device=cu_seqlens.device)
    return q_padded, k_padded, v_padded, g_padded, beta_padded, seq_ranges, cu_padded


def _prepare_chunk_indices_list(cu_seqlens: torch.Tensor, chunk_size: int) -> list[int]:
    chunk_indices: list[int] = []
    compact_seq_idx = 0
    for seq_idx in range(cu_seqlens.numel() - 1):
        seq_len = int(cu_seqlens[seq_idx + 1].item() - cu_seqlens[seq_idx].item())
        num_chunks = _ceil_div(seq_len, chunk_size) if seq_len > 0 else 0
        if num_chunks == 0:
            continue
        for chunk_idx in range(num_chunks):
            chunk_indices.extend((compact_seq_idx, chunk_idx))
        compact_seq_idx += 1
    return chunk_indices


def _compute_kernel_inputs_from_torch_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the original torch WY prefix and return AscendC kernel layout."""
    batch_size, padded_tokens, _, k_dim = k.shape
    num_v_heads = v.shape[2]
    value_dim = v.shape[-1]
    num_chunks = padded_tokens // chunk_size

    q_kernel = q.transpose(1, 2).contiguous()
    k_kernel = k.transpose(1, 2).contiguous()

    key = _expand_qk_to_v_heads(k, num_v_heads).transpose(1, 2).contiguous().to(torch.float32)
    value = v.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    key = key.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_dim)
    value = value.reshape(batch_size, num_v_heads, num_chunks, chunk_size, value_dim)
    g = g.reshape(batch_size, num_v_heads, num_chunks, chunk_size).cumsum(dim=-1)
    beta = beta.reshape(batch_size, num_v_heads, num_chunks, chunk_size)

    lower_decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()
    k_beta = key * beta.unsqueeze(-1)
    if is_rc_device():
        attn_chunks: list[torch.Tensor] = []
        for c in range(num_chunks):
            k_beta_c = k_beta[:, :, c]  # [B, Hv, S, D]
            key_c = key[:, :, c]  # [B, Hv, S, D]
            key_c_t = key_c.transpose(-1, -2).contiguous()  # [B, Hv, D, S]
            lower_decay_c = lower_decay[:, :, c]  # [B, Hv, S, S]
            attn_c = -(k_beta_c @ key_c_t * lower_decay_c)
            attn_chunks.append(attn_c)
        attn = torch.stack(attn_chunks, dim=2)  # [B, Hv, C, S, S]
    else:
        attn = -(k_beta @ key.transpose(-1, -2) * lower_decay)
    mask_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=0,
    )
    attn = attn.masked_fill(mask_diag, 0)
    for row_idx in range(1, chunk_size):
        row = attn[..., row_idx, :row_idx].clone()
        sub = attn[..., :row_idx, :row_idx].clone()
        attn[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ (value * beta.unsqueeze(-1))
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    u_kernel = value.reshape(batch_size, num_v_heads, padded_tokens, value_dim).to(torch.float16).contiguous()
    w_kernel = k_cumdecay.reshape(batch_size, num_v_heads, padded_tokens, k_dim).to(torch.float16).contiguous()
    g_kernel = g.reshape(batch_size, num_v_heads, padded_tokens).contiguous()
    return q_kernel, k_kernel, w_kernel, u_kernel, g_kernel


def _unpad_chunk_output(
    out: torch.Tensor,
    seq_ranges: list[tuple[int, int, int]],
    total_tokens: int,
    input_was_tnd: bool,
    is_varlen: bool,
) -> torch.Tensor:
    if is_varlen:
        unpadded = out.new_empty((1, total_tokens, *out.shape[2:]))
        padded_cursor = 0
        for _, start, end in seq_ranges:
            seq_len = end - start
            padded_len = _ceil_div(seq_len, CHUNK_SIZE) * CHUNK_SIZE if seq_len > 0 else 0
            if seq_len > 0:
                unpadded[:, start:end] = out[:, padded_cursor : padded_cursor + seq_len]
            padded_cursor += padded_len
        return unpadded[0] if input_was_tnd else unpadded

    seq_len = total_tokens
    return out[:, :seq_len]


def chunk_gated_delta_rule_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reference-only implementation with vLLM-compatible interface.
    Internal math follows Transformers torch_chunk_gated_delta_rule flow.

    The 310P production path must use the AscendC chunk-gdr kernels and does
    not fall back to this implementation.
    """
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in the reference implementation.")
    q, k, v, g, beta, input_was_tnd = _normalize_chunk_inputs(q, k, v, g, beta, cu_seqlens)

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")

    batch_size, total_tokens, h_qk, k_dim = q.shape
    h_v = v.shape[2]
    v_dim = v.shape[-1]
    if k.shape != q.shape:
        raise ValueError("q and k shapes must match.")
    if g.shape != beta.shape or g.shape[:2] != (batch_size, total_tokens) or g.shape[2] != h_v:
        raise ValueError("g/beta must have shape [B, T, Hv] matching v.")

    seq_ranges = _iter_seq_ranges(batch_size, total_tokens, cu_seqlens)
    num_states = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
    if initial_state is not None:
        states = initial_state.to(torch.float32).clone()
    else:
        states = torch.zeros(num_states, h_v, v_dim, k_dim, dtype=torch.float32, device=q.device)

    out = torch.zeros_like(v)
    for seq_idx, start, end in seq_ranges:
        seq_len = end - start
        if seq_len <= 0:
            continue

        b_idx = 0 if (cu_seqlens is not None and batch_size == 1) else seq_idx

        q_seq = _expand_qk_to_v_heads(q[b_idx, start:end], h_v).unsqueeze(0)
        k_seq = _expand_qk_to_v_heads(k[b_idx, start:end], h_v).unsqueeze(0)
        v_seq = v[b_idx, start:end].unsqueeze(0)
        g_seq = g[b_idx, start:end].unsqueeze(0)
        beta_seq = beta[b_idx, start:end].unsqueeze(0)
        init_seq_state = states[seq_idx].unsqueeze(0)

        out_seq, final_state = _torch_chunk_gated_delta_rule_chunked(
            query=q_seq,
            key=k_seq,
            value=v_seq,
            g=g_seq,
            beta=beta_seq,
            chunk_size=CHUNK_SIZE,
            scale=scale,
            initial_state=init_seq_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        out[b_idx, start:end] = out_seq[0]
        states[seq_idx] = final_state[0]

    if input_was_tnd:
        out = out[0]

    if output_final_state:
        return out, states
    return out, None


def chunk_gated_delta_rule_310(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """310P chunk GDN path backed by AscendC fwd_h/fwd_o kernels.

    Triton is unavailable on 310P, so the local WY preparation is done with
    torch ops and the inter-chunk state/output matmuls are delegated to the
    custom AscendC kernels.
    """
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in 310P chunk path.")
    if g is None or beta is None:
        raise RuntimeError("g and beta are required for the AscendC chunk-gdr path.")

    q, k, v, g, beta, input_was_tnd = _normalize_chunk_inputs(q, k, v, g, beta, cu_seqlens)
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")
    if k.shape != q.shape:
        raise ValueError("q and k shapes must match.")
    if g.shape != beta.shape or g.shape[:2] != q.shape[:2] or g.shape[2] != v.shape[2]:
        raise ValueError("g/beta must have shape [B, T, Hv] matching v.")

    _require_ascend_chunk_ops(q, k, v)

    q = _maybe_l2norm(q, use_qk_l2norm_in_kernel)
    k = _maybe_l2norm(k, use_qk_l2norm_in_kernel)

    original_tokens = v.shape[1]
    if cu_seqlens is None:
        q_pad, k_pad, v_pad, g_pad, beta_pad, seq_ranges, cu_kernel = _pad_bthd_to_chunk(q, k, v, g, beta, CHUNK_SIZE)
        cu_list = None
        chunk_indices_list = None
        num_states = q.shape[0]
    else:
        q_pad, k_pad, v_pad, g_pad, beta_pad, seq_ranges, cu_kernel = _pad_varlen_to_chunk(
            q, k, v, g, beta, cu_seqlens.to(torch.int64).cpu(), CHUNK_SIZE
        )
        assert cu_kernel is not None
        cu_list = cu_kernel.tolist()
        chunk_indices_list = _prepare_chunk_indices_list(cu_kernel, CHUNK_SIZE)
        num_states = cu_seqlens.numel() - 1

    expected_state_shape = (num_states, v.shape[2], v.shape[-1], k.shape[-1])
    if initial_state is not None:
        if initial_state.device != q.device:
            raise RuntimeError(f"initial_state must be on {q.device}, got {initial_state.device}.")
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(f"initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}.")

    if q_pad.shape[1] == 0:
        empty_out = v.new_empty((0, *v.shape[2:])) if input_was_tnd else v.new_empty(v.shape)
        final_state = initial_state if output_final_state else None
        return empty_out, final_state

    scale = k.shape[-1] ** -0.5 if scale is None else scale
    q_kernel, k_kernel, w_kernel, u_kernel, g_kernel = _compute_kernel_inputs_from_torch_wy(
        q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE
    )

    if initial_state is None:
        state = torch.zeros(
            num_states,
            v.shape[2],
            v.shape[-1],
            k.shape[-1],
            dtype=torch.float32,
            device=v.device,
        )
    else:
        state = initial_state
    state_kernel = state.transpose(-1, -2).contiguous()

    h, v_new, final_state_kernel = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k_kernel,
        w_kernel,
        u_kernel,
        g=g_kernel,
        gk=None,
        initial_state=state_kernel,
        output_final_state=output_final_state,
        chunk_size=CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=cu_list,
        chunk_indices=chunk_indices_list,
        use_exp2=False,
        transpose_state_layout=False,
    )
    o_kernel = torch.ops._C_ascend.chunk_fwd_o(
        q_kernel,
        k_kernel,
        v_new,
        h,
        scale,
        g=g_kernel,
        g_gamma=None,
        cu_seqlens=cu_list,
        chunk_indices=chunk_indices_list,
        chunk_size=CHUNK_SIZE,
        transpose_state_layout=False,
    )

    out = o_kernel.transpose(1, 2).contiguous().to(v.dtype)
    out = _unpad_chunk_output(out, seq_ranges, original_tokens, input_was_tnd, cu_seqlens is not None)
    if not output_final_state:
        return out, None
    final_state = final_state_kernel.transpose(-1, -2).contiguous()
    return out, final_state
