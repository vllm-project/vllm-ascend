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

CHUNK_SIZE = 64


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=dim, keepdim=True) + eps)


def _expand_qk_to_v_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    """
    Expand q/k heads to match v heads for grouped-value-attention semantics.
    x: [L, Hqk, D] -> [L, Hv, D]
    """
    h_qk = x.shape[1]
    if h_qk == num_v_heads:
        return x
    if num_v_heads % h_qk != 0:
        raise ValueError(f"Invalid grouped heads: Hqk={h_qk}, Hv={num_v_heads}.")
    group_size = num_v_heads // h_qk
    return x.repeat_interleave(group_size, dim=1)


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
    initial_state: [B, H, K, V]
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

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
    scale = 1 / (query.shape[-1] ** 0.5)
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
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_inter_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        inter_state = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = inter_state + attn_inter_chunk @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    310P fallback with vLLM-compatible interface.
    Internal math follows Transformers torch_chunk_gated_delta_rule flow.
    """
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in 310P fallback.")
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
        states = torch.zeros(num_states, h_v, k_dim, v_dim, dtype=torch.float32, device=q.device)

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
