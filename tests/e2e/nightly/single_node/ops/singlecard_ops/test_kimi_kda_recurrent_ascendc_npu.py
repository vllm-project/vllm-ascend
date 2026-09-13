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
# This file is a part of the vllm-ascend project.

"""Kimi K3 recurrent-KDA reference and direct AscendC accuracy coverage."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def _flatten_bsnd(x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    if layout != "BSND":
        raise ValueError("layout must be BSND or TND")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _restore_layout(x: torch.Tensor, ref: torch.Tensor, layout: str) -> torch.Tensor:
    return x if layout == "TND" else x.reshape(ref.shape)


def _seq_ranges(total_tokens: int, cu_seqlens: Sequence[int]) -> list[tuple[int, int]]:
    offsets = [int(offset) for offset in cu_seqlens]
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least two cumulative offsets")
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    if offsets[-1] != total_tokens:
        raise ValueError("the last cu_seqlens offset must equal the packed token count")
    return list(zip(offsets, offsets[1:]))


def _state_slot(ssm_state_indices: torch.Tensor, seq_idx: int, start: int, token: int) -> int:
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[token].item())
    if ssm_state_indices.ndim == 2:
        return int(ssm_state_indices[seq_idx, token - start].item())
    raise ValueError("ssm_state_indices must be packed [T] or speculative [seq_num,max_step]")


def recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: Sequence[int],
    ssm_state_indices: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    layout: str = "BSND",
    scale: float | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    state_v_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    del output_final_state
    if not state_v_first:
        raise ValueError("reference only supports state_v_first=True")

    q_flat = _flatten_bsnd(q, layout).float()
    k_flat = _flatten_bsnd(k, layout).float()
    v_flat = _flatten_bsnd(v, layout).float()
    g_flat = _flatten_bsnd(g, layout).float()
    beta_flat = _flatten_bsnd(beta, layout).float()
    total_tokens, h, dk = q_flat.shape
    _, hv, dv = v_flat.shape
    scale = dk**-0.5 if scale is None else scale

    if use_qk_l2norm_in_kernel:
        q_flat = F.normalize(q_flat, p=2, dim=-1)
        k_flat = F.normalize(k_flat, p=2, dim=-1)
    q_flat = q_flat * scale

    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        gate = g_flat
        if dt_bias is not None:
            gate = gate + dt_bias.float().reshape(hv, dk).unsqueeze(0)
        exp_a = torch.exp(A_log.float()).reshape(1, hv, 1)
        gate = lower_bound * torch.sigmoid(exp_a * gate) if safe_gate else -exp_a * F.softplus(gate)
    else:
        gate = g_flat
    gate_decay = torch.exp(gate.float())

    beta_eff = beta_flat
    if use_beta_sigmoid_in_kernel:
        beta_eff = torch.sigmoid(beta_eff)
        if allow_neg_eigval:
            beta_eff = beta_eff * 2.0

    ranges = _seq_ranges(total_tokens, cu_seqlens)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    state = (
        torch.zeros((len(ranges), hv, dv, dk), dtype=torch.float32, device=q.device)
        if initial_state is None
        else initial_state.float().clone()
    )
    out_flat = torch.zeros_like(v_flat, dtype=torch.float32)

    for seq_idx, (start, end) in enumerate(ranges):
        if start == end:
            continue
        state_slot = seq_idx
        if ssm_state_indices is not None:
            token = start
            if num_accepted_tokens is not None:
                token = start + int(num_accepted_tokens[seq_idx].item()) - 1
            state_slot = _state_slot(ssm_state_indices, seq_idx, start, token)
        for hv_idx in range(hv):
            h_idx = hv_idx // (hv // h)
            state_cur = state[state_slot, hv_idx].clone()
            for token in range(start, end):
                state_cur = state_cur * gate_decay[token, hv_idx].unsqueeze(0)
                delta = v_flat[token, hv_idx] - torch.mv(state_cur, k_flat[token, h_idx])
                state_cur = state_cur + torch.outer(delta * beta_eff[token, hv_idx], k_flat[token, h_idx])
                out_flat[token, hv_idx] = torch.mv(state_cur, q_flat[token, h_idx])
                out_slot = (
                    _state_slot(ssm_state_indices, seq_idx, start, token) if ssm_state_indices is not None else seq_idx
                )
                state[out_slot, hv_idx] = state_cur

    return _restore_layout(out_flat.to(q.dtype), v, layout), state.to(state_dtype)


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_non_contiguous_state_pool():
    """Preserve a strided cache view while updating only selected Kimi slots."""
    torch.manual_seed(20260806)
    device = torch.device("npu")
    batch, heads, dim = 4, 6, 128
    state_capacity = 17
    cu_seqlens_host = list(range(batch + 1))
    state_indices_cpu = torch.tensor([9, 2, 15, 4], dtype=torch.int64)

    q_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn_like(q_cpu)
    raw_gate_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16) * 0.25
    beta_cpu = torch.rand(1, batch, heads, dtype=torch.float32).sigmoid()
    state_cpu = torch.randn(state_capacity, heads, dim, dim, dtype=torch.float32) * 0.01
    a_log_cpu = torch.randn(heads, dtype=torch.float32) * 0.05
    dt_bias_cpu = torch.randn(heads, dim, dtype=torch.float32) * 0.05

    ref_out, ref_state = recurrent_kda_reference(
        q_cpu,
        k_cpu,
        v_cpu,
        raw_gate_cpu,
        beta_cpu,
        state_cpu,
        cu_seqlens=cu_seqlens_host,
        ssm_state_indices=state_indices_cpu,
        A_log=a_log_cpu,
        dt_bias=dt_bias_cpu,
        layout="BSND",
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
    )

    state_pool = torch.full(
        (state_capacity + 1, 2, heads, dim, dim),
        7.0,
        dtype=torch.float32,
        device=device,
    )
    state_view = state_pool[1:, 0]
    state_view.copy_(state_cpu.to(device))
    guard_layer = state_pool[1:, 1].clone()
    state_before = state_view.clone()
    state_stride = state_view.stride()
    state_storage = state_view.untyped_storage().data_ptr()
    assert not state_view.is_contiguous()
    assert state_view.storage_offset() > 0

    out = torch.ops._C_ascend.recurrent_kda(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        raw_gate_cpu.to(device),
        beta_cpu.to(device),
        state_view,
        torch.tensor(cu_seqlens_host, dtype=torch.int32, device=device),
        state_indices_cpu.to(device),
        a_log_cpu.to(device),
        dt_bias_cpu.to(device),
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
    )
    torch.npu.synchronize()

    assert state_view.stride() == state_stride
    assert state_view.untyped_storage().data_ptr() == state_storage
    torch.testing.assert_close(out.cpu(), ref_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state_view.cpu(), ref_state, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state_pool[1:, 1], guard_layer, rtol=0, atol=0)
    used_slots = set(state_indices_cpu.tolist())
    untouched_slots = [slot for slot in range(state_capacity) if slot not in used_slots]
    torch.testing.assert_close(state_view[untouched_slots], state_before[untouched_slots], rtol=0, atol=0)
