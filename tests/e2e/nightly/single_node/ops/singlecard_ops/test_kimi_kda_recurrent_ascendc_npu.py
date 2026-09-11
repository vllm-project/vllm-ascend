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

"""Kimi K3 TP16 recurrent-KDA reference and AscendC accuracy coverage."""

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
def test_kimi_k3_tp16_recurrent_kda_bsnd_single_token_decode_wrapper():
    """Exercise the non-spec Kimi decode wrapper with one cache slot per request."""
    from torch import nn

    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention

    torch.manual_seed(20260723)
    device = torch.device("npu")
    batch, heads, dim = 4, 6, 128
    cu_seqlens_host = list(range(batch + 1))
    q_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn_like(q_cpu)
    # The projection feeding Kimi's decode wrapper produces BF16. The aclnn
    # path must preserve the Triton gate preprocessing contract for that dtype.
    raw_gate_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16) * 0.25
    beta_cpu = torch.rand(1, batch, heads, dtype=torch.float32).sigmoid()
    state_slots = 17
    state_cpu = torch.randn(state_slots, heads, dim, dim, dtype=torch.float32) * 0.01
    state_indices_cpu = torch.tensor([9, 2, 15, 4], dtype=torch.int64)
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

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = dim
    layer.gate_lower_bound = -5.0
    layer.A_log = nn.Parameter(a_log_cpu.reshape(1, 1, heads, 1).to(device))
    layer.dt_bias = nn.Parameter(dt_bias_cpu.to(device))
    state_npu = state_cpu.to(device)
    out = layer._run_recurrent(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        raw_gate_cpu.to(device),
        beta_cpu.to(device),
        state_npu,
        torch.tensor(cu_seqlens_host, dtype=torch.int32, device=device),
        state_indices_cpu.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), ref_out, rtol=0.02, atol=0.02)
    final_state = state_npu.cpu()
    torch.testing.assert_close(final_state, ref_state, rtol=0.02, atol=0.02)
    untouched = [slot for slot in range(state_slots) if slot not in state_indices_cpu.tolist()]
    torch.testing.assert_close(final_state[untouched], state_cpu[untouched], rtol=0, atol=0)


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
    # Model-runner cache slots may be a strided view with a non-zero storage
    # offset. Keep the adjacent layer as a guard against an incorrect write.
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
    torch.testing.assert_close(
        state_view[untouched_slots],
        state_before[untouched_slots],
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_multistep_decode_matches_triton_and_reference():
    """Reuse real decode cache slots across steps, as the model wrapper does."""
    from torch import nn

    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention
    from vllm_ascend.ops.triton.kda.kda import fused_recurrent_kda

    torch.manual_seed(20260724)
    device = torch.device("npu")
    steps, batch, heads, dim = 8, 4, 6, 128
    state_slots = 17
    state_indices_cpu = torch.tensor([9, 2, 15, 4], dtype=torch.int64)
    cu_seqlens_cpu = torch.arange(batch + 1, dtype=torch.int32)

    # K3 checkpoints use a much wider decay range than the old small-random
    # smoke. Exercise saturated and nonsaturated safe-gate rows together.
    a_log_cpu = torch.linspace(0.0, 3.0, heads, dtype=torch.float32)
    dt_bias_cpu = torch.linspace(-6.0, 2.0, heads * dim, dtype=torch.float32).reshape(heads, dim)
    initial_state_cpu = torch.randn(state_slots, heads, dim, dim, dtype=torch.float32) * 0.1

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = dim
    layer.gate_lower_bound = -5.0
    layer.A_log = nn.Parameter(a_log_cpu.reshape(1, 1, heads, 1).to(device))
    layer.dt_bias = nn.Parameter(dt_bias_cpu.to(device))

    ascendc_state = initial_state_cpu.to(device)
    graph_state = initial_state_cpu.to(device)
    triton_state = initial_state_cpu.to(device)
    reference_state = initial_state_cpu
    cu_seqlens = cu_seqlens_cpu.to(device)
    state_indices = state_indices_cpu.to(device)

    graph_q = torch.zeros(1, batch, heads, dim, dtype=torch.bfloat16, device=device)
    graph_k = torch.zeros_like(graph_q)
    graph_v = torch.zeros_like(graph_q)
    graph_raw_gate = torch.zeros_like(graph_q)
    graph_beta = torch.zeros(1, batch, heads, dtype=torch.float32, device=device)
    layer._run_recurrent(
        graph_q,
        graph_k,
        graph_v,
        graph_raw_gate,
        graph_beta,
        graph_state,
        cu_seqlens,
        state_indices,
    )
    torch.npu.synchronize()
    graph_state.copy_(initial_state_cpu)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_out = layer._run_recurrent(
            graph_q,
            graph_k,
            graph_v,
            graph_raw_gate,
            graph_beta,
            graph_state,
            cu_seqlens,
            state_indices,
        )
    torch.npu.synchronize()
    graph_state.copy_(initial_state_cpu)

    for step in range(steps):
        q_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16)
        k_cpu = torch.randn_like(q_cpu)
        v_cpu = torch.randn_like(q_cpu)
        raw_gate_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16) * 2.0
        beta_cpu = torch.rand(1, batch, heads, dtype=torch.float32).sigmoid()

        reference_out, reference_state = recurrent_kda_reference(
            q_cpu,
            k_cpu,
            v_cpu,
            raw_gate_cpu,
            beta_cpu,
            reference_state,
            cu_seqlens=cu_seqlens_cpu.tolist(),
            ssm_state_indices=state_indices_cpu,
            A_log=a_log_cpu,
            dt_bias=dt_bias_cpu,
            layout="BSND",
            scale=dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
        )

        q, k, v = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu))
        raw_gate = raw_gate_cpu.to(device)
        beta = beta_cpu.to(device)
        graph_q.copy_(q)
        graph_k.copy_(k)
        graph_v.copy_(v)
        graph_raw_gate.copy_(raw_gate)
        graph_beta.copy_(beta)
        graph.replay()
        ascendc_out = layer._run_recurrent(
            q,
            k,
            v,
            raw_gate,
            beta,
            ascendc_state,
            cu_seqlens,
            state_indices,
        )
        triton_out, _ = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=layer._recurrent_gate(raw_gate),
            beta=beta,
            initial_state=triton_state,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(ascendc_out.cpu(), triton_out.cpu(), rtol=0.02, atol=0.02)
        torch.testing.assert_close(ascendc_out.cpu(), reference_out, rtol=0.02, atol=0.02)
        torch.testing.assert_close(ascendc_state.cpu(), triton_state.cpu(), rtol=0.02, atol=0.02)
        torch.testing.assert_close(ascendc_state.cpu(), reference_state, rtol=0.02, atol=0.02)
        torch.testing.assert_close(graph_out.cpu(), triton_out.cpu(), rtol=0.02, atol=0.02)
        torch.testing.assert_close(graph_state.cpu(), triton_state.cpu(), rtol=0.02, atol=0.02)


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_compiled_wrapper_updates_state_across_steps():
    """Cover the AOT functionalization used before full-decode graph capture."""
    import npugraph_ex as nge
    from torch import nn

    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention

    torch.manual_seed(20260725)
    device = torch.device("npu")
    steps, batch, heads, dim = 3, 4, 6, 128
    state_indices_cpu = torch.tensor([9, 2, 15, 4], dtype=torch.int64)
    cu_seqlens_cpu = torch.arange(batch + 1, dtype=torch.int32)
    a_log_cpu = torch.linspace(-0.43, 1.26, heads, dtype=torch.float32)
    dt_bias_cpu = torch.linspace(-9.0, -1.47, heads * dim, dtype=torch.float32).reshape(heads, dim)
    initial_state_cpu = torch.randn(17, heads, dim, dim, dtype=torch.float32) * 0.1

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = dim
    layer.gate_lower_bound = -5.0
    layer.A_log = nn.Parameter(a_log_cpu.reshape(1, 1, heads, 1).to(device))
    layer.dt_bias = nn.Parameter(dt_bias_cpu.to(device))

    class DecodeWrapper(nn.Module):
        def __init__(self, attention: nn.Module) -> None:
            super().__init__()
            self.attention = attention

        def forward(self, q, k, v, raw_gate, beta, state, cu_seqlens, state_indices):
            return self.attention._run_recurrent(
                q,
                k,
                v,
                raw_gate,
                beta,
                state,
                cu_seqlens,
                state_indices,
            )

    config = nge.CompilerConfig()
    compiled = torch.compile(
        DecodeWrapper(layer),
        backend=nge.get_npu_backend(compiler_config=config),
        dynamic=False,
    )
    compiled_state = initial_state_cpu.to(device)
    reference_state = initial_state_cpu
    cu_seqlens = cu_seqlens_cpu.to(device)
    state_indices = state_indices_cpu.to(device)

    for _ in range(steps):
        q_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16)
        k_cpu = torch.randn_like(q_cpu)
        v_cpu = torch.randn_like(q_cpu)
        raw_gate_cpu = torch.randn(1, batch, heads, dim, dtype=torch.bfloat16) * 2.0
        beta_cpu = torch.rand(1, batch, heads, dtype=torch.float32).sigmoid()
        reference_out, reference_state = recurrent_kda_reference(
            q_cpu,
            k_cpu,
            v_cpu,
            raw_gate_cpu,
            beta_cpu,
            reference_state,
            cu_seqlens=cu_seqlens_cpu.tolist(),
            ssm_state_indices=state_indices_cpu,
            A_log=a_log_cpu,
            dt_bias=dt_bias_cpu,
            layout="BSND",
            scale=dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
        )
        compiled_out = compiled(
            q_cpu.to(device),
            k_cpu.to(device),
            v_cpu.to(device),
            raw_gate_cpu.to(device),
            beta_cpu.to(device),
            compiled_state,
            cu_seqlens,
            state_indices,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(compiled_out.cpu(), reference_out, rtol=0.02, atol=0.02)
        torch.testing.assert_close(compiled_state.cpu(), reference_state, rtol=0.02, atol=0.02)


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_safe_gate_and_spec_state_slots():
    torch.manual_seed(20260721)
    device = torch.device("npu")
    dtype = torch.bfloat16

    # Kimi K3 has 96 KDA heads globally. TP16 exposes six heads to each
    # recurrent kernel invocation. cu_seqlens follows the FLA prefix-sum format.
    cu_seqlens_host = [0, 1, 4, 4, 5]
    tokens, heads, dim = cu_seqlens_host[-1], 6, 128
    q_cpu = torch.randn(1, tokens, heads, dim, dtype=dtype)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn_like(q_cpu)
    raw_gate_cpu = torch.randn(1, tokens, heads, dim, dtype=torch.float32) * 0.25
    beta_cpu = torch.rand(1, tokens, heads, dtype=torch.float32).sigmoid()
    # The production cache is a capacity-sized pool, not one state per active
    # sequence. Non-contiguous slots exercise the same indirection contract as
    # the Triton implementation and make accidental seq-index addressing fail.
    # Active sequences own disjoint slots; draft steps within one sequence may
    # reuse a slot.
    state_slots = 13
    state_cpu = torch.randn(state_slots, heads, dim, dim, dtype=torch.float32) * 0.01
    a_log_cpu = torch.randn(heads, dtype=torch.float32) * 0.05
    dt_bias_cpu = torch.randn(heads * dim, dtype=torch.float32) * 0.05
    state_indices_cpu = torch.tensor(
        [[8, 8, 8, 8, 8], [2, 11, 6, 2, 2], [4, 4, 4, 4, 4], [1, 3, 5, 7, 9]],
        dtype=torch.int64,
    )
    accepted_cpu = torch.tensor([1, 2, 0, 1], dtype=torch.int64)

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
        num_accepted_tokens=accepted_cpu,
        layout="BSND",
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )

    state_npu = state_cpu.to(device)
    out = torch.ops._C_ascend.recurrent_kda(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        raw_gate_cpu.to(device),
        beta_cpu.to(device),
        state_npu,
        torch.tensor(cu_seqlens_host, dtype=torch.int32, device=device),
        state_indices_cpu.to(device),
        a_log_cpu.to(device),
        dt_bias_cpu.to(device),
        num_accepted_tokens=accepted_cpu.to(device),
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), ref_out, rtol=0.02, atol=0.02)
    final_state_cpu = state_npu.cpu()
    torch.testing.assert_close(final_state_cpu, ref_state, rtol=0.02, atol=0.02)
    touched = {1, 2, 6, 8, 11}
    untouched = [slot for slot in range(state_slots) if slot not in touched]
    torch.testing.assert_close(final_state_cpu[untouched], state_cpu[untouched], rtol=0, atol=0)


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_full_decode_graph_padding():
    """Cover one real decode token replayed in a graph with capacity 16."""
    from vllm_ascend.ops.triton.kda.kda import fused_kda_gate, fused_recurrent_kda

    torch.manual_seed(20260723)
    device = torch.device("npu")
    graph_tokens, active_tokens, heads, dim = 16, 1, 6, 128
    active_slot, state_capacity = 2, 17

    q_cpu = torch.zeros(1, graph_tokens, heads, dim, dtype=torch.bfloat16)
    k_cpu = torch.zeros_like(q_cpu)
    v_cpu = torch.zeros_like(q_cpu)
    raw_gate_cpu = torch.zeros_like(q_cpu)
    beta_cpu = torch.zeros(1, graph_tokens, heads, dtype=torch.float32)
    q_cpu[:, :active_tokens].normal_()
    k_cpu[:, :active_tokens].normal_()
    v_cpu[:, :active_tokens].normal_()
    raw_gate_cpu[:, :active_tokens].normal_(std=0.25)
    beta_cpu[:, :active_tokens].uniform_().sigmoid_()

    state_cpu = torch.randn(state_capacity, heads, dim, dim, dtype=torch.float32) * 0.01
    a_log_cpu = torch.linspace(-0.43, 1.26, heads, dtype=torch.float32)
    dt_bias_cpu = torch.linspace(-9.0, -1.47, heads * dim, dtype=torch.float32)
    # FULL_DECODE_ONLY pads graph metadata with zero-length rows. The QKV
    # tensors retain the capture capacity, while the terminal cumulative
    # offset is the number of real tokens rather than graph_tokens.
    cu_seqlens_cpu = torch.tensor([0, 1] + [1] * (graph_tokens - 1), dtype=torch.int32)
    state_indices_cpu = torch.tensor([active_slot] + [0] * (graph_tokens - 1), dtype=torch.int64)

    reference_out, reference_state = recurrent_kda_reference(
        q_cpu[:, :active_tokens],
        k_cpu[:, :active_tokens],
        v_cpu[:, :active_tokens],
        raw_gate_cpu[:, :active_tokens],
        beta_cpu[:, :active_tokens],
        state_cpu,
        cu_seqlens=[0, active_tokens],
        ssm_state_indices=state_indices_cpu[:active_tokens],
        A_log=a_log_cpu,
        dt_bias=dt_bias_cpu,
        layout="BSND",
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
    )

    q, k, v = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu))
    raw_gate = raw_gate_cpu.to(device)
    beta = beta_cpu.to(device)
    cu_seqlens = cu_seqlens_cpu.to(device)
    state_indices = state_indices_cpu.to(device)
    a_log = a_log_cpu.to(device)
    dt_bias = dt_bias_cpu.to(device)
    ascendc_state = state_cpu.to(device)
    triton_state = state_cpu.to(device)

    ascendc_out = torch.ops._C_ascend.recurrent_kda(
        q,
        k,
        v,
        raw_gate,
        beta,
        ascendc_state,
        cu_seqlens,
        state_indices,
        a_log,
        dt_bias,
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
    )
    gate = fused_kda_gate(
        raw_gate.reshape(graph_tokens, heads * dim),
        a_log.reshape(1, 1, heads, 1),
        dim,
        g_bias=dt_bias,
        safe_gate=True,
        lower_bound=-5.0,
    ).unsqueeze(0)
    triton_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=triton_state,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
    )
    torch.npu.synchronize()

    assert isinstance(ascendc_out, torch.Tensor)
    assert torch.isfinite(ascendc_out).all()
    # A graph-padded call must update the active cache line while skipping
    # zero-length rows. The kernel clears the unused static output tail.
    padding_out = ascendc_out[:, active_tokens:].cpu()
    torch.testing.assert_close(
        padding_out,
        torch.zeros_like(padding_out),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(ascendc_state.cpu(), triton_state.cpu(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(ascendc_state.cpu(), reference_state, rtol=0.02, atol=0.02)
    torch.testing.assert_close(
        ascendc_out[:, :active_tokens].cpu(),
        triton_out[:, :active_tokens].cpu(),
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        ascendc_out[:, :active_tokens].cpu(),
        reference_out,
        rtol=0.02,
        atol=0.02,
    )


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_bsnd_mtp_lengths_1_to_8():
    torch.manual_seed(20260722)
    device = torch.device("npu")
    lengths = list(range(1, 9))
    cu_seqlens_host = [0]
    for length in lengths:
        cu_seqlens_host.append(cu_seqlens_host[-1] + length)

    tokens, heads, dim = cu_seqlens_host[-1], 6, 128
    q_cpu = torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn_like(q_cpu)
    raw_gate_cpu = torch.randn(1, tokens, heads, dim, dtype=torch.float32) * 0.25
    beta_cpu = torch.rand(1, tokens, heads, dtype=torch.float32)
    state_cpu = torch.randn(16, heads, dim, dim, dtype=torch.float32) * 0.01
    slots = [8, 2, 11, 6, 4, 15, 1, 13]
    state_indices_cpu = torch.tensor([[slot] * 8 for slot in slots], dtype=torch.int64)
    accepted_cpu = torch.tensor(lengths, dtype=torch.int64)
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
        num_accepted_tokens=accepted_cpu,
        layout="BSND",
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
    )

    state_npu = state_cpu.to(device)
    out = torch.ops._C_ascend.recurrent_kda(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        raw_gate_cpu.to(device),
        beta_cpu.to(device),
        state_npu,
        torch.tensor(cu_seqlens_host, dtype=torch.int32, device=device),
        state_indices_cpu.to(device),
        a_log_cpu.to(device),
        dt_bias_cpu.to(device),
        num_accepted_tokens=accepted_cpu.to(device),
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), ref_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state_npu.cpu(), ref_state, rtol=0.02, atol=0.02)


@torch.inference_mode()
def test_kimi_k3_recurrent_kda_decode_wrapper_aclgraph_replay():
    """Replay the decode wrapper after changing graph-static device metadata buffers."""
    from torch import nn

    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention

    torch.manual_seed(20260724)
    device = torch.device("npu")
    tokens, heads, dim = 4, 6, 128
    q = torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn(1, tokens, heads, dim, dtype=torch.float32, device=device) * 0.25
    beta = torch.rand(1, tokens, heads, dtype=torch.float32, device=device)
    state_initial = torch.randn(12, heads, dim, dim, dtype=torch.float32, device=device) * 0.01
    state_graph = state_initial.clone()
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    state_indices = torch.tensor(
        [[1, 2, 2, 2, 2, 2, 2, 2], [4, 5, 5, 5, 5, 5, 5, 5]],
        dtype=torch.int64,
        device=device,
    )
    accepted = torch.tensor([1, 2], dtype=torch.int64, device=device)

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = dim
    layer.gate_lower_bound = -5.0
    layer.A_log = nn.Parameter(torch.randn(1, 1, heads, 1, dtype=torch.float32, device=device) * 0.05)
    layer.dt_bias = nn.Parameter(torch.randn(heads, dim, dtype=torch.float32, device=device) * 0.05)

    def invoke(state: torch.Tensor) -> torch.Tensor:
        return layer._run_recurrent(
            q,
            k,
            v,
            raw_gate,
            beta,
            state,
            cu_seqlens,
            state_indices,
            num_accepted_tokens=accepted,
        )

    invoke(state_graph)
    torch.npu.synchronize()
    state_graph.copy_(state_initial)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_out = invoke(state_graph)
    torch.npu.synchronize()

    q.copy_(torch.randn_like(q))
    k.copy_(torch.randn_like(k))
    v.copy_(torch.randn_like(v))
    raw_gate.copy_(torch.randn_like(raw_gate) * 0.25)
    beta.copy_(torch.rand_like(beta))
    cu_seqlens.copy_(torch.tensor([0, 1, 2], dtype=torch.int32, device=device))
    state_indices.copy_(
        torch.tensor(
            [[7, 8, 8, 8, 8, 8, 8, 8], [3, 6, 6, 6, 6, 6, 6, 6]],
            dtype=torch.int64,
            device=device,
        )
    )
    accepted.copy_(torch.tensor([1, 1], dtype=torch.int64, device=device))
    state_graph.copy_(state_initial)
    graph.replay()
    torch.npu.synchronize()

    state_eager = state_initial.clone()
    eager_out = invoke(state_eager)
    torch.npu.synchronize()
    graph_out_cpu = graph_out.cpu()
    torch.testing.assert_close(graph_out_cpu, eager_out.cpu(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(state_graph.cpu(), state_eager.cpu(), rtol=0.02, atol=0.02)
    padding_out = graph_out_cpu[:, 2:]
    torch.testing.assert_close(
        padding_out,
        torch.zeros_like(padding_out),
        rtol=0,
        atol=0,
    )

    # Replaying the same graph with no active tokens must clear the complete
    # static output without touching the state pool.
    cu_seqlens.zero_()
    accepted.zero_()
    state_graph.copy_(state_initial)
    graph.replay()
    torch.npu.synchronize()
    graph_out_cpu = graph_out.cpu()
    torch.testing.assert_close(graph_out_cpu, torch.zeros_like(graph_out_cpu), rtol=0, atol=0)
    torch.testing.assert_close(state_graph.cpu(), state_initial.cpu(), rtol=0, atol=0)
