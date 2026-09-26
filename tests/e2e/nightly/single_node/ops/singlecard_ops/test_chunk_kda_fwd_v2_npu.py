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

"""Numerical coverage for the migrated chunk-KDA V2 flags and fused fallback."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

_CHUNK_SIZE = 64
_HEAD_DIM = 128


def _l2norm(x, epsilon):
    return (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + epsilon)).to(x.dtype)


def _input_layout(x, layout):
    if layout == "BSND":
        return x.contiguous()
    if layout == "BNSD":
        return x.transpose(1, 2).contiguous()
    if layout == "TND":
        return x[0].contiguous()
    assert layout == "NTD"
    return x[0].transpose(0, 1).contiguous()


def _recurrent_reference(q, k, v, gate, beta, initial_state, cu_seqlens=None):
    """Token recurrence independent of the kernel's chunk/matrix algorithm."""
    q, k, v, gate, beta = (x.float() for x in (q, k, v, gate, beta))
    group_size = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(group_size, dim=2)
    k = k.repeat_interleave(group_size, dim=2)
    output = torch.empty_like(v)
    final_state = initial_state.clone()
    sequences = (
        [(batch, 0, q.shape[1]) for batch in range(q.shape[0])]
        if cu_seqlens is None
        else [(0, start, end) for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
    )
    for sequence, (batch, start, end) in enumerate(sequences):
        state = initial_state[sequence].clone()
        for token in range(start, end):
            state *= gate[batch, token].exp().unsqueeze(-1)
            key = k[batch, token]
            residual = v[batch, token] - torch.einsum("hk,hkv->hv", key, state)
            state += torch.einsum("hk,hv->hkv", key * beta[batch, token, :, None], residual)
            output[batch, token] = torch.einsum("hk,hkv->hv", q[batch, token] * q.shape[-1] ** -0.5, state)
        final_state[sequence] = state
    return output, final_state


@pytest.mark.parametrize(
    ("layout", "cu_seqlens", "state_v_first", "gate_mode"),
    [
        pytest.param("BSND", None, False, "safe", id="dense-batch-safe"),
        pytest.param("BNSD", None, True, "softplus", id="dense-head-major-softplus"),
        pytest.param("TND", (0, 3, 68, 131), True, "safe", id="packed-tails-safe"),
        pytest.param("NTD", (0, 3, 68, 131), False, "activated", id="packed-head-major-activated"),
    ],
)
@pytest.mark.parametrize(
    "flags",
    [
        pytest.param({}, id="default-fused-baseline"),
        pytest.param({"epsilon": 1e-3, "use_qk_l2norm_in_kernel": True}, id="l2norm-epsilon"),
        pytest.param({"use_beta_sigmoid_in_kernel": True}, id="beta-sigmoid"),
        pytest.param(
            {
                "epsilon": 1e-3,
                "use_qk_l2norm_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": True,
                "use_exp2": False,
            },
            id="l2norm-negative-eigenvalues-natural-exp",
        ),
    ],
)
@torch.inference_mode()
def test_chunk_kda_v2_flags_match_recurrence(layout, cu_seqlens, state_v_first, gate_mode, flags):
    torch.manual_seed(20260923)
    batch_size, tokens, query_heads, value_heads = 1 if cu_seqlens else 2, 131, 1, 2
    q = torch.randn(batch_size, tokens, query_heads, _HEAD_DIM).to(torch.bfloat16)
    k = torch.randn_like(q)
    if flags.get("use_qk_l2norm_in_kernel", False):
        # Make epsilon material so ignoring the non-default value fails accuracy.
        q, k = q * 0.003, k * 0.003
        q_ref, k_ref = (_l2norm(x, flags["epsilon"]) for x in (q, k))
    else:
        q, k = (_l2norm(x, 1e-6) for x in (q, k))
        q_ref, k_ref = q, k
    v = (torch.randn(batch_size, tokens, value_heads, _HEAD_DIM) * 0.2).to(torch.bfloat16)
    raw_gate = (torch.randn_like(v.float()) * 0.2).to(torch.bfloat16)
    a_log = torch.linspace(-0.2, 0.2, value_heads)
    dt_bias = torch.linspace(-4.0, -2.0, value_heads * _HEAD_DIM)
    shifted_gate = raw_gate.float() + dt_bias.view(1, 1, value_heads, _HEAD_DIM)
    if gate_mode == "softplus":
        gate = -a_log.exp().view(1, 1, value_heads, 1) * torch.nn.functional.softplus(shifted_gate)
    else:
        gate = -5.0 * torch.sigmoid(a_log.exp().view(1, 1, value_heads, 1) * shifted_gate)
    use_gate_in_kernel = gate_mode != "activated"
    gate_input = raw_gate if use_gate_in_kernel else gate
    beta = torch.randn(batch_size, tokens, value_heads)
    if cu_seqlens is not None:
        beta = beta.to(torch.bfloat16)
    if flags.get("use_beta_sigmoid_in_kernel", False):
        beta_ref = beta.float().sigmoid() * (2 if flags.get("allow_neg_eigval", False) else 1)
    else:
        beta = beta.sigmoid()
        beta_ref = beta
    sequence_count = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
    initial_kv = torch.randn(sequence_count, value_heads, _HEAD_DIM, _HEAD_DIM) * 0.03
    expected_output, expected_state = _recurrent_reference(q_ref, k_ref, v, gate, beta_ref, initial_kv, cu_seqlens)
    initial_state = initial_kv.transpose(-1, -2).contiguous() if state_v_first else initial_kv.clone()
    state_npu = initial_state.npu()
    result = torch.ops._C_ascend.chunk_kda_fwd(
        *(_input_layout(x, layout).npu() for x in (q, k, v, gate_input, beta)),
        _HEAD_DIM**-0.5,
        _CHUNK_SIZE,
        layout=layout,
        initial_state=state_npu,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        safe_gate=gate_mode != "softplus",
        lower_bound=-5.0,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=a_log.npu() if use_gate_in_kernel else None,
        dt_bias=dt_bias.npu() if use_gate_in_kernel else None,
        state_v_first=state_v_first,
        disable_recompute=state_v_first,
        **flags,
    )
    if layout in ("TND", "NTD"):
        expected_output = expected_output[0]
    if state_v_first:
        expected_state = expected_state.transpose(-1, -2).contiguous()
    assert len(result) == 12
    assert result[11] is state_npu
    assert (result[2] is not None) == (not use_gate_in_kernel or state_v_first)
    assert all((result[index] is not None) == state_v_first for index in range(5, 11))
    for output in result:
        if output is not None:
            assert torch.isfinite(output).all().item()
    torch.testing.assert_close(state_npu.cpu(), initial_state, rtol=0, atol=0)
    torch.testing.assert_close(result[0].float().cpu(), expected_output, rtol=3e-2, atol=2e-3)
    torch.testing.assert_close(result[1].cpu(), expected_state, rtol=3e-2, atol=5e-3)


@pytest.mark.parametrize(
    ("dtype", "head_dim", "chunk_size", "cu_seqlens"),
    [
        pytest.param(torch.float16, 128, 64, (0, 65), id="fp16"),
        pytest.param(torch.bfloat16, 64, 64, (0, 65), id="head-dim-64"),
        pytest.param(torch.bfloat16, 128, 128, (0, 65), id="chunk-size-128"),
        pytest.param(torch.bfloat16, 128, 64, (0, 0, 65), id="empty-sequence"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_default_flags_fallback_matches_recurrence(dtype, head_dim, chunk_size, cu_seqlens):
    torch.manual_seed(20260923)
    q = _l2norm(torch.randn(1, 65, 1, head_dim).to(dtype), 1e-6)
    k = _l2norm(torch.randn_like(q), 1e-6)
    v = torch.randn_like(q) * 0.2
    gate = -torch.rand(q.shape) * 0.03
    beta = torch.rand(1, 65, 1)
    state = torch.randn(len(cu_seqlens) - 1, 1, head_dim, head_dim) * 0.03
    expected_output, expected_state = _recurrent_reference(q, k, v, gate, beta, state, cu_seqlens)
    result = torch.ops._C_ascend.chunk_kda_fwd(
        *(x.npu() for x in (q, k, v, gate, beta)),
        head_dim**-0.5,
        chunk_size,
        initial_state=state.npu(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(result[0].float().cpu(), expected_output, rtol=3e-2, atol=2e-3)
    torch.testing.assert_close(result[1].cpu(), expected_state, rtol=3e-2, atol=5e-3)
    if cu_seqlens[0] == cu_seqlens[1]:
        torch.testing.assert_close(result[1][0].cpu(), state[0], rtol=0, atol=0)


@pytest.mark.parametrize(("key_dim", "value_dim"), [(64, 128), (128, 64), (128, 256), (96, 96)])
def test_chunk_kda_rejects_unsupported_head_dimensions(key_dim, value_dim):
    q = torch.empty((1, 64, 1, key_dim), dtype=torch.bfloat16, device="npu")
    v = torch.empty((1, 64, 1, value_dim), dtype=q.dtype, device=q.device)
    gate = torch.empty_like(q, dtype=torch.float32)
    beta = torch.empty((1, 64, 1), dtype=torch.float32, device=q.device)
    with pytest.raises(RuntimeError, match="K.*V|head.*dim"):
        torch.ops._C_ascend.chunk_kda_fwd(q, q, v, gate, beta, key_dim**-0.5, _CHUNK_SIZE)


@pytest.mark.parametrize(
    ("dtype", "head_dim", "chunk_size", "cu_seqlens"),
    [
        pytest.param(torch.float16, 128, 64, None, id="fp16"),
        pytest.param(torch.bfloat16, 64, 64, None, id="head-dim-64"),
        pytest.param(torch.bfloat16, 128, 128, None, id="chunk-size-128"),
        pytest.param(torch.bfloat16, 128, 64, (0, 0, 64), id="empty-sequence"),
    ],
)
@pytest.mark.parametrize(
    "flags",
    [
        {"use_qk_l2norm_in_kernel": True},
        {"use_beta_sigmoid_in_kernel": True},
        {"use_exp2": False},
    ],
)
def test_chunk_kda_rejects_nondefault_flags_outside_v2(dtype, head_dim, chunk_size, cu_seqlens, flags):
    q = torch.empty((1, 64, 1, head_dim), dtype=dtype, device="npu")
    gate = torch.empty_like(q, dtype=torch.float32)
    beta = torch.empty((1, 64, 1), dtype=torch.float32, device=q.device)
    with pytest.raises(RuntimeError, match="V2|BF16|bfloat16"):
        torch.ops._C_ascend.chunk_kda_fwd(
            q, q, q, gate, beta, head_dim**-0.5, chunk_size, cu_seqlens=cu_seqlens, **flags
        )


def test_chunk_kda_rejects_negative_eigenvalues_without_beta_sigmoid():
    q = torch.empty((1, 64, 1, _HEAD_DIM), dtype=torch.bfloat16, device="npu")
    gate = torch.empty_like(q, dtype=torch.float32)
    beta = torch.empty((1, 64, 1), dtype=torch.float32, device=q.device)
    with pytest.raises(RuntimeError, match="allow_neg_eigval.*use_beta_sigmoid_in_kernel"):
        torch.ops._C_ascend.chunk_kda_fwd(q, q, q, gate, beta, _HEAD_DIM**-0.5, _CHUNK_SIZE, allow_neg_eigval=True)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf"), 1e-300, 1e300])
def test_chunk_kda_rejects_invalid_epsilon(epsilon):
    q = torch.empty((1, 64, 1, _HEAD_DIM), dtype=torch.bfloat16, device="npu")
    gate = torch.empty_like(q, dtype=torch.float32)
    beta = torch.empty((1, 64, 1), dtype=torch.float32, device=q.device)
    with pytest.raises(RuntimeError, match="epsilon must be a positive finite float32"):
        torch.ops._C_ascend.chunk_kda_fwd(q, q, q, gate, beta, _HEAD_DIM**-0.5, _CHUNK_SIZE, epsilon=epsilon)
