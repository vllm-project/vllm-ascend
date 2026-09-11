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

"""Kimi K3 integration coverage for the fused AscendC prefill operator."""

import importlib
import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

CHUNK_KDA_OUTPUT_NAMES = (
    "o",
    "final_state",
    "gk",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state",
)


def _has_chunk_kda_op() -> bool:
    if hasattr(torch.ops._C_ascend, "chunk_kda_fwd"):
        return True
    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    except ImportError:
        return False
    return hasattr(torch.ops._C_ascend, "chunk_kda_fwd")


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)).to(dtype)


def _naive_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = v.dtype
    q, k, v, gate, beta = (x.float() for x in (q, k, v, gate, beta))
    state = initial_state_kv.float().clone()
    out = torch.empty_like(v)
    scale = q.shape[-1] ** -0.5
    for token in range(q.shape[1]):
        state *= gate[:, token].exp().unsqueeze(-1)
        residual = v[:, token] - torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        state += torch.einsum("bhk,bhv->bhkv", beta[:, token].unsqueeze(-1) * k[:, token], residual)
        out[:, token] = torch.einsum("bhk,bhkv->bhv", q[:, token] * scale, state)
    return out.to(dtype), state


def _chunked_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state_vk: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    q, k, v, gate, beta, initial_state_vk = (x.detach().cpu() for x in (q, k, v, gate, beta, initial_state_vk))
    batch_size, tokens, heads, head_dim = q.shape
    assert k.shape == q.shape == v.shape == gate.shape
    assert beta.shape == (batch_size, tokens, heads)

    dtype = q.dtype
    scale = head_dim**-0.5
    chunk_size = 64
    chunk_count = math.ceil(tokens / chunk_size)
    gk = torch.empty((batch_size, heads, tokens, head_dim), dtype=torch.float32)
    aqk = torch.zeros((batch_size, heads, tokens, chunk_size), dtype=dtype)
    akk = torch.zeros_like(aqk)
    w = torch.empty((batch_size, heads, tokens, head_dim), dtype=dtype)
    u = torch.empty_like(w)
    qg = torch.empty_like(w)
    kg = torch.empty_like(w)
    v_new = torch.empty_like(w)
    h = torch.empty((batch_size, chunk_count, heads, head_dim, head_dim), dtype=dtype)
    final_state_vk = torch.empty_like(initial_state_vk, dtype=torch.float32)
    o = torch.empty_like(v)

    for batch_idx in range(batch_size):
        for head_idx in range(heads):
            state_kv = initial_state_vk[batch_idx, head_idx].float().transpose(-1, -2).contiguous()
            for chunk_idx in range(chunk_count):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, tokens)
                chunk_tokens = end - start
                causal = torch.ones((chunk_tokens, chunk_tokens), dtype=torch.bool).tril()
                strict_causal = torch.ones_like(causal).tril(diagonal=-1)
                eye = torch.eye(chunk_tokens, dtype=torch.float32)
                q_block = q[batch_idx, start:end, head_idx].float()
                k_block = k[batch_idx, start:end, head_idx].float()
                v_block = v[batch_idx, start:end, head_idx].float()
                beta_block = beta[batch_idx, start:end, head_idx].float()
                gk_block = torch.cumsum(gate[batch_idx, start:end, head_idx].float(), dim=0) / math.log(2.0)
                relative_gate = gk_block[:, None, :] - gk_block[None, :, :]
                gate_factor = torch.exp2(relative_gate.masked_fill(~causal[:, :, None], 0.0))
                qk = torch.einsum("ik,jk,ijk->ij", q_block, k_block, gate_factor) * scale
                kk = torch.einsum("ik,jk,ijk->ij", k_block, k_block, gate_factor)
                aqk_block = torch.where(causal, qk, 0.0)
                strict_kk = torch.where(strict_causal, kk * beta_block[:, None], 0.0)
                akk_block = torch.linalg.solve_triangular(eye + strict_kk, eye, upper=False)

                k_beta_g = k_block * beta_block[:, None] * torch.exp2(gk_block)
                w_block = akk_block @ k_beta_g
                u_block = akk_block @ (v_block * beta_block[:, None])
                qg_block = q_block * torch.exp2(gk_block)
                kg_block = k_block * torch.exp2(gk_block[-1][None, :] - gk_block)
                v_new_block = u_block - w_block @ state_kv

                gk[batch_idx, head_idx, start:end] = gk_block
                aqk[batch_idx, head_idx, start:end, :chunk_tokens] = aqk_block.to(dtype)
                akk[batch_idx, head_idx, start:end, :chunk_tokens] = akk_block.to(dtype)
                w[batch_idx, head_idx, start:end] = w_block.to(dtype)
                u[batch_idx, head_idx, start:end] = u_block.to(dtype)
                qg[batch_idx, head_idx, start:end] = qg_block.to(dtype)
                kg[batch_idx, head_idx, start:end] = kg_block.to(dtype)
                v_new[batch_idx, head_idx, start:end] = v_new_block.to(dtype)
                h[batch_idx, chunk_idx, head_idx] = state_kv.transpose(-1, -2).to(dtype)
                o[batch_idx, start:end, head_idx] = (qg_block @ state_kv * scale + aqk_block @ v_new_block).to(dtype)
                state_kv = torch.exp2(gk_block[-1])[:, None] * state_kv + kg_block.T @ v_new_block
            final_state_vk[batch_idx, head_idx] = state_kv.transpose(-1, -2)

    return o, final_state_vk, gk, aqk, akk, w, u, qg, kg, v_new, h, initial_state_vk


def _assert_chunk_kda_outputs_close(actual, expected, retained_indices):
    assert len(actual) == len(expected) == len(CHUNK_KDA_OUTPUT_NAMES)
    for index, (name, expected_output) in enumerate(zip(CHUNK_KDA_OUTPUT_NAMES, expected)):
        if index not in retained_indices:
            assert actual[index] is None, f"{name} must be None"
            continue
        assert actual[index] is not None, f"{name} must be retained"
        torch.testing.assert_close(
            actual[index].detach().cpu(),
            expected_output,
            rtol=3e-2,
            atol=3e-2,
            msg=name,
        )


def _is_ascend_950() -> bool:
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_kimi_k3_safe_gate_prefill_and_transposed_state_layout():
    if not _has_chunk_kda_op():
        pytest.skip("requires the fused chunk KDA AscendC operator")

    torch.manual_seed(20260720)
    tokens, heads, head_dim = 64, 1, 128
    dtype = torch.float16
    q = _l2norm(torch.randn(1, tokens, heads, head_dim, dtype=dtype, device="npu"))
    k = _l2norm(torch.randn_like(q))
    v = torch.randn_like(q) * 0.05
    raw_gate = torch.randn(1, tokens, heads, head_dim, dtype=torch.float32, device="npu") * 0.1
    beta = torch.rand(1, tokens, heads, dtype=torch.float32, device="npu").sigmoid()
    a_log = torch.randn(heads, dtype=torch.float32, device="npu") * 0.05
    dt_bias = torch.randn(heads * head_dim, dtype=torch.float32, device="npu") * 0.05
    cache_vk = torch.randn(1, heads, head_dim, head_dim, dtype=torch.float32, device="npu") * 0.01
    lower_bound = -5.0
    cu_seqlens = (0, tokens)
    chunk_indices = (0, 0)

    initial_state_kv = cache_vk.transpose(-1, -2).contiguous()
    got = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        raw_gate,
        beta,
        head_dim**-0.5,
        64,
        layout="BSND",
        initial_state=cache_vk,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )

    retained = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        head_dim**-0.5,
        64,
        layout="BSND",
        initial_state=cache_vk,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        return_intermediate_states=True,
        state_v_first=True,
    )

    safe_gate = lower_bound * torch.sigmoid(
        (raw_gate.float() + dt_bias.view(1, 1, heads, head_dim)) * a_log.exp().view(1, 1, heads, 1)
    )
    expected_out, expected_state_kv = _naive_kda(q, k, v, safe_gate, beta, initial_state_kv)
    expected = _chunked_kda_reference(q, k, v, safe_gate, beta, cache_vk)

    torch.testing.assert_close(got[0], expected_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(got[1].transpose(-1, -2), expected_state_kv, rtol=3e-2, atol=3e-2)
    _assert_chunk_kda_outputs_close(got, expected, retained_indices={0, 1, 3, 4, 11})
    _assert_chunk_kda_outputs_close(
        retained,
        expected,
        retained_indices=set(range(len(CHUNK_KDA_OUTPUT_NAMES))),
    )
    assert got[11] is cache_vk
    assert retained[11] is cache_vk
    # The vLLM decode cache remains [H,V,K] after crossing the AscendC boundary.
    cache_vk.copy_(got[1])
    torch.testing.assert_close(cache_vk.transpose(-1, -2), expected_state_kv, rtol=3e-2, atol=3e-2)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(("tokens", "heads"), [(65, 1), (131, 6)])
@torch.inference_mode()
def test_kimi_k3_a5_multichunk_all_outputs_match_reference(tokens, heads):
    if not _has_chunk_kda_op():
        pytest.skip("requires the fused chunk KDA AscendC operator")
    if not _is_ascend_950():
        pytest.skip("requires an Ascend 950 device")

    torch.manual_seed(20260819)
    head_dim = 128
    dtype = torch.bfloat16
    q = _l2norm(torch.randn(1, tokens, heads, head_dim, dtype=dtype, device="npu"))
    k = _l2norm(torch.randn_like(q))
    v = torch.randn_like(q) * 0.05
    raw_gate = torch.randn(1, tokens, heads, head_dim, dtype=torch.float32, device="npu") * 0.1
    beta = torch.rand(1, tokens, heads, dtype=torch.float32, device="npu").sigmoid()
    a_log = torch.randn(heads, dtype=torch.float32, device="npu") * 0.05
    dt_bias = torch.randn(heads * head_dim, dtype=torch.float32, device="npu") * 0.05
    cache_vk = torch.randn(1, heads, head_dim, head_dim, dtype=torch.float32, device="npu") * 0.01
    chunk_indices = tuple(value for chunk_id in range(math.ceil(tokens / 64)) for value in (0, chunk_id))

    result = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        head_dim**-0.5,
        64,
        layout="BSND",
        initial_state=cache_vk,
        output_final_state=True,
        cu_seqlens=(0, tokens),
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )
    retained = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        head_dim**-0.5,
        64,
        layout="BSND",
        initial_state=cache_vk,
        output_final_state=True,
        cu_seqlens=(0, tokens),
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        return_intermediate_states=True,
        state_v_first=True,
    )

    safe_gate = -5.0 * torch.sigmoid(
        (raw_gate.float() + dt_bias.view(1, 1, heads, head_dim)) * a_log.exp().view(1, 1, heads, 1)
    )
    expected = _chunked_kda_reference(q, k, v, safe_gate, beta, cache_vk)
    _assert_chunk_kda_outputs_close(result, expected, retained_indices={0, 1, 3, 4, 11})
    _assert_chunk_kda_outputs_close(
        retained,
        expected,
        retained_indices=set(range(len(CHUNK_KDA_OUTPUT_NAMES))),
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("layout", ["BSND", "TND"])
@torch.inference_mode()
def test_kimi_k3_a5_model_prefill_profile_shape(layout):
    if not _has_chunk_kda_op():
        pytest.skip("requires the fused chunk KDA AscendC operator")
    if not _is_ascend_950():
        pytest.skip("requires an Ascend 950 device")

    torch.manual_seed(20260819)
    tokens, heads, head_dim = 8191, 12, 128
    dtype = torch.bfloat16
    q_bsnd = torch.full(
        (1, tokens, heads, head_dim),
        1.0 / math.sqrt(head_dim),
        dtype=dtype,
        device="npu",
    )
    k_bsnd = torch.full_like(q_bsnd, 1.0 / math.sqrt(head_dim))
    v_bsnd = torch.zeros_like(q_bsnd)
    raw_gate_bsnd = torch.zeros((1, tokens, heads, head_dim), dtype=torch.float32, device="npu")
    beta_bsnd = torch.full((1, tokens, heads), 0.5, dtype=torch.float32, device="npu")
    output_shape: tuple[int, ...]
    matrix_shape: tuple[int, ...]
    if layout == "TND":
        q, k, v = q_bsnd[0], k_bsnd[0], v_bsnd[0]
        raw_gate, beta = raw_gate_bsnd[0], beta_bsnd[0]
        output_shape = (tokens, heads, head_dim)
        matrix_shape = (heads, tokens, 64)
    else:
        q, k, v = q_bsnd, k_bsnd, v_bsnd
        raw_gate, beta = raw_gate_bsnd, beta_bsnd
        output_shape = (1, tokens, heads, head_dim)
        matrix_shape = (1, heads, tokens, 64)
    a_log = torch.zeros(heads, dtype=torch.float32, device="npu")
    dt_bias = torch.linspace(
        -9.0,
        -1.47,
        heads * head_dim,
        dtype=torch.float32,
        device="npu",
    )
    cache_vk = (
        torch.eye(head_dim, dtype=torch.float32, device="npu").view(1, 1, head_dim, head_dim).repeat(1, heads, 1, 1)
        * 0.01
    )
    cu_seqlens = (0, tokens)
    chunk_indices = tuple(value for chunk_id in range(math.ceil(tokens / 64)) for value in (0, chunk_id))

    result = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        head_dim**-0.5,
        64,
        layout=layout,
        initial_state=cache_vk,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )
    retained = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        head_dim**-0.5,
        64,
        layout=layout,
        initial_state=cache_vk,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        disable_recompute=True,
        return_intermediate_states=True,
        state_v_first=True,
    )

    assert len(result) == len(CHUNK_KDA_OUTPUT_NAMES)
    assert result[0].shape == output_shape
    assert result[1].shape == cache_vk.shape
    assert result[3].shape == result[4].shape == matrix_shape
    assert result[11] is cache_vk
    for index in (0, 1, 3, 4):
        assert torch.isfinite(result[index]).all().item(), CHUNK_KDA_OUTPUT_NAMES[index]
    for index in (2, 5, 6, 7, 8, 9, 10):
        assert result[index] is None, CHUNK_KDA_OUTPUT_NAMES[index]

    token_head_shape = (heads, tokens, head_dim)
    stored_token_head_shape = (1,) + token_head_shape if layout == "BSND" else token_head_shape
    retained_shapes = (
        output_shape,
        cache_vk.shape,
        stored_token_head_shape,
        matrix_shape,
        matrix_shape,
        stored_token_head_shape,
        stored_token_head_shape,
        stored_token_head_shape,
        stored_token_head_shape,
        stored_token_head_shape,
        (
            (1, math.ceil(tokens / 64), heads, head_dim, head_dim)
            if layout == "BSND"
            else (math.ceil(tokens / 64), heads, head_dim, head_dim)
        ),
        cache_vk.shape,
    )
    assert len(retained) == len(retained_shapes) == len(CHUNK_KDA_OUTPUT_NAMES)
    for name, output, shape in zip(CHUNK_KDA_OUTPUT_NAMES, retained, retained_shapes):
        assert output is not None, name
        assert output.shape == shape, name
        assert torch.isfinite(output).all().item(), name
    assert retained[11] is cache_vk
