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

"""Kimi K3 integration coverage for the AscendC prefill operators."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


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


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_kimi_k3_safe_gate_prefill_and_transposed_state_layout():
    if not hasattr(torch.ops._C_ascend, "kda_gate_cumsum") or not hasattr(torch.ops._C_ascend, "chunk_kda_fwd"):
        pytest.skip("requires the KDA AscendC operators")

    torch.manual_seed(20260720)
    tokens, heads, head_dim = 64, 1, 128
    dtype = torch.float16
    q = _l2norm(torch.randn(1, tokens, heads, head_dim, dtype=dtype, device="npu"))
    k = _l2norm(torch.randn_like(q))
    v = torch.randn_like(q) * 0.05
    raw_gate = torch.randn_like(q) * 0.1
    beta = torch.rand(1, tokens, heads, dtype=torch.float32, device="npu").sigmoid()
    a_log = torch.randn(heads, dtype=torch.float32, device="npu") * 0.05
    dt_bias = torch.randn(heads * head_dim, dtype=torch.float32, device="npu") * 0.05
    cache_vk = torch.randn(1, heads, head_dim, head_dim, dtype=torch.float32, device="npu") * 0.01
    lower_bound = -5.0
    cu_seqlens = (0, tokens)
    chunk_indices = (0, 0)

    gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
        raw_gate,
        64,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        layout="BSND",
    )
    initial_state_kv = cache_vk.transpose(-1, -2).contiguous()
    got = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        gate_cumsum,
        beta,
        head_dim**-0.5,
        64,
        layout="BSND",
        initial_state=initial_state_kv,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        return_intermediate=False,
    )

    safe_gate = lower_bound * torch.sigmoid(
        (raw_gate.float() + dt_bias.view(1, 1, heads, head_dim)) * a_log.exp().view(1, 1, heads, 1)
    )
    expected_out, expected_state_kv = _naive_kda(q, k, v, safe_gate, beta, initial_state_kv)

    torch.testing.assert_close(got[0], expected_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(got[1], expected_state_kv, rtol=3e-2, atol=3e-2)
    # The vLLM decode cache remains [H,V,K] after crossing the AscendC boundary.
    cache_vk.copy_(got[1].transpose(-1, -2))
    torch.testing.assert_close(cache_vk.transpose(-1, -2), expected_state_kv, rtol=3e-2, atol=3e-2)
