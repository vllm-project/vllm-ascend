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

"""FLA-style numerical coverage for the standalone Triton KDA chunk op."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from torch import nn

from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention
from vllm_ascend.ops.triton.kda.kda import chunk_kda
from vllm_ascend.ops.triton.kda.utils import prepare_chunk_indices, prepare_chunk_offsets


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)).to(dtype)


def _naive_recurrent_kda(
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


def _assert_rmse(name: str, actual: torch.Tensor, expected: torch.Tensor, threshold: float) -> None:
    difference = (actual.float() - expected.float()).flatten()
    rmse_ratio = difference.square().mean().sqrt() / (expected.float().flatten().square().mean().sqrt() + 1e-8)
    print(f"{name}: rmse ratio={rmse_ratio.item():.6f}, threshold={threshold}")
    assert not torch.isnan(actual).any(), f"{name} contains NaN"
    assert rmse_ratio.item() < threshold


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_kimi_k3_triton_chunk_varlen_safe_gate_and_canonical_state():
    """D=128/chunk64, varlen, initial/final state, and K3 safe gate."""
    torch.manual_seed(20260720)
    cu_seqlens_host = (0, 17, 64)
    tokens, heads, head_dim = cu_seqlens_host[-1], 1, 128
    dtype = torch.float16

    q = torch.randn(1, tokens, heads, head_dim, dtype=dtype, device="npu")
    k = torch.randn_like(q)
    v = torch.randn_like(q) * 0.05
    raw_gate = torch.randn_like(q) * 0.1
    beta = torch.rand(1, tokens, heads, dtype=torch.float32, device="npu").sigmoid()
    # chunk_kda intentionally reuses its contiguous v input as the output
    # buffer, so preserve the source values for the recurrent reference.
    v_before = v.clone()

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = head_dim
    layer.gate_lower_bound = -5.0
    layer.A_log = nn.Parameter(torch.randn(1, 1, heads, 1, dtype=torch.float32, device="npu") * 0.05)
    layer.dt_bias = nn.Parameter(torch.randn(heads * head_dim, dtype=torch.float32, device="npu") * 0.05)

    cache_vk = torch.randn(5, heads, head_dim, head_dim, dtype=torch.float32, device="npu") * 0.01
    state_indices = torch.tensor([1, 3], dtype=torch.int64, device="npu")
    cache_before = cache_vk.clone()
    has_initial_state = torch.ones(2, dtype=torch.bool, device="npu")

    cu_seqlens_cpu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    prebuilt = SimpleNamespace(
        chunk_indices_chunk64=prepare_chunk_indices(cu_seqlens_cpu, 64).to("npu"),
        chunk_offsets_chunk64=prepare_chunk_offsets(cu_seqlens_cpu, 64).to("npu"),
    )

    initial_state_vk = cache_vk[state_indices].contiguous()
    initial_state_vk[~has_initial_state] = 0
    actual_out, final_state_vk = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=layer._recurrent_gate(raw_gate),
        beta=beta,
        initial_state=initial_state_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_cpu.to(device="npu", dtype=torch.int64),
        prebuilt_meta=prebuilt,
    )
    cache_vk[state_indices] = final_state_vk

    safe_gate = layer.gate_lower_bound * torch.sigmoid(
        (raw_gate.float() + layer.dt_bias.view(1, 1, heads, head_dim)) * layer.A_log.exp().view(1, 1, heads, 1)
    )
    expected_outputs = []
    expected_cache = cache_before.clone()
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens_host, cu_seqlens_host[1:])):
        slot = state_indices[seq_idx]
        initial_state_kv = cache_before[slot].transpose(-1, -2).unsqueeze(0)
        expected_out, expected_state_kv = _naive_recurrent_kda(
            _l2norm(q[:, start:end]),
            _l2norm(k[:, start:end]),
            v_before[:, start:end],
            safe_gate[:, start:end],
            beta[:, start:end],
            initial_state_kv,
        )
        expected_outputs.append(expected_out)
        expected_cache[slot] = expected_state_kv.squeeze(0).transpose(-1, -2)
    expected_out = torch.cat(expected_outputs, dim=1)

    _assert_rmse("output", actual_out, expected_out, 0.01)
    _assert_rmse("final_state", cache_vk[state_indices], expected_cache[state_indices], 0.01)
    untouched = torch.tensor([0, 2, 4], dtype=torch.int64, device="npu")
    torch.testing.assert_close(cache_vk[untouched], cache_before[untouched], rtol=0, atol=0)
