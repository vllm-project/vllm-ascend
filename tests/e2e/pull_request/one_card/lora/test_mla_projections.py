# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Absorbed MLA LoRA increments against independently merged CPU weights."""
# ruff: noqa: E402

from types import SimpleNamespace

import pytest
import torch
from torch import nn

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("MLA LoRA acceptance requires a real NPU", allow_module_level=True)

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.lora.utils import AscendColumnParallelLinearWithLoRA


@pytest.mark.parametrize("prefill", [False, True])
def test_sfa_absorbed_query_and_value_use_real_adapter_factors(prefill):
    torch.npu.set_device(0)
    torch.manual_seed(54)
    device = torch.device("npu:0")
    heads, qdim, vdim, latent, rank = 2, 16, 16, 32, 8
    wrapper = object.__new__(AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(wrapper)
    wrapper.lora_config = SimpleNamespace(fully_sharded_loras=False)
    wrapper.tp_size = 1
    wrapper.punica_wrapper = SimpleNamespace()
    a = torch.randn(rank, latent, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(heads, qdim + vdim, rank, device=device, dtype=torch.bfloat16) * 0.1
    wrapper.lora_a_stacked = (a[None, None],)
    wrapper.lora_b_stacked = (b.reshape(1, 1, -1, rank),)
    base = torch.randn(heads, qdim + vdim, latent, device=device, dtype=torch.bfloat16) * 0.1
    attributes = dict(
        local_num_heads=heads,
        qk_head_dim=qdim,
        qk_nope_head_dim=qdim,
        qk_rope_head_dim=0,
        kv_lora_rank=latent,
        v_head_dim=vdim,
        W_UK_T=base[:, :qdim].contiguous(),
        W_UV=base[:, qdim:].transpose(1, 2).contiguous(),
    )
    owner = object.__new__(AscendSFAImpl)
    owner.q_lora_rank = None
    owner._init_projections({"q_proj": lambda x: (x, None), "kv_b_proj": wrapper}, lora_enabled=True)
    for name, value in attributes.items():
        setattr(owner, name, value)
    count = 3 if prefill else 1
    q = torch.randn(count, heads * qdim, device=device, dtype=torch.bfloat16) * 0.1
    x = torch.randn(count, heads, latent, device=device, dtype=torch.bfloat16) * 0.1
    effective = base.cpu().double() + b.cpu().double() @ a.cpu().double()
    for slot in (0, -1, 0):
        wrapper.punica_wrapper.token_lora_indices = torch.full((count,), slot, device=device)
        actual_q, _ = AscendSFAImpl._q_proj_and_k_up_proj(owner, q)
        actual_v = AscendSFAImpl._v_up_proj(owner, x).reshape(count, heads, vdim)
        torch.npu.synchronize()
        expected_q, expected_v = [], []
        weight = base.cpu().double() if slot < 0 else effective
        for row in range(count):
            expected_q.append(torch.einsum("hq,hqc->hc", q[row].cpu().double().view(heads, qdim), weight[:, :qdim]))
            expected_v.append(torch.einsum("hc,hvc->hv", x[row].cpu().double(), weight[:, qdim:]))
        torch.testing.assert_close(actual_q.cpu().double(), torch.stack(expected_q), atol=5e-4, rtol=3e-2)
        torch.testing.assert_close(actual_v.cpu().double(), torch.stack(expected_v), atol=5e-4, rtol=3e-2)
