# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu
from torch.nn import functional as F

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Atlas A3 kernel")


@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("tokens", [1, 3, 129])
def test_dsv41_mla_prolog(heads, tokens):
    _check_mla_prolog(heads, tokens, replay=False)


def test_dsv41_mla_prolog_aclgraph_replay():
    _check_mla_prolog(heads=16, tokens=3, replay=True)


def _check_mla_prolog(heads, tokens, *, replay):
    torch.manual_seed(2026)
    device = "npu"
    dtype = torch.bfloat16
    epsilon = 1e-20
    hidden, rank, head_dim, block_size, rope_dim = 5120, 1280, 512, 128, 64

    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    wqa = torch.randn(rank, hidden, dtype=dtype, device=device) / hidden**0.5
    wkv = torch.randn(head_dim, hidden, dtype=dtype, device=device) / hidden**0.5
    gamma_qa = torch.ones(rank, dtype=dtype, device=device)
    gamma_kv = torch.ones(head_dim, dtype=dtype, device=device)
    wqb = torch.randint(-16, 16, (rank, heads * head_dim), dtype=torch.int8, device=device)
    wqb = torch_npu.npu_format_cast(wqb, ACL_FORMAT_FRACTAL_NZ)
    wqb_scale = torch.full((heads * head_dim,), 0.003, dtype=dtype, device=device)
    angles = torch.randn(tokens, 1, 1, rope_dim // 2, device=device).repeat_interleave(2, dim=-1)
    cos, sin = angles.cos(), angles.sin()

    # Start in the last row of page zero. Cross the page boundary and include
    # invalid negative and positive INT64 slots while keeping all valid writes distinct.
    pages = 3
    valid_slots = list(range(block_size - 1, block_size - 1 + tokens))
    if tokens >= 3:
        valid_slots[-2:] = [-1, 1 << 33]
    slots = torch.tensor(valid_slots, dtype=torch.int64, device=device)
    # Model cache views can have gaps between pages.
    cache = torch.full((pages, block_size + 2, 1, head_dim), 17, dtype=dtype, device=device)
    args = (x, wqa, gamma_qa, wqb, wqb_scale, wkv, gamma_kv, cos, sin)

    def reference():
        qr, _ = torch_npu.npu_rms_norm(F.linear(x, wqa), gamma_qa, epsilon)
        kv, _ = torch_npu.npu_rms_norm(F.linear(x, wkv), gamma_kv, epsilon)
        q_int8, token_scale = torch_npu.npu_dynamic_quant(qr)
        q = torch_npu.npu_quant_matmul(q_int8, wqb, wqb_scale, pertoken_scale=token_scale, output_dtype=dtype)
        q = q.view(tokens, heads, head_dim)
        kv = kv.view(tokens, 1, head_dim)
        for tensor in (q, kv):
            DeviceOperator.apply_partial_rotary_inplace(tensor, cos, sin, start=head_dim - rope_dim, end=head_dim)
        return q, qr, kv

    def fused_write():
        return torch.ops._C_ascend.dsv41_mla_prolog(*args, slots, cache[:, :block_size], epsilon)

    def check_cache(q, qr, expected_slots):
        q_ref, qr_ref, kv_ref = reference()
        reference_cache = torch.full_like(cache, 17)
        for index, slot in enumerate(expected_slots):
            if 0 <= slot < pages * block_size:
                reference_cache[slot // block_size, slot % block_size] = kv_ref[index]
        torch.testing.assert_close(qr, qr_ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(q, q_ref, rtol=0.03, atol=0.03)
        torch.testing.assert_close(cache, reference_cache, rtol=0.02, atol=0.02)
        unwritten = reference_cache[..., 0] == 17
        torch.testing.assert_close(cache[unwritten], reference_cache[unwritten], rtol=0, atol=0)

    if replay:
        # Capture with dummy slots, then exercise two different live-token masks.
        slots.fill_(-1)
        fused_write()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            q, qr = fused_write()

        for replay_slots in ([127, 128, -1], [256, -1, -1]):
            x.normal_()
            angles = torch.randn_like(angles[..., ::2]).repeat_interleave(2, dim=-1)
            cos.copy_(angles.cos())
            sin.copy_(angles.sin())
            slots.copy_(torch.tensor(replay_slots, dtype=slots.dtype, device=device))
            cache.fill_(17)
            graph.replay()
            check_cache(q, qr, replay_slots)
    else:
        check_cache(*fused_write(), valid_slots)
        q_ref, qr_ref, kv_ref = reference()
        q, qr, kv = torch.ops._C_ascend.dsv41_mla_prolog_no_cache(*args, epsilon)
        torch.testing.assert_close(qr, qr_ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(q, q_ref, rtol=0.03, atol=0.03)
        torch.testing.assert_close(kv, kv_ref.squeeze(1), rtol=0.02, atol=0.02)
