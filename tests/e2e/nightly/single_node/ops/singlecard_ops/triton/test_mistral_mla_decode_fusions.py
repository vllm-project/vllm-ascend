# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mistral_mla_k_rope import try_slice_k_rope
from vllm_ascend.ops.triton.mistral_mla_q import try_ssmax_split_rope

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")


@pytest.mark.parametrize("tokens", [1, 4, 8, 9])
def test_mistral_mla_decode_fusions_match_reference(tokens):
    ssmax = pytest.importorskip("vllm.models.mistral.amd.ssmax")
    torch.manual_seed(1024)
    heads = 8
    scaling = ssmax.SSMaxScaling(heads, tp_sharded=False).npu().bfloat16()
    with torch.no_grad():
        scaling.s.copy_(0.02 + torch.rand_like(scaling.s) * 0.08)
        scaling.b.copy_(0.8 + torch.rand_like(scaling.b) * 0.4)
    query = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device="npu")
    kv = torch.randn(tokens, 1, 1, 576, dtype=torch.bfloat16, device="npu")
    log_positions = torch.arange(16384, 16384 + tokens, dtype=torch.float32, device="npu").add(1).log()
    angles = torch.randn(tokens, 1, 1, 32, device="npu")
    cos = angles.cos().repeat(1, 1, 1, 2).bfloat16()
    sin = angles.sin().repeat(1, 1, 1, 2).bfloat16()

    actual_query = try_ssmax_split_rope(query, log_positions, scaling.s, scaling.b, cos, sin)
    actual_key = try_slice_k_rope(kv, cos, sin)
    if tokens == 9:
        assert actual_query is None
        assert actual_key is None
        return

    expected_nope, expected_pe = scaling(query, log_positions).split(64, dim=-1)
    expected_query_pe = torch_npu.npu_interleave_rope(expected_pe.unsqueeze(2), cos, sin).squeeze(2)
    expected_key = torch_npu.npu_interleave_rope(kv[..., 512:], cos, sin)
    assert actual_query is not None and actual_key is not None
    torch.testing.assert_close(actual_query[0], expected_nope, rtol=0, atol=0)
    torch.testing.assert_close(actual_query[1], expected_query_pe, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
