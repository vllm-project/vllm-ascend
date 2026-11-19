# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

import vllm_ascend.ops  # noqa: F401 -- initialize the operator registry before the device adaptor
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="A5 FLA Chunk32 integration")
@pytest.mark.parametrize("head_first", [True, False], ids=["BNSD", "BSND"])
def test_chunk32_matches_chunk64(head_first):
    device = torch.device("npu:0")
    rng = torch.Generator().manual_seed(17)
    # Packed lengths 31, 0, 33, 65 cross both tile boundaries.
    offsets = torch.tensor([0, 31, 31, 64, 129], dtype=torch.int32)
    meta = _build_non_spec_chunked_prefill_metadata(None, offsets, device, build_device_metadata=False)
    assert meta.chunk_indices_chunk32_host == (0, 0, 1, 0, 1, 1, 2, 0, 2, 1, 2, 2)

    def randn(*shape):
        return torch.randn(shape, generator=rng).bfloat16().to(device)

    qkv = [randn(1, heads, 129, 128).transpose(1, 2) for heads in (8, 8, 24)]
    if not head_first:
        qkv = [x.contiguous() for x in qkv]
    g = -torch.rand(1, 129, 24, generator=rng).to(device) * 0.2
    b = randn(1, 129, 24)
    state = randn(4, 24, 128, 128) * 0.05
    args = (state, 128**-0.5, meta, chunk_gated_delta_rule_fwd)
    ref = DeviceOperator.fla_gdn_prefill(*qkv, g, b.sigmoid(), *args)
    actual = DeviceOperator.fla_gdn_prefill(*qkv, g, b, *args, chunk_size=32)
    torch.testing.assert_close(actual[1][1], state[1], rtol=0, atol=0)
    for got, expected in zip(actual, ref):
        error = (got.float() - expected.float()).norm() / expected.float().norm()
        assert error.item() < 0.03
