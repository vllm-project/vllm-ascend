# SPDX-License-Identifier: Apache-2.0
from itertools import accumulate

import pytest
import torch
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

import vllm_ascend.ops  # noqa: F401 -- initialize the operator registry before the device adaptor
from vllm_ascend.attention.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware import AscendDeviceType


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="A3 FLA Chunk64 integration")
def test_packed_chunk64_matches_independent_sequences():
    lengths = [63, 0, 64, 65]
    tokens = sum(lengths)
    device = torch.device("npu:0")
    rng = torch.Generator().manual_seed(2048)

    def randn(*shape):
        return torch.randn(shape, generator=rng).bfloat16().to(device)

    q = randn(1, tokens, 8, 128)
    k = randn(1, tokens, 8, 128)
    v = randn(1, tokens, 24, 128)
    g = -torch.rand(1, tokens, 24, generator=rng).to(device) * 0.12
    beta = randn(1, tokens, 24).sigmoid()
    state = randn(len(lengths), 24, 128, 128) * 0.08

    def prefill(inputs, initial_state, offsets):
        meta = _build_non_spec_chunked_prefill_metadata(
            None, torch.tensor(offsets, dtype=torch.int32), device, build_device_metadata=False
        )
        return DeviceOperator.fla_gdn_prefill(*inputs, initial_state, 128**-0.5, meta, chunk_gated_delta_rule_fwd)

    inputs = (q, k, v, g, beta)
    offsets = list(accumulate(lengths, initial=0))
    output, final_state = prefill(inputs, state.clone(), offsets)
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            torch.testing.assert_close(final_state[seq], state[seq], rtol=0, atol=0)
            continue
        single = tuple(x[:, start:end].contiguous() for x in inputs)
        expected_o, expected_state = prefill(single, state[seq : seq + 1].clone(), [0, end - start])
        torch.testing.assert_close(output[:, start:end], expected_o, rtol=0.016, atol=0.001)
        torch.testing.assert_close(final_state[seq], expected_state[0], rtol=0.016, atol=0.002)
