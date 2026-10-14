# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.rotary_embedding.packed_qk_rope import packed_qk_rope_
from vllm.triton_utils import tl, triton

from vllm_ascend.patch.worker import patch_triton  # noqa: F401


@pytest.mark.parametrize("value, expected", [(0, 1), (1, 1), (3, 4), (63, 64), (128, 128), (129, 256)])
def test_next_power_of_two_host_and_constexpr(value, expected):
    assert triton.next_power_of_2(value) == expected
    assert triton.next_power_of_2(tl.constexpr(value)) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [5, 32])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_packed_qk_rope_matches_reference(dtype, seq_len, head_dim):
    generator = torch.Generator().manual_seed(42)
    original = torch.randn(seq_len, 3, 2, head_dim, generator=generator, dtype=dtype)
    angles = torch.randn(seq_len, head_dim // 2, generator=generator)
    freqs = torch.polar(torch.ones_like(angles), angles)

    expected = original.clone()
    qk = original[:, :2].float().reshape(seq_len, 2, 2, head_dim // 2, 2)
    cos = freqs.real[:, None, None, :]
    sin = freqs.imag[:, None, None, :]
    rotated = torch.stack((qk[..., 0] * cos - qk[..., 1] * sin, qk[..., 0] * sin + qk[..., 1] * cos), dim=-1)
    expected[:, :2] = rotated.reshape(seq_len, 2, 2, head_dim).to(dtype)

    actual = original.npu()
    packed_qk_rope_(actual, freqs.npu())
    actual = actual.cpu()
    tolerance = torch.finfo(dtype).eps
    torch.testing.assert_close(actual[:, :2], expected[:, :2], rtol=tolerance, atol=tolerance)
    # The packed kernel must leave the V slice untouched.
    torch.testing.assert_close(actual[:, 2], original[:, 2], rtol=0, atol=0)
