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

from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ops.rotary_embedding import AscendApplyRotaryEmb

SEQ_LEN = 4
NUM_HEADS = 2
HEAD_SIZE = 72
PARTIAL_ROTARY_DIM = 32


def _make_op(is_neox_style):
    with set_current_vllm_config(VllmConfig()):
        return AscendApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=is_neox_style,
            enable_fp32_compute=True,
        )


def _mock_rotary_mul(values, cos, sin, rotary_mode):
    if rotary_mode == "half":
        first, second = values.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    else:
        assert rotary_mode == "interleave"
        rotated = torch.stack((-values[..., 1::2], values[..., ::2]), dim=-1).flatten(-2)
    return values * cos + rotated * sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rotary_dim", [PARTIAL_ROTARY_DIM, HEAD_SIZE])
@pytest.mark.parametrize("is_neox_style", [False, True])
def test_fused_rope_matches_complex_reference(dtype, batched, strided, rotary_dim, is_neox_style):
    """The fused mode and coefficient layout must match the model's pairing."""
    generator = torch.Generator().manual_seed(1024)
    shape: tuple[int, ...] = (SEQ_LEN, NUM_HEADS, HEAD_SIZE * (2 if strided else 1))
    if batched:
        shape = (2, *shape)
    x = torch.randn(shape, generator=generator).to(dtype)
    if strided:
        x = x[..., ::2]
    angles = torch.randn(SEQ_LEN, rotary_dim // 2, generator=generator)
    cos, sin = angles.cos(), angles.sin()

    # Rotate independently via complex multiplication; Kimi uses adjacent pairs.
    values = x[..., :rotary_dim].float()
    if is_neox_style:
        first, second = values.chunk(2, dim=-1)
        pairs = torch.stack((first, second), dim=-1)
    else:
        pairs = values.contiguous().reshape(*x.shape[:-1], rotary_dim // 2, 2)
    frequencies = torch.complex(cos, sin).unsqueeze(-2)
    complex_output = torch.view_as_real(torch.view_as_complex(pairs) * frequencies)
    if is_neox_style:
        rotated = torch.cat((complex_output[..., 0], complex_output[..., 1]), dim=-1).to(dtype)
    else:
        rotated = complex_output.flatten(-2).to(dtype)
    expected = torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    op = _make_op(is_neox_style=is_neox_style)
    with patch(
        "vllm_ascend.ops.rotary_embedding.torch_npu.npu_rotary_mul",
        side_effect=_mock_rotary_mul,
        create=True,
    ) as kernel:
        actual = op.forward_oot(x, cos, sin)
        kernel.assert_called_once()
        assert kernel.call_args.kwargs == {"rotary_mode": "half" if is_neox_style else "interleave"}
        _, kernel_cos, kernel_sin = kernel.call_args.args
        expected_cos = torch.cat((cos, cos), dim=-1) if is_neox_style else cos.repeat_interleave(2, dim=-1)
        expected_sin = torch.cat((sin, sin), dim=-1) if is_neox_style else sin.repeat_interleave(2, dim=-1)
        torch.testing.assert_close(kernel_cos, expected_cos.reshape(1, SEQ_LEN, 1, rotary_dim))
        torch.testing.assert_close(kernel_sin, expected_sin.reshape(1, SEQ_LEN, 1, rotary_dim))

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[..., rotary_dim:], x[..., rotary_dim:], rtol=0, atol=0)


@pytest.mark.parametrize("rotary_dim", [PARTIAL_ROTARY_DIM, HEAD_SIZE])
def test_neox_rope_keeps_fused_kernel(rotary_dim):
    x = torch.randn(SEQ_LEN, NUM_HEADS, HEAD_SIZE).to(torch.bfloat16)
    cos = torch.ones(SEQ_LEN, rotary_dim // 2)
    sin = torch.zeros_like(cos)
    op = _make_op(is_neox_style=True)

    with patch(
        "vllm_ascend.ops.rotary_embedding.torch_npu.npu_rotary_mul",
        side_effect=_mock_rotary_mul,
        create=True,
    ) as kernel:
        actual = op.forward_oot(x, cos, sin)
        kernel.assert_called_once()
        assert kernel.call_args.kwargs == {"rotary_mode": "half"}
        values, kernel_cos, kernel_sin = kernel.call_args.args

    assert values.shape == (1, SEQ_LEN, NUM_HEADS, rotary_dim)
    assert values.dtype == torch.float32
    assert kernel_cos.shape == kernel_sin.shape == (1, SEQ_LEN, 1, rotary_dim)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_interleaved_rope_rejects_oversized_rotary_dim():
    x = torch.zeros(SEQ_LEN, NUM_HEADS, HEAD_SIZE)
    cos = torch.ones(SEQ_LEN, HEAD_SIZE // 2 + 1)
    sin = torch.zeros_like(cos)
    op = _make_op(is_neox_style=False)

    with pytest.raises(ValueError, match="must not exceed head_dim"):
        op.forward_oot(x, cos, sin)
