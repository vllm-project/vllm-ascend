# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small-head SISO coverage for default dimensions and existing input modes."""

import pytest
import torch

from vllm_ascend.ops.triton.rope import rope_forward_triton_siso


def _siso_reference(query, cos, sin, rotary_dim, is_neox_style):
    rotated = query[..., :rotary_dim].float()
    if is_neox_style:
        first, second = rotated.chunk(2, dim=-1)
        perpendicular = torch.cat((-second, first), dim=-1)
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
    else:
        perpendicular = torch.stack((-rotated[..., 1::2], rotated[..., ::2]), dim=-1).flatten(-2)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    result = rotated * cos.float().unsqueeze(1) + perpendicular * sin.float().unsqueeze(1)
    return torch.cat((result.to(query.dtype), query[..., rotary_dim:]), dim=-1)


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotary_dim", [64, 96, 128])
@pytest.mark.parametrize("mode", ["default", "sentinel", "explicit", "cache"])
@torch.inference_mode()
def test_rope_siso_dimension_modes(is_neox_style, dtype, rotary_dim, mode):
    # Keep the head count small to avoid the separate large-head UB limit.
    generator = torch.Generator(device="cpu").manual_seed(0)
    query = torch.randn(4, 2, 128, generator=generator, dtype=dtype, device="cpu")
    cache = torch.randn(8, rotary_dim, generator=generator, dtype=dtype, device="cpu")
    positions = torch.tensor([7, 1, 4, 0], dtype=torch.int64, device="cpu")
    cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
    expected = _siso_reference(query, cos, sin, rotary_dim, is_neox_style)
    actual_input = query.to("npu:0")
    kwargs = {"is_neox_style": is_neox_style}
    if mode == "cache":
        kwargs.update(cos_sin_cache=cache.to("npu:0"), positions=positions.to("npu:0"), rope_dim=rotary_dim)
    else:
        kwargs.update(cos=cos.to("npu:0"), sin=sin.to("npu:0"))
        if mode != "default":
            kwargs["rope_dim"] = -1 if mode == "sentinel" else rotary_dim
    actual = rope_forward_triton_siso(actual_input, **kwargs)
    assert actual.data_ptr() == actual_input.data_ptr()
    actual_cpu = actual.cpu()
    torch.testing.assert_close(actual_cpu, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_cpu[..., rotary_dim:], query[..., rotary_dim:], atol=0, rtol=0)
