# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton.dcp import dcp_a2a


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("dtype,packed_extra", [(torch.float32, 1), (torch.bfloat16, 4)])
def test_deferred_fake_layout(scatter_dim, dtype, packed_extra):
    output = torch.empty(8, 12, 96, dtype=dtype, device="meta")
    lse = torch.empty(8, 12, 1, device="meta")
    actual = dcp_a2a.dcp_a2a_fused_fake(output, lse, 2, scatter_dim, "dcp", defer_combine=True)
    assert actual.shape == (2, output.shape[scatter_dim] // 2, output.shape[1 - scatter_dim], 96 + packed_extra)
    assert actual.dtype == dtype


@pytest.mark.parametrize("scatter_size", [1, 2])
def test_deferred_exchange_skips_combine(monkeypatch, scatter_size):
    send = torch.empty(scatter_size, 2, 3, 5)
    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", Mock(return_value=send))
    collective = Mock()
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", collective)
    combine = Mock(side_effect=AssertionError("deferred exchange must not combine"))
    monkeypatch.setattr(dcp_a2a, "fused_dcp_lse_combine", combine)
    result = dcp_a2a.dcp_a2a_fused_combine(
        torch.empty(2, 3, 4), torch.empty(2, 3, 1), scatter_size, 1, object(), defer_combine=True
    )
    assert result.shape == send.shape
    assert collective.call_count == int(scatter_size > 1)
    combine.assert_not_called()
    if scatter_size == 1:
        assert result is send


@pytest.mark.parametrize(
    "tokens,heads,dim,world,has_local,head_major",
    [
        (5, 1, 512, 8, True, False),
        (5, 8, 512, 8, True, False),
        (41, 8, 128, 8, True, False),
        (41, 7, 128, 3, True, False),
        (20, 8, 64, 4, False, False),
        (41, 12, 512, 8, True, True),
    ],
)
def test_combine_launch_supports_generic_rows(tokens, heads, dim, world, has_local, head_major):
    recv = torch.empty(world, heads, tokens, dim + 4, dtype=torch.bfloat16, device="meta")
    output = torch.empty(tokens, heads, dim, dtype=torch.bfloat16, device="meta")
    local = (output.transpose(0, 1).contiguous().transpose(0, 1) if head_major else output) if has_local else None
    local_lse = torch.empty(tokens, heads, 1, device="meta") if has_local else None
    launch = Mock()
    dcp_a2a._launch_fused_dcp_lse_combine(recv, output, dim, 1, False, local, local_lse, 40, launch)
    launch.assert_called_once()
    kernel, grid, *args = launch.call_args.args
    assert kernel is dcp_a2a._fused_dcp_lse_combine_bf16_head_scatter_kernel
    assert grid == (min(tokens * heads, 40),)
    assert "BLOCK_ROWS" not in launch.call_args.kwargs
    assert "RANK_UNROLL" not in launch.call_args.kwargs
    assert args[4:7] == list(local.stride() if has_local else (0, 0, 0))


@pytest.mark.parametrize(
    "dtype,scatter_dim,dim,return_lse,feature_stride",
    [
        (torch.float32, 1, 256, False, 1),
        (torch.float16, 1, 256, False, 1),
        (torch.bfloat16, 0, 256, False, 1),
        (torch.bfloat16, 1, 513, False, 1),
        (torch.bfloat16, 1, 256, False, 2),
        (torch.float32, 1, 256, True, 1),
    ],
)
def test_combine_launch_preserves_scalar_modes(dtype, scatter_dim, dim, return_lse, feature_stride):
    tokens, heads = 41, 8
    packed_dim = dim + (1 if dtype == torch.float32 else 4)
    shape = (8, heads, tokens, packed_dim) if scatter_dim else (8, tokens, heads, packed_dim)
    recv = torch.empty(shape, dtype=dtype, device="meta")
    output = torch.empty(tokens, heads, dim + int(return_lse), dtype=dtype, device="meta")
    local = torch.empty_strided(
        (tokens, heads, dim),
        (heads * dim * feature_stride, dim * feature_stride, feature_stride),
        dtype=dtype,
        device="meta",
    )
    local_lse = torch.empty(tokens, heads, 1, device="meta")
    launch = Mock()
    dcp_a2a._launch_fused_dcp_lse_combine(recv, output, dim, scatter_dim, return_lse, local, local_lse, 40, launch)
    launch.assert_called_once()
    assert launch.call_args.args[0] is dcp_a2a._fused_dcp_lse_combine_kernel
