# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

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


def _tensor(shape, dtype=torch.bfloat16):
    from math import prod

    strides = tuple(prod(shape[index + 1 :]) for index in range(len(shape)))
    return SimpleNamespace(
        ndim=len(shape),
        shape=shape,
        dtype=dtype,
        device=SimpleNamespace(type="npu"),
        is_contiguous=lambda: True,
        stride=lambda: strides,
    )


@pytest.mark.parametrize("has_local", [False, True])
@pytest.mark.parametrize("tokens,batched", [(8, False), (32, True), (40, True), (128, True)])
def test_head_scatter_batch_dispatch_with_optional_local(monkeypatch, has_local, tokens, batched):
    recv, output = _tensor((8, 12, tokens, 516)), _tensor((tokens, 12, 512))
    local = _tensor((tokens, 12, 512)) if has_local else None
    lse = _tensor((tokens, 12, 1), torch.float32) if has_local else None
    batch, single = MagicMock(), MagicMock()
    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_batched_kernel", batch)
    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_kernel", single)
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: True)
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 56)
    monkeypatch.setattr(dcp_a2a.triton, "next_power_of_2", lambda x: 1 << (x - 1).bit_length(), raising=False)
    monkeypatch.setattr(dcp_a2a.torch, "empty", Mock(return_value=output))
    assert dcp_a2a.fused_dcp_lse_combine(recv, 512, 1, local_output=local, local_lse=lse) is output
    selected, unused = (batch, single) if batched else (single, batch)
    selected.__getitem__.assert_called_once_with((56,))
    unused.__getitem__.assert_not_called()
    assert selected.__getitem__.return_value.call_args.kwargs["HAS_LOCAL"] is has_local
