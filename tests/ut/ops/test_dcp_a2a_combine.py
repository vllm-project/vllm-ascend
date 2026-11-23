# SPDX-License-Identifier: Apache-2.0

from math import prod
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


@pytest.mark.parametrize("scatter_size", [1, 2])
@pytest.mark.parametrize("use_pcp", [False, True])
def test_packed_exchange_preserves_collective_and_combine_order(monkeypatch, scatter_size, use_pcp):
    send = torch.arange(scatter_size * 2 * 3 * 5, dtype=torch.float32).reshape(scatter_size, 2, 3, 5)
    operations = []
    scatter_group = object()
    gathered = torch.empty(scatter_size * 2, 2, 3, 5)
    output = torch.empty(3, 2, 4)

    def exchange(recv, payload, *, group):
        assert payload is send and group is scatter_group
        operations.append("exchange")
        recv.copy_(payload)

    def gather(recv, *, dim):
        torch.testing.assert_close(recv, send)
        assert dim == 0
        operations.append("gather")
        return gathered

    def combine(recv, head_dim, *, scatter_dim, return_lse, vector_core_budget):
        if use_pcp:
            assert recv is gathered
        else:
            torch.testing.assert_close(recv, send)
        assert (head_dim, scatter_dim, return_lse, vector_core_budget) == (4, 1, False, 24)
        operations.append("combine")
        return output

    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", Mock(side_effect=AssertionError("must not repack")))
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", exchange)
    monkeypatch.setattr(dcp_a2a, "fused_dcp_lse_combine", combine)
    pcp_group = SimpleNamespace(all_gather=gather) if use_pcp else None
    assert (
        dcp_a2a.exchange_and_combine_dcp_output(send, 4, 1, scatter_group, pcp_group=pcp_group, vector_core_budget=24)
        is output
    )
    expected = (["exchange"] if scatter_size > 1 else []) + (["gather"] if use_pcp else []) + ["combine"]
    assert operations == expected


def test_packed_exchange_requires_group_before_collective(monkeypatch):
    collective = Mock()
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", collective)
    with pytest.raises(ValueError, match="explicit All2All group"):
        dcp_a2a.exchange_and_combine_dcp_output(torch.empty(2, 2, 3, 5), 4, 1, None)
    collective.assert_not_called()


@pytest.mark.parametrize(
    "shape,head_dim,scatter_dim,return_lse,defer_combine,error",
    [
        ((2, 3, 5), 4, 1, False, False, "contiguous 4D"),
        ((2, 2, 3, 5), 4, 2, False, False, "scatter_dim"),
        ((2, 2, 3, 5), 0, 1, False, False, "positive integer head_dim"),
        ((2, 2, 3, 5), 8, 1, False, False, "LSE packing"),
        ((0, 2, 3, 5), 4, 1, False, False, "source rank"),
        ((2, 2, 3, 5), 4, 1, True, True, "mutually exclusive"),
    ],
)
def test_packed_exchange_rejects_invalid_payload(shape, head_dim, scatter_dim, return_lse, defer_combine, error):
    with pytest.raises((ValueError, RuntimeError), match=error):
        dcp_a2a.exchange_and_combine_dcp_output(
            torch.empty(shape), head_dim, scatter_dim, None, return_lse=return_lse, defer_combine=defer_combine
        )


def _tensor(shape, dtype=torch.bfloat16):
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
@pytest.mark.parametrize("budget,grid", [(None, 56), (24, 24), (128, 56)])
def test_head_scatter_batch_dispatch_with_optional_local(monkeypatch, has_local, tokens, batched, budget, grid):
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
    assert (
        dcp_a2a.fused_dcp_lse_combine(recv, 512, 1, local_output=local, local_lse=lse, vector_core_budget=budget)
        is output
    )
    selected, unused = (batch, single) if batched else (single, batch)
    selected.__getitem__.assert_called_once_with((grid,))
    unused.__getitem__.assert_not_called()
    assert selected.__getitem__.return_value.call_args.kwargs["HAS_LOCAL"] is has_local


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "8"])
def test_combine_rejects_invalid_vector_budget(budget):
    with pytest.raises(ValueError, match="vector_core_budget must be a positive integer"):
        dcp_a2a.fused_dcp_lse_combine(_tensor((8, 12, 8, 516)), 512, 1, vector_core_budget=budget)


def test_combine_grid_does_not_exceed_rows(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(dcp_a2a, "_fused_dcp_lse_combine_kernel", kernel)
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: False)
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 56)
    monkeypatch.setattr(dcp_a2a.triton, "next_power_of_2", lambda x: 1 << (x - 1).bit_length(), raising=False)
    monkeypatch.setattr(dcp_a2a.torch, "empty", Mock(return_value=_tensor((3, 2, 512))))
    dcp_a2a.fused_dcp_lse_combine(_tensor((8, 2, 3, 516)), 512, 1, vector_core_budget=24)
    kernel.__getitem__.assert_called_once_with((6,))
