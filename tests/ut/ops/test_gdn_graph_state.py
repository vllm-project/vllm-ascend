# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton.v2.mamba import graph_state


@pytest.mark.parametrize("width", [1, 3, 8, 17])
@pytest.mark.parametrize("reset_spec", [False, True])
def test_graph_state_writes_group_ids_and_clears_padding(width, reset_spec):
    updater = graph_state.GDNGraphStateUpdater()
    sources = torch.arange(24 * 8 * (width + 2), dtype=torch.int32).view(24, 8, width + 2)
    destinations = [torch.full((8, width), -99, dtype=torch.int32) for _ in range(24)]
    clears = [torch.full((8, 4), 99, dtype=torch.int32) if reset_spec else None for _ in range(24)]
    addresses = [dst.data_ptr() for dst in destinations]
    for actual_rows, graph_rows in [(5, 8), (1, 4), (0, 4), (8, 8)]:
        sources.add_(1000)
        before = [dst.clone() for dst in destinations]
        updater.apply(
            [
                (src[:, :width], dst, actual_rows, graph_rows, 0, clear)
                for src, dst, clear in zip(sources, destinations, clears)
            ]
        )
        for src, dst, previous, clear in zip(sources, destinations, before, clears):
            torch.testing.assert_close(dst[:actual_rows], src[:actual_rows, :width], rtol=0, atol=0)
            assert torch.all(dst[actual_rows:graph_rows] == 0)
            torch.testing.assert_close(dst[graph_rows:], previous[graph_rows:])
            if clear is not None:
                assert torch.all(clear[:graph_rows] == -1)
        assert [dst.data_ptr() for dst in destinations] == addresses


def test_graph_state_validates_all_groups_before_writing():
    source, destination = torch.ones(2, dtype=torch.int32), torch.full((4,), -99, dtype=torch.int32)
    with pytest.raises(AssertionError):
        graph_state.GDNGraphStateUpdater().apply(
            [(source, destination, 2, 4, 0, None), (source, torch.empty(1, dtype=torch.int32), 2, 4, 0, None)]
        )
    assert torch.all(destination == -99)


def test_graph_state_only_reads_destination_columns():
    source = torch.arange(24, dtype=torch.int32).view(4, 6)
    destination = torch.full((4, 3), -99, dtype=torch.int32)
    graph_state.GDNGraphStateUpdater().apply([(source, destination, 2, 4, 0, None)])
    torch.testing.assert_close(destination[:2], source[:2, :3])
    assert torch.all(destination[2:] == 0)


def test_graph_state_launches_once_and_reuses_pointer_table(monkeypatch):
    updater = graph_state.GDNGraphStateUpdater()
    source, destination = MagicMock(), MagicMock()
    source.dtype = destination.dtype = torch.int32
    source.device = destination.device = SimpleNamespace(type="npu")
    source.ndim = destination.ndim = 2
    source.size.side_effect = destination.size.side_effect = lambda dim: (8, 4)[dim]
    source.stride.side_effect = lambda dim: (12, 1)[dim]
    source.data_ptr.return_value, destination.data_ptr.return_value = 1024, 2048
    destination.is_contiguous.return_value = True
    host, pointers, kernel = MagicMock(), object(), MagicMock()
    host.to.return_value = pointers
    tensor = MagicMock(return_value=host)
    monkeypatch.setattr(graph_state.torch, "tensor", tensor)
    monkeypatch.setattr(graph_state, "_graph_state_kernel", kernel)
    for count in (5, 1, 0):
        updater.apply([(source, destination, count, 8, 0, None)])
    tensor.assert_called_once()
    host.to.assert_called_once_with(destination.device, non_blocking=True)
    assert kernel.__getitem__.return_value.call_count == 3
    for call in kernel.__getitem__.return_value.call_args_list:
        assert call.args[0] is pointers
        assert call.kwargs["SOURCE_ROW_STRIDE"] == 12
        assert call.kwargs["multibuffer"] is False and call.kwargs["unit_flag"] is False
