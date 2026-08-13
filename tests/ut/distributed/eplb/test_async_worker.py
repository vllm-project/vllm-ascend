# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.eplb import async_worker as eplb_async_worker
from vllm_ascend.distributed.eplb.state import AscendEplbState


class _CycleComplete(Exception):
    pass


class _OneCycleEvent:
    def __init__(self):
        self.wait_count = 0

    def wait(self, stream):
        self.wait_count += 1
        if self.wait_count > 1:
            raise _CycleComplete


def _run_one_cycle(monkeypatch, old_map, new_map):
    pending_layers: list[int] = []
    completed_cycles: list[int] = []
    transfer_metadata = object()
    transfer_layer = MagicMock(return_value=transfer_metadata)
    all_reduce = MagicMock()

    class _ConsumedEvent:
        def wait(self, stream):
            result = model_state.pending_result
            if result.transfer_metadata is eplb_async_worker.NO_TRANSFER_CYCLE_COMPLETE:
                completed_cycles.append(result.layer_idx)
                model_state.rebalanced = False
            else:
                pending_layers.append(result.layer_idx)
            model_state.pending_result = None

    stream = MagicMock()
    model_state = SimpleNamespace(
        communicator=MagicMock(),
        model=SimpleNamespace(
            num_moe_layers=old_map.shape[0],
            expert_weights=[[object()]] * old_map.shape[0],
        ),
        physical_to_logical_map=old_map,
        expert_buffer=[object()],
        rebalanced=True,
        pending_result=None,
    )
    state = SimpleNamespace(
        rearrange_event=_OneCycleEvent(),
        is_async=True,
        model_states={"model": model_state},
    )
    device_group = MagicMock()
    device_group.rank.return_value = 0
    cpu_group = MagicMock()
    cpu_group.size.return_value = 1
    coordinator = SimpleNamespace(device_group=device_group, cpu_group=cpu_group)

    monkeypatch.setattr(eplb_async_worker, "get_eplb_group", lambda: coordinator)
    monkeypatch.setattr(eplb_async_worker, "_run_rebalance_experts", lambda *args: new_map)
    monkeypatch.setattr(eplb_async_worker, "transfer_layer", transfer_layer)
    monkeypatch.setattr(eplb_async_worker, "CpuGpuEvent", _ConsumedEvent)
    monkeypatch.setattr(eplb_async_worker.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(eplb_async_worker.torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(_CycleComplete):
        eplb_async_worker.transfer_run_periodically(cast(AscendEplbState, state), stream)

    return model_state, stream, transfer_layer, all_reduce, pending_layers, completed_cycles


def test_async_worker_skips_fully_unchanged_cycle(monkeypatch):
    placement = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    model_state, stream, transfer_layer, all_reduce, pending_layers, completed_cycles = _run_one_cycle(
        monkeypatch,
        placement,
        placement.clone(),
    )

    transfer_layer.assert_not_called()
    stream.synchronize.assert_not_called()
    all_reduce.assert_not_called()
    assert pending_layers == []
    assert completed_cycles == [1]
    assert model_state.rebalanced is False


def test_async_worker_transfers_only_changed_layers(monkeypatch):
    old_map = torch.tensor([[0, 1], [0, 1], [1, 0]], dtype=torch.int32)
    new_map = torch.tensor([[0, 1], [1, 0], [1, 0]], dtype=torch.int32)
    model_state, stream, transfer_layer, all_reduce, pending_layers, completed_cycles = _run_one_cycle(
        monkeypatch,
        old_map,
        new_map,
    )

    transfer_layer.assert_called_once()
    assert transfer_layer.call_args.kwargs["layer_idx"] == 1
    stream.synchronize.assert_called_once_with()
    all_reduce.assert_called_once()
    assert pending_layers == [1]
    assert completed_cycles == [2]
