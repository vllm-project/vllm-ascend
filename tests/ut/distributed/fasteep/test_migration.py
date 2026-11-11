# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unchanged placements require no device work or distributed rendezvous."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.fasteep import migration


@pytest.mark.parametrize("verify", [False, True])
def test_unchanged_surviving_experts_skip_device_work(monkeypatch, verify):
    synchronize = Mock()
    monkeypatch.setattr(migration.torch.npu, "synchronize", synchronize)
    comms = Mock()
    comms.gather_objects.return_value = [True] * 3
    # Removing a redundant third rank leaves every surviving expert in place.
    before = [[0, 1, -1, -1], [-1, -1, 0, 1], [0, 1, -1, -1]]
    after = before[:2]
    layers = [object()] * 3
    mover = migration.ExpertMigration(layers, 0, comms, verify=verify)
    assert mover.execute([before] * 3, [after] * 3) == 0
    synchronize.assert_not_called()
    comms.barrier.assert_not_called()
    comms.gather_objects.assert_not_called()
    comms.begin_transition.assert_not_called()


def test_nonparticipant_still_joins_changed_layer_barrier(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(migration.torch.npu, "synchronize", synchronize)
    comms = Mock()
    mover = migration.ExpertMigration([object()], 2, comms)
    before = [[0, 1, -1, -1], [-1, -1, 0, 1], [0, 1, -1, -1]]
    after = [before[0], [-1, -1, 1, 0]]
    # Rank 2 neither sends nor receives, but must wait for other ranks.
    assert mover.execute([before], [after]) == 0
    synchronize.assert_called_once()
    comms.barrier.assert_called_once()
    comms.begin_transition.assert_not_called()


def test_peer_migration_bootstraps_on_nonparticipating_rank(monkeypatch):
    monkeypatch.setattr(migration.torch.npu, "synchronize", Mock())
    comms = Mock()
    # Rank 3 is idle while ranks 1 and 2 exchange a missing expert.
    before = [[0, 1, -1, -1], [-1, 0, 1, -1], [-1, -1, 0, 1], [0, -1, -1, 1]]
    after = [[0, 1, -1, -1], [-1, -1, 1, 0]]
    mover = migration.ExpertMigration([object()], 3, comms)
    assert mover.execute([before], [after]) == 0
    comms.begin_transition.assert_called_once()
    comms.barrier.assert_called_once()


def test_local_cycle_preserves_sources_without_peer_bootstrap(monkeypatch):
    monkeypatch.setattr(migration.torch.npu, "synchronize", Mock())
    monkeypatch.setattr(migration.torch_npu, "npu_format_cast", lambda tensor, fmt: tensor)
    layer = SimpleNamespace(w13_weight=torch.tensor([[1.0], [2.0]]), w2_weight=torch.tensor([[10.0], [20.0]]))
    comms = Mock()
    comms.gather_objects.return_value = [True]
    mover = migration.ExpertMigration([layer], 0, comms, verify=True)
    assert mover.execute([[[0, 1]]], [[[1, 0]]]) == 4
    torch.testing.assert_close(layer.w13_weight, torch.tensor([[2.0], [1.0]]))
    torch.testing.assert_close(layer.w2_weight, torch.tensor([[20.0], [10.0]]))
    comms.begin_transition.assert_not_called()
    comms.barrier.assert_called_once()
