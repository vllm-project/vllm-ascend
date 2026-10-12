#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

"""UT: runtime_config.dist (sync-group selection + task-bus collectives)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm_ascend.observability.runtime_config.dist as dist
from vllm_ascend.observability.runtime_config.dist import (
    broadcast_when_due,
    sync_due_bits,
    sync_due_bits_from_src,
    sync_task_bus,
)

# ---- sync group ----


def test_sync_group_none_when_not_last_pp(monkeypatch):
    pp = SimpleNamespace(is_last_rank=False)
    tp = SimpleNamespace(world_size=2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: pp,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: tp,
    )
    assert dist._runtime_config_sync_group_or_none() is None


def test_sync_group_tp_when_last_pp_tp_gt1(monkeypatch):
    pp = SimpleNamespace(is_last_rank=True)
    tp = SimpleNamespace(world_size=2, is_first_rank=True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: pp,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: tp,
    )
    assert dist._runtime_config_sync_group_or_none() is tp


def test_sync_group_none_when_last_pp_tp1(monkeypatch):
    pp = SimpleNamespace(is_last_rank=True)
    tp = SimpleNamespace(world_size=1)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: pp,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: tp,
    )
    assert dist._runtime_config_sync_group_or_none() is None


def test_sync_group_none_when_pp_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no pp")

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        _boom,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        MagicMock(),
    )
    assert dist._runtime_config_sync_group_or_none() is None


# ---- task bus ----


def test_sync_due_bits_none_group_returns_local():
    assert sync_due_bits(None, [False, True, False]) == [False, True, False]
    assert sync_due_bits(None, []) == []


def test_sync_due_bits_world_size_one():
    group = SimpleNamespace(world_size=1)
    assert sync_due_bits(group, [True, False]) == [True, False]


def test_broadcast_when_due_false_is_noop():
    assert broadcast_when_due(None, due=False, payload={"x": 1}) is None


def test_broadcast_when_due_single_process_uses_payload():
    assert broadcast_when_due(None, due=True, payload={"x": 1}) == {"x": 1}
    assert broadcast_when_due(None, due=True, build_payload=lambda: {"y": 2}) == {"y": 2}


def test_sync_task_bus_idle():
    assert sync_task_bus(None, due_local=False, payload={"a": 1}) is None
    assert sync_task_bus(None, due_local=True, payload={"a": 1}) == {"a": 1}


def _patch_dist(monkeypatch, fake_broadcast):
    import torch

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda g: [7, 8])


def test_sync_due_bits_from_src_none_group_returns_local():
    assert sync_due_bits_from_src(None, [False, True]) == [False, True]
    assert sync_due_bits_from_src(None, [], wave_idx=3) == []


def test_sync_due_bits_from_src_world_size_one():
    group = SimpleNamespace(world_size=1)
    assert sync_due_bits_from_src(group, [True, False], wave_idx=7) == [True, False]


def test_sync_due_bits_from_src_source_packs_payload(monkeypatch):
    seen = {}

    def fake_broadcast(tensor, src=0, group=None):
        seen["src"] = src
        seen["payload"] = [float(x) for x in tensor.tolist()]

    _patch_dist(monkeypatch, fake_broadcast)
    group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object())
    out = sync_due_bits_from_src(group, [True, False], wave_idx=(1 << 24) + 5)
    assert out == [True, False]
    assert seen["src"] == 7
    assert seen["payload"][0] == 5.0  # wave_idx compared modulo 2**24
    assert seen["payload"][1:] == [1.0, 0.0]


def test_sync_due_bits_from_src_receiver_applies_source_bits(monkeypatch):
    def fake_broadcast(tensor, src=0, group=None):
        tensor[0] = 9.0
        tensor[1] = 0.0
        tensor[2] = 1.0

    _patch_dist(monkeypatch, fake_broadcast)
    group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=object())
    assert sync_due_bits_from_src(group, [False, False], wave_idx=9) == [False, True]


def test_sync_due_bits_from_src_wave_mismatch_raises(monkeypatch):
    def fake_broadcast(tensor, src=0, group=None):
        tensor[0] = 8.0  # source is one wave behind

    _patch_dist(monkeypatch, fake_broadcast)
    group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=object())
    with pytest.raises(RuntimeError, match="misalignment"):
        sync_due_bits_from_src(group, [False, False], wave_idx=9)


def test_sync_due_bits_from_src_wraparound_matches(monkeypatch):
    def fake_broadcast(tensor, src=0, group=None):
        tensor[0] = 5.0

    _patch_dist(monkeypatch, fake_broadcast)
    group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=object())
    assert sync_due_bits_from_src(group, [False, False], wave_idx=(1 << 24) + 5) == [
        False,
        False,
    ]


def test_sync_due_bits_from_src_unset_wave_skips_assert(monkeypatch):
    _patch_dist(monkeypatch, lambda t, src=0, group=None: None)
    group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=object())
    assert sync_due_bits_from_src(group, [True, False]) == [False, False]
