# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side store clients must use their own DP shard's device."""

from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import base


def _config(*, assigned, dp=0, tp=1):
    return SimpleNamespace(
        assigned_physical_gpu_ids=assigned,
        data_parallel_rank=dp,
        data_parallel_rank_local=dp,
        tensor_parallel_size=tp,
    )


def test_scheduler_device_follows_dp_shard(monkeypatch):
    monkeypatch.setattr(base, "set_assigned_physical_gpu_ids", lambda ids: None)
    monkeypatch.setattr(
        base.current_platform,
        "logical_device_id_to_visible_device_id",
        lambda index: 100 + index,
    )

    assert base.get_scheduler_device_id(_config(assigned=list(range(16)), dp=0)) == 100
    assert base.get_scheduler_device_id(_config(assigned=list(range(16)), dp=5)) == 105
    # TP groups start at dp_rank * tp_size.
    assert base.get_scheduler_device_id(_config(assigned=list(range(16)), dp=3, tp=2)) == 106
    # A short assignment list clamps to its last device instead of indexing past it.
    assert base.get_scheduler_device_id(_config(assigned=[4, 5], dp=9)) == 101


def test_scheduler_device_without_assignment_uses_current_device(monkeypatch):
    monkeypatch.setattr(base.torch.npu, "current_device", lambda: 7)
    assert base.get_scheduler_device_id(_config(assigned=None, dp=4)) == 7
