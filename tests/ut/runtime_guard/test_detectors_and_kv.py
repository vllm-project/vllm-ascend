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

# mypy: ignore-errors
"""P0 UT: token_repeat pure logic + kv dump_kv empty-block safety."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.runtime_guard.action.actions import ActionContext
from vllm_ascend.runtime_guard.detector.manager import AfterSampleCpuSnapshot, DetectorManager
from vllm_ascend.runtime_guard.detector.token_repeat import (
    TokenRepeatDetector,
    TokenRepeatState,
    push_token_repeat,
)
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader, _slice_blocks
from vllm_ascend.runtime_guard.request_state import RequestGuardStore


def _dump_rc(tmp_path: Path, **extra) -> SimpleNamespace:
    base = dict(
        dump_get=lambda k, d=None: 0 if k == "free_headroom_bytes" else d,
        dump_root=lambda: str(tmp_path),
        report_save_sensitive_info=lambda: False,
        report_decode_token_ids=lambda: True,
        report_max_prompt_token_ids=lambda: 1000,
        report_max_output_token_ids=lambda: 1000,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _dump_ctx(
    *,
    tmp_path: Path,
    incident: Incident,
    quota: MagicMock | None = None,
    kv_reader: MagicMock | KvCacheReader | None = None,
    runner: SimpleNamespace | None = None,
    detail: dict | None = None,
    action_overrides: dict | None = None,
    rank_tag: str = "tp0",
    rc: SimpleNamespace | None = None,
) -> ActionContext:
    if quota is None:
        quota = MagicMock()
        quota.try_consume.return_value = True
    if runner is None:
        guard = MagicMock()
        guard.queue_kv_dump.side_effect = lambda job: True
        runner = SimpleNamespace(
            tp_rank=0,
            dp_rank=0,
            dcp_rank=0,
            dcp_size=1,
            runtime_guard=guard,
        )
    if kv_reader is None:
        kv_reader = MagicMock()
        kv_reader.estimate_dump_bytes.return_value = 0
    return ActionContext(
        incident=incident,
        runner=runner,
        runtime_config=rc or _dump_rc(tmp_path),  # type: ignore[arg-type]
        report_writer=MagicMock(),
        kv_reader=kv_reader,
        quota=quota,
        rank_tag=rank_tag,
        detail=detail or {},
        action_overrides=action_overrides or {},
    )


def test_push_token_repeat_scores_and_ignore():
    st = TokenRepeatState()
    ignore: set[int] = set()
    for tid in (1, 2, 3, 4):
        assert push_token_repeat(st, tid, window=8, ignore=ignore) == 0
    assert push_token_repeat(st, 1, window=8, ignore=ignore) == 1
    assert st.repeat_sum >= 1

    ignored = TokenRepeatState()
    assert push_token_repeat(ignored, 0, window=8, ignore={0}) == 0
    assert len(ignored.content) == 0


def test_w1_1_r04_same_wave_double_fold_keeps_seen_eq_oc(tmp_path: Path):
    """W1-1/R-04: same-wave duplicate after-sample must not inflate content_tokens_seen.

    Store append dedupes identical chunks; folding must follow Store only so
    ``content_tokens_seen`` stays aligned with ``output_token_count``.
    """
    from vllm_ascend.runtime_config.config import RuntimeConfig

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    tr = cfg._data["detector"]["token_repeat"]
    tr["enabled"] = True
    tr["window"] = 32
    tr["repeat_sum_threshold"] = 9999
    tr["min_tokens"] = 1

    det = TokenRepeatDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    det.refresh_from_config()
    io = RequestIoSnapshotManager.get()
    rid = "r1"

    def one_step(tok: int, *, double: bool) -> None:
        row = [[tok]]
        io.append_batch([rid], row)
        det.check_all(None, req_ids=[rid])
        if double:
            io.append_batch([rid], row)
            det.check_all(None, req_ids=[rid])

    for i in range(5):
        one_step(100 + i, double=False)
    for i in range(5):
        one_step(200 + i, double=True)

    st = RequestGuardStore.get().get_state(rid)
    assert st is not None
    oc = len(st.output_token_ids)
    seen = det._states[rid].content_tokens_seen
    snap = io.snapshot(None, rid, None, include_token_ids=False, use_cache=False)
    assert oc == 10
    assert seen == oc
    assert snap.output_token_count == oc


def test_w1_1_run_after_sample_cpu_folds_store_only(tmp_path: Path):
    """DetectorManager CPU path must not re-fold frozen sampled rows past Store."""
    from vllm_ascend.runtime_config.config import RuntimeConfig

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    cfg._data["detector"]["token_repeat"]["enabled"] = True
    cfg._data["detector"]["token_repeat"]["repeat_sum_threshold"] = 9999
    cfg._data["detector"]["token_repeat"]["min_tokens"] = 1
    cfg._data["detector"]["output_substring"]["enabled"] = False

    runner = SimpleNamespace(tp_rank=0, input_batch=None)
    mgr = DetectorManager(runtime_config=cfg, runner=runner)
    mgr.apply_runtime_config()
    io = RequestIoSnapshotManager.get()
    rid = "poem"

    io.append_batch([rid], [[7]])
    snap = AfterSampleCpuSnapshot(
        req_ids=[rid],
        sampled_token_ids=[[7]],
        skip_req_ids=None,
    )
    mgr.run_after_sample_cpu(snap)
    # Same-wave duplicate: Store skips append; CPU job must not double-fold.
    io.append_batch([rid], [[7]])
    mgr.run_after_sample_cpu(snap)

    st = RequestGuardStore.get().get_state(rid)
    assert st is not None
    assert len(st.output_token_ids) == 1
    assert mgr.get("token_repeat")._states[rid].content_tokens_seen == 1  # type: ignore[attr-defined]


def test_slice_blocks_selects_requested_and_empty():
    t = torch.arange(0, 4 * 8).reshape(4, 8).float()
    assert _slice_blocks(t, []) is None
    sliced = _slice_blocks(t, [1, 3])
    assert sliced is not None
    out, used = sliced
    assert used == [1, 3]
    assert out.shape == (2, 8)  # keep [n_blocks, block_size, ...]
    assert torch.equal(out[0], t[1])
    assert torch.equal(out[1], t[3])


def test_slice_blocks_partial_out_of_range_reports_used_ids():
    # B'5c: payload metadata must reflect the blocks actually dumped.
    t = torch.arange(0, 4 * 8).reshape(4, 8).float()
    sliced = _slice_blocks(t, [1, 9])  # 9 out of range
    assert sliced is not None
    out, used = sliced
    assert used == [1]
    assert out.shape == (1, 8)
    assert torch.equal(out[0], t[1])


def test_snapshot_skips_empty_block_ids(tmp_path: Path):
    runner = SimpleNamespace(
        kv_caches={"layers.0": torch.randn(8, 16, 4)},
    )
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[],
        out_dir=tmp_path / "kv",
    )
    assert snaps == []


def test_snapshot_request_blocks_writes_req_slice(tmp_path: Path):
    cache = torch.randn(4, 8, 2)
    runner = SimpleNamespace(kv_caches={"L0": cache})
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[0, 2],
        out_dir=tmp_path / "kv",
    )
    assert len(snaps) == 1
    assert snaps[0].payload["tensor"].shape == (2, 8, 2)
    written = KvCacheReader.write_snapshots(snaps)
    assert len(written) == 1
    assert Path(written[0]).is_file()


def test_estimate_dump_bytes_scales_with_blocks():
    cache = torch.zeros(4, 8, 2)
    reader = KvCacheReader(SimpleNamespace(kv_caches={"L0": cache}))
    empty = reader.estimate_dump_bytes(block_ids=[])
    two = reader.estimate_dump_bytes(block_ids=[0, 1])
    assert empty == 0
    assert two == int(cache.nbytes) // 2


def test_dump_kv_skips_when_free_below_payload_plus_headroom(tmp_path, monkeypatch):
    from vllm_ascend.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction

    cache = torch.zeros(4, 8, 2)
    reader = KvCacheReader(SimpleNamespace(kv_caches={"L0": cache}))
    estimated = reader.estimate_dump_bytes(block_ids=[0])
    tp_size = 4
    headroom = int(_DEFAULTS["dump"]["free_headroom_bytes"])
    monkeypatch.setattr(
        "vllm_ascend.runtime_guard.action.actions.free_bytes_at",
        lambda _path: estimated + headroom - 1,
    )
    monkeypatch.setattr(
        "vllm_ascend.runtime_guard.action.actions.runner_tp_world_size",
        lambda _runner: tp_size,
    )
    quota = MagicMock()
    rc = SimpleNamespace(
        dump_get=lambda k, d=None: headroom if k == "free_headroom_bytes" else d,
        dump_root=lambda: str(tmp_path),
    )
    ctx = _dump_ctx(
        tmp_path=tmp_path,
        incident=Incident(incident_type="token_repeat", req_id="r1", block_ids=[0]),
        quota=quota,
        kv_reader=reader,
        runner=SimpleNamespace(tp_size=tp_size),
        rc=rc,
        rank_tag="tp0",
    )
    DumpKvAction().run(ctx)
    quota.try_consume.assert_not_called()


def test_dump_kv_skips_when_request_finished(tmp_path):
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        RequestGuardStore.get().mark_finished(["r1"], wave=0)
        quota = MagicMock()
        ctx = _dump_ctx(
            tmp_path=tmp_path,
            incident=Incident(incident_type="token_repeat", req_id="r1", block_ids=[0]),
            quota=quota,
            rank_tag="dp0_tp0_pp0_cp0",
            rc=_dump_rc(tmp_path, dump_get=lambda k, d=None: d),
        )
        DumpKvAction().run(ctx)
        quota.try_consume.assert_not_called()
        markers = list(tmp_path.glob("**/dump_skipped.json"))
        assert len(markers) == 1
        marker = markers[0]
        assert marker.is_file()
        data = json.loads(marker.read_text(encoding="utf-8"))
        assert data["reason"] == "finished_or_reaped"
        assert data["stage"] == "arm"
        assert data["req_id"] == "r1"
        assert "wave_unknown" in str(marker)
    finally:
        RequestGuardStore.reset_for_tests()


def test_dump_kv_writes_request_info_json(tmp_path):
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        ctx = _dump_ctx(
            tmp_path=tmp_path,
            incident=Incident(
                incident_type="token_repeat",
                req_id="r1",
                block_ids=[0, 1],
                wave=7,
                detail={"repeat_sum": 99},
            ),
            detail={
                "repeat_sum": 99,
                "prompt_token_count": 3,
                "output_token_count": 5,
                "block_ids": [0, 1],
            },
            rank_tag="dp0_tp0_pp0_cp0",
        )
        DumpKvAction().run(ctx)
        guard = ctx.runner.runtime_guard
        guard.queue_kv_dump.assert_called_once()
        info = tmp_path / "token_repeat" / "r1" / "wave_7" / "request_info.json"
        assert info.is_file()
        data = json.loads(info.read_text(encoding="utf-8"))
        assert data["req_id"] == "r1"
        assert data["incident_type"] == "token_repeat"
        assert data["dump_arm_wave"] == 7
        assert data["block_ids"] == [0, 1]
        assert data["detail"]["repeat_sum"] == 99
        assert data["detail"]["prompt_token_count"] == 3
        assert "prompt_token_ids" not in data["detail"]
    finally:
        RequestGuardStore.reset_for_tests()


def test_write_kv_dump_skipped_marker(tmp_path):
    from vllm_ascend.runtime_guard.dump_io import write_kv_dump_skipped

    path = write_kv_dump_skipped(
        tmp_path,
        req_id="r2",
        incident_type="logits_finite",
        reason="finished_or_reaped",
        stage="drain",
        rank_tag="dp0_tp0_pp1_cp0",
        wave=3,
    )
    assert path is not None and path.is_file()
    assert path.name == "dump_skipped.json"
    assert "wave_3" in str(path)
    assert "dp0_tp0_pp1_cp0" in str(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["stage"] == "drain"
    assert data["reason"] == "finished_or_reaped"
    assert data["incident_type"] == "logits_finite"


def test_write_kv_dump_skipped_arm_reasons(tmp_path):
    from vllm_ascend.runtime_guard.dump_io import write_kv_dump_skipped

    path = write_kv_dump_skipped(
        tmp_path,
        req_id="__manual_trigger__",
        incident_type="manual_trigger",
        reason="no_dump_targets",
        stage="arm",
        rank_tag="dp0_tp0_pp0_cp0",
        wave=4,
        detail={"scope": "all_requests"},
    )
    assert path is not None and path.is_file()
    assert path.name == "dump_skipped.json"
    assert "wave_4" in str(path)
    assert "dp0_tp0_pp0_cp0" in str(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["reason"] == "no_dump_targets"
    assert data["dump_arm_wave"] == 4
    assert data["detail"]["scope"] == "all_requests"


def test_dump_kv_writes_skipped_when_no_targets(tmp_path, monkeypatch):
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        monkeypatch.setattr(
            "vllm_ascend.runtime_guard.manual_trigger.iter_local_request_rows",
            lambda _runner: [],
        )
        ctx = _dump_ctx(
            tmp_path=tmp_path,
            incident=Incident(
                incident_type=MANUAL_TRIGGER_TYPE,
                req_id="__manual_trigger__",
                consume_quota=False,
                wave=9,
            ),
            action_overrides={"dump_kv": {"scope": "all_requests"}},
        )
        DumpKvAction().run(ctx)
        markers = list(tmp_path.glob("**/dump_skipped.json"))
        assert len(markers) == 1
        marker = markers[0]
        assert "wave_9" in str(marker)
        assert json.loads(marker.read_text(encoding="utf-8"))["reason"] == "no_dump_targets"
    finally:
        RequestGuardStore.reset_for_tests()


def test_dump_kv_all_requests_enumerates_local_batch(tmp_path, monkeypatch):
    """scope=all_requests lists live reqs via iter_local_request_rows."""
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        guard = MagicMock()
        guard.queue_kv_dump.side_effect = lambda job: True
        runner = SimpleNamespace(
            tp_rank=0,
            dp_rank=0,
            dcp_rank=0,
            dcp_size=1,
            runtime_guard=guard,
            input_batch=SimpleNamespace(req_ids=["r1", "r2"]),
        )
        monkeypatch.setattr(
            "vllm_ascend.runtime_guard.kv_block_meta.block_ids_for_request",
            lambda _runner, req_id, req_idx=None, **kw: [0] if req_id == "r1" else [1, 2],
        )
        ctx = _dump_ctx(
            tmp_path=tmp_path,
            incident=Incident(
                incident_type="token_repeat",
                req_id="r1",
                block_ids=[0],
                consume_quota=True,
            ),
            runner=runner,
            detail={"repeat_sum": 1},
            action_overrides={"dump_kv": {"scope": "all_requests"}},
        )
        DumpKvAction().run(ctx)
        assert guard.queue_kv_dump.call_count == 2
        req_ids = {c.args[0]["req_id"] for c in guard.queue_kv_dump.call_args_list}
        assert req_ids == {"r1", "r2"}
        arm_ids = {c.args[0]["arm_id"] for c in guard.queue_kv_dump.call_args_list}
        assert len(arm_ids) == 1
        assert (tmp_path / "token_repeat" / "r1" / "wave_unknown" / "request_info.json").is_file()
        assert (tmp_path / "token_repeat" / "r2" / "wave_unknown" / "request_info.json").is_file()
    finally:
        RequestGuardStore.reset_for_tests()


def test_queue_kv_dump_dedupes_req_id_same_step(caplog):
    import logging

    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

    proc = SimpleNamespace(_kv_dump_jobs=[])
    with caplog.at_level(logging.INFO, logger="vllm_ascend.runtime_guard.processor_dump"):
        assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r1", "arm_id": "a", "wave": 1}) is True
        assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r1", "arm_id": "b", "wave": 1}) is False
    assert any("already pending" in r.message for r in caplog.records)
    # Different arm wave may queue again (previous job not yet drained).
    assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r1", "arm_id": "c", "wave": 2}) is True
    assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r2", "arm_id": "b", "wave": 1}) is True
    assert [(j["req_id"], j["wave"]) for j in proc._kv_dump_jobs] == [
        ("r1", 1),
        ("r1", 2),
        ("r2", 1),
    ]


def test_bug4_block_ids_v2_without_input_batch():
    """BUG-4: after sample_tokens clears execute_model_state, still resolve via req_states."""
    import numpy as np

    from vllm_ascend.runtime_guard.kv_block_meta import block_ids_for_request

    # state slots 0..2; req lives at persistent index 2 with 3 blocks
    num_blocks = SimpleNamespace(np=np.array([[0, 0, 3]], dtype=np.int32))
    entry = SimpleNamespace(
        np=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [10, 11, 12, 0],
            ],
            dtype=np.int32,
        ),
        gpu=None,
    )
    block_tables = SimpleNamespace(num_blocks=num_blocks, block_tables=[entry])
    req_states = SimpleNamespace(req_id_to_index={"cmpl-a": 2})
    runner = SimpleNamespace(
        requests=None,
        input_batch=None,
        execute_model_state=None,
        block_tables=block_tables,
        req_states=req_states,
    )
    assert block_ids_for_request(runner, "cmpl-a") == [10, 11, 12]
    assert block_ids_for_request(runner, "missing") == []


def test_bug4_block_ids_v2_gpu_row_when_no_host_np():
    """StagedWriteTensor-style: only ``.gpu``, sync row to host."""
    import numpy as np

    from vllm_ascend.runtime_guard.kv_block_meta import block_ids_for_request

    num_blocks = SimpleNamespace(np=np.array([[2]], dtype=np.int32))
    gpu_row = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    entry = SimpleNamespace(np=None, gpu=gpu_row)
    block_tables = SimpleNamespace(num_blocks=num_blocks, block_tables=[entry])
    runner = SimpleNamespace(
        requests=None,
        input_batch=None,
        execute_model_state=None,
        block_tables=block_tables,
        req_states=SimpleNamespace(req_id_to_index={"r0": 0}),
    )
    assert block_ids_for_request(runner, "r0") == [7, 8]
