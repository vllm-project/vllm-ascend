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

from vllm_ascend.observability.runtime_guard.action.actions import ActionContext
from vllm_ascend.observability.runtime_guard.detector.manager import AfterSampleCpuSnapshot, DetectorManager
from vllm_ascend.observability.runtime_guard.detector.token_repeat import (
    TokenRepeatDetector,
    TokenRepeatState,
    push_token_repeat,
)
from vllm_ascend.observability.runtime_guard.dump import KvCacheReader, _slice_blocks
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager
from vllm_ascend.observability.runtime_guard.state import Incident, RequestGuardStore


def _dump_rc(tmp_path: Path, **extra) -> SimpleNamespace:
    base = dict(
        dump_root=lambda: str(tmp_path),
        report_save_sensitive_info=lambda: False,
        report_decode_token_ids=lambda: False,
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


def test_token_repeat_detector_hit_and_miss(tmp_path: Path):
    """S4: synthetic repeats alert; unique ids miss; disabled stays quiet."""
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
    )
    tr = cfg._data["detector"]["token_repeat"]
    tr["enabled"] = True
    tr["window"] = 8
    tr["repeat_sum_threshold"] = 4
    tr["min_tokens"] = 3
    tr["consecutive_hits"] = 1

    det = TokenRepeatDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    det.refresh_from_config()

    miss: list = []
    for tid in range(10, 20):
        miss = det.check_all([[tid]], req_ids=["uniq"])
    assert miss == []

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    det.clear_finished("uniq")
    # Same token repeatedly → high repeat_sum after warmup.
    hit: list = []
    for _ in range(12):
        alerts = det.check_all([[7]], req_ids=["rep"])
        if alerts:
            hit = alerts
            break
    assert len(hit) == 1
    assert hit[0].incident_type == "token_repeat"
    assert hit[0].req_id == "rep"
    assert hit[0].is_ill is True

    # Already alerted: further repeats do not re-emit.
    assert det.check_all([[7]], req_ids=["rep"]) == []

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    tr["enabled"] = False
    cfg._version = float(cfg.config_version) + 1e-6
    det.refresh_from_config()
    for _ in range(20):
        assert det.check_all([[7]], req_ids=["off"]) == []


def test_w1_1_r04_second_append_is_kept_and_seen_follows_store(tmp_path: Path):
    """A second append of the same row is another token.

    MR v2 writes each step once. Folding follows Store, so
    ``content_tokens_seen`` stays equal to ``output_token_count``.
    """
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
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
    assert oc == 15
    assert seen == oc
    assert snap.output_token_count == oc


def test_w1_1_run_after_sample_cpu_folds_store_only(tmp_path: Path):
    """DetectorManager CPU path must not re-fold frozen sampled rows past Store."""
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
    )
    cfg._data["detector"]["token_repeat"]["enabled"] = True
    cfg._data["detector"]["token_repeat"]["repeat_sum_threshold"] = 9999
    cfg._data["detector"]["token_repeat"]["min_tokens"] = 1

    runner = SimpleNamespace(tp_rank=0, input_batch=None)
    mgr = DetectorManager(runtime_config=cfg, runner=runner)
    mgr.apply_runtime_config()
    io = RequestIoSnapshotManager.get()
    rid = "poem"

    io.append_batch([rid], [[7]])
    snap = AfterSampleCpuSnapshot(
        req_ids=[rid],
        has_nonempty_sampled=True,
        skip_req_ids=None,
    )
    mgr.run_after_sample_cpu(snap)
    # A second append is kept. The CPU path folds Store only, so seen matches.
    io.append_batch([rid], [[7]])
    mgr.run_after_sample_cpu(snap)

    st = RequestGuardStore.get().get_state(rid)
    assert st is not None
    assert len(st.output_token_ids) == 2
    assert mgr.get("token_repeat")._states[rid].content_tokens_seen == 2  # type: ignore[attr-defined]


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
    # Payload metadata must reflect the blocks actually dumped.
    t = torch.arange(0, 4 * 8).reshape(4, 8).float()
    sliced = _slice_blocks(t, [1, 9])  # 9 out of range
    assert sliced is not None
    out, used = sliced
    assert used == [1]
    assert out.shape == (1, 8)
    assert torch.equal(out[0], t[1])


def test_multi_group_dump_uses_per_group_block_ids(tmp_path: Path):
    """Hybrid: group0 and group1 must not share group0's page list."""
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    g0 = torch.arange(0, 4 * 4, dtype=torch.float32).reshape(4, 4)
    g1 = torch.arange(100, 100 + 4 * 4, dtype=torch.float32).reshape(4, 4)
    # Fill distinctive rows: g0 block1 → 4..7; g1 block2 → 108..111
    requests = {
        "r1": SimpleNamespace(block_ids=([1], [2])),
    }
    groups = [
        SimpleNamespace(layer_names=["L0"]),
        SimpleNamespace(layer_names=["L1"]),
    ]
    runner = SimpleNamespace(
        requests=requests,
        kv_caches={"L0": g0, "L1": g1},
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        block_tables=SimpleNamespace(blocks_per_kv_block=[1, 1]),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(
        req_id="r1",
        block_ids=[1],  # arm-time group0 hint
        out_dir=tmp_path / "kv",
    )
    by_layer = {s.payload["layer"]: s.payload for s in snaps}
    assert set(by_layer) == {"L0", "L1"}
    assert by_layer["L0"]["block_ids"] == [1]
    assert by_layer["L1"]["block_ids"] == [2]
    assert torch.equal(by_layer["L0"]["tensor"][0], g0[1])
    assert torch.equal(by_layer["L1"]["tensor"][0], g1[2])


def test_dump_uses_armed_rows_for_estimate_and_delayed_copy(tmp_path):
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor

    pages = torch.arange(8, dtype=torch.float32).view(-1, 1)
    state = SimpleNamespace(block_ids=([1], [2, 3, 4, 5], []))
    guard = MagicMock()
    runner = SimpleNamespace(
        kv_caches={"A": pages, "B": pages + 100, "C": pages + 200},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=[name]) for name in ("A", "B", "C")]
        ),
        block_tables=SimpleNamespace(blocks_per_kv_block=1),
        requests={"r1": state},
        runtime_guard=guard,
    )
    reader = KvCacheReader(runner)
    ctx = _dump_ctx(
        tmp_path=tmp_path,
        incident=Incident(incident_type="token_repeat", req_id="r1", block_ids=[1], wave=7),
        runner=runner,
        kv_reader=reader,
    )
    DumpKvAction().run(ctx)
    job = guard.queue_kv_dump.call_args.args[0]
    rows = job["manager_block_ids_by_group"]
    assert rows == [[1], [2, 3, 4, 5], []]
    estimate = reader.estimate_dump_bytes(manager_block_ids_by_group=rows)
    state.block_ids[0].append(6)
    state.block_ids[1].clear()
    state.block_ids[2].append(7)
    proc = SimpleNamespace(
        runner=runner,
        runtime_config=ctx.runtime_config,
        quota=ctx.quota,
        action_executor=SimpleNamespace(_kv_reader=reader, submit_heavy=None),
        wave_tracker=SimpleNamespace(current_wave=lambda: 8),
    )
    RuntimeGuardProcessor._run_kv_dumps(proc, [job])
    shards = [torch.load(path, weights_only=True) for path in tmp_path.rglob("*.pt")]
    assert {p["layer"]: p["block_ids"] for p in shards} == {"A": [1], "B": [2, 3, 4, 5]}
    assert all((p["dump_arm_wave"], p["dump_copy_wave"]) == (7, 8) for p in shards)
    assert estimate == sum(p["tensor"].nbytes for p in shards) == 20
    values = {p["layer"]: p["tensor"].flatten().tolist() for p in shards}
    assert values == {"A": [1], "B": [102, 103, 104, 105]}
    info = json.loads(next(tmp_path.rglob("request_info.json")).read_text(encoding="utf-8"))
    assert info["manager_block_ids_by_group"] == rows


def test_draft_layer_in_non_zero_group_uses_that_group(tmp_path: Path):
    """Spec draft resident in group1 must use group1 block_ids (not group0)."""
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    target = torch.zeros(4, 4)
    draft = torch.arange(0, 4 * 4, dtype=torch.float32).reshape(4, 4)
    requests = {"r1": SimpleNamespace(block_ids=([0], [3]))}
    groups = [
        SimpleNamespace(layer_names=["target.attn"]),
        SimpleNamespace(layer_names=["draft.attn"]),
    ]
    runner = SimpleNamespace(
        requests=requests,
        kv_caches={"target.attn": target, "draft.attn": draft},
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        block_tables=SimpleNamespace(blocks_per_kv_block=[1, 1]),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(req_id="r1", block_ids=[0], out_dir=tmp_path / "kv")
    draft_snap = next(s for s in snaps if s.payload["layer"] == "draft.attn")
    assert draft_snap.payload["block_ids"] == [3]
    assert draft_snap.payload["kv_cache_group"] == 1
    assert torch.equal(draft_snap.payload["tensor"][0], draft[3])


def test_dump_skips_when_kv_cache_groups_missing(tmp_path: Path):
    """Empty layer→group map must not silently dump with group-0 pages."""
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    cache = torch.arange(0, 4 * 4, dtype=torch.float32).reshape(4, 4)
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1],))},
        kv_caches={"L0": cache},
        kv_cache_config=None,
        block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(
        req_id="r1",
        block_ids=[1],
        out_dir=tmp_path / "kv",
    )
    assert snaps == []


def test_dump_skips_layer_missing_from_group_map(tmp_path: Path):
    """Layer not registered in kv_cache_groups is skipped; mapped layers still dump."""
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    g0 = torch.arange(0, 4 * 4, dtype=torch.float32).reshape(4, 4)
    orphan = torch.arange(100, 100 + 4 * 4, dtype=torch.float32).reshape(4, 4)
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1], [2]))},
        kv_caches={"L0": g0, "orphan.attn": orphan},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["L0"])],
        ),
        block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(
        req_id="r1",
        block_ids=[1],
        out_dir=tmp_path / "kv",
    )
    assert [s.payload["layer"] for s in snaps] == ["L0"]
    assert snaps[0].payload["kv_cache_group"] == 0
    assert snaps[0].payload["block_ids"] == [1]
    assert torch.equal(snaps[0].payload["tensor"][0], g0[1])


def test_explicit_group0_layer_still_dumps(tmp_path: Path):
    """True group-0 membership (name in map → 0) must still produce a dump."""
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    cache = torch.arange(0, 4 * 4, dtype=torch.float32).reshape(4, 4)
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([2],))},
        kv_caches={"L0": cache},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["L0"])],
        ),
        block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(
        req_id="r1",
        block_ids=[2],
        out_dir=tmp_path / "kv",
    )
    assert len(snaps) == 1
    assert snaps[0].payload["kv_cache_group"] == 0
    assert snaps[0].payload["block_ids"] == [2]
    assert torch.equal(snaps[0].payload["tensor"][0], cache[2])


def test_expand_manager_block_ids_kernel_ratio():
    from vllm_ascend.observability.runtime_guard.dump import expand_manager_block_ids

    assert expand_manager_block_ids([1, 3], 1) == [1, 3]
    assert expand_manager_block_ids([1], 2) == [2, 3]
    assert expand_manager_block_ids([0, 1], 2) == [0, 1, 2, 3]


def test_manager_block_ids_expanded_when_blocks_per_kv_gt_1():
    from vllm_ascend.observability.runtime_guard.dump import block_ids_for_request

    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1],))},
        block_tables=SimpleNamespace(blocks_per_kv_block=[2]),
        input_batch=None,
        execute_model_state=None,
        req_states=None,
        kv_cache_config=None,
    )
    assert block_ids_for_request(runner, "r1", kv_cache_group=0) == [2, 3]


def test_capture_expanded_cpu_rows_preserves_repeated_manager_blocks():
    import numpy as np

    from vllm_ascend.observability.runtime_guard.dump import request_manager_blocks

    runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"r": 0}),
        block_tables=SimpleNamespace(
            blocks_per_kv_block=[2, 3],
            block_tables_cpu=[np.array([[2, 3, 2, 3, 6, 7]]), np.array([[9, 10, 11]])],
            num_blocks_np=np.array([[6], [3]]),
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None, None]),
    )
    assert request_manager_blocks(runner, "r") == [[1, 1, 3], [3]]
    runner.block_tables.num_blocks_np[0, 0] = 0
    assert request_manager_blocks(runner, "r") == [[], [3]]


def test_scalar_blocks_per_kv_block_expands_not_silent_r1():
    """Gemini: scalar ratio must expand (not TypeError→R=1)."""
    from vllm_ascend.observability.runtime_guard.dump import (
        _blocks_per_kv_block_for_group,
        block_ids_for_request,
    )

    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1],))},
        block_tables=SimpleNamespace(blocks_per_kv_block=2),  # scalar, not [2]
        input_batch=None,
        execute_model_state=None,
        req_states=None,
        kv_cache_config=None,
    )
    assert _blocks_per_kv_block_for_group(runner, 0) == (2, False)
    assert _blocks_per_kv_block_for_group(runner, 1) == (2, False)
    assert block_ids_for_request(runner, "r1", kv_cache_group=0) == [2, 3]


def test_numpy_scalar_blocks_per_kv_block_shared_across_groups():
    """Nit #4: numpy scalar uses the same 'shared R' contract as Python int."""
    import numpy as np

    from vllm_ascend.observability.runtime_guard.dump import _blocks_per_kv_block_for_group

    runner = SimpleNamespace(
        block_tables=SimpleNamespace(blocks_per_kv_block=np.int64(2)),
        kv_cache_config=None,
    )
    assert _blocks_per_kv_block_for_group(runner, 0) == (2, False)
    assert _blocks_per_kv_block_for_group(runner, 1) == (2, False)


def test_blocks_per_kv_index_error_warns_and_marks_unknown(monkeypatch):
    """Nit #4 / Minor #1: short per-group sequence must not silent-R=1."""
    from vllm_ascend.observability.runtime_guard import dump as dump_mod
    from vllm_ascend.observability.runtime_guard.dump import _blocks_per_kv_block_for_group

    calls: list[tuple] = []
    monkeypatch.setattr(dump_mod.logger, "warning_once", lambda *a, **_k: calls.append(a))
    runner = SimpleNamespace(
        block_tables=SimpleNamespace(blocks_per_kv_block=[2]),  # only group 0
        kv_cache_config=None,
    )
    assert _blocks_per_kv_block_for_group(runner, 1) == (1, True)
    assert calls and any("index out of range" in str(a) for a in calls[0])


def test_derive_ratio_uses_physical_states_not_logical_tokens():
    """Alex #3: dump tensor dim0 uses storage R, not logical block_size//kernel."""
    import numpy as np

    from vllm_ascend.observability.runtime_guard.dump import (
        _blocks_per_kv_block_for_group,
        block_ids_for_request,
    )

    # logical would be 128/32=4; storage layout is 64/32=2
    spec = SimpleNamespace(block_size=128, num_states=64)
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1],))},
        block_tables=SimpleNamespace(),  # no blocks_per_kv_block (main path)
        kernel_block_sizes=[32],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]),
        attn_groups=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    assert _blocks_per_kv_block_for_group(runner, 0) == (2, False)
    assert block_ids_for_request(runner, "r1", kv_cache_group=0) == [2, 3]

    # Manager-granularity public tables must expand too when attr is absent.
    entry = SimpleNamespace(np=np.array([[5, 6]], dtype=np.int32), gpu=None)
    runner_bt = SimpleNamespace(
        requests=None,
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=np.array([[2]], dtype=np.int32)),
            block_tables=[entry],
        ),
        kernel_block_sizes=[32],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]),
        attn_groups=None,
        input_batch=None,
        execute_model_state=None,
        req_states=SimpleNamespace(req_id_to_index={"r1": 0}),
    )
    assert block_ids_for_request(runner_bt, "r1", kv_cache_group=0) == [10, 11, 12, 13]


def test_underivable_compressed_ratio_warns_and_payload_marks_unknown(tmp_path: Path, monkeypatch):
    """Minor #1: compressed spec without kernel size → warn + expansion=unknown."""
    from vllm_ascend.observability.runtime_guard import dump as dump_mod
    from vllm_ascend.observability.runtime_guard.dump import (
        KvCacheReader,
        _blocks_per_kv_block_for_group,
    )

    calls: list[tuple] = []
    monkeypatch.setattr(dump_mod.logger, "warning_once", lambda *a, **_k: calls.append(a))

    # storage != block_size ⇒ compressed; no kernel_block_sizes ⇒ underivable
    spec = SimpleNamespace(block_size=128, num_states=64)
    cache = torch.arange(0, 4 * 8, dtype=torch.float32).reshape(4, 8)
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1],))},
        block_tables=SimpleNamespace(),
        kernel_block_sizes=None,
        attn_groups=None,
        kv_caches={"L0": cache},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["L0"], kv_cache_spec=spec)],
        ),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
        tp_rank=0,
        dp_rank=0,
        dcp_rank=0,
        dcp_size=1,
    )
    assert _blocks_per_kv_block_for_group(runner, 0) == (1, True)
    assert calls and "underivable" in calls[0][0]

    snaps = KvCacheReader(runner).snapshot_request_blocks(
        req_id="r1",
        block_ids=[1],
        out_dir=tmp_path / "kv",
    )
    assert len(snaps) == 1
    assert snaps[0].payload.get("expansion") == "unknown"


def test_kv_source_logs_coverage_when_counts_differ(monkeypatch):
    """Alex #4: surface layer-count mismatch between forward_context and flat list."""
    from vllm_ascend.observability.runtime_guard import dump as dump_mod
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    calls: list[tuple] = []
    monkeypatch.setattr(dump_mod.logger, "warning_once", lambda *a, **_k: calls.append(a))

    k = torch.arange(0, 2 * 4, dtype=torch.float32).reshape(2, 4)
    v = torch.arange(50, 50 + 2 * 4, dtype=torch.float32).reshape(2, 4)
    layer = SimpleNamespace(kv_cache=(k, v))
    runner = SimpleNamespace(
        kv_caches=[k, v, k],  # 3 flat entries vs 1 forward_context layer
        compilation_config=SimpleNamespace(static_forward_context={"L0": layer}),
    )
    src = KvCacheReader(runner)._kv_sources()
    assert src[0][0] == "static_forward_context"
    assert calls and "static_forward_context has" in calls[0][0]


def test_kv_source_prefers_forward_context_tuple_parts(tmp_path: Path):
    """C8-style tuple on static_forward_context keeps part names (not flat list)."""
    from vllm_ascend.observability.runtime_guard.dump import KvCacheReader

    k = torch.arange(0, 2 * 4, dtype=torch.float32).reshape(2, 4)
    v = torch.arange(50, 50 + 2 * 4, dtype=torch.float32).reshape(2, 4)
    scale = torch.arange(90, 90 + 2 * 2, dtype=torch.float32).reshape(2, 2)
    layer = SimpleNamespace(kv_cache=(k, v, scale))
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([1],))},
        kv_caches=[k, v, scale],  # flattened Ascend bind — must NOT be preferred
        compilation_config=SimpleNamespace(static_forward_context={"L0": layer}),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["L0"])]),
        block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(req_id="r1", block_ids=[1], out_dir=tmp_path / "kv")
    layers = [s.payload["layer"] for s in snaps]
    assert layers == ["L0[0]", "L0[1]", "L0[2]"]
    assert snaps[0].payload["source"] == "static_forward_context"
    assert torch.equal(snaps[0].payload["tensor"][0], k[1])


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
    runner = SimpleNamespace(
        kv_caches={"L0": cache},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["L0"])],
        ),
        requests={"r1": SimpleNamespace(block_ids=([0, 2],))},
        block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        compilation_config=None,
        input_batch=None,
        execute_model_state=None,
        req_states=None,
    )
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


def test_write_snapshots_torch_save_failure_writes_dump_skipped(tmp_path: Path, monkeypatch):
    from vllm_ascend.observability.runtime_guard.dump import KvDumpSnapshot

    dump_root = tmp_path / "dump"
    snap = KvDumpSnapshot(
        path=tmp_path / "out" / "r1_L0_req.pt",
        payload={
            "req_id": "r1",
            "incident_type": "token_repeat",
            "dump_root": str(dump_root),
            "dump_arm_wave": 3,
            "rank_tag": "dp0_tp0_pp0_cp0",
            "layer": "L0",
            "tensor": torch.zeros(1),
        },
    )

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(
        "vllm_ascend.observability.runtime_guard.dump.torch.save",
        _boom,
    )
    assert KvCacheReader.write_snapshots([snap]) == []
    marker = dump_root / "token_repeat" / "r1" / "wave_3" / "dp0_tp0_pp0_cp0" / "dump_skipped.json"
    assert marker.is_file()
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["reason"] == "torch_save_failed"
    assert data["stage"] == "drain"
    assert "OSError" in data["detail"]["error"]


def test_free_bytes_at_walks_to_existing_ancestor(tmp_path: Path):
    from vllm_ascend.observability.runtime_guard.dump import free_bytes_at

    nested = tmp_path / "a" / "b" / "c" / "missing_file"
    free = free_bytes_at(nested)
    assert free is not None
    assert free >= 0


def test_free_bytes_at_returns_none_on_statvfs_error(tmp_path: Path, monkeypatch):
    from vllm_ascend.observability.runtime_guard import dump as dump_mod

    def _boom(_path):
        raise OSError("no statvfs")

    monkeypatch.setattr(dump_mod.os, "statvfs", _boom)
    assert dump_mod.free_bytes_at(tmp_path) is None


def test_estimate_dump_bytes_scales_with_blocks():
    cache = torch.zeros(4, 8, 2)
    reader = KvCacheReader(
        SimpleNamespace(
            kv_caches={"L0": cache},
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["L0"])]),
            block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        )
    )
    empty = reader.estimate_dump_bytes(manager_block_ids_by_group=[[]])
    two = reader.estimate_dump_bytes(manager_block_ids_by_group=[[0, 1]])
    assert empty == 0
    assert two == int(cache.nbytes) // 2


def test_dump_kv_skips_when_free_below_payload_plus_headroom(tmp_path, monkeypatch):
    from vllm_ascend.observability.runtime_config._defaults import DUMP_FREE_HEADROOM_BYTES
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction

    cache = torch.zeros(4, 8, 2)
    reader = KvCacheReader(
        SimpleNamespace(
            kv_caches={"L0": cache},
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["L0"])]),
            block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
        )
    )
    estimated = reader.estimate_dump_bytes(manager_block_ids_by_group=[[0]])
    tp_size = 4
    headroom = DUMP_FREE_HEADROOM_BYTES
    monkeypatch.setattr(
        "vllm_ascend.observability.runtime_guard.action.actions.free_bytes_at",
        lambda _path: estimated + headroom - 1,
    )
    monkeypatch.setattr(
        "vllm_ascend.observability.runtime_guard.action.actions.runner_tp_world_size",
        lambda _runner: tp_size,
    )
    quota = MagicMock()
    rc = SimpleNamespace(
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


def test_dump_kv_arms_when_request_finished_stamps_flag(tmp_path):
    """Finished-but-not-reaped may still arm; request_finished_at_dump is set."""
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.mark_finished(["r1"], wave=0)

        def _queue(job):
            store.add_dump_jobs([job["req_id"]])
            return True

        guard = MagicMock()
        guard.queue_kv_dump.side_effect = _queue
        runner = SimpleNamespace(
            tp_rank=0,
            dp_rank=0,
            dcp_rank=0,
            dcp_size=1,
            runtime_guard=guard,
        )
        ctx = _dump_ctx(
            tmp_path=tmp_path,
            incident=Incident(
                incident_type="token_repeat",
                req_id="r1",
                block_ids=[0],
                wave=3,
            ),
            runner=runner,
            rank_tag="dp0_tp0_pp0_cp0",
            rc=_dump_rc(tmp_path),
        )
        DumpKvAction().run(ctx)
        guard.queue_kv_dump.assert_called_once()
        job = guard.queue_kv_dump.call_args.args[0]
        assert job["request_finished_at_dump"] is True
        assert job["block_ids"] == [0]
        info = tmp_path / "token_repeat" / "r1" / "wave_3" / "request_info.json"
        assert info.is_file()
        data = json.loads(info.read_text(encoding="utf-8"))
        assert data["request_finished_at_dump"] is True
        assert store.get_state("r1") is not None
        assert store.get_state("r1").dump_jobs >= 1
    finally:
        RequestGuardStore.reset_for_tests()


def test_dump_kv_skips_when_request_reaped(tmp_path):
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.mark_finished(["r1"], wave=0)
        store.clear("r1")
        assert store.kv_dump_allowed("r1") is False
        quota = MagicMock()
        ctx = _dump_ctx(
            tmp_path=tmp_path,
            incident=Incident(incident_type="token_repeat", req_id="r1", block_ids=[0]),
            quota=quota,
            rank_tag="dp0_tp0_pp0_cp0",
            rc=_dump_rc(tmp_path),
        )
        DumpKvAction().run(ctx)
        quota.try_consume.assert_not_called()
        markers = list(tmp_path.glob("**/dump_skipped.json"))
        assert len(markers) == 1
        data = json.loads(markers[0].read_text(encoding="utf-8"))
        assert data["reason"] == "reaped"
        assert data["stage"] == "arm"
        assert data["req_id"] == "r1"
    finally:
        RequestGuardStore.reset_for_tests()


def test_dump_kv_writes_request_info_json(tmp_path):
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

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
        assert data["request_finished_at_dump"] is False
        assert data["detail"]["repeat_sum"] == 99
        assert data["detail"]["prompt_token_count"] == 3
        assert "prompt_token_ids" not in data["detail"]
    finally:
        RequestGuardStore.reset_for_tests()


def test_write_kv_dump_skipped_marker(tmp_path):
    from vllm_ascend.observability.runtime_guard.dump import write_kv_dump_skipped

    path = write_kv_dump_skipped(
        tmp_path,
        req_id="r2",
        incident_type="logits_finite",
        reason="reaped",
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
    assert data["reason"] == "reaped"
    assert data["incident_type"] == "logits_finite"


def test_write_kv_dump_skipped_arm_reasons(tmp_path):
    from vllm_ascend.observability.runtime_guard.dump import write_kv_dump_skipped

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
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.observability.runtime_guard.state import MANUAL_TRIGGER_TYPE, RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        monkeypatch.setattr(
            "vllm_ascend.observability.runtime_guard.state.iter_local_request_rows",
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
    from vllm_ascend.observability.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

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
            "vllm_ascend.observability.runtime_guard.dump.block_ids_for_request",
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


def test_queue_kv_dump_dedupes_req_id_same_step():
    from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor

    from ._helpers import capture_logger_text

    proc = SimpleNamespace(_kv_dump_jobs=[])
    with capture_logger_text("vllm_ascend.observability.runtime_guard.processor_dump") as buf:
        assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r1", "arm_id": "a", "wave": 1}) is True
        assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r1", "arm_id": "b", "wave": 1}) is False
    assert "already pending" in buf.getvalue()
    # Different arm wave may queue again (previous job not yet drained).
    assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r1", "arm_id": "c", "wave": 2}) is True
    assert RuntimeGuardProcessor.queue_kv_dump(proc, {"req_id": "r2", "arm_id": "b", "wave": 1}) is True
    assert [(j["req_id"], j["wave"]) for j in proc._kv_dump_jobs] == [
        ("r1", 1),
        ("r1", 2),
        ("r2", 1),
    ]


def test_mrv1_host_mirror_table_is_not_used():
    """Alex #2: do not return manager-granularity ids from MRV1 host-mirror tables."""
    import numpy as np

    from vllm_ascend.observability.runtime_guard.dump import block_ids_for_request

    # Tempting MRV1-style batch table with manager ids — must be ignored.
    host_table = SimpleNamespace(
        num_blocks_per_row=[2],
        block_table=SimpleNamespace(np=np.array([[1, 2, 0]], dtype=np.int32)),
    )
    input_batch = SimpleNamespace(
        req_id_to_index={"r1": 0},
        req_ids=["r1"],
        block_table=SimpleNamespace(block_tables=[host_table]),
        idx_mapping_np=None,  # no V2 state mapping → cannot resolve via block_tables
    )
    runner = SimpleNamespace(
        requests=None,
        input_batch=input_batch,
        execute_model_state=None,
        block_tables=None,
        req_states=None,
    )
    assert block_ids_for_request(runner, "r1") == []


def test_bug4_block_ids_v2_without_input_batch():
    """BUG-4: after sample_tokens clears execute_model_state, still resolve via req_states."""
    import numpy as np

    from vllm_ascend.observability.runtime_guard.dump import block_ids_for_request

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

    from vllm_ascend.observability.runtime_guard.dump import block_ids_for_request

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
