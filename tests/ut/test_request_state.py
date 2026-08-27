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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.ut.dfx_test_utils import capture_vllm_ascend_logs, make_dfx_config
from vllm_ascend.dfx.detector.manager import DetectorManager
from vllm_ascend.dfx.dfx_types import DumpFinishMeta
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.dfx.request_state import RequestDfxStore


def test_store_clear_drops_shared_fields_and_detectors(tmp_path):
    RequestDfxStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    store = RequestDfxStore.get()
    io = RequestIoSnapshotManager.get()

    io.append_output("r1", [1, 2, 3])
    store.set_filter_allowed("r1", True)
    store.record_sample_waves(["r1"], wave=4)
    store.mark_stopped_after_alert("r1")
    store.set_dump_finish("r1", DumpFinishMeta(dump_arm_wave=1, dump_activate_wave=2))

    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1"]), requests={})
    mgr = DetectorManager(dfx_config=make_dfx_config(tmp_path), runner=runner)
    det = mgr.get("token_repeat")
    assert det is not None
    det.clear_finished = MagicMock()  # type: ignore[method-assign]

    # take dump_finish before clear (processor write path)
    meta = store.take_dump_finish("r1")
    assert meta is not None
    assert meta.dump_arm_wave == 1
    assert io.cumulative_output_ids("r1") == [1, 2, 3]

    popped = store.clear("r1", detectors=mgr)
    assert popped is not None
    assert popped.output_token_ids == [1, 2, 3]
    assert "r1" not in store
    assert io.cumulative_output_count("r1") == 0
    assert store.get_filter_allowed("r1") is None
    assert store.take_sample_wave("r1") is None
    assert store.stopped_req_ids() == set()
    assert store.take_dump_finish("r1") is None
    det.clear_finished.assert_called_once_with("r1")


def test_sample_wave_fifo_via_store():
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.record_sample_waves(["r1"], 1)
    store.record_sample_waves(["r1"], 2)
    assert store.take_sample_wave("r1") == 1
    assert store.take_sample_wave("r1") == 2
    assert store.take_sample_wave("r1") is None


def test_store_on_clear_hook_does_not_import_io():
    """Store.clear must notify hooks without importing io_snapshot."""
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    seen: list[str] = []
    store.register_on_clear(seen.append)
    store.get_or_create("r1")
    store.clear("r1")
    assert seen == ["r1"]


def test_store_clear_drops_io_snapshot_wave_cache():
    """IoSnapshotManager registers on_clear so finish clears stale report cache."""
    RequestDfxStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    store = RequestDfxStore.get()
    io = RequestIoSnapshotManager.get()
    io.append_output("r1", [1, 2])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_output_token_ids=[[-1]]),
        requests={},
    )
    assert io.snapshot(runner, "r1", 0, include_token_ids=True).output_token_ids == [1, 2]
    assert "r1|1" in io._cache
    store.clear("r1")
    assert "r1|1" not in io._cache
    assert io.cumulative_output_count("r1") == 0


def test_io_get_rebinds_on_clear_after_store_reset():
    """If Store is reset while Io singleton survives, get() re-registers the hook."""
    RequestIoSnapshotManager.reset_for_tests()
    io = RequestIoSnapshotManager.get()
    io.append_output("r1", [7])
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_output_token_ids=[]), requests={})
    io.snapshot(runner, "r1", 0, include_token_ids=True)
    # Simulate partial test reset: Store gone, Io instance still alive.
    RequestDfxStore.reset_for_tests()
    # Rebind + restore a state entry so clear has something to pop.
    io2 = RequestIoSnapshotManager.get()
    assert io2 is io
    store = RequestDfxStore.get()
    store.append_output_ids("r1", [7])
    io._cache["r1|1"] = io._cache.get("r1|1") or MagicMock()
    store.clear("r1")
    assert "r1|1" not in io._cache


def test_deferred_reap_waits_for_sample_wave_drain():
    """finished + pending stamp → keep state; after take → reapable."""
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.append_output_ids("r1", [1])
    store.mark_finished(["r1"], wave=5)
    store.record_sample_waves(["r1"], 5)  # last stamp after mark (runner order)
    assert store.is_finished("r1")
    assert store.list_reapable(current_wave=5) == []
    assert store.take_sample_wave("r1") == 5
    assert store.list_reapable(current_wave=5) == ["r1"]
    # Same object across mark/record (no double-create).
    state = store.get_state("r1")
    assert state is not None
    store.record_sample_waves(["r1"], 6)
    assert store.get_state("r1") is state


def test_deferred_reap_force_after_max_deferred_waves():
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.max_deferred_waves = 2
    store.mark_finished(["r1"], wave=10)
    store.record_sample_waves(["r1"], 10)
    assert store.list_reapable(current_wave=11) == []
    assert store.list_reapable(current_wave=12) == ["r1"]


def test_on_clear_hook_logs_exception(caplog):
    import logging

    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()

    def _boom(_rid: str) -> None:
        raise RuntimeError("cache clear failed")

    store.register_on_clear(_boom)
    store.get_or_create("r1")
    with capture_vllm_ascend_logs(caplog, logging.ERROR):
        store.clear("r1")
    assert any("on_clear hook failed" in r.getMessage() for r in caplog.records)


def test_ready_to_reap_and_empty_guards():
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    assert store.get_state("") is None
    assert store.is_finished("") is False
    assert store.ready_to_reap("", current_wave=1) is False
    assert store.ready_to_reap("missing", current_wave=1) is False
    assert store.clear("") is None
    assert store.take_sample_wave("") is None
    assert store.sample_wave_pending("missing") == 0
    store.mark_finished(None, wave=1)
    store.mark_finished([""], wave=1)
    store.record_sample_waves(None, 1)
    store.record_sample_waves(["", "r1"], 1)
    store.mark_finished(["r1"], wave=1)
    assert store.is_finished("r1")
    assert store.sample_wave_pending("r1") == 1
    assert store.ready_to_reap("r1", current_wave=1) is False
    store.take_sample_wave("r1")
    assert store.ready_to_reap("r1", current_wave=1) is True


def test_force_reap_clear_warns_on_small_leftover_waves(caplog):
    import logging

    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.max_deferred_waves = 1
    store.mark_finished(["r1"], wave=3)
    store.record_sample_waves(["r1"], 3)
    assert store.list_reapable(current_wave=4) == ["r1"]
    with capture_vllm_ascend_logs(caplog, logging.WARNING):
        store.clear("r1")
    assert any("leftover sample_waves" in r.getMessage() for r in caplog.records)


def test_force_reap_clear_debug_on_large_leftover_waves(caplog):
    """Large FIFO leftover (legacy async non-consumer) must not WARNING-spam."""
    import logging

    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.max_deferred_waves = 2
    store.mark_finished(["r1"], wave=1)
    for w in range(10):
        store.record_sample_waves(["r1"], w)
    assert store.list_reapable(current_wave=3) == ["r1"]
    with capture_vllm_ascend_logs(caplog, logging.DEBUG):
        store.clear("r1")
    leftover = [r for r in caplog.records if "leftover sample_waves" in r.getMessage()]
    assert leftover
    assert all(r.levelno < logging.WARNING for r in leftover)


def test_processor_check_after_sample_reaps_finished(tmp_path):
    from vllm_ascend.dfx.processor import DfxProcessor

    RequestDfxStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    store = RequestDfxStore.get()
    io = RequestIoSnapshotManager.get()
    io.append_output("r1", [9])

    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(tp_rank=0, use_async_scheduling=False, requests={}, input_batch=None)
    proc.dfx_config = MagicMock()
    proc.dfx_config.log_print_output_on_finish.return_value = False
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.detectors = MagicMock()
    proc.detectors.check_after_sample.return_value = []
    proc.detectors.get.return_value = None
    proc.dumper = MagicMock()
    proc.dumper.current_wave.return_value = 2
    proc.dumper.take_sample_wave.return_value = None
    proc.report_writer = MagicMock()

    proc.mark_finished(["r1"])
    assert "r1" in store
    proc.check_after_sample(sampled_token_ids=[[9]], logprobs_lists=None, req_ids=["r1"])
    assert "r1" not in store
    proc.detectors.clear_finished.assert_called_with("r1")


def test_post_reap_late_append_stamps_finished_for_reap():
    """S14/R1: late async append after clear() must not leave a zombie state."""
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.append_output_ids("r1", [1])
    store.mark_finished(["r1"], wave=1)
    store.clear("r1")
    assert "r1" not in store

    store.append_output_ids("r1", [2])
    assert store.is_finished("r1")
    assert store.list_reapable(current_wave=1) == ["r1"]
    store.clear("r1")
    assert "r1" not in store


def test_reaped_discarded_on_new_live_sample_wave():
    """Id reuse: record_sample_waves clears reaped tracking for a new live req."""
    RequestDfxStore.reset_for_tests()
    store = RequestDfxStore.get()
    store.mark_finished(["r1"], wave=1)
    store.clear("r1")
    store.record_sample_waves(["r1"], wave=2)
    store.append_output_ids("r1", [9])
    assert not store.is_finished("r1")


def test_processor_sync_for_step_reaps_idle_finished():
    from vllm_ascend.dfx.processor import DfxProcessor

    RequestDfxStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    store = RequestDfxStore.get()
    store.mark_finished(["idle1"], wave=1)
    assert store.list_reapable(current_wave=1) == ["idle1"]

    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(tp_rank=0, dp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.hot_reload_enabled = False
    proc.dfx_config.log_print_output_on_finish.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.manual_triggers = MagicMock()
    proc.manual_triggers.consume_once.return_value = None
    proc.maybe_print_input_token_ids_once = MagicMock(return_value=False)  # type: ignore[method-assign]
    proc.detectors = MagicMock()
    proc.dumper = MagicMock()
    proc.dumper.current_wave.return_value = 1
    proc.dumper.advance_wave.return_value = 1
    proc.dumper.sync_dump_pending_or.return_value = False
    proc.dumper.take_dump_finish_meta.return_value = None
    proc.report_writer = MagicMock()
    proc._report_tokenizer = None

    proc.sync_for_step(allow_arm=True)
    assert "idle1" not in store
