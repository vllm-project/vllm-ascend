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

from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager, output_token_count_for_request
from vllm_ascend.dfx.processor import DfxProcessor


def test_append_output_builds_cumulative_and_dedupes_suffix():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [1, 2, -1, 3])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3]
    # Same wave again (spec + sample) → no double append.
    mgr.append_output("r1", [1, 2, 3])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3]
    # New chunk in the same wave may legitimately repeat the previous last token.
    mgr.append_output("r1", [3])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3, 3]
    mgr.append_output("r1", [4, 5])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3, 3, 4, 5]
    mgr.clear_req("r1")
    assert mgr.cumulative_output_ids("r1") == []


def test_append_output_keeps_identical_chunk_across_waves():
    """Consecutive steps with the same accepted ids must both be recorded."""
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [9, 9, 9])
    assert mgr.cumulative_output_ids("r1") == [9, 9, 9]
    # Next engine wave (sync_for_step → clear_wave_cache) resets dedupe frontier.
    mgr.clear_wave_cache()
    mgr.append_output("r1", [9, 9, 9])
    assert mgr.cumulative_output_ids("r1") == [9, 9, 9, 9, 9, 9]


def test_snapshot_prefers_cumulative_over_placeholder_batch():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [10, 20, 30])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_output_token_ids=[[-1, -1, -1]]),
        requests={},
    )
    snap = mgr.snapshot(runner, "r1", 0, include_token_ids=True, use_cache=False)
    assert snap.output_token_ids == [10, 20, 30]
    assert snap.output_token_count == 3
    assert output_token_count_for_request(runner, "r1", 0) == 3


def test_append_batch_builds_cumulative_per_req():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    # Batch rows are aligned to req_ids; -1 placeholders are stripped per row.
    mgr.append_batch(["r1", "r2", ""], [[1, 2], [5, -1, 6], [7]])
    assert mgr.cumulative_output_ids("r1") == [1, 2]
    assert mgr.cumulative_output_ids("r2") == [5, 6]
    # Empty req_id rows are skipped, not appended.
    mgr.append_batch(None, [[9]])
    mgr.append_batch(["r3"], None)
    assert mgr.cumulative_output_ids("r3") == []


def test_clear_wave_cache_resets_snapshot_cache_only():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [1, 2])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_output_token_ids=[[-1]]),
        requests={},
    )
    # First snapshot populates the wave cache.
    assert mgr.snapshot(runner, "r1", 0, include_token_ids=True).output_token_ids == [1, 2]
    # Same wave: cache is served, so the new token is not yet visible.
    mgr.append_output("r1", [3])
    assert mgr.snapshot(runner, "r1", 0, include_token_ids=True).output_token_ids == [1, 2]
    # clear_wave_cache drops snapshot cache + same-wave append dedupe frontier;
    # cumulative output persists.
    mgr.clear_wave_cache()
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3]
    snap = mgr.snapshot(runner, "r1", 0, include_token_ids=True)
    assert snap.output_token_ids == [1, 2, 3]


def test_clear_finished_clears_cumulative():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [1, 2])
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.detectors = MagicMock()
    proc.dfx_config = MagicMock()
    proc.dfx_config.log_print_output_on_finish.return_value = False
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.take_dump_finish_meta.return_value = None
    proc.report_writer = MagicMock()
    proc.clear_finished(["r1"])
    proc.detectors.clear_finished.assert_called_once_with("r1")
    assert mgr.cumulative_output_ids("r1") == []
