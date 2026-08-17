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

from vllm_ascend.dfx.kv_block_meta import (
    KvBlockMetaTracker,
    block_ids_for_request,
    touched_block_ids,
)


def test_touched_block_ids_decode_and_prefill():
    ids = [10, 11, 12, 13]
    assert touched_block_ids(ids, block_size=16, num_computed_before=48, num_scheduled=1) == [13]
    assert touched_block_ids(ids, block_size=16, num_computed_before=0, num_scheduled=48) == [10, 11, 12]
    assert touched_block_ids([], block_size=16, num_computed_before=0, num_scheduled=1) == []


def test_block_ids_for_request_from_req_state():
    runner = SimpleNamespace(
        requests={"r1": SimpleNamespace(block_ids=([7, 8, 9],))},
        input_batch=None,
    )
    assert block_ids_for_request(runner, "r1") == [7, 8, 9]


def test_kv_block_meta_tracker_record_and_detail():
    KvBlockMetaTracker.reset_for_tests()
    tr = KvBlockMetaTracker.get()
    tr.record_writes("req-a", [1, 2], wave=5)
    tr.record_writes("req-b", [2], wave=6)
    assert tr.last_write_wave(1) == 5
    assert tr.last_writer_req_id(1) == "req-a"
    assert tr.last_write_wave(2) == 6
    assert tr.last_writer_req_id(2) == "req-b"
    detail = tr.blocks_detail([1, 2], include_wave=True, include_writer=True)
    assert detail == [
        {"block_id": 1, "last_write_wave": 5, "last_writer_req_id": "req-a"},
        {"block_id": 2, "last_write_wave": 6, "last_writer_req_id": "req-b"},
    ]
    wave_only = tr.blocks_detail([1], include_wave=True, include_writer=False)
    assert wave_only == [{"block_id": 1, "last_write_wave": 5}]
