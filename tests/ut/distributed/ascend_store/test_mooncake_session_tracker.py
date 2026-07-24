#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)


def test_commit_promotes_put_key_to_every_request_owner():
    tracker = MooncakeSessionTracker()
    tracker.register_put_keys("r1", [("shared", 0)])
    tracker.register_put_keys("r2", [("shared", 1)])

    tracker.commit_put_keys(["shared"])

    assert tracker.prepare_load_entries("r1", []) == [("shared", 0)]
    assert tracker.prepare_load_entries("r2", []) == [("shared", 1)]


def test_new_complete_key_replaces_partial_key_for_same_block():
    tracker = MooncakeSessionTracker()
    tracker.register_put_keys("r1", [("partial", 1)])
    tracker.commit_put_keys(["partial"])
    tracker.register_put_keys("r1", [("complete", 1)])
    tracker.commit_put_keys(["complete"])

    assert tracker.prepare_load_entries("r1", []) == [("complete", 1)]


def test_shared_key_ends_only_after_last_request_owner_releases_it():
    tracker = MooncakeSessionTracker()
    tracker.prepare_load_entries("r1", [("shared", 0)])
    tracker.prepare_load_entries("r2", [("shared", 0)])
    tracker.record_get_result("shared", {"r1", "r2"}, succeeded=True)

    assert tracker.release_terminal({"r1"}) == []
    assert tracker.release_terminal({"r2"}) == ["shared"]
    assert tracker.release_terminal({"r2"}) == []


def test_failed_renewal_clears_owners_but_retains_desired_keys():
    tracker = MooncakeSessionTracker()
    tracker.prepare_load_entries("r1", [("shared", 0)])
    tracker.register_put_keys("r1", [("pending", 1)])
    tracker.record_get_result("shared", {"r1"}, succeeded=True)

    tracker.record_get_result("shared", {"r1"}, succeeded=False)

    assert tracker.release_for_retry({"r1"}) == []
    tracker.commit_put_keys(["pending"])
    assert tracker.prepare_load_entries("r1", []) == [
        ("shared", 0),
        ("pending", 1),
    ]


def test_failed_get_attempt_keeps_unrelated_owner_and_cleans_unowned_key():
    tracker = MooncakeSessionTracker()
    tracker.prepare_load_entries("old-owner", [("shared", 0)])
    tracker.prepare_load_entries(
        "new-owner",
        [("shared", 0), ("new-key", 1)],
    )
    tracker.record_get_result(
        "shared",
        {"old-owner", "new-owner"},
        succeeded=True,
    )

    keys_to_end = tracker.release_failed_get_attempts(
        {
            "shared": {"new-owner"},
            "new-key": {"new-owner"},
        }
    )

    assert keys_to_end == ["new-key"]
    assert tracker.release_terminal({"old-owner"}) == ["shared"]
    assert tracker.prepare_load_entries("new-owner", []) == [
        ("shared", 0),
        ("new-key", 1),
    ]


def test_dropping_request_removes_pending_put_ownership():
    tracker = MooncakeSessionTracker()
    tracker.register_put_keys("r1", [("pending", 0)])

    tracker.release_terminal({"r1"})
    tracker.commit_put_keys(["pending"])

    assert tracker.prepare_load_entries("r1", []) == []
