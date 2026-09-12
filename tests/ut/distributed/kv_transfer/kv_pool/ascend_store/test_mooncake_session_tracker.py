# SPDX-License-Identifier: Apache-2.0

import unittest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)


class TestMooncakeSessionTracker(unittest.TestCase):
    def test_commit_promotes_shared_put_key_to_every_request_owner(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("shared", 0)])
        tracker.register_put_keys("r2", [("shared", 1)])

        tracker.commit_put_keys(["shared"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("shared", 0)])
        self.assertEqual(tracker.prepare_load_entries("r2", []), [("shared", 1)])

    def test_complete_key_replaces_partial_key_for_the_same_block(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("partial", 1)])
        tracker.commit_put_keys(["partial"])
        tracker.register_put_keys("r1", [("complete", 1)])
        tracker.commit_put_keys(["complete"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("complete", 1)])

    def test_shared_get_ends_only_after_the_last_owner_releases_it(self):
        tracker = MooncakeSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.prepare_load_entries("r2", [("shared", 0)])
        tracker.record_get_result("shared", {"r1", "r2"}, succeeded=True)

        self.assertEqual(tracker.release_terminal({"r1"}), [])
        self.assertEqual(tracker.release_terminal({"r2"}), ["shared"])
        self.assertEqual(tracker.release_terminal({"r2"}), [])

    def test_failed_renewal_retains_desired_keys_for_retry(self):
        tracker = MooncakeSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.register_put_keys("r1", [("pending", 1)])
        tracker.record_get_result("shared", {"r1"}, succeeded=True)

        tracker.record_get_result("shared", {"r1"}, succeeded=False)
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.release_for_retry({"r1"}), [])
        self.assertEqual(
            tracker.prepare_load_entries("r1", []),
            [("shared", 0), ("pending", 1)],
        )

    def test_failed_get_attempt_preserves_unrelated_shared_owner(self):
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

        self.assertEqual(keys_to_end, ["new-key"])
        self.assertEqual(tracker.release_terminal({"old-owner"}), ["shared"])
        self.assertEqual(
            tracker.prepare_load_entries("new-owner", []),
            [("shared", 0), ("new-key", 1)],
        )

    def test_terminal_request_loses_pending_put_ownership(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("pending", 0)])

        tracker.release_terminal({"r1"})
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [])

    def test_chunk_commit_retry_and_terminal_cleanup(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("k0", 0)])
        tracker.commit_put_keys(["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_for_retry({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_terminal({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])
