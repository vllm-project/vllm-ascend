# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for resident-cache request incarnation identities."""

import unittest
from zlib import adler32

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.request_identity import RequestIdentityRegistry


class TestRequestIdentityRegistry(unittest.TestCase):
    def test_adler32_collisions_are_different_owners(self):
        request_ids = ["234054828099429543", "471754017250656876", "304399536143223879"]
        self.assertEqual({adler32(req_id.encode("utf-8")) for req_id in request_ids}, {585106354})
        registry = RequestIdentityRegistry()

        owners = [registry.get_or_create(req_id) for req_id in request_ids]

        self.assertEqual(len(set(owners)), len(request_ids))
        self.assertTrue(all(owner > 0 for owner in owners))
        self.assertEqual([registry.get_or_create(req_id) for req_id in request_ids[::-1]], owners[::-1])

    def test_unscheduled_request_and_interleaved_draft_steps_keep_identity(self):
        registry = RequestIdentityRegistry()
        target_owner = registry.get_or_create("target")
        other_owner = registry.get_or_create("other")

        for _ in range(4):
            self.assertEqual(registry.get_or_create("other"), other_owner)
        # Absence from intervening batches does not end the local lifecycle.
        self.assertEqual(registry.get_or_create("target"), target_owner)
        registry.remove("other")
        self.assertEqual(registry.get_or_create("target"), target_owner)

    def test_finished_or_cancelled_request_gets_new_incarnation(self):
        registry = RequestIdentityRegistry()
        old_owner = registry.get_or_create("reused-id")

        registry.remove("reused-id")
        registry.remove("reused-id")  # Duplicate finish notifications are harmless.
        new_owner = registry.get_or_create("reused-id")

        self.assertGreater(new_owner, old_owner)
        self.assertEqual(registry.get_or_create("reused-id"), new_owner)

    def test_cleanup_bounds_mapping_without_recycling_owner_ids(self):
        registry = RequestIdentityRegistry()
        owner = 0
        for step in range(1000):
            req_id = str(step)
            new_owner = registry.get_or_create(req_id)
            self.assertGreater(new_owner, owner)
            registry.remove(req_id)
            owner = new_owner
        self.assertEqual(len(registry._owner_ids), 0)
        self.assertGreater(registry.get_or_create("new"), owner)

    def test_owners_above_uint32_are_not_truncated(self):
        registry = RequestIdentityRegistry()
        registry._next_owner_id = 1 << 32

        owner = registry.get_or_create("large")

        self.assertEqual(owner, 1 << 32)
        self.assertEqual(registry.get_or_create("large"), owner)
        self.assertGreater(registry.get_or_create("next"), owner)

    def test_int64_limit_fails_closed_without_reusing_old_ids(self):
        registry = RequestIdentityRegistry()
        registry._next_owner_id = (1 << 63) - 1
        last_owner = registry.get_or_create("last")
        self.assertEqual(last_owner, (1 << 63) - 1)

        for req_id in ("overflow", "another"):
            with self.subTest(req_id=req_id), self.assertRaises(OverflowError):
                registry.get_or_create(req_id)
        self.assertEqual(registry.get_or_create("last"), last_owner)
        registry.remove("last")
        self.assertEqual(len(registry._owner_ids), 0)
        with self.assertRaises(OverflowError):
            registry.get_or_create("last")

    def test_non_string_ids_cannot_fall_back_to_batch_row(self):
        registry = RequestIdentityRegistry()
        for req_id in (None, 0, 1, b"request", object()):
            with self.subTest(req_id=req_id), self.assertRaises((ValueError, TypeError)):
                registry.get_or_create(req_id)
        self.assertEqual(len(registry._owner_ids), 0)
        self.assertEqual(registry.get_or_create("first"), 1)

    def test_dummy_domain_excludes_padding_empty_and_real_owners(self):
        registry = RequestIdentityRegistry()
        real_owner = registry.get_or_create("real")
        dummy_owners = [registry.dummy_owner_id(row) for row in range(4)]

        self.assertEqual(dummy_owners, [-2, -3, -4, -5])
        self.assertTrue(set(dummy_owners).isdisjoint({0, -1, real_owner}))
        self.assertEqual(registry.get_or_create("next"), real_owner + 1)


if __name__ == "__main__":
    unittest.main()
