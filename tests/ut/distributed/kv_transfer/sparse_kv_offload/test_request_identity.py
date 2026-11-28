# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for resident-cache request incarnation identities."""

import unittest
from unittest.mock import Mock, patch
from zlib import adler32

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import request_identity as identity_module
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

    def test_int64_limit_wraps_after_invalidating_resident_state(self):
        invalidate = Mock()
        registry = RequestIdentityRegistry(on_wrap=invalidate)
        registry._next_owner_id = (1 << 63) - 1
        last_owner = registry.get_or_create("last")
        self.assertEqual(last_owner, (1 << 63) - 1)
        invalidate.assert_not_called()

        self.assertEqual(registry.get_or_create("overflow"), 1)
        invalidate.assert_called_once_with()
        self.assertEqual(registry.get_or_create("another"), 2)
        self.assertEqual(registry.get_or_create("last"), last_owner)

    def test_wrap_keeps_live_and_subsequently_retired_owners_reserved(self):
        registry = RequestIdentityRegistry()
        registry.get_or_create("retired")
        retained_owner = registry.get_or_create("temporarily-unscheduled")
        registry.remove("retired")
        registry._next_owner_id = (1 << 63) - 1
        registry.get_or_create("last")

        self.assertEqual(registry.get_or_create("wrapped"), 1)
        self.assertEqual(registry.get_or_create("temporarily-unscheduled"), retained_owner)
        # A request can populate resident KV after wrap and then finish. Its
        # owner must remain unavailable until the next invalidation cycle.
        registry.remove("temporarily-unscheduled")
        self.assertEqual(registry.get_or_create("new"), 3)

    def test_repeated_wraps_invalidate_retired_resident_owners(self):
        resident_owners: set[int] = set()
        invalidate = Mock(side_effect=resident_owners.clear)
        registry = RequestIdentityRegistry(on_wrap=invalidate)
        live_owner = registry.get_or_create("live")
        with patch.object(identity_module, "MAX_OWNER_ID", 5):
            for step in range(30):
                owner = registry.get_or_create(str(step))
                self.assertTrue(1 <= owner <= 5)
                self.assertNotEqual(owner, live_owner)
                self.assertNotIn(owner, resident_owners)
                resident_owners.update((live_owner, owner))
                registry.remove(str(step))
                self.assertEqual(registry.get_or_create("live"), live_owner)
        self.assertGreater(invalidate.call_count, 1)

    def test_failed_invalidation_does_not_publish_a_recycled_owner(self):
        invalidate = Mock(side_effect=RuntimeError("invalidation failed"))
        registry = RequestIdentityRegistry(on_wrap=invalidate)
        registry._next_owner_id = (1 << 63) - 1
        last_owner = registry.get_or_create("last")
        with self.assertRaisesRegex(RuntimeError, "invalidation failed"):
            registry.get_or_create("overflow")
        self.assertNotIn("overflow", registry._owner_ids)
        self.assertEqual(registry.get_or_create("last"), last_owner)
        invalidate.side_effect = None
        self.assertEqual(registry.get_or_create("overflow"), 1)
        registry.remove("last")
        self.assertEqual(registry.get_or_create("last"), 2)

    def test_non_string_ids_cannot_fall_back_to_batch_row(self):
        registry = RequestIdentityRegistry()
        for req_id in (None, 0, 1, b"request", object()):
            with self.subTest(req_id=req_id), self.assertRaises((ValueError, TypeError)):
                # Deliberately pass invalid types to exercise runtime validation.
                registry.get_or_create(req_id)  # type: ignore[arg-type]
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
