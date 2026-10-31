# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolBackendExecutor, PoolReadRange


class TestPoolBackendExecutor(unittest.TestCase):
    def backend(self, plane="gva"):
        backend = MagicMock()
        backend.layerwise_data_plane = plane
        backend.absent_read_lease_keys.return_value = set()
        backend.batch_add_lease.return_value = [0]
        backend.batch_remove_lease.return_value = 0
        backend.batch_get_start.return_value = [0]
        backend.batch_get_end.return_value = 0
        backend.batch_get_key_info.return_value = [SimpleNamespace(size=lambda: 16, gva_list=lambda: [1000])]
        backend.batch_copy_gva.return_value = 0
        backend.batch_copy_get.return_value = [0]
        return backend

    def test_gva_offsets_and_bounded_batches(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000, max_bytes=4)
        executor.restore([PoolReadRange("key", 16, 3, 2000, 9)])
        self.assertEqual(
            backend.batch_copy_gva.call_args_list,
            [
                call([1003], [2000], [4], 1),
                call([1007], [2004], [4], 1),
                call([1011], [2008], [1], 1),
            ],
        )
        backend.batch_add_lease.assert_called_once_with(["key"], 10000)
        backend.batch_remove_lease.assert_called_once_with(["key"])

    def test_range_sessions_share_native_protection(self):
        backend = self.backend("block_key")
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with executor._reading(["key"]):
            executor.restore([PoolReadRange("key", 16, 2, 2000, 5)])
            backend.batch_get_end.assert_not_called()
        backend.batch_get_start.assert_called_once_with(["key"])
        backend.batch_get_end.assert_called_once_with(["key"])
        backend.batch_copy_get.assert_called_once_with(["key"], [[2000]], [[5]], [[2]])

    def test_copy_failure_cannot_report_success_and_releases_readers(self):
        for plane in ("gva", "block_key"):
            with self.subTest(plane=plane):
                backend = self.backend(plane)
                backend.batch_copy_gva.return_value = -1
                backend.batch_copy_get.return_value = [-1]
                executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    executor.restore([PoolReadRange("key", 16, 0, 2000, 16)])
                self.assertFalse(executor._readers)
                if plane == "gva":
                    backend.batch_remove_lease.assert_called_once()
                else:
                    backend.batch_get_end.assert_called_once()

    def test_wrong_object_size_rejected_before_copy(self):
        backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            PoolBackendExecutor(backend, lease_ttl_ms=10000).restore([PoolReadRange("key", 20, 0, 2000, 10)])
        backend.batch_copy_gva.assert_not_called()
        backend.batch_remove_lease.assert_called_once()

    def test_failed_start_rolls_back_only_new_sessions(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with executor._reading(["existing"]):
            backend.batch_add_lease.return_value = [-1]
            with self.assertRaisesRegex(RuntimeError, "acquisition failed"), executor._reading(["existing", "new"]):
                self.fail("failed acquisition entered read scope")
            backend.batch_remove_lease.assert_called_once_with(["new"])
            self.assertEqual(executor._readers, {"existing": 1})

    def test_missing_key_does_not_release_nonexistent_lease_or_poison_next_request(self):
        backend = self.backend()
        backend.absent_read_lease_keys.return_value = {"absent"}
        backend.batch_add_lease.side_effect = [[0, -3102], [0]]
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
            executor.restore([PoolReadRange(key, 16, 0, 2000, 4) for key in ("present", "absent")])
        backend.batch_remove_lease.assert_called_once_with(["present"])
        backend.batch_copy_gva.assert_not_called()
        self.assertFalse(executor._lease_counts)
        self.assertFalse(executor._pending_release)
        executor.restore([PoolReadRange("present", 16, 0, 2000, 4)])
        executor.close()
        self.assertEqual(backend.batch_remove_lease.call_count, 2)

    def test_malformed_missing_key_response_keeps_ambiguous_cleanup(self):
        backend = self.backend()
        backend.absent_read_lease_keys.return_value = {"first", "second"}
        backend.batch_add_lease.return_value = [-3102]
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with self.assertRaisesRegex(RuntimeError, "1 results for 2 keys"):
            executor.restore([PoolReadRange(key, 16, 0, 2000, 4) for key in ("first", "second")])
        backend.batch_remove_lease.assert_called_once_with(["first", "second"])
        backend.absent_read_lease_keys.assert_not_called()
        executor.close()

    def test_invalid_ranges_never_submit_io(self):
        for args in (("key", 16, 10, 2000, 7), ("key", 16, -1, 2000, 1), ("", 16, 0, 2000, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                PoolReadRange(*args)

    def test_failed_acquire_cleanup_also_quarantines_executor(self):
        backend = self.backend()
        backend.batch_add_lease.return_value = [-1]
        backend.batch_remove_lease.return_value = -1
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            executor.restore([PoolReadRange("key", 16, 0, 2000, 4)])
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            executor.restore([PoolReadRange("other", 16, 0, 2000, 4)])

    def test_failed_release_quarantines_executor(self):
        backend = self.backend()
        backend.batch_remove_lease.return_value = -1
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            executor.restore([PoolReadRange("key", 16, 0, 2000, 4)])
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            executor.restore([PoolReadRange("key", 16, 0, 2000, 4)])

    def test_failed_release_retried_on_close_without_live_readers(self):
        for plane in ("gva", "block_key"):
            for failed_acquire in (False, True):
                with self.subTest(plane=plane, failed_acquire=failed_acquire):
                    backend = self.backend(plane)
                    acquire = backend.batch_add_lease if plane == "gva" else backend.batch_get_start
                    release = backend.batch_remove_lease if plane == "gva" else backend.batch_get_end
                    acquire.return_value = [-1] if failed_acquire else [0]
                    release.return_value = -1
                    executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
                    with self.assertRaisesRegex(RuntimeError, "release failed"):
                        executor.restore([PoolReadRange("key", 16, 0, 2000, 4)])
                    self.assertFalse(executor._readers)
                    self.assertEqual(executor._pending_release, {"key"})
                    with self.assertRaisesRegex(RuntimeError, "release failed"):
                        executor.close()
                    self.assertEqual(executor._pending_release, {"key"})
                    release.return_value = 0
                    executor.close()
                    self.assertFalse(executor._pending_release)
                    self.assertEqual(release.call_count, 3)

    def test_thread_start_failure_keeps_failed_rollback_for_close(self):
        backend = self.backend()
        backend.batch_remove_lease.side_effect = [-1, 0]
        executor = PoolBackendExecutor(backend, lease_ttl_ms=10000)
        with (
            patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start")),
            self.assertRaisesRegex(RuntimeError, "release failed"),
        ):
            executor.restore([PoolReadRange("key", 16, 0, 2000, 4)])
        self.assertFalse(executor._readers)
        self.assertEqual(executor._pending_release, {"key"})
        backend.batch_copy_gva.assert_not_called()
        executor.close()
        self.assertFalse(executor._pending_release)

    def test_other_cleanup_failure_and_early_close_do_not_stop_active_renewal(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=90)
        renewed = threading.Event()

        def release(keys):
            return -1 if keys == ["failed"] else 0

        def acquire(keys, ttl):
            if keys == ["active"] and executor._pending_release:
                renewed.set()
            return [0] * len(keys)

        backend.batch_add_lease.side_effect = acquire
        backend.batch_remove_lease.side_effect = release
        try:
            with executor._reading(["active"]):
                with self.assertRaisesRegex(RuntimeError, "release failed"), executor._reading(["failed"]):
                    pass
                with self.assertRaisesRegex(RuntimeError, "must drain"):
                    executor.close()
                self.assertFalse(executor._closed)
                self.assertTrue(renewed.wait(timeout=5))
                self.assertEqual(executor._readers, {"active": 1})
                self.assertEqual(executor._pending_release, {"failed"})
        finally:
            backend.batch_remove_lease.side_effect = None
            backend.batch_remove_lease.return_value = 0
            executor.close()
        self.assertFalse(executor._pending_release)

    def test_conflicting_object_sizes_rejected_before_acquire(self):
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            PoolBackendExecutor(backend, lease_ttl_ms=10000).restore(
                [
                    PoolReadRange("key", 16, 0, 2000, 4),
                    PoolReadRange("key", 32, 0, 3000, 4),
                ]
            )
        backend.batch_add_lease.assert_not_called()

    def test_prefill_retention_and_restore_share_protection_and_idle_renewal(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=90)
        renewed = threading.Event()
        operations = []

        def acquire(keys, ttl):
            operations.append(object())
            if len(operations) > 1:
                renewed.set()
            return [0]

        def release(keys):
            operations.pop(0)
            return 0

        backend.batch_add_lease.side_effect = acquire
        backend.batch_remove_lease.side_effect = release
        try:
            executor.acquire_read_protection(["key"])
            executor.restore([PoolReadRange("key", 16, 0, 2000, 4)])
            self.assertEqual(executor._readers, {"key": 1})
            self.assertTrue(renewed.wait(timeout=5))
            executor.release_read_protection(["key"])
            self.assertFalse(operations)
            self.assertFalse(executor._readers)
        finally:
            executor.close()

    def test_slow_copy_renews_lease_until_native_io_drains(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=90)
        copying, renewed, release = threading.Event(), threading.Event(), threading.Event()
        native_operations = []

        def add_lease(keys, ttl):
            native_operations.append(object())
            self.assertLessEqual(len(native_operations), 2)
            if copying.is_set():
                renewed.set()
            return [0]

        def remove_lease(keys):
            self.assertTrue(native_operations)
            native_operations.pop(0)
            if copying.is_set() and not release.is_set():
                self.assertTrue(native_operations, "Renewal must not leave an unprotected live copy")
            return 0

        def copy(*_):
            copying.set()
            self.assertTrue(release.wait(timeout=5))
            return 0

        backend.batch_add_lease.side_effect = add_lease
        backend.batch_remove_lease.side_effect = remove_lease
        backend.batch_copy_gva.side_effect = copy
        with ThreadPoolExecutor(max_workers=1) as queue:
            future = queue.submit(executor.restore, [PoolReadRange("key", 16, 0, 2000, 16)])
            try:
                self.assertTrue(renewed.wait(timeout=5))
                with executor._lock:
                    self.assertEqual(executor._lease_counts, {"key": 1})
                    self.assertEqual(backend.batch_remove_lease.call_count, backend.batch_add_lease.call_count - 1)
            finally:
                release.set()
            future.result(timeout=5)
        executor.close()
        self.assertEqual(backend.batch_remove_lease.call_count, backend.batch_add_lease.call_count)
        self.assertFalse(executor._lease_counts)
        self.assertFalse(native_operations)
        count = backend.batch_add_lease.call_count
        self.assertGreaterEqual(count, 2)
        self.assertIsNone(executor._renew_thread)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executor.restore([PoolReadRange("other", 16, 0, 2000, 16)])
        self.assertEqual(backend.batch_add_lease.call_count, count)

    def test_failed_renewal_fails_restore_after_copy_drains_and_quarantines(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=90, max_bytes=4)
        copying, renewal_failed, release = threading.Event(), threading.Event(), threading.Event()

        def add_lease(keys, ttl):
            if copying.is_set():
                renewal_failed.set()
                return [-1]
            return [0]

        def copy(*_):
            copying.set()
            self.assertTrue(release.wait(timeout=5))
            return 0

        backend.batch_add_lease.side_effect = add_lease
        backend.batch_copy_gva.side_effect = copy
        with ThreadPoolExecutor(max_workers=1) as queue:
            future = queue.submit(executor.restore, [PoolReadRange("key", 16, 0, 2000, 16)])
            try:
                self.assertTrue(renewal_failed.wait(timeout=5))
                # Let the renewal thread record its failure under its lock.
                thread = executor._renew_thread
                if thread is not None:
                    thread.join(timeout=5)
                self.assertFalse(future.done())
                backend.batch_remove_lease.assert_not_called()
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "renewal failed"):
                future.result(timeout=5)
        self.assertEqual(backend.batch_remove_lease.call_count, 2)
        # The first 4-byte batch must drain, but the remaining three batches
        # cannot be submitted after the renewal thread reports failure.
        self.assertEqual(backend.batch_copy_gva.call_count, 1)
        self.assertFalse(executor._lease_counts)
        with self.assertRaisesRegex(RuntimeError, "unresolved lease renewal"):
            executor.restore([PoolReadRange("other", 16, 0, 2000, 16)])
        executor.close()

    def test_failed_renewal_during_metadata_lookup_prevents_first_copy(self):
        backend = self.backend()
        executor = PoolBackendExecutor(backend, lease_ttl_ms=90)
        looking_up, renewal_failed, release = threading.Event(), threading.Event(), threading.Event()
        infos = backend.batch_get_key_info.return_value

        def add_lease(keys, ttl):
            if looking_up.is_set():
                renewal_failed.set()
                return [-1]
            return [0]

        def get_info(*args, **kwargs):
            looking_up.set()
            self.assertTrue(release.wait(timeout=5))
            return infos

        backend.batch_add_lease.side_effect = add_lease
        backend.batch_get_key_info.side_effect = get_info
        with ThreadPoolExecutor(max_workers=1) as queue:
            future = queue.submit(executor.restore, [PoolReadRange("key", 16, 0, 2000, 16)])
            try:
                self.assertTrue(renewal_failed.wait(timeout=5))
                thread = executor._renew_thread
                if thread is not None:
                    thread.join(timeout=5)
                self.assertFalse(future.done())
                backend.batch_remove_lease.assert_not_called()
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "renewal failed"):
                future.result(timeout=5)
        backend.batch_copy_gva.assert_not_called()
        self.assertEqual(backend.batch_remove_lease.call_count, 2)
        self.assertFalse(executor._lease_counts)
        executor.close()
