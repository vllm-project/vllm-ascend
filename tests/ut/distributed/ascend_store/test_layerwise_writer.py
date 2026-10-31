# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import unittest
from unittest.mock import MagicMock, call

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import BatchResultShapeError
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import (
    LayerwiseAllocationPlan,
    LayerwiseWriteCompletion,
    LayerwiseWriter,
    LayerwiseWriteSession,
    LayerWriteRange,
)


class TestLayerwiseAllocationPlan(unittest.TestCase):
    def test_deduplicates_objects_and_preserves_original_slot_results(self):
        allocate = MagicMock(side_effect=[[100, (1 << 64) - 1], [0]])
        keys = ["a", "b", "a", "c"]
        sizes = [64, 128, 64, 256]
        plan = LayerwiseAllocationPlan(keys, sizes, max_keys=2)
        # Queued work must retain its own key/size plan.
        keys[0], sizes[0] = "changed", 1
        self.assertEqual(plan.allocate_gvas(allocate, 500), [100, (1 << 64) - 1, 100, 0])
        self.assertEqual(allocate.call_args_list, [call(["a", "b"], [64, 128], 500), call(["c"], [256], 500)])

    def test_range_sessions_preserve_partial_failure_and_aliases(self):
        start = MagicMock(side_effect=[[0], [-7]])
        plan = LayerwiseAllocationPlan(["a", "b", "a"], [64, 128, 64], max_keys=1)
        self.assertEqual(plan.start_ranges(start), [0, -7, 0])
        self.assertEqual(start.call_args_list, [call(["a"], [64]), call(["b"], [128])])

    def test_malformed_later_batch_stops_before_allocating_more_objects(self):
        for malformed in (None, [], [True], [1.5], [0, 1]):
            with self.subTest(malformed=malformed):
                start = MagicMock(side_effect=[[0], malformed, [0]])
                plan = LayerwiseAllocationPlan(["a", "b", "c"], [1, 2, 3], max_keys=1)
                with self.assertRaises(BatchResultShapeError):
                    plan.start_ranges(start)
                self.assertEqual(start.call_count, 2)

    def test_invalid_plans_fail_before_native_calls(self):
        for keys, sizes in (
            (["a"], []),
            ([""], [1]),
            (["a"], [True]),
            (["a"], [0]),
            (["a"], [1.5]),
            (["a", "a"], [1, 2]),
        ):
            with self.subTest(keys=keys, sizes=sizes), self.assertRaises(ValueError):
                LayerwiseAllocationPlan(keys, sizes)
        with self.assertRaises(ValueError):
            LayerwiseAllocationPlan(["a"], [1], max_keys=True)
        allocate = MagicMock()
        with self.assertRaises(ValueError):
            LayerwiseAllocationPlan(["a"], [1]).allocate_gvas(allocate, -1)
        allocate.assert_not_called()

    def test_empty_plan_does_not_open_native_sessions(self):
        start = MagicMock()
        plan = LayerwiseAllocationPlan([], [])
        self.assertEqual(plan.allocate_gvas(start, 0), [])
        self.assertEqual(plan.start_ranges(start), [])
        start.assert_not_called()


class TestLayerwiseWriteSession(unittest.TestCase):
    def test_event_and_native_copy_finish_before_layer_can_publish(self):
        completion = LayerwiseWriteCompletion({"layer": {"key"}})
        session = LayerwiseWriteSession(completion)
        copy_entered, allow_copy_return = threading.Event(), threading.Event()
        returned = threading.Event()
        event = MagicMock()

        def copy(*args):
            event.synchronize.assert_called_once()
            copy_entered.set()
            if not allow_copy_return.wait(5):
                raise TimeoutError("Test did not release native copy")
            return 0

        def execute():
            session.write_gva("layer", {"key"}, event, copy, [100], [200], [4])
            returned.set()

        worker = threading.Thread(target=execute)
        worker.start()
        try:
            self.assertTrue(copy_entered.wait(5))
            self.assertFalse(returned.is_set())
            finish = MagicMock(return_value=[0])
            with self.assertRaisesRegex(ValueError, "Required object layers"):
                session.publish(["key"], finish)
            finish.assert_not_called()
        finally:
            allow_copy_return.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertTrue(returned.is_set())
        self.assertEqual(session.publish(["key"], finish), [0])

    def test_wrong_step_rejected_before_event_or_native_copy(self):
        session = LayerwiseWriteSession(LayerwiseWriteCompletion({"layer": {"key"}}))
        event, copy = MagicMock(), MagicMock(return_value=0)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            session.write_gva("other_layer", {"key"}, event, copy, [100], [200], [4])
        event.synchronize.assert_not_called()
        copy.assert_not_called()

    def test_shared_key_partial_failure_never_publishes_failed_key(self):
        session = LayerwiseWriteSession(LayerwiseWriteCompletion({"layer": {"good", "bad"}}))
        copy = MagicMock(side_effect=[[2], [-1], [2], [2]])
        results = session.write_ranges(
            "layer",
            ["good", "bad"],
            MagicMock(),
            copy,
            [[100], [200]],
            [[4], [4]],
            [[0], [0]],
            max_batch_bytes=2,
            max_packet_bytes=2,
            stop_on_error=False,
        )
        # The second range of good fails; later successful batches cannot
        # overwrite that failure, even though bad completes all its bytes.
        self.assertEqual(results, [-1, 4])
        finish = MagicMock(return_value=[0])
        self.assertEqual(session.publish(["bad"], finish), [0])
        with self.assertRaisesRegex(ValueError, "Required object layers"):
            session.publish(["good"], finish)
        finish.assert_called_once_with(["bad"])

    def test_cancelled_during_compute_wait_never_starts_copy(self):
        for plane in ("gva", "key"):
            with self.subTest(plane=plane):
                session = LayerwiseWriteSession(LayerwiseWriteCompletion({"layer": {"key"}}))
                cancelled = threading.Event()
                event = MagicMock()
                event.synchronize.side_effect = cancelled.set
                copy = MagicMock()
                if plane == "gva":
                    result = session.write_gva(
                        "layer", {"key"}, event, copy, [100], [200], [4], cancelled=cancelled.is_set
                    )
                else:
                    result = session.write_ranges(
                        "layer", ["key"], event, copy, [[200]], [[4]], [[0]], cancelled=cancelled.is_set
                    )
                self.assertIsNone(result)
                copy.assert_not_called()
                with self.assertRaisesRegex(ValueError, "Required object layers"):
                    session.publish(["key"], MagicMock())

    def test_cancel_during_native_copy_drains_current_batch_and_skips_later_batches(self):
        for plane in ("gva", "key"):
            with self.subTest(plane=plane):
                self._check_cancel_between_native_batches(plane)

    def _check_cancel_between_native_batches(self, plane):
        session = LayerwiseWriteSession(LayerwiseWriteCompletion({"layer": {"key"}}))
        cancelled, entered, release, returned = (threading.Event() for _ in range(4))
        results, errors = [], []

        def native_copy(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the first native batch")
            return 0 if plane == "gva" else [2]

        copy = MagicMock(side_effect=native_copy)

        def execute():
            try:
                if plane == "gva":
                    result = session.write_gva(
                        "layer",
                        {"key"},
                        MagicMock(),
                        copy,
                        [100],
                        [200],
                        [8],
                        max_batch_bytes=2,
                        cancelled=cancelled.is_set,
                    )
                else:
                    result = session.write_ranges(
                        "layer",
                        ["key"],
                        MagicMock(),
                        copy,
                        [[200]],
                        [[8]],
                        [[0]],
                        max_batch_bytes=2,
                        cancelled=cancelled.is_set,
                    )
                results.append(result)
            except BaseException as exc:
                errors.append(exc)
            finally:
                returned.set()

        worker = threading.Thread(target=execute)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            cancelled.set()
            self.assertFalse(returned.is_set())
            self.assertEqual(copy.call_count, 1)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [None])
        self.assertEqual(copy.call_count, 1)
        finish = MagicMock(return_value=[0])
        with self.assertRaisesRegex(ValueError, "Required object layers"):
            session.publish(["key"], finish)
        finish.assert_not_called()


class TestLayerwiseWriter(unittest.TestCase):
    def test_partial_key_completion_does_not_mark_failed_siblings_complete(self):
        completion = LayerwiseWriteCompletion({"main": {"a", "b"}, "draft": {"a", "b"}})
        completion.record_keys("main", {"a", "b"})
        completion.record_keys("draft", {"a"})
        completion.require_complete({"a"})
        with self.assertRaisesRegex(ValueError, "Required object layers"):
            completion.require_complete({"b"})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            completion.record_keys("draft", {"a", "b"})
        completion.record_keys("draft", {"b"})
        completion.require_complete({"a", "b"})

    def test_completion_checks_each_objects_own_required_layers(self):
        completion = LayerwiseWriteCompletion({"main": {"combined"}, "draft": {"combined", "draft_only"}})
        completion.record("draft", {"combined", "draft_only"})
        completion.require_complete({"draft_only"})
        with self.assertRaisesRegex(ValueError, "Required object layers"):
            completion.require_complete({"combined"})
        with self.assertRaisesRegex(ValueError, "outside"):
            completion.require_complete({"another_step"})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            completion.record("main", {"wrong_key"})
        completion.record("main", {"combined"})
        completion.require_complete({"combined", "draft_only"})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            completion.record("main", {"combined"})

    def backend(self, plane="gva"):
        backend = MagicMock()
        backend.layerwise_data_plane = plane
        backend.batch_alloc.return_value = [10000]
        backend.batch_write_finish.return_value = [0]
        backend.batch_copy_gva.return_value = 0
        backend.batch_put_start.return_value = [0]
        backend.batch_copy_put.return_value = [0]
        backend.batch_commit.return_value = [0]
        backend.batch_revoke.return_value = [0]
        return backend

    def plan(self, writer):
        return writer.prepare(
            {"object": 8},
            {
                "main": (LayerWriteRange("object", 100, 0, 4),),
                "draft": (LayerWriteRange("object", 200, 4, 4),),
            },
        )

    def shared_plan(self, writer):
        return writer.prepare(
            {"prefix": 8},
            {"main": (LayerWriteRange("prefix", 100, 0, 4),), "draft": (LayerWriteRange("prefix", 200, 4, 4),)},
            shared_objects=frozenset({"prefix"}),
        )

    def test_committed_shared_object_is_protected_without_allocating_or_writing(self):
        backend = self.backend()
        backend.batch_is_readable.return_value = [True]
        backend.batch_add_lease.return_value = [0]
        backend.batch_remove_lease.return_value = 0
        info = MagicMock()
        info.size.return_value = 8
        backend.batch_get_key_info.return_value = [info]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            handle = self.shared_plan(writer)
            writer.submit_layer(handle, "main", MagicMock())
            writer.submit_layer(handle, "draft", MagicMock())
            writer.finish(handle).result(timeout=5)
            backend.batch_alloc.assert_not_called()
            backend.batch_copy_gva.assert_not_called()
            backend.batch_write_finish.assert_not_called()
            backend.batch_add_lease.assert_called_once_with(["prefix"], 10000)
            backend.batch_remove_lease.assert_called_once_with(["prefix"])
        finally:
            writer.close()

    def test_cancel_shared_writer_never_revokes_another_producers_prefix(self):
        backend = self.backend()
        backend.batch_is_readable.return_value = [False]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            handle = self.shared_plan(writer)
            handle.copy_futures[0].result(timeout=5)
            # A different producer can commit the idempotently allocated key
            # before this attempt is cancelled. Neither abort primitive is safe.
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                writer.cancel(handle).result(timeout=5)
            backend.batch_revoke.assert_not_called()
            backend.batch_write_finish.assert_not_called()
            self.assertFalse(writer._handles)
        finally:
            writer.close()

    def test_shared_borrow_rejects_wrong_size_and_releases_its_protection(self):
        backend = self.backend()
        backend.batch_is_readable.return_value = [True]
        backend.batch_add_lease.return_value = [0]
        backend.batch_remove_lease.return_value = 0
        info = MagicMock()
        info.size.return_value = 9
        backend.batch_get_key_info.return_value = [info]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            handle = self.shared_plan(writer)
            writer.submit_layer(handle, "main", MagicMock())
            writer.submit_layer(handle, "draft", MagicMock())
            with self.assertRaisesRegex(RuntimeError, "attempt failed"):
                writer.finish(handle).result(timeout=5)
            backend.batch_remove_lease.assert_called_once_with(["prefix"])
            backend.batch_alloc.assert_not_called()
            backend.batch_copy_gva.assert_not_called()
            backend.batch_revoke.assert_not_called()
        finally:
            writer.close()

    def test_same_shared_key_has_independent_local_attempts(self):
        backend = self.backend()
        backend.batch_is_readable.return_value = [False]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            first, second = self.shared_plan(writer), self.shared_plan(writer)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                writer.cancel(first).result(timeout=5)
            writer.submit_layer(second, "main", MagicMock())
            writer.submit_layer(second, "draft", MagicMock())
            writer.finish(second).result(timeout=5)
            backend.batch_write_finish.assert_called_once_with(["prefix"], [0])
            backend.batch_revoke.assert_not_called()
        finally:
            writer.close()

    def test_waits_for_every_required_layer_before_commit(self):
        for plane in ("gva", "block_key"):
            with self.subTest(plane=plane):
                backend = self.backend(plane)
                writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
                try:
                    handle = self.plan(writer)
                    event = MagicMock()
                    writer.submit_layer(handle, "main", event)
                    with self.assertRaisesRegex(ValueError, "Required layers"):
                        writer.finish(handle)
                    backend.batch_commit.assert_not_called()
                    backend.batch_write_finish.assert_not_called()
                    writer.submit_layer(handle, "draft", event)
                    writer.finish(handle).result(timeout=5)
                    self.assertEqual(event.synchronize.call_count, 2)
                    if plane == "gva":
                        self.assertEqual(
                            backend.batch_copy_gva.call_args_list,
                            [
                                call([10000], [100], [4], 0),
                                call([10004], [200], [4], 0),
                            ],
                        )
                        backend.batch_write_finish.assert_called_once_with(["object"], [0])
                    else:
                        backend.batch_commit.assert_called_once_with(["object"])
                finally:
                    writer.close()

    def test_failed_copy_aborts_without_publishing_success(self):
        backend = self.backend()
        backend.batch_copy_gva.return_value = -1
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            handle = self.plan(writer)
            writer.submit_layer(handle, "main", MagicMock())
            writer.submit_layer(handle, "draft", MagicMock())
            with self.assertRaisesRegex(RuntimeError, "attempt failed"):
                writer.finish(handle).result(timeout=5)
            backend.batch_write_finish.assert_called_once_with(["object"], [-1])
        finally:
            writer.close()

    def test_invalid_or_unknown_gva_allocation_is_revoked_without_copy(self):
        for response in ([0], [-1], [(1 << 64) - 1], None, [], [True], RuntimeError("RPC reply lost")):
            with self.subTest(response=response):
                backend = self.backend()
                if isinstance(response, Exception):
                    backend.batch_alloc.side_effect = response
                else:
                    backend.batch_alloc.return_value = response
                writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
                try:
                    handle = self.plan(writer)
                    writer.submit_layer(handle, "main", MagicMock())
                    writer.submit_layer(handle, "draft", MagicMock())
                    with self.assertRaisesRegex(RuntimeError, "attempt failed"):
                        writer.finish(handle).result(timeout=5)
                    backend.batch_revoke.assert_called_once_with(["object"])
                    backend.batch_copy_gva.assert_not_called()
                    backend.batch_write_finish.assert_not_called()
                    self.assertFalse(writer._active)
                finally:
                    writer.close()

    def test_partial_gva_allocation_cleans_untracked_and_registered_objects(self):
        backend = self.backend()
        backend.batch_alloc.return_value = [10000, 0]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            handle = writer.prepare(
                {"registered": 4, "untracked": 4},
                {"main": (LayerWriteRange("registered", 100, 0, 4), LayerWriteRange("untracked", 200, 0, 4))},
            )
            writer.submit_layer(handle, "main", MagicMock())
            with self.assertRaisesRegex(RuntimeError, "attempt failed"):
                writer.finish(handle).result(timeout=5)
            backend.batch_revoke.assert_called_once_with(["untracked"])
            backend.batch_write_finish.assert_called_once_with(["registered"], [-1])
            backend.batch_copy_gva.assert_not_called()
        finally:
            writer.close()

    def test_untracked_allocation_cleanup_retains_only_failed_keys_for_retry(self):
        backend = self.backend()
        backend.batch_alloc.return_value = [0, 0]
        backend.batch_revoke.side_effect = [[0, -7], [0]]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        handle = writer.prepare(
            {"a": 4, "b": 4}, {"main": (LayerWriteRange("a", 100, 0, 4), LayerWriteRange("b", 200, 0, 4))}
        )
        writer.submit_layer(handle, "main", MagicMock())
        with self.assertRaisesRegex(RuntimeError, "allocation cleanup failed"):
            writer.finish(handle).result(timeout=5)
        self.assertEqual(handle.untracked_allocations, ["b"])
        self.assertIs(writer._active["b"], handle)
        writer.close()
        self.assertEqual(backend.batch_revoke.call_args_list, [call(["a", "b"]), call(["b"])])
        self.assertFalse(writer._active)

    def test_positive_range_counts_commit_but_positive_session_status_does_not(self):
        for commit_status in (0, 1):
            with self.subTest(commit_status=commit_status):
                backend = self.backend("block_key")
                backend.batch_copy_put.side_effect = lambda keys, buffers, sizes, offsets: [sum(row) for row in sizes]
                backend.batch_commit.return_value = [commit_status]
                writer = LayerwiseWriter(backend, lease_ttl_ms=10000, max_bytes=2)
                try:
                    handle = self.plan(writer)
                    writer.submit_layer(handle, "main", MagicMock())
                    writer.submit_layer(handle, "draft", MagicMock())
                    result = writer.finish(handle)
                    if commit_status == 0:
                        result.result(timeout=5)
                        backend.batch_revoke.assert_not_called()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Pool write commit failed"):
                            result.result(timeout=5)
                        backend.batch_revoke.assert_called_once_with(["object"])
                    self.assertEqual(backend.batch_copy_put.call_count, 4)
                    backend.batch_commit.assert_called_once_with(["object"])
                finally:
                    writer.close()

    def test_expired_write_never_commits_or_submits_later_batches(self):
        for expiration in ("allocation", "compute", "copy", "commit_queue"):
            with self.subTest(expiration=expiration):
                now = [0.0]
                backend = self.backend()
                writer = LayerwiseWriter(backend, lease_ttl_ms=1000, max_bytes=2, clock=lambda now=now: now[0])

                def expire(now=now):
                    now[0] = 1.0

                def allocate(*args, expiration=expiration, expire=expire):
                    if expiration == "allocation":
                        expire()
                    return [10000]

                def copy(*args, expiration=expiration, expire=expire):
                    if expiration == "copy":
                        expire()
                    return 0

                backend.batch_alloc.side_effect = allocate
                backend.batch_copy_gva.side_effect = copy
                try:
                    handle = writer.prepare({"object": 8}, {"main": (LayerWriteRange("object", 100, 0, 8),)})
                    event = MagicMock()
                    if expiration == "compute":
                        event.synchronize.side_effect = expire
                    writer.submit_layer(handle, "main", event)
                    if expiration == "commit_queue":
                        handle.copy_futures[-1].result(timeout=5)
                        expire()
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        writer.finish(handle).result(timeout=5)
                    backend.batch_write_finish.assert_called_once_with(["object"], [-1])
                    self.assertEqual(
                        backend.batch_copy_gva.call_count,
                        {"allocation": 0, "compute": 0, "copy": 1, "commit_queue": 4}[expiration],
                    )
                    backend.batch_add_lease.assert_not_called()
                    self.assertFalse(writer._active)
                finally:
                    writer.close()

    def test_key_backend_uses_its_own_session_lifetime(self):
        now = [0.0]
        backend = self.backend("block_key")
        writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda: now[0])
        try:
            handle = self.plan(writer)
            handle.copy_futures[0].result(timeout=5)
            now[0] = 100.0
            writer.submit_layer(handle, "main", MagicMock())
            writer.submit_layer(handle, "draft", MagicMock())
            writer.finish(handle).result(timeout=5)
            backend.batch_commit.assert_called_once_with(["object"])
        finally:
            writer.close()

    def test_layer_batches_multiple_objects_before_commit(self):
        for plane in ("gva", "block_key"):
            with self.subTest(plane=plane):
                backend = self.backend(plane)
                backend.batch_alloc.return_value = [10000, 20000]
                backend.batch_write_finish.return_value = [0, 0]
                backend.batch_put_start.return_value = [0, 0]
                backend.batch_copy_put.return_value = [0, 0]
                backend.batch_commit.return_value = [0, 0]
                writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
                try:
                    handle = writer.prepare(
                        {"first": 4, "second": 4},
                        {"main": (LayerWriteRange("first", 100, 0, 4), LayerWriteRange("second", 200, 0, 4))},
                    )
                    writer.submit_layer(handle, "main", MagicMock())
                    writer.finish(handle).result(timeout=5)
                    if plane == "gva":
                        backend.batch_copy_gva.assert_called_once_with([10000, 20000], [100, 200], [4, 4], 0)
                        backend.batch_write_finish.assert_called_once_with(["first", "second"], [0, 0])
                    else:
                        backend.batch_copy_put.assert_called_once_with(
                            ["first", "second"], [[100], [200]], [[4], [4]], [[0], [0]]
                        )
                        backend.batch_commit.assert_called_once_with(["first", "second"])
                finally:
                    writer.close()

    def test_cancel_drains_before_releasing_source_ownership(self):
        backend = self.backend("block_key")
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        try:
            handle = self.plan(writer)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                writer.cancel(handle).result(timeout=5)
            backend.batch_commit.assert_not_called()
            backend.batch_revoke.assert_called_once_with(["object"])
            self.assertFalse(writer._active)
        finally:
            writer.close()

    def test_expired_write_tracker_uses_private_revoke_without_publishing(self):
        now = [0.0]
        backend = self.backend()
        backend.batch_write_finish.return_value = [-3000]
        writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda: now[0])
        handle = self.plan(writer)
        handle.copy_futures[0].result(timeout=5)
        event = MagicMock()
        event.synchronize.side_effect = lambda: now.__setitem__(0, 2.0)
        try:
            writer.submit_layer(handle, "main", event)
            writer.submit_layer(handle, "draft", MagicMock())
            with self.assertRaisesRegex(RuntimeError, "write attempt failed") as failure:
                writer.finish(handle).result(timeout=5)
            self.assertIsInstance(failure.exception.__cause__, TimeoutError)
            backend.batch_copy_gva.assert_not_called()
            backend.batch_write_finish.assert_called_once_with(["object"], [-1])
            backend.batch_revoke.assert_called_once_with(["object"])
            self.assertFalse(writer._active)
        finally:
            writer.close()

    def test_expired_private_revoke_failure_retains_ownership_for_retry(self):
        now = [0.0]
        backend = self.backend()
        backend.batch_write_finish.return_value = [-3000]
        backend.batch_revoke.return_value = [-7]
        writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda: now[0])
        handle = self.plan(writer)
        handle.copy_futures[0].result(timeout=5)
        now[0] = 2.0
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            writer.cancel(handle).result(timeout=5)
        self.assertIs(writer._active["object"], handle)
        with self.assertRaisesRegex(RuntimeError, "unresolved cleanup"):
            self.plan(writer)
        backend.batch_revoke.return_value = [0]
        writer.close()
        self.assertEqual(backend.batch_revoke.call_count, 2)
        self.assertFalse(writer._active)

    def test_live_write_cleanup_failure_does_not_delete_objects_by_key(self):
        backend = self.backend()
        backend.batch_write_finish.side_effect = [[-3000], [0]]
        writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda: 0.0)
        handle = self.plan(writer)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            writer.cancel(handle).result(timeout=5)
        writer.close()
        backend.batch_revoke.assert_not_called()

    def test_expired_revoke_only_retries_failed_keys_and_checks_alignment(self):
        for revoke_result in ([0], [], [True], [0, 0]):
            with self.subTest(revoke_result=revoke_result):
                backend = self.backend()
                backend.batch_alloc.return_value = [10000, 20000]
                backend.batch_write_finish.side_effect = lambda keys, statuses: [
                    0 if key == "a" else -3000 for key in keys
                ]
                backend.batch_revoke.return_value = revoke_result
                now = [0.0]
                writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda now=now: now[0])
                handle = writer.prepare(
                    {"a": 4, "b": 4},
                    {"layer": (LayerWriteRange("a", 100, 0, 4), LayerWriteRange("b", 200, 0, 4))},
                )
                handle.copy_futures[0].result(timeout=5)
                now[0] = 2.0
                try:
                    if revoke_result == [0] and type(revoke_result[0]) is int:
                        with self.assertRaisesRegex(RuntimeError, "cancelled"):
                            writer.cancel(handle).result(timeout=5)
                        self.assertFalse(writer._active)
                    else:
                        with self.assertRaises(BatchResultShapeError):
                            writer.cancel(handle).result(timeout=5)
                        self.assertIs(writer._active["b"], handle)
                        self.assertEqual(handle.started, ["b"])
                    backend.batch_revoke.assert_called_once_with(["b"])
                finally:
                    backend.batch_revoke.return_value = [0]
                    writer.close()
                if revoke_result != [0] or type(revoke_result[0]) is not int:
                    self.assertEqual(
                        backend.batch_write_finish.call_args_list,
                        [call(["a", "b"], [-1, -1]), call(["b"], [-1])],
                    )

    def test_partial_abort_retries_only_failed_sessions(self):
        for plane in ("gva", "block_key"):
            with self.subTest(plane=plane):
                backend = self.backend(plane)
                backend.batch_alloc.return_value = [10000, 20000]
                backend.batch_put_start.return_value = [0, 0]
                cleanup = backend.batch_write_finish if plane == "gva" else backend.batch_revoke
                remaining = {"a", "b"}

                def revoke(keys, *args, remaining=remaining, cleanup=cleanup):
                    # Simulate a backend whose successful cleanup removes its
                    # session: retrying an already removed key stays an error.
                    result = []
                    for key in keys:
                        if key not in remaining or (key == "b" and cleanup.call_count == 1):
                            result.append(-7)
                        else:
                            remaining.remove(key)
                            result.append(0)
                    return result

                cleanup.side_effect = revoke
                writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda: 0.0)
                handle = writer.prepare(
                    {"a": 4, "b": 4},
                    {"layer": (LayerWriteRange("a", 100, 0, 4), LayerWriteRange("b", 200, 0, 4))},
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                        writer.cancel(handle).result(timeout=5)
                    self.assertEqual(handle.started, ["b"])
                    self.assertEqual(remaining, {"b"})
                finally:
                    writer.close()
                self.assertEqual([args.args[0] for args in cleanup.call_args_list], [["a", "b"], ["b"]])
                self.assertFalse(remaining)
                self.assertFalse(writer._active)

    def test_ambiguous_abort_retains_all_sessions_for_retry(self):
        for plane in ("gva", "block_key"):
            for reply in (None, [], [0], [0, True], RuntimeError("lost reply")):
                with self.subTest(plane=plane, reply=reply):
                    backend = self.backend(plane)
                    backend.batch_alloc.return_value = [10000, 20000]
                    backend.batch_put_start.return_value = [0, 0]
                    cleanup = backend.batch_write_finish if plane == "gva" else backend.batch_revoke
                    cleanup.side_effect = [reply, [0, 0]]
                    writer = LayerwiseWriter(backend, lease_ttl_ms=1000, clock=lambda: 0.0)
                    handle = writer.prepare(
                        {"a": 4, "b": 4},
                        {"layer": (LayerWriteRange("a", 100, 0, 4), LayerWriteRange("b", 200, 0, 4))},
                    )
                    try:
                        with self.assertRaises((BatchResultShapeError, RuntimeError)):
                            writer.cancel(handle).result(timeout=5)
                        self.assertEqual(handle.started, ["a", "b"])
                    finally:
                        writer.close()
                    self.assertFalse(writer._active)

    def test_failed_rollback_retains_keys_until_close_retries(self):
        for plane in ("gva", "block_key"):
            with self.subTest(plane=plane):
                backend = self.backend(plane)
                cleanup = backend.batch_write_finish if plane == "gva" else backend.batch_revoke
                cleanup.side_effect = [[-1], [0]]
                writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
                handle = self.plan(writer)
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    writer.cancel(handle).result(timeout=5)
                self.assertIs(writer._active["object"], handle)
                with self.assertRaisesRegex(RuntimeError, "unresolved cleanup"):
                    self.plan(writer)
                writer.close()
                self.assertEqual(cleanup.call_count, 2)
                self.assertFalse(writer._active)

    def test_close_keeps_failed_cleanup_ownership_and_retries_later(self):
        backend = self.backend("block_key")
        backend.batch_revoke.return_value = [-1]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        handle = self.plan(writer)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            writer.cancel(handle).result(timeout=5)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            writer.close()
        self.assertIs(writer._active["object"], handle)
        backend.batch_revoke.return_value = [0]
        writer.close()
        self.assertFalse(writer._active)

    def test_close_waits_for_native_copy_before_retrying_rollback(self):
        backend = self.backend()
        entered, release = threading.Event(), threading.Event()

        def copy(*args):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release the native copy")
            return 0

        backend.batch_copy_gva.side_effect = copy
        backend.batch_write_finish.side_effect = [[-1], [0]]
        writer = LayerwiseWriter(backend, lease_ttl_ms=10000)
        handle = self.plan(writer)
        writer.submit_layer(handle, "main", MagicMock())
        errors = []

        def close():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)

        closer = threading.Thread(target=close)
        try:
            self.assertTrue(entered.wait(timeout=5))
            closer.start()
            self.assertTrue(handle.cancelled.wait(timeout=5))
            backend.batch_write_finish.assert_not_called()
            self.assertTrue(closer.is_alive())
        finally:
            release.set()
            if closer.ident is not None:
                closer.join(timeout=5)
            else:
                writer.close()
        self.assertFalse(closer.is_alive())
        self.assertFalse(errors)
        self.assertEqual(backend.batch_write_finish.call_count, 2)
        self.assertFalse(writer._active)

    def test_rejects_incomplete_or_overlapping_object_plan(self):
        writer = LayerwiseWriter(self.backend(), lease_ttl_ms=10000)
        try:
            for offset in (3, 5):
                with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "coverage"):
                    writer.prepare(
                        {"object": 8},
                        {
                            "main": (
                                LayerWriteRange("object", 100, 0, 4),
                                LayerWriteRange("object", 200, offset, 4),
                            )
                        },
                    )
        finally:
            writer.close()

    def test_duplicate_layer_is_not_written_twice(self):
        writer = LayerwiseWriter(self.backend(), lease_ttl_ms=10000)
        try:
            handle = self.plan(writer)
            writer.submit_layer(handle, "main", MagicMock())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                writer.submit_layer(handle, "main", MagicMock())
            writer.submit_layer(handle, "draft", MagicMock())
            writer.finish(handle).result(timeout=5)
        finally:
            writer.close()
