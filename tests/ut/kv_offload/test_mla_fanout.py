# SPDX-License-Identifier: Apache-2.0
import json
import multiprocessing
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mla_fanout import MLAFanout, reader_blocks, transferred_blocks


def request(local=None):
    return {
        "request_id": "decode-request",
        "remote_request_id": "prefill-request",
        "offset": 0,
        "transfer_attempt": "attempt-1",
        "remote_block_ids": [[10, 11, 12], [20]],
        "local_block_ids": [[2, 3, 4], [7]] if local is None else local,
    }


def run_process(config, rank, memory, copied, done):
    fanout = MLAFanout(config, rank, 2)
    meta = request()

    def pull():
        if rank != 0:
            raise AssertionError("reader made a redundant cross-node read")
        memory[0] = 42

    def read(record):
        if reader_blocks(meta, record) != meta["local_block_ids"]:
            raise AssertionError("wrong leader block table")
        copied.value = memory[0]

    fanout.transfer(meta, pull, read, {})
    done.set()


class TestMLAFanout(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = {
            "shared_dir": self.temporary.name,
            "session_id": "unique-engine-startup",
            "ready_timeout": 1,
            "ack_timeout": 1,
        }

    def test_block_mapping_handles_different_allocations_and_prefix_hits(self):
        meta = request([[100, 101], []])
        publication = {"source_blocks": [[10, 11, 12], [20]], "local_blocks": [[2, 3, 4], [7]]}
        self.assertEqual(transferred_blocks(meta), [[11, 12], []])
        self.assertEqual(reader_blocks(meta, publication), [[3, 4], []])

    def test_missing_leader_coverage_is_rejected(self):
        with self.assertRaises(KeyError):
            reader_blocks(request(), {"source_blocks": [[11, 12], [20]], "local_blocks": [[3, 4], [7]]})

    def test_inconsistent_block_table_is_rejected(self):
        with self.assertRaises(ValueError):
            reader_blocks(request(), {"source_blocks": [[10, 11, 12], [20]], "local_blocks": [[2], [7]]})

    def test_invalid_configuration(self):
        for bad in (
            {"leader_rank": -1},
            {"leader_rank": 2},
            {"ready_timeout": 0},
            {"ack_timeout": float("nan")},
            {"ack_timeout": float("inf")},
            {"shared_dir": None},
            {"session_id": ""},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MLAFanout(dict(self.config, **bad), 0, 2)

    def test_eight_ranks_pull_from_prefill_only_once(self):
        leader_pull = MagicMock()
        reader_pull = MagicMock()
        read = MagicMock()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for rank in range(8):
                fanout = MLAFanout(self.config, rank, 8)
                meta = dict(request(), remote_handshake_port=31000 + rank)
                futures.append(pool.submit(fanout.transfer, meta, leader_pull if rank == 0 else reader_pull, read, {}))
            for future in futures:
                future.result(timeout=3)
        leader_pull.assert_called_once()
        reader_pull.assert_not_called()
        self.assertEqual(read.call_count, 7)
        self.assertFalse(list(Path(self.temporary.name).rglob("*.json")))
        self.assertFalse(list(Path(self.temporary.name).rglob("*.ack-*")))

    def test_reader_error_or_missing_metadata_acknowledges_then_falls_back(self):
        for error in (
            RuntimeError("TE failed"),
            KeyError("rpc_port"),
            ValueError("layout mismatch"),
            AssertionError("reader failed"),
        ):
            with self.subTest(error=error), ThreadPoolExecutor(max_workers=2) as pool:
                leader = MLAFanout(self.config, 0, 2)
                reader = MLAFanout(self.config, 1, 2)
                fallback = MagicMock()
                future = pool.submit(leader.transfer, request(), MagicMock(), MagicMock(), {})
                reader.transfer(request(), fallback, MagicMock(side_effect=error), {})
                future.result(timeout=2)
                fallback.assert_called_once()
                self.assertFalse(list(leader.directory.glob("*.ack-*")))

    def test_ready_timeout_sends_early_ack_and_falls_back(self):
        config = dict(self.config, ready_timeout=0.02)
        reader = MLAFanout(config, 1, 2)
        fallback = MagicMock()
        reader.transfer(request(), fallback, MagicMock(), {})
        fallback.assert_called_once()
        leader = MLAFanout(config, 0, 2)
        started = time.monotonic()
        leader.transfer(request(), MagicMock(), MagicMock(), {})
        self.assertLess(time.monotonic() - started, 0.5)

    def test_expired_publication_is_not_read(self):
        reader = MLAFanout(self.config, 1, 2)
        key = reader._key(request())
        (reader.directory / f"{key}.json").write_text(
            json.dumps({"deadline": time.monotonic() - 1, "transfer_attempt": "attempt-1"})
        )
        read, pull = MagicMock(), MagicMock()
        reader.transfer(request(), pull, read, {})
        read.assert_not_called()
        pull.assert_called_once()
        self.assertEqual(reader.stats()["fallback_expired_publication"], 1)

    def test_publication_attempt_is_validated_before_read(self):
        reader = MLAFanout(self.config, 1, 2)
        key = reader._key(request())
        (reader.directory / f"{key}.json").write_text(
            json.dumps({"deadline": time.monotonic() + 10, "transfer_attempt": "wrong-attempt"})
        )
        read, pull = MagicMock(), MagicMock()
        reader.transfer(request(), pull, read, {})
        read.assert_not_called()
        pull.assert_called_once()
        self.assertEqual(reader.stats()["fallback_local_read_error"], 1)
        self.assertEqual(reader.stats()["fallback"], 1)

    def test_sessions_do_not_share_records(self):
        first = MLAFanout(self.config, 0, 2)
        second = MLAFanout(dict(self.config, session_id="next-startup"), 1, 2)
        self.assertNotEqual(first.directory, second.directory)

    def test_missing_attempt_falls_back_without_publishing(self):
        for rank in (0, 1):
            fanout = MLAFanout(self.config, rank, 2)
            meta = request()
            del meta["transfer_attempt"]
            pull, read = MagicMock(), MagicMock()
            fanout.transfer(meta, pull, read, {})
            pull.assert_called_once()
            read.assert_not_called()
            self.assertEqual(fanout.stats()["fallback_missing_attempt"], 1)
            self.assertFalse(list(fanout.directory.glob("*.json")))

    def test_overlapping_attempts_do_not_read_ack_or_cleanup_each_other(self):
        leader = MLAFanout(self.config, 0, 2)
        reader = MLAFanout(self.config, 1, 2)
        old, new = request(), dict(request(), transfer_attempt="attempt-2")
        entered, release = threading.Event(), threading.Event()
        published_new = threading.Event()
        self.assertNotEqual(leader._key(old), leader._key(new))
        original_publish = leader._publish

        def publish(path, record):
            original_publish(path, record)
            if record["transfer_attempt"] == "attempt-2":
                published_new.set()

        leader._publish = publish

        def read_old(record):
            self.assertEqual(record["transfer_attempt"], "attempt-1")
            entered.set()
            self.assertTrue(release.wait(timeout=2))

        with ThreadPoolExecutor(max_workers=3) as pool:
            old_leader = pool.submit(leader.transfer, old, MagicMock(), MagicMock(), {})
            old_reader = pool.submit(reader.transfer, old, MagicMock(), read_old, {})
            try:
                self.assertTrue(entered.wait(timeout=1))
                new_leader = pool.submit(leader.transfer, new, MagicMock(), MagicMock(), {})
                self.assertTrue(published_new.wait(timeout=1))
                self.assertFalse(new_leader.done(), "old attempt ACK must not complete the new attempt")
                self.assertFalse(old_leader.done())
                self.assertTrue((leader.directory / f"{leader._key(old)}.json").exists())
            finally:
                release.set()
            old_reader.result(timeout=2)
            old_leader.result(timeout=2)
            self.assertTrue((leader.directory / f"{leader._key(new)}.json").exists())
            read, fallback = MagicMock(), MagicMock()
            reader.transfer(new, fallback, read, {})
            read.assert_called_once()
            self.assertEqual(read.call_args.args[0]["transfer_attempt"], "attempt-2")
            fallback.assert_not_called()
            new_leader.result(timeout=2)

    def test_old_ready_and_ack_are_ignored_by_retry(self):
        config = dict(self.config, ready_timeout=0.02, ack_timeout=0.02)
        leader = MLAFanout(config, 0, 2)
        old, new = request(), dict(request(), transfer_attempt="attempt-2")
        key = leader._key(old)
        ready = leader.directory / f"{key}.json"
        ack = leader.directory / f"{key}.ack-1"
        ready.write_text(json.dumps({"deadline": time.monotonic() + 10, "transfer_attempt": "attempt-1"}))
        ack.touch()
        pull, read = MagicMock(), MagicMock()
        reader = MLAFanout(config, 1, 2)
        reader.transfer(new, pull, read, {})
        read.assert_not_called()
        pull.assert_called_once()
        self.assertEqual(reader.stats()["fallback_ready_timeout"], 1)
        leader.transfer(dict(new, transfer_attempt="attempt-3"), MagicMock(), MagicMock(), {})
        self.assertEqual(leader.stats()["ack_timeout"], 1)
        self.assertTrue(ready.exists())
        self.assertTrue(ack.exists())

    def test_pin_wait_is_observable_without_releasing_kv(self):
        leader = MLAFanout(self.config, 0, 2)
        reader = MLAFanout(self.config, 1, 2)
        key = leader._key(request())

        def take_exclusive_lock():
            with leader._lock(key, exclusive=True):
                pass

        with (
            patch.object(sys.modules[MLAFanout.__module__], "PIN_WARNING_INTERVAL", 0.01),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            with reader._lock(key):
                future = pool.submit(take_exclusive_lock)
                time.sleep(0.05)
                self.assertFalse(future.done())
                self.assertGreater(leader.stats().get("pin_wait_warnings", 0), 0)
            future.result(timeout=1)

    def test_leader_transfer_failure_never_publishes_ready(self):
        leader = MLAFanout(self.config, 0, 2)
        with self.assertRaises(RuntimeError):
            leader.transfer(request(), MagicMock(side_effect=RuntimeError("P failed")), MagicMock(), {})
        self.assertFalse(list(leader.directory.glob("*.json")))

    def test_ack_deadline_does_not_release_an_active_reader(self):
        config = dict(self.config, ack_timeout=0.1)
        leader = MLAFanout(config, 0, 2)
        reader = MLAFanout(config, 1, 2)
        started, release = threading.Event(), threading.Event()

        def read(record):
            started.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test reader was never released")

        with ThreadPoolExecutor(max_workers=2) as pool:
            leader_future = pool.submit(leader.transfer, request(), MagicMock(), MagicMock(), {})
            reader_future = pool.submit(reader.transfer, request(), MagicMock(), read, {})
            try:
                self.assertTrue(started.wait(timeout=1))
                time.sleep(0.15)
                self.assertFalse(leader_future.done(), "leader released KV underneath the reader")
            finally:
                release.set()
            reader_future.result(timeout=2)
            leader_future.result(timeout=2)

    def test_missing_reader_is_bounded_and_late_reader_falls_back(self):
        config = dict(self.config, ack_timeout=0.02, ready_timeout=0.02)
        leader = MLAFanout(config, 0, 2)
        leader.transfer(request(), MagicMock(), MagicMock(), {})
        read, pull = MagicMock(), MagicMock()
        MLAFanout(config, 1, 2).transfer(request(), pull, read, {})
        read.assert_not_called()
        pull.assert_called_once()

    def test_flock_coordinates_separate_processes(self):
        ctx = multiprocessing.get_context("fork")
        memory = ctx.Array("i", [0])
        copied = ctx.Value("i", 0)
        events = [ctx.Event(), ctx.Event()]
        processes = [
            ctx.Process(target=run_process, args=(self.config, rank, memory, copied, events[rank])) for rank in (1, 0)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=3)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(event.is_set() for event in events))
            self.assertEqual(copied.value, 42)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)

    def test_broken_local_rendezvous_still_falls_back_to_prefill(self):
        reader = MLAFanout(self.config, 1, 2)
        read, pull = MagicMock(), MagicMock()
        with patch.object(reader, "_lock", side_effect=OSError("filesystem unavailable")):
            reader.transfer(request(), pull, read, {})
        read.assert_not_called()
        pull.assert_called_once()

    def test_invalid_ready_metadata_is_acknowledged_before_fallback(self):
        reader = MLAFanout(self.config, 1, 2)
        key = reader._key(request())
        (reader.directory / f"{key}.json").write_text("invalid-json")
        read, pull = MagicMock(), MagicMock()
        reader.transfer(request(), pull, read, {})
        read.assert_not_called()
        pull.assert_called_once()
        self.assertTrue((reader.directory / f"{key}.ack-1").exists())
