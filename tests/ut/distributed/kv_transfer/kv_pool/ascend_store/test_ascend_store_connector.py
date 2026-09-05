# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import types
import unittest
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_events import KVCacheEvent
from vllm.v1.serial_utils import MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
    AscendStoreKVEvents,
)


@pytest.fixture
def lookup_server(monkeypatch):
    context, socket, thread = MagicMock(), MagicMock(), MagicMock()
    thread.is_alive.return_value = False
    constructor = MagicMock(return_value=thread)
    monkeypatch.setattr(connector_module.zmq, "Context", MagicMock(return_value=context))
    monkeypatch.setattr(connector_module, "make_zmq_socket", MagicMock(return_value=socket))
    monkeypatch.setattr(connector_module.threading, "Thread", constructor)
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        kv_transfer_config=types.SimpleNamespace(kv_connector_extra_config={}),
    )
    worker = MagicMock(lookup_scheduler=MagicMock(return_value=12))
    server = connector_module.LookupKeyServer(worker, config)
    return server, worker, socket, context, thread, constructor


def test_lookup_server_decodes_frames_and_encodes_hit_length(lookup_server):
    server, worker, socket, context, thread, constructor = lookup_server
    encoder = MsgpackEncoder()
    socket.recv_multipart.return_value = [
        (16).to_bytes(4, "big"),
        encoder.encode([0, 1])[0],
        (8).to_bytes(4, "big"),
        *encoder.encode([b"a", b"b"]),
    ]
    socket.send.side_effect = lambda response: setattr(server, "running", False)
    constructor.call_args.kwargs["target"]()
    worker.lookup_scheduler.assert_called_once_with(16, [b"a", b"b"], [0, 1], use_layerwise=False, hbm_hit_tokens=8)
    socket.send.assert_called_once_with((12).to_bytes(4, "big"))
    thread.start.assert_called_once_with()
    server.close()
    socket.close.assert_called_once_with(linger=0)


def test_lookup_server_close_stops_thread_and_releases_context(lookup_server):
    server, _, socket, context, thread, _ = lookup_server
    thread.is_alive.return_value = False
    server.close()
    assert server.running is False
    thread.join.assert_called_once()
    socket.close.assert_called_once_with(linger=0)
    context.term.assert_called_once_with()


def test_lookup_server_receive_timeout_observes_shutdown(lookup_server):
    server, worker, socket, context, thread, constructor = lookup_server

    def timeout():
        server.running = False
        raise connector_module.zmq.Again()

    socket.recv_multipart.side_effect = lambda **kwargs: timeout()
    constructor.call_args.kwargs["target"]()
    worker.lookup_scheduler.assert_not_called()
    socket.setsockopt.assert_called_once_with(connector_module.zmq.RCVTIMEO, 100)
    server.close()
    context.term.assert_called_once_with()


def test_lookup_server_close_does_not_destroy_socket_used_by_pending_lookup(lookup_server):
    server, _, socket, context, thread, _ = lookup_server
    thread.is_alive.return_value = True
    with pytest.raises(TimeoutError, match="lookup thread did not stop"):
        server.close()
    thread.join.assert_called_once_with(timeout=1)
    socket.close.assert_not_called()
    context.term.assert_not_called()
    thread.is_alive.return_value = False
    server.close()
    socket.close.assert_called_once_with(linger=0)
    context.term.assert_called_once_with()


def test_connector_scheduler_and_worker_metadata_delegation():
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    connector.connector_scheduler = MagicMock()
    connector.connector_worker = MagicMock()
    request, pool = object(), object()
    assert (
        connector.request_finished_all_groups(request, ([1], [2]))
        is connector.connector_scheduler.request_finished_all_groups.return_value
    )
    connector.connector_scheduler.request_finished_all_groups.assert_called_once_with(request, ([1], [2]))
    connector.bind_gpu_block_pool(pool)
    connector.connector_scheduler.bind_gpu_block_pool.assert_called_once_with(pool)
    assert (
        connector.build_connector_worker_meta() is connector.connector_worker.build_connector_worker_meta.return_value
    )


def test_connector_mamba_copy_failure_clears_buffers():
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    buffers = object()
    connector._mamba_copy_bufs = buffers
    with (
        patch.object(
            connector_module.mamba_utils, "finish_mamba_copy_by_layer", side_effect=RuntimeError("copy failed")
        ),
        pytest.raises(RuntimeError, match="copy failed"),
    ):
        connector.finish_mamba_state_copy()
    assert connector._mamba_copy_bufs is None
    connector.finish_mamba_state_copy()


def test_worker_connector_ignores_empty_output_and_repeated_finish_without_forward():
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    connector.connector_scheduler = None
    connector._kv_cache_events = None
    connector.update_connector_output(types.SimpleNamespace(kv_cache_events=None))
    assert connector._kv_cache_events is None
    connector.connector_worker = MagicMock(get_finished=MagicMock(return_value=({"a"}, {"b"})))
    connector._connector_metadata = object()
    connector._current_step_has_real_forward = False
    assert connector.get_finished({"a"}) == ({"a"}, {"b"})
    connector.connector_worker.ensure_store_initialized.assert_not_called()


def test_legacy_connector_name_keeps_scheduler_configuration(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    config = TestAscendStoreConnector()._make_vllm_config()
    config.kv_transfer_config.kv_connector = "MooncakeConnectorStoreV1"
    scheduler = MagicMock()
    monkeypatch.setattr(connector_module, "KVPoolScheduler", scheduler)
    connector = AscendStoreConnector(config, KVConnectorRole.SCHEDULER, types.SimpleNamespace())
    assert connector.connector_scheduler is scheduler.return_value
    assert scheduler.call_args.args[0] is config


def test_connector_store_initialization_failure_resets_forward_flag():
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    connector.connector_worker = MagicMock()
    connector._connector_metadata = object()
    connector._current_step_has_real_forward = True
    connector.connector_worker.ensure_store_initialized.side_effect = RuntimeError("init failed")
    with pytest.raises(RuntimeError, match="init failed"):
        connector.get_finished(set())
    assert connector._current_step_has_real_forward is False
    connector.connector_worker.get_finished.assert_not_called()


def _mock_events(num_workers=1):
    events = AscendStoreKVEvents(num_workers=num_workers)
    events._aggregator = MagicMock()
    return events


class TestAscendStoreKVEvents(unittest.TestCase):
    def test_event_lifecycle(self):
        ev = _mock_events()
        mock_events = [MagicMock(spec=KVCacheEvent), MagicMock(spec=KVCacheEvent)]
        ev.add_events(mock_events)
        ev._aggregator.get_all_events.return_value = mock_events
        self.assertEqual(ev.get_all_events(), mock_events)
        self.assertIn("AscendStoreKVEvents", repr(ev))

        ev.clear_events()
        ev._aggregator.clear_events.assert_called_once()
        ev._aggregator.reset_workers.assert_called_once()

    def test_worker_aggregation(self):
        ev = _mock_events()
        ev.increment_workers(3)
        ev._aggregator.increment_workers.assert_called_once_with(3)
        ev._aggregator.get_number_of_workers.return_value = 5
        self.assertEqual(ev.get_number_of_workers(), 5)

        common = [MagicMock()]
        ev._aggregator.get_common_events.return_value = common
        self.assertIs(ev.aggregate(), ev)
        ev._aggregator.clear_events.assert_called_once()
        ev._aggregator.add_events.assert_called_once_with(common)
        ev._aggregator.reset_workers.assert_called_once()


class TestAscendStoreConnector(unittest.TestCase):
    def _make_vllm_config(self, kv_role="kv_producer", extra_config=None):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector = "AscendStoreConnector"
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        config.parallel_config.rank = 0
        return config

    def test_pp_handshake_metadata_is_ignored(self):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        metadata = {
            (0, 0): MagicMock(),
            (1, 0): MagicMock(),
        }
        original_metadata = metadata.copy()

        result = connector.set_xfer_handshake_metadata_pp_aware(metadata)

        self.assertIsNone(result)
        self.assertEqual(metadata, original_metadata)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_init_scheduler_role(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        _connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        mock_scheduler_cls.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_init_worker_role(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        _connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        mock_worker_cls.assert_called_once()
        mock_lookup_cls.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_scheduler_methods_delegate(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        mock_sched = mock_scheduler_cls.return_value

        # get_num_new_matched_tokens
        mock_sched.get_num_new_matched_tokens.return_value = (10, False)
        result = connector.get_num_new_matched_tokens(MagicMock(), 5)
        self.assertEqual(result, (10, False))

        # update_state_after_alloc
        connector.update_state_after_alloc(MagicMock(), MagicMock(), 10)
        mock_sched.update_state_after_alloc.assert_called_once()

        # build_connector_meta
        connector.build_connector_meta(MagicMock())
        mock_sched.build_connector_meta.assert_called_once()

        # request_finished
        mock_sched.request_finished.return_value = (True, None)
        result = connector.request_finished(MagicMock(), [1, 2])
        self.assertEqual(result, (True, None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_update_connector_output_accumulates_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        output = MagicMock()
        output.kv_cache_events = None
        connector.update_connector_output(output)
        self.assertIsNone(connector._kv_cache_events)

        for _ in range(2):
            events = _mock_events()
            events._aggregator.get_all_events.return_value = [MagicMock()]
            events._aggregator.get_number_of_workers.return_value = 1
            output.kv_cache_events = events
            connector.update_connector_output(output)
        self.assertIsNotNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_take_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        # No events
        result = list(connector.take_events())
        self.assertEqual(result, [])

        # With events
        events = _mock_events(num_workers=1)
        mock_event = MagicMock()
        events._aggregator.get_common_events.return_value = [mock_event]
        events._aggregator.get_all_events.return_value = [mock_event]
        connector._kv_cache_events = events
        result = list(connector.take_events())
        self.assertEqual(len(result), 1)
        self.assertIsNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_worker_methods(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        mock_worker = mock_worker_cls.return_value

        # register_kv_caches
        connector.register_kv_caches({"layer1": MagicMock()})
        mock_worker.register_kv_caches.assert_called_once()

        # start_load_kv
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())
        connector.start_load_kv(MagicMock())
        mock_worker.start_load_kv.assert_called_once()

        # wait_for_save (non-consumer)
        connector.kv_role = "kv_producer"
        connector.use_layerwise = False
        connector.wait_for_save()
        mock_worker.wait_for_save.assert_called_once()

        # get_finished
        mock_worker.get_finished.return_value = ({"r1"}, {"r2"})
        done_s, done_r = connector.get_finished({"r1"})
        self.assertEqual(done_s, {"r1"})

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_layerwise_methods_return_early(self, mock_worker_cls, mock_lookup_cls):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        cases = [
            ("wait_for_layer_load", "kv_both", False),
            ("save_kv_layer", "kv_both", False),
            ("save_kv_layer", "kv_consumer", True),
            ("wait_for_save", "kv_consumer", False),
        ]
        for method_name, kv_role, use_layerwise in cases:
            with self.subTest(method=method_name, kv_role=kv_role, use_layerwise=use_layerwise):
                worker = mock_worker_cls.return_value
                worker.reset_mock()
                config = self._make_vllm_config(
                    kv_role=kv_role,
                    extra_config={"use_layerwise": use_layerwise},
                )
                connector = AscendStoreConnector(
                    vllm_config=config,
                    role=KVConnectorRole.WORKER,
                    kv_cache_config=None,
                )
                if method_name == "wait_for_layer_load":
                    connector.wait_for_layer_load("layer_0")
                elif method_name == "save_kv_layer":
                    connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
                else:
                    connector.wait_for_save()
                getattr(worker, method_name).assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_set_external_slot_release_waiter_worker_gates(self, mock_worker_cls, mock_lookup_cls):
        """Regression guard for the #14465 / #15291 connector flag.

        The connector must stay a pure forwarder: it no longer derives
        the layerwise gate itself (#14465 dropped the copy that this method
        read, crashing MultiConnector init; #15291 restored it). The gate
        now lives in KVPoolWorker.set_external_slot_release_waiter, so
        this test also pins that the connector keeps no flag of its own.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        config = self._make_vllm_config(
            extra_config={"use_layerwise": True, "backend": "mooncake"},
        )
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        worker = mock_worker_cls.return_value

        # Non-GVA backend: the worker gate rejects, the connector relays False.
        worker.set_external_slot_release_waiter.return_value = False
        self.assertFalse(connector.set_external_slot_release_waiter(lambda _l: None))
        worker.set_external_slot_release_waiter.assert_called_once()

        # GVA backend: the worker gate accepts, the connector relays True and
        # passes the waiter through unchanged.
        waiter = MagicMock()
        worker.set_external_slot_release_waiter.reset_mock()
        worker.set_external_slot_release_waiter.return_value = True
        self.assertTrue(connector.set_external_slot_release_waiter(waiter))
        worker.set_external_slot_release_waiter.assert_called_once_with(waiter)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_set_external_slot_release_waiter_scheduler_role(self, mock_scheduler_cls):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        config = self._make_vllm_config()
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        self.assertFalse(connector.set_external_slot_release_waiter(lambda _l: None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_save_kv_layer_not_layerwise(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(extra_config={"use_layerwise": False})
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
        # Should return immediately

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_save_kv_layer_consumer(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(kv_role="kv_consumer", extra_config={"use_layerwise": True})
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
        # Consumer should not save

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_save_kv_layer_consumer_with_put_enabled(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(
            kv_role="kv_consumer",
            extra_config={
                "use_layerwise": True,
                "consumer_is_to_put": True,
            },
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())

        connector.save_kv_layer("layer_0", MagicMock(), MagicMock())

        mock_worker_cls.return_value.save_kv_layer.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_wait_for_save_consumer(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(kv_role="kv_consumer")
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector.wait_for_save()
        mock_worker_cls.return_value.wait_for_save.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_get_kv_connector_kv_cache_events_empty(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        for events, expected_type in (([], type(None)), ([MagicMock()], AscendStoreKVEvents)):
            mock_worker_cls.return_value.get_kv_events.return_value = events
            self.assertIsInstance(connector.get_kv_connector_kv_cache_events(), expected_type)


class TestAscendStoreConnectorLayerwise(unittest.TestCase):
    """Test connector methods that are specific to layerwise mode."""

    connector_mod: types.ModuleType

    @classmethod
    def setUpClass(cls):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector

        cls.connector_mod = ascend_store_connector

    def test_requires_piecewise_for_cudagraph(self):
        cases = [
            ({"use_layerwise": True}, True),
            ({"use_layerwise": False}, False),
            ({}, False),
        ]
        for config, expected in cases:
            with self.subTest(config=config):
                self.assertEqual(
                    self.connector_mod.AscendStoreConnector.requires_piecewise_for_cudagraph(config),
                    expected,
                )

    def test_layerwise_worker_paths(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        with (
            patch.object(self.connector_mod, "KVPoolWorker") as mock_worker_cls,
            patch.object(self.connector_mod, "LookupKeyServer") as _mock_lookup_cls,
        ):
            config = MagicMock()
            config.kv_transfer_config.kv_role = "kv_producer"
            config.kv_transfer_config.kv_connector = "AscendStoreConnector"
            config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": True}
            config.parallel_config.rank = 0

            connector = self.connector_mod.AscendStoreConnector(
                vllm_config=config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            connector.wait_for_save()
            mock_worker_cls.return_value.wait_for_save.assert_not_called()
            connector._get_connector_metadata = MagicMock(return_value=MagicMock())
            connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
            mock_worker_cls.return_value.save_kv_layer.assert_called_once()

            config.kv_transfer_config.kv_role = "kv_consumer"
            connector = self.connector_mod.AscendStoreConnector(
                vllm_config=config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            connector.wait_for_layer_load("layer_0")
            mock_worker_cls.return_value.wait_for_layer_load.assert_called_once()

    def test_mamba_state_copy_runs_after_layer_load(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        call_order = []
        with (
            patch.object(self.connector_mod, "KVPoolWorker") as mock_worker_cls,
            patch.object(self.connector_mod, "LookupKeyServer"),
            patch.object(
                self.connector_mod.mamba_utils,
                "do_mamba_copy_block_for_layer",
                side_effect=lambda *_: call_order.append("copy"),
                create=True,
            ),
            patch.object(
                self.connector_mod.mamba_utils,
                "prepare_mamba_copy_by_layer",
                create=True,
            ) as prepare_copy,
            patch.object(
                self.connector_mod.mamba_utils,
                "finish_mamba_copy_by_layer",
                create=True,
            ) as finish_copy,
        ):
            config = MagicMock()
            config.kv_transfer_config.kv_role = "kv_consumer"
            config.kv_transfer_config.kv_connector = "AscendStoreConnector"
            config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": True}
            config.parallel_config.rank = 0
            mock_worker_cls.return_value.wait_for_layer_load.side_effect = lambda: call_order.append("load")

            connector = self.connector_mod.AscendStoreConnector(
                vllm_config=config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            copy_bufs = MagicMock()
            self.assertTrue(connector.prepare_mamba_state_copy(copy_bufs))

            connector.wait_for_layer_load("layers.0.linear_attn")
            connector.finish_mamba_state_copy()

            self.assertEqual(call_order, ["load", "copy"])
            prepare_copy.assert_called_once_with(copy_bufs)
            finish_copy.assert_called_once_with(copy_bufs)
            self.assertIsNone(connector._mamba_copy_bufs)

    def test_non_layerwise_connector_keeps_batched_mamba_copy(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        with (
            patch.object(self.connector_mod, "KVPoolWorker"),
            patch.object(self.connector_mod, "LookupKeyServer"),
        ):
            config = MagicMock()
            config.kv_transfer_config.kv_role = "kv_consumer"
            config.kv_transfer_config.kv_connector = "AscendStoreConnector"
            config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": False}
            config.parallel_config.rank = 0
            connector = self.connector_mod.AscendStoreConnector(
                vllm_config=config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )

            self.assertFalse(connector.prepare_mamba_state_copy(MagicMock()))


class TestAscendStoreConnectorContract(unittest.TestCase):
    """Regression tests for connector-level load failure reporting."""

    def test_get_block_ids_with_load_errors_forwards_to_worker(self):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = MagicMock()
        connector.connector_worker.get_block_ids_with_load_errors.return_value = {3, 7}

        result = connector.get_block_ids_with_load_errors()

        self.assertEqual(result, {3, 7})
        connector.connector_worker.get_block_ids_with_load_errors.assert_called_once_with()
