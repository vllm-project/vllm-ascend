import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cloudpickle  # type: ignore
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache import (
    KVCacheClient,
    KVCacheMethod,
    ServiceSessionExpiredError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import WorkerKVCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import (
    decode_scheduler_session,
    decode_worker_session,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerBusyError,
)

KV_CACHE_CLIENT_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.client"


def _make_vllm_config():
    from types import SimpleNamespace

    hf_config = SimpleNamespace(num_hidden_layers=2, model_type="llama")
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            max_model_len=1024,
            hf_text_config=hf_config,
            hf_config=hf_config,
            use_mla=False,
            get_num_layers=lambda _parallel_config: 2,
            get_total_num_kv_heads=lambda: 1,
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            world_size=1,
            data_parallel_rank=0,
            data_parallel_index=0,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="engine-0",
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={},
        ),
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def _make_request():
    from types import SimpleNamespace

    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(16)),
        block_hashes=[bytes.fromhex("01" * 32)],
        num_tokens=16,
    )


def test_recovery_reuses_the_same_session_after_initial_registration_failure() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [MPRequestTimeoutError("timeout"), [b"OK"]]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert not client.register_scheduler(_make_vllm_config(), None, 0)
            client._maintain_lease()
            assert client.is_registered

            register_calls = [
                request_call
                for request_call in rpc_client.request.call_args_list
                if request_call.args[0] == KVCacheMethod.REGISTER_SCHEDULER
            ]
            first_registration = cloudpickle.loads(register_calls[0].args[1][-1])
            second_registration = cloudpickle.loads(register_calls[1].args[1][-1])

            assert first_registration.session_id == second_registration.session_id
            assert first_registration.identity == second_registration.identity
            rpc_client.ping.assert_not_called()
        finally:
            client.close()


def test_stale_session_during_recovery_becomes_terminal_for_kv_cache_client() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [
            [b"OK"],
            MPRequestTimeoutError("renew timeout"),
            MPRemoteError("StaleSessionError: old session"),
        ]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_scheduler(_make_vllm_config(), None, 0)
            client._maintain_lease()
            assert not client.is_registered
            client._maintain_lease()

            with pytest.raises(ServiceSessionExpiredError, match="superseded"):
                client.lookup(_make_request(), 0)
        finally:
            client.close()


def test_lookup_maps_remote_stale_session_to_terminal_client_state() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [[b"OK"], MPRemoteError("StaleSessionError: old session")]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_scheduler(_make_vllm_config(), None, 0)

            with pytest.raises(ServiceSessionExpiredError, match="StaleSessionError"):
                client.lookup(_make_request(), 0)

            assert not client.is_registered
            with pytest.raises(ServiceSessionExpiredError, match="superseded"):
                client.lookup(_make_request(), 0)
        finally:
            client.close()


def test_close_stops_lease_loop_then_unregisters_the_same_session() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.return_value = [b"OK"]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        assert client.register_scheduler(_make_vllm_config(), None, 0)

        register_call = rpc_client.request.call_args_list[0]
        registration = cloudpickle.loads(register_call.args[1][-1])
        lease_stopped = False

        def stop_lease_loop() -> None:
            nonlocal lease_stopped
            lease_stopped = True

        def unregister(method, payloads, timeout_ms):
            assert method == KVCacheMethod.UNREGISTER_SCHEDULER
            assert lease_stopped
            return [b"OK"]

        rpc_client.request.side_effect = unregister
        with patch.object(client, "_stop_lease_loop", side_effect=stop_lease_loop) as stop:
            client.close()

        stop.assert_called_once_with()
        unregister_call = rpc_client.request.call_args_list[-1]
        _, session_id = decode_scheduler_session(unregister_call.args[1])
        assert session_id == registration.session_id
        rpc_client.close.assert_called_once_with()


def test_concurrent_registration_attempts_share_one_request() -> None:
    registration_started = threading.Event()
    finish_registration = threading.Event()

    def register(method, payloads, timeout_ms):
        assert method == KVCacheMethod.REGISTER_SCHEDULER
        registration_started.set()
        assert finish_registration.wait(1)
        return [b"OK"]

    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = register
        client = KVCacheClient("tcp://127.0.0.1:12345")

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(client.register_scheduler, _make_vllm_config(), None, 0)
                assert registration_started.wait(1)
                second = executor.submit(client._try_register)
                finish_registration.set()

                assert first.result()
                assert second.result()

            assert rpc_client.request.call_count == 1
        finally:
            finish_registration.set()
            client.close()


def test_close_waits_for_registration_before_unregistering() -> None:
    registration_started = threading.Event()
    finish_registration = threading.Event()
    close_started = threading.Event()

    def request(method, payloads, timeout_ms):
        if method == KVCacheMethod.REGISTER_SCHEDULER:
            registration_started.set()
            assert finish_registration.wait(1)
        return [b"OK"]

    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = request
        client = KVCacheClient("tcp://127.0.0.1:12345")

        with (
            patch.object(client, "_stop_lease_loop", side_effect=close_started.set),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            register_future = executor.submit(client.register_scheduler, _make_vllm_config(), None, 0)
            assert registration_started.wait(1)
            close_future = executor.submit(client.close)
            try:
                assert close_started.wait(1)
                rpc_client.close.assert_not_called()
            finally:
                finish_registration.set()

            assert not register_future.result()
            close_future.result()

        methods = [call.args[0] for call in rpc_client.request.call_args_list]
        assert methods == [KVCacheMethod.REGISTER_SCHEDULER, KVCacheMethod.UNREGISTER_SCHEDULER]
        rpc_client.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("service_type", "renew_method"),
    [("scheduler", KVCacheMethod.RENEW_SCHEDULER), ("worker", KVCacheMethod.RENEW_WORKER)],
)
def test_service_renewal_uses_the_registered_session(service_type: str, renew_method: KVCacheMethod) -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [[b"OK"], [b"OK"]]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            if service_type == "scheduler":
                assert client.register_scheduler(_make_vllm_config(), None, 0)
            else:
                assert client.register_worker(_make_vllm_config(), None)
            registration = cloudpickle.loads(rpc_client.request.call_args_list[0].args[1][-1])
            client._maintain_lease()

            renew_call = rpc_client.request.call_args_list[1]
            assert renew_call.args[0] == renew_method
            decode_session = decode_scheduler_session if service_type == "scheduler" else decode_worker_session
            _, session_id = decode_session(renew_call.args[1])
            assert session_id == registration.session_id
        finally:
            client.close()


def test_worker_cache_spec_is_registered_again_after_service_recovers() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [
            [b"OK"],
            MPRequestTimeoutError("cache registration timeout"),
            [b"OK"],
            [b"OK"],
        ]
        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_worker(_make_vllm_config(), None)
            spec = WorkerKVCacheSpec(caches={"layer.0": ()}, storages=())
            assert not client.register_kv_caches(spec)
            assert not client.is_registered

            client._maintain_lease()

            assert client.is_registered
            methods = [call.args[0] for call in rpc_client.request.call_args_list]
            assert methods == [
                KVCacheMethod.REGISTER_WORKER,
                KVCacheMethod.REGISTER_KV_CACHES,
                KVCacheMethod.REGISTER_WORKER,
                KVCacheMethod.REGISTER_KV_CACHES,
            ]
        finally:
            client.close()


def test_missing_service_during_renewal_reregisters_the_same_session() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [
            [b"OK"],
            MPRemoteError("ServiceNotRegisteredError: service lease expired"),
            [b"OK"],
        ]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_scheduler(_make_vllm_config(), None, 0)
            client._maintain_lease()

            register_calls = [
                request_call
                for request_call in rpc_client.request.call_args_list
                if request_call.args[0] == KVCacheMethod.REGISTER_SCHEDULER
            ]
            assert len(register_calls) == 2
            first_registration = cloudpickle.loads(register_calls[0].args[1][-1])
            second_registration = cloudpickle.loads(register_calls[1].args[1][-1])
            assert first_registration.session_id == second_registration.session_id
            assert client.is_registered
        finally:
            client.close()


def test_unregistered_client_retries_registration_from_lease_loop() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [MPServerBusyError("busy"), [b"OK"]]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert not client.register_scheduler(_make_vllm_config(), None, 0)
            client._maintain_lease()
            assert client.is_registered
        finally:
            client.close()


def test_stale_service_renewal_supersedes_client() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [[b"OK"], MPRemoteError("StaleSessionError: old session")]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_scheduler(_make_vllm_config(), None, 0)
            client._maintain_lease()

            assert not client.is_registered
            with pytest.raises(ServiceSessionExpiredError, match="superseded"):
                client.lookup(_make_request(), 0)
        finally:
            client.close()
