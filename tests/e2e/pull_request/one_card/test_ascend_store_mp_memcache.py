"""Real Memcache E2E for the AscendStore MP backend."""

import contextlib
import multiprocessing
import os
import socket
import time
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.pull_request.one_card.test_ascend_store_mp_ipc import (
    _allocate_mooncake_test_caches,
    _ModelConfig,
    _receive,
    _run_kv_cache_server,
    _stop_process,
    _wait_for_active_export,
    _wait_until_connected,
    _wait_until_registered,
)
from tests.e2e.pull_request.one_card.test_ascend_store_mp_model_smoke import (
    _HIT_LOG_PATTERN,
    _PROMPT,
    _build_llm,
    _generate_once,
    _model_path,
    _run_smoke_server,
    _wait_for_prefix_cache_reset,
)

_MEMCACHE_START_TIMEOUT_S = 30.0
_MEMCACHE_STOP_TIMEOUT_S = 10.0
_MEMCACHE_LAYERWISE_LEASE_TTL_MS = 30_000
_MEMCACHE_LAYERWISE_WORLD_SIZE = 256


def _require_memcache() -> None:
    try:
        import memcache_hybrid  # type: ignore[import-not-found, import-untyped]  # noqa: F401
        import memfabric_hybrid  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("memfabric_hybrid and memcache_hybrid are required for this E2E test") from exc


def _write_memcache_configs(
    tmp_path: Path,
    meta_port: int,
    config_store_port: int,
    metrics_port: int,
    protocol: str = "host_shm",
    world_size: int = 1,
    lease_ttl_ms: int | None = None,
) -> tuple[Path, Path]:
    # memcache_hybrid 1.1.x rejects ock.mmc.log_path in the local config, so
    # library logs go to the built-in default location instead of tmp_path.
    meta_config_path = tmp_path / "mmc-meta.conf"
    local_config_path = tmp_path / "mmc-local.conf"
    meta_config = [
        f"ock.mmc.meta_service_url = tcp://127.0.0.1:{meta_port}",
        f"ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:{config_store_port}",
        f"ock.mmc.meta_service.metrics_url = http://127.0.0.1:{metrics_port}",
        "ock.mmc.log_level = error",
    ]
    if lease_ttl_ms is not None:
        meta_config.insert(3, f"ock.mmc.meta.lease_ttl_ms = {lease_ttl_ms}")
    meta_config_path.write_text("\n".join((*meta_config, "")), encoding="utf-8")
    local_config_path.write_text(
        "\n".join(
            (
                f"ock.mmc.meta_service_url = tcp://127.0.0.1:{meta_port}",
                f"ock.mmc.local_service.config_store_url = tcp://127.0.0.1:{config_store_port}",
                "ock.mmc.log_level = error",
                f"ock.mmc.local_service.world_size = {world_size}",
                f"ock.mmc.local_service.protocol = {protocol}",
                "ock.mmc.local_service.dram.size = 1GB",
                "",
            )
        ),
        encoding="utf-8",
    )
    return meta_config_path, local_config_path


def _run_memcache_meta_service() -> None:
    from memcache_hybrid import MetaService

    MetaService.main()


def _wait_for_memcache_meta_service(process: BaseProcess, port: int) -> None:
    deadline = time.monotonic() + _MEMCACHE_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.exitcode is not None:
            raise RuntimeError(f"Memcache MetaService exited with code {process.exitcode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("Timed out waiting for Memcache MetaService")


def _stop_memcache_meta_service(process: BaseProcess | None) -> bool:
    if process is None:
        return False

    was_running = process.is_alive()
    if was_running:
        process.terminate()
    process.join(_MEMCACHE_STOP_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        process.join(_MEMCACHE_STOP_TIMEOUT_S)
    return was_running


def _make_memcache_worker_config(server_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=_ModelConfig(),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            rank=0,
            world_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="ascend-store-mp-memcache-test",
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"backend": "memcache", "kv_cache_server_url": server_url},
            is_kv_producer=True,
        ),
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def test_real_memcache_backend_store_and_retrieve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import torch_npu  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.utils.network_utils import get_open_port

    from tests.e2e.conftest import cleanup_dist_env_and_memory
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector import (
        AscendStoreMPConnector,
    )
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
        AscendConnectorMetadata,
        LoadSpec,
        ReqMeta,
    )

    _require_memcache()
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available for the Memcache E2E test")

    context = multiprocessing.get_context("spawn")
    endpoint_connection, endpoint_child_connection = context.Pipe()
    control_connection, control_child_connection = context.Pipe()
    meta_service = None
    server = None
    connector = None
    first_layer = None
    second_layer = None
    failure: BaseException | None = None
    server_exitcode = None
    server_forced = False
    meta_service_running = False

    meta_port = get_open_port()
    config_store_port = get_open_port()
    metrics_port = get_open_port()
    meta_config_path, local_config_path = _write_memcache_configs(tmp_path, meta_port, config_store_port, metrics_port)
    monkeypatch.setenv("MMC_META_CONFIG_PATH", str(meta_config_path))
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(local_config_path))

    try:
        meta_service_process = context.Process(target=_run_memcache_meta_service, name="memcache-meta-service")
        meta_service_process.start()
        meta_service = meta_service_process
        _wait_for_memcache_meta_service(meta_service, meta_port)

        server_process = context.Process(
            target=_run_kv_cache_server,
            args=(endpoint_child_connection, None, control_child_connection),
            name="kv-cache-memcache-server",
        )
        server_process.start()
        server = server_process
        endpoint_child_connection.close()
        control_child_connection.close()

        server_status, server_result = _receive(endpoint_connection, "KV cache server")
        if server_status != "ready":
            raise RuntimeError(f"KV cache server failed to start:\n{server_result}")

        torch.npu.set_device(0)
        connector = AscendStoreMPConnector(
            _make_memcache_worker_config(server_result),
            KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        _wait_until_connected(connector._kv_cache_client)
        _wait_until_registered(connector._kv_cache_client)

        first_layer, second_layer = _allocate_mooncake_test_caches()
        torch.npu.synchronize()
        connector.register_kv_caches({"model.layers.0.attn": first_layer, "model.layers.1.attn": second_layer})
        _wait_for_active_export(connector)

        store_metadata = AscendConnectorMetadata(set(), set(), delayed_free_req_ids={"store-request"})
        store_metadata.add_request(
            ReqMeta(
                "store-request",
                token_len_chunk=16,
                block_ids=[1],
                block_hashes=["real-memcache-hash"],
                can_save=True,
            )
        )
        connector.bind_connector_metadata(store_metadata)
        connector.wait_for_save()
        assert connector.get_finished({"store-request"}) == ({"store-request"}, set())
        connector.clear_connector_metadata()

        first_layer.zero_()
        second_layer.zero_()
        torch.npu.synchronize()

        load_metadata = AscendConnectorMetadata(set(), set())
        load_metadata.add_request(
            ReqMeta(
                "load-request",
                token_len_chunk=16,
                block_ids=[1],
                block_hashes=["real-memcache-hash"],
                load_spec=LoadSpec(0, 16, True),
            )
        )
        connector.bind_connector_metadata(load_metadata)
        connector.start_load_kv(None)
        torch.npu.synchronize()

        expected_first_layer = torch.zeros_like(first_layer, device="cpu")
        expected_first_layer[1].fill_(13)
        expected_second_layer = torch.zeros_like(second_layer, device="cpu")
        expected_second_layer[1].fill_(17)
        assert torch.equal(first_layer.cpu(), expected_first_layer)
        assert torch.equal(second_layer.cpu(), expected_second_layer)
        assert connector.get_block_ids_with_load_errors() == set()
        connector.clear_connector_metadata()
    except BaseException as exc:
        failure = exc
    finally:
        endpoint_connection.close()
        endpoint_child_connection.close()
        try:
            if connector is not None:
                connector.shutdown()
        except BaseException as exc:
            if failure is None:
                failure = exc
        connector = None
        if server is not None:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                control_connection.send("stop")
        control_connection.close()
        control_child_connection.close()
        try:
            server_exitcode, server_forced = _stop_process(server)
        finally:
            meta_service_running = _stop_memcache_meta_service(meta_service)
        first_layer = None
        second_layer = None
        cleanup_dist_env_and_memory()

    if failure is not None:
        raise failure
    if server_forced:
        pytest.fail("KV cache server did not stop after closing the Memcache Worker")
    assert server_exitcode == 0
    if not meta_service_running:
        pytest.fail("Memcache MetaService exited before test cleanup")


def test_real_model_memcache_layerwise_hit_and_retrieve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import torch_npu  # noqa: F401
    from vllm.utils.network_utils import get_open_port

    from tests.e2e.conftest import cleanup_dist_env_and_memory
    from vllm_ascend.ascend_config import clear_ascend_config

    _require_memcache()
    model_path = _model_path()
    if model_path is None:
        pytest.skip("Set ASCEND_STORE_MP_SMOKE_MODEL to a local model path to run this smoke test")
    assert model_path is not None
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    if os.getenv("PYTHONHASHSEED") != "0":
        pytest.fail("PYTHONHASHSEED=0 must be set before pytest starts for Memcache key consistency")

    context = multiprocessing.get_context("spawn")
    endpoint_connection, endpoint_child_connection = context.Pipe()
    control_connection, control_child_connection = context.Pipe()
    server_log = tmp_path / "kv_cache_server.log"
    meta_service = None
    server = None
    llm = None
    npu_initialized = False
    failure: BaseException | None = None
    server_exitcode = None
    server_forced = False
    meta_service_running = False

    meta_port = get_open_port()
    config_store_port = get_open_port()
    metrics_port = get_open_port()
    meta_config_path, local_config_path = _write_memcache_configs(
        tmp_path,
        meta_port,
        config_store_port,
        metrics_port,
        protocol="device_sdma",
        world_size=_MEMCACHE_LAYERWISE_WORLD_SIZE,
        lease_ttl_ms=_MEMCACHE_LAYERWISE_LEASE_TTL_MS,
    )
    monkeypatch.setenv("MMC_META_CONFIG_PATH", str(meta_config_path))
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(local_config_path))

    try:
        meta_service = context.Process(target=_run_memcache_meta_service, name="memcache-layerwise-meta-service")
        meta_service.start()
        _wait_for_memcache_meta_service(meta_service, meta_port)

        server = context.Process(
            target=_run_smoke_server,
            args=(endpoint_child_connection, control_child_connection, str(server_log)),
            name="kv-cache-memcache-layerwise-server",
        )
        server.start()
        endpoint_child_connection.close()
        control_child_connection.close()
        server_status, server_result = _receive(endpoint_connection, "KV cache server")
        if server_status != "ready":
            raise RuntimeError(f"KV cache server failed to start:\n{server_result}")

        torch.npu.set_device(0)
        npu_initialized = True
        llm = _build_llm(model_path, server_result, monkeypatch, use_layerwise=True, backend="memcache")
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        prompt_token_count = len(llm.get_tokenizer().encode(_PROMPT))
        assert prompt_token_count >= 2 * block_size, (
            f"Smoke prompt has {prompt_token_count} tokens, but at least {2 * block_size} are required"
        )

        first_output = _generate_once(llm)
        assert first_output.strip(), "First generation produced empty output"
        _wait_for_prefix_cache_reset(llm)
        second_output = _generate_once(llm)

        assert second_output == first_output, "Retrieved KV changed the greedy output"
        hits = [int(value) for value in _HIT_LOG_PATTERN.findall(server_log.read_text())]
        assert hits and max(hits) > 0, "No external Memcache KV pool hit was recorded by the server"
    except BaseException as exc:
        failure = exc
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except BaseException as exc:
                if failure is None:
                    failure = exc
            llm = None
        if npu_initialized:
            clear_ascend_config()
            cleanup_dist_env_and_memory()
        endpoint_connection.close()
        endpoint_child_connection.close()
        control_child_connection.close()
        if server is not None:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                control_connection.send("stop")
        control_connection.close()
        try:
            server_exitcode, server_forced = _stop_process(server)
        finally:
            meta_service_running = _stop_memcache_meta_service(meta_service)

    if failure is not None:
        raise failure
    if server_forced:
        pytest.fail("KV cache server did not stop gracefully after the Memcache layerwise smoke run")
    assert server_exitcode == 0
    if not meta_service_running:
        pytest.fail("Memcache MetaService exited before test cleanup")
