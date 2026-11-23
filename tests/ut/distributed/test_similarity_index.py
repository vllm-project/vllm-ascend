# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-only index tests; also runnable with pytest --noconftest."""

import importlib.util
import multiprocessing
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import zmq

# Standalone metadata tests bypass only the package's runtime logging setup
# when vLLM is absent. Production package initialization stays unchanged.
if importlib.util.find_spec("vllm") is None and "vllm_ascend" not in sys.modules:
    _test_package = ModuleType("vllm_ascend")
    _test_package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
    _test_package.__spec__ = importlib.util.spec_from_loader("vllm_ascend", loader=None, is_package=True)
    sys.modules["vllm_ascend"] = _test_package

from vllm_ascend.distributed.ec_transfer.similarity_cache_config import SimilarityCacheConfig  # noqa: E402
from vllm_ascend.distributed.ec_transfer.similarity_index import LocalSimilarityIndex  # noqa: E402
from vllm_ascend.distributed.ec_transfer.similarity_index_service import (  # noqa: E402
    SimilarityIndexClient,
    SimilarityIndexConfigurationError,
    SimilarityIndexServer,
    SimilarityIndexUnavailable,
    create_similarity_index,
)


def _serve(connection, stop, endpoint="tcp://127.0.0.1:*"):
    with SimilarityIndexServer(endpoint, "test") as server:
        connection.send(server.endpoint)
        connection.close()
        server.serve_forever(stop)


def _client(endpoint, **kwargs):
    config = SimilarityCacheConfig(
        enabled=True,
        shared_index_endpoint=endpoint,
        shared_index_namespace="test",
        shared_index_timeout_ms=200,
        **kwargs,
    )
    return SimilarityIndexClient(config, model_id="model", merge_size=2)


def _scheduler(endpoint, connection, publish):
    client = _client(endpoint)
    try:
        if publish:
            client.batch_register_phashes([("a", 1 << 63, "resized-a"), ("b", (1 << 63) | 1, "resized-b")])
            client.record_similarity_scores("a", [(0.999, "b")])
            connection.send(True)
        else:
            recalled = client.batch_recall([(1 << 63) | 2])[0]
            entries = client.batch_get_entries(["a", "b"])
            connection.send((recalled, [dict(item.ssim_similar_items) for item in entries]))
    finally:
        client.close()
        connection.close()


@pytest.fixture
def server():
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    stop = ctx.Event()
    process = ctx.Process(target=_serve, args=(child, stop))
    process.start()
    child.close()
    try:
        assert parent.poll(10), "index server did not start"
        yield parent.recv(), process, stop
    finally:
        stop.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()


def test_schedulers_in_distinct_processes_share_recall_and_scores(server):
    endpoint, _, _ = server
    ctx = multiprocessing.get_context("spawn")
    results = []
    for publish in (True, False):
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_scheduler, args=(endpoint, child, publish))
        process.start()
        child.close()
        try:
            assert parent.poll(10), "scheduler did not return"
            results.append(parent.recv())
        finally:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()
        assert process.exitcode == 0
    assert results == [True, (["resized-a", "resized-b"], [{"b": 0.999}, {"a": 0.999}])]


def test_registration_eviction_and_snapshots_are_consistent():
    index = LocalSimilarityIndex(SimilarityCacheConfig(max_capacity=20, eviction_threshold=1, lower_watermark=0.5))
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda i: index.batch_register_phashes([(str(i), i % 4, f"key-{i}")]), range(40)))
    assert len(index.similarity_cache) == 10
    surviving_keys = {key for item in index.similarity_cache.values() for key in item.resized_keys}
    assert set(index.batch_recall([0])[0]) == surviving_keys
    assert set(index.phash_to_resized_mm_hashes) == {item.phash for item in index.similarity_cache.values()}
    for bucket in index._phash_band_index.values():
        assert bucket <= set(index.phash_to_resized_mm_hashes)
    name = next(iter(index.similarity_cache))
    entry = index.batch_get_entries([name])[0]
    entry.ssim_similar_items["alien"] = 1.0
    entry.resized_keys.clear()
    assert not index.batch_get_entries([name])[0].ssim_similar_items
    assert index.batch_get_entries([name])[0].resized_keys


def test_clients_do_not_cache_misses_or_share_shutdown(server):
    endpoint, _, _ = server
    first, second = _client(endpoint), _client(endpoint)
    try:
        assert second.batch_recall([0]) == [[]]
        first.batch_register_phashes([("a", 0, "key-a")])
        first.close()
        assert second.batch_recall([0]) == [["key-a"]]
        second.batch_register_phashes([("a", 0, "key-a")])
        assert second.batch_recall([0]) == [["key-a"]]
    finally:
        first.close()
        second.close()


def test_service_rejects_incompatible_clients(server):
    endpoint, _, _ = server
    client = _client(endpoint)
    try:
        with pytest.raises(SimilarityIndexConfigurationError, match="compatibility"):
            _client(endpoint, max_hamming=0)
        config = replace(client.config, shared_index_namespace="another")
        with pytest.raises(SimilarityIndexConfigurationError, match="namespace"):
            SimilarityIndexClient(config, model_id="model", merge_size=2)
        with pytest.raises(SimilarityIndexConfigurationError, match="compatibility"):
            SimilarityIndexClient(client.config, model_id="different-model", merge_size=2)
    finally:
        client.close()


def test_timeout_resets_socket_and_reconnects_after_restart(server):
    endpoint, process, stop = server
    client = _client(endpoint)
    client.batch_register_phashes([("a", 0, "key-a")])
    stop.set()
    process.join(5)
    assert not process.is_alive()
    started = time.monotonic()
    with pytest.raises(SimilarityIndexUnavailable):
        client.batch_recall([0])
    assert time.monotonic() - started < 2
    with pytest.raises(SimilarityIndexUnavailable):
        client.batch_recall([0])
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    restart_stop = ctx.Event()
    restarted = ctx.Process(target=_serve, args=(child, restart_stop, endpoint))
    restarted.start()
    child.close()
    try:
        assert parent.poll(10)
        assert parent.recv() == endpoint
        time.sleep(client.config.shared_index_timeout_ms / 1000)
        # Restarted metadata is cold; clients must not keep stale responses.
        assert client.batch_recall([0]) == [[]]
        client.batch_register_phashes([("b", 0, "key-b")])
        assert client.batch_recall([0]) == [["key-b"]]
    finally:
        client.close()
        restart_stop.set()
        restarted.join(5)
        if restarted.is_alive():
            restarted.terminate()
            restarted.join(5)
        parent.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"shared_index_endpoint": "http://localhost:9000"},
        {"shared_index_namespace": ""},
        {"shared_index_timeout_ms": 0},
        {"shared_index_timeout_ms": True},
    ],
)
def test_shared_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError, match="shared_index"):
        SimilarityCacheConfig(**settings)


def test_shared_config_is_read_from_connector_extra_config():
    settings = {
        "enabled": True,
        "shared_index_endpoint": "tcp://localhost:5558",
        "shared_index_namespace": "deployment",
        "shared_index_timeout_ms": 50,
    }
    vllm_config = SimpleNamespace(
        ec_transfer_config=SimpleNamespace(get_from_extra_config=lambda key, default: settings)
    )
    config = SimilarityCacheConfig.from_vllm_config(vllm_config)
    assert config.shared_index_endpoint == settings["shared_index_endpoint"]
    assert config.shared_index_namespace == settings["shared_index_namespace"]
    assert config.shared_index_timeout_ms == 50


def test_embedded_owner_shares_index_and_releases_address():
    # A temporary ZMQ bind chooses a test port; the embedded factory must bind
    # this address itself, without a separate service process.
    with zmq.Context() as context, context.socket(zmq.REP) as reservation:
        reservation.bind("tcp://127.0.0.1:*")
        endpoint = reservation.getsockopt_string(zmq.LAST_ENDPOINT)
    config = SimilarityCacheConfig(
        enabled=True, shared_index_endpoint=endpoint, shared_index_namespace="test", shared_index_timeout_ms=200
    )
    owner = create_similarity_index(
        config, SimpleNamespace(data_parallel_index=0, data_parallel_rank=0), model_id="model", merge_size=2
    )
    follower = None
    try:
        # Dense models reset compute ranks to zero. The serving index must
        # prevent this follower from trying to bind the owner's address.
        follower = create_similarity_index(
            config, SimpleNamespace(data_parallel_index=1, data_parallel_rank=0), model_id="model", merge_size=2
        )
        owner.batch_register_phashes([("a", 0, "key-a")])
        assert follower.batch_recall([0]) == [["key-a"]]
        follower.close()
        assert owner.batch_recall([0]) == [["key-a"]]
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_scheduler, args=(endpoint, child, True))
        process.start()
        child.close()
        try:
            assert parent.poll(10)
            assert parent.recv() is True
        finally:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()
        assert process.exitcode == 0
        assert owner.batch_get_entries(["b"])[0].ssim_similar_items == {"a": 0.999}
    finally:
        if follower is not None:
            follower.close()
        owner.close()
    # Shutdown joins the thread and closes its socket before returning.
    restarted = create_similarity_index(config, SimpleNamespace(data_parallel_index=0), model_id="model", merge_size=2)
    try:
        assert restarted.batch_recall([0]) == [[]]
    finally:
        restarted.close()


def test_external_mode_does_not_start_a_second_server(server):
    endpoint, _, _ = server
    config = SimilarityCacheConfig(
        enabled=True, shared_index_endpoint=endpoint, shared_index_namespace="test", shared_index_mode="external"
    )
    client = create_similarity_index(config, SimpleNamespace(data_parallel_index=0), model_id="model", merge_size=2)
    try:
        client.batch_register_phashes([("a", 0, "key-a")])
        assert client.batch_recall([0]) == [["key-a"]]
    finally:
        client.close()
