# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration regressions exercised through the real initialization paths."""

import unittest
from unittest.mock import patch

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import get_backend_class
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.config import (
    DEFAULT_KEY_PREFETCH_LAYERS,
    parse_layerwise_prefetch_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    build_layerwise_cache_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

INVALID_PREFETCH_VALUES: tuple[object, ...] = (
    True,
    False,
    1.0,
    1.5,
    float("inf"),
    float("nan"),
    "",
    " ",
    "abc",
    "1.0",
    "1e1",
    "1_0",
    "0x10",
    "+",
    "++1",
    "１２",
    "-0",
    0,
    -1,
    "0",
    "-1",
    None,
    [],
    {},
)
CONNECTOR_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"


@pytest.fixture
def case():
    test = unittest.TestCase()
    yield test
    test.doCleanups()


@pytest.mark.parametrize("value, expected", [(1, 1), (2**63, 2**63), ("1", 1), (" 003 ", 3), ("+2", 2)])
@pytest.mark.parametrize("backend", ["memcache", "mooncake"])
def test_worker_parses_positive_prefetch_without_changing_requested_window(case, backend, value, expected):
    worker = make_worker(
        case, use_layerwise=True, extra_config={"backend": backend, "layerwise_prefetch_layers": value}
    )
    assert worker.num_prefetch_layers == expected


@pytest.mark.parametrize("value", INVALID_PREFETCH_VALUES)
@pytest.mark.parametrize("backend", ["memcache", "mooncake"])
def test_invalid_prefetch_fails_before_backend_or_transfer_threads(case, backend, value):
    config = {"backend": backend, "use_layerwise": True, "layerwise_prefetch_layers": value}
    vllm_config = make_config(extra_config=config)
    with (
        patch.object(BaseKVPoolWorker, "_init_backend") as backend_init,
        patch("threading.Thread.start") as start,
        patch.object(LayerwiseKVPoolWorker, "wait_for_layer_load") as wait,
        patch(f"{CONNECTOR_MODULE}.create_kv_pool_worker") as worker_factory,
        patch(f"{CONNECTOR_MODULE}.create_kv_pool_scheduler") as scheduler_factory,
        patch.object(get_backend_class(backend), "create_scheduler_client") as scheduler_client,
    ):
        for role in ("SCHEDULER", "WORKER"):
            with pytest.raises(ValueError, match="layerwise_prefetch_layers must be a positive integer"):
                AscendStoreConnector(vllm_config, role)
        with pytest.raises(ValueError, match="layerwise_prefetch_layers must be a positive integer"):
            make_worker(case, use_layerwise=True, extra_config=config)
        with pytest.raises(ValueError, match="layerwise_prefetch_layers must be a positive integer"):
            LayerwiseKVPoolScheduler(vllm_config, True)
        backend_init.assert_not_called()
        scheduler_client.assert_not_called()
        worker_factory.assert_not_called()
        scheduler_factory.assert_not_called()
        start.assert_not_called()
        wait.assert_not_called()


@pytest.mark.parametrize("value", INVALID_PREFETCH_VALUES)
def test_cache_planner_uses_the_same_strict_prefetch_parser(value):
    with pytest.raises(ValueError, match="layerwise_prefetch_layers must be a positive integer"):
        build_layerwise_cache_layout(8, {"layerwise_prefetch_layers": value})


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("backend, default", [("memcache", 8), ("mooncake", DEFAULT_KEY_PREFETCH_LAYERS)])
def test_supported_role_defaults(case, backend, default, role):
    worker = make_worker(case, kv_role=role, num_layers=12, use_layerwise=True, extra_config={"backend": backend})
    assert worker.num_prefetch_layers == default
    assert worker.kv_role == role
    assert worker._transfer_threads_started is False


@pytest.mark.parametrize("buffers, expected", [(1, 1), (3, 3), (8, 8), (12, 8)])
def test_gva_default_is_capped_by_shared_buffer_count(case, buffers, expected):
    worker = make_worker(
        case,
        num_layers=12,
        use_layerwise=True,
        extra_config={"backend": "memcache", "layerwise_num_shared_buffers": buffers},
    )
    assert worker.num_prefetch_layers == expected
    assert build_layerwise_cache_layout(12, {"layerwise_num_shared_buffers": buffers}).num_prefetch_layers == expected


@pytest.mark.parametrize(
    "layerwise, backend", [(False, "memcache"), (False, "mooncake"), (True, "memcache"), (True, "mooncake")]
)
def test_discard_partial_chunks_defaults_to_true_and_preserves_override(layerwise, backend):
    for extra in ({}, {"discard_partial_chunks": False}):
        config = make_config(extra_config={"backend": backend, **extra})
        config.kv_transfer_config.get_from_extra_config.side_effect = lambda name, default, config=config: (
            config.kv_transfer_config.kv_connector_extra_config.get(name, default)
        )
        with (
            patch.object(get_backend_class(backend), "create_scheduler_client"),
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"),
        ):
            scheduler = (LayerwiseKVPoolScheduler if layerwise else KVPoolScheduler)(config, layerwise)
        assert scheduler._discard_partial_chunks is extra.get("discard_partial_chunks", True)


@pytest.mark.parametrize("backend", ["yuanrong", "unknown"])
def test_unsupported_backend_fails_on_all_initialization_paths(case, backend):
    config = make_config(extra_config={"backend": backend, "use_layerwise": True})
    with patch("threading.Thread.start") as start:
        for role in ("SCHEDULER", "WORKER"):
            with pytest.raises(ValueError, match=f"Backend '{backend}'.*does not support layerwise"):
                AscendStoreConnector(config, role)
        with pytest.raises(ValueError, match=f"Backend '{backend}'.*does not support layerwise"):
            make_worker(case, use_layerwise=True, extra_config={"backend": backend})
        with pytest.raises(ValueError, match=f"Backend '{backend}'.*does not support layerwise"):
            LayerwiseKVPoolScheduler(config, True)
        start.assert_not_called()


@pytest.mark.parametrize(
    "option, value", [("layerwise_num_shared_buffers", 2), ("layerwise_independent_layers", "all")]
)
def test_mooncake_rejects_gva_only_options_before_client_creation(case, option, value):
    extra = {"backend": "mooncake", option: value}
    with (
        patch.object(BaseKVPoolWorker, "_init_backend") as backend_init,
        pytest.raises(ValueError, match=option),
    ):
        make_worker(case, use_layerwise=True, extra_config=extra)
    backend_init.assert_not_called()
    with pytest.raises(ValueError, match=option):
        LayerwiseKVPoolScheduler(make_config(extra_config=extra), True)


def test_absent_prefetch_uses_the_path_default():
    assert parse_layerwise_prefetch_layers({}, 1) == 1
    assert parse_layerwise_prefetch_layers({}, 8) == 8


def test_disabled_layerwise_worker_ignores_unused_block_key_prefetch_config(case):
    # The factory selects the ordinary worker when disabled. Exercise direct
    # construction as well, retaining the existing subclass constructor behavior.
    with patch(
        "tests.ut.distributed.ascend_store.test_pool_worker.create_kv_pool_worker",
        side_effect=LayerwiseKVPoolWorker,
    ):
        worker = make_worker(
            case,
            use_layerwise=False,
            extra_config={"backend": "mooncake", "layerwise_prefetch_layers": 0},
        )
    assert worker.num_prefetch_layers == DEFAULT_KEY_PREFETCH_LAYERS


@pytest.mark.parametrize("backend", ["memcache", "mooncake"])
@pytest.mark.parametrize("role, peer", [("kv_producer", "decode"), ("kv_consumer", "prefill")])
def test_layerwise_tp_mismatch_remains_an_initialization_error(case, backend, role, peer):
    with (
        patch.object(BaseKVPoolWorker, "_init_backend") as backend_init,
        patch("threading.Thread.start") as start,
        pytest.raises(ValueError, match=r"TP mismatch \(local_tp=2, peer_tp=4\)"),
    ):
        make_worker(
            case,
            use_layerwise=True,
            kv_role=role,
            tp_size=2,
            num_kv_heads=8,
            extra_config={"backend": backend, f"{peer}_tp_size": 4},
        )
    backend_init.assert_not_called()
    start.assert_not_called()


@pytest.mark.parametrize("backend", ["memcache", "mooncake"])
@pytest.mark.parametrize("role, peer", [("kv_producer", "decode"), ("kv_consumer", "prefill")])
@pytest.mark.parametrize("parallelism", ["dcp", "pcp"])
def test_layerwise_cp_layout_mismatch_fails_before_threads(case, backend, role, peer, parallelism):
    with (
        patch.object(BaseKVPoolWorker, "_init_backend") as backend_init,
        patch("threading.Thread.start") as start,
        pytest.raises(ValueError, match="Decode-context-parallel mismatch.*local dcp_size=1, local pcp_size=1"),
    ):
        make_worker(
            case,
            use_layerwise=True,
            kv_role=role,
            extra_config={"backend": backend, f"{peer}_{parallelism}_size": 2},
        )
    backend_init.assert_not_called()
    start.assert_not_called()
