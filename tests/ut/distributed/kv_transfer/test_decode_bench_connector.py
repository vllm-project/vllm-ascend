# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorScheduler,
    DecodeBenchConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.distributed.kv_transfer.decode_bench_connector import (
    AscendDecodeBenchConnector,
    AscendDecodeBenchConnectorScheduler,
    AscendDecodeBenchConnectorWorker,
)


def _config(startup_fill=False):
    extra = {"startup_fill": startup_fill, "fill_mean": 0.015, "fill_std": 0.0}
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kv_transfer_config=SimpleNamespace(
            kv_connector="DecodeBenchConnector",
            kv_connector_module_path=None,
            get_from_extra_config=lambda name, default: extra.get(name, default),
        ),
    )


def _cache_config(wrapped=False, upstream=False):
    tail = (
        CircularBufferSpec(block_size=7, num_kv_heads=1, head_size=4, dtype=torch.float32)
        if upstream
        else AscendIndexerKPoolTailSpec(
            block_size=7,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            sliding_window=4,
            compress_ratio=4,
        )
    )
    if wrapped:
        tail = UniformTypeKVCacheSpecs(block_size=7, kv_cache_specs={"tail": tail})
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["attention"],
                kv_cache_spec=FullAttentionSpec(block_size=1024, num_kv_heads=1, head_size=4, dtype=torch.float32),
            ),
            KVCacheGroupSpec(layer_names=["tail"], kv_cache_spec=tail),
        ],
    )


def _blocks(attention=(2, 3, 4, 5, 6, 7), tail=(0, 9)):
    return SimpleNamespace(
        blocks=tuple(
            [SimpleNamespace(block_id=block_id, is_null=block_id == 0) for block_id in group]
            for group in (attention, tail)
        )
    )


def test_reproduces_upstream_857_versus_one_failure():
    scheduler = DecodeBenchConnectorScheduler(_config(), _cache_config())
    with pytest.raises(AssertionError, match=r"group=1, range=\[0, 857\), allocated=1"):
        scheduler.update_state_after_alloc(
            SimpleNamespace(request_id="original", num_tokens=6000), _blocks(tail=(9,)), 5999
        )


@pytest.mark.parametrize("startup_fill", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_selects_actual_circular_blocks_and_preserves_specs(startup_fill, wrapped):
    cache = _cache_config(wrapped=wrapped)
    original_spec = cache.kv_cache_groups[1].kv_cache_spec
    scheduler = AscendDecodeBenchConnectorScheduler(_config(startup_fill), cache)
    request = SimpleNamespace(request_id="fixed", num_tokens=6000)
    scheduler.update_state_after_alloc(request, _blocks(), 5999)
    meta = scheduler.build_connector_meta(None)
    expected_attention = [] if startup_fill else [2, 3, 4, 5, 6, 7]
    assert meta.reqs_to_fill == {"fixed": ((expected_attention, [9]), 5999)}
    assert cache.kv_cache_groups[1].kv_cache_spec is original_spec
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert not scheduler.build_connector_meta(None).reqs_to_fill
    scheduler.request_finished(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (5999, False)


def test_ordinary_attention_still_checks_allocation():
    scheduler = AscendDecodeBenchConnectorScheduler(_config(), _cache_config())
    with pytest.raises(AssertionError, match="group=0"):
        scheduler.update_state_after_alloc(
            SimpleNamespace(request_id="short", num_tokens=6000), _blocks(attention=(2,)), 5999
        )


def test_cached_prefix_selection_remains_unchanged():
    scheduler = AscendDecodeBenchConnectorScheduler(_config(), _cache_config())
    scheduler.update_state_after_alloc(SimpleNamespace(request_id="prefix", num_tokens=6000), _blocks(), 879)
    assert scheduler.build_connector_meta(None).reqs_to_fill == {"prefix": (([7], [9]), 879)}


def test_no_external_tokens_and_null_only_circular_blocks():
    scheduler = AscendDecodeBenchConnectorScheduler(_config(), _cache_config())
    request = SimpleNamespace(request_id="empty", num_tokens=6000)
    scheduler.update_state_after_alloc(request, _blocks(), 0)
    assert not scheduler.build_connector_meta(None).reqs_to_fill
    scheduler.update_state_after_alloc(request, _blocks(tail=(0,)), 5999)
    assert scheduler.build_connector_meta(None).reqs_to_fill["empty"][0][1] == []


@pytest.mark.parametrize("startup_fill", [False, True])
@pytest.mark.parametrize("container", ["tensor", "list", "tuple"])
def test_zeroes_only_requested_circular_rows_and_preserves_padding(startup_fill, container):
    config, cache = _config(startup_fill), _cache_config()
    scheduler = AscendDecodeBenchConnectorScheduler(config, cache)
    worker = AscendDecodeBenchConnectorWorker(config, cache)
    base = torch.full((16, 9, 10), 123.0)
    view = base[:, :7, :8]
    tail = view if container == "tensor" else [view] if container == "list" else (view,)
    attention = torch.full((16, 2, 4), 123.0)
    worker.register_kv_caches({"attention": attention, "tail": tail})
    # Circular caches must not participate in whole-pool startup fill.
    assert torch.equal(base, torch.full_like(base, 123.0))
    for request_id, block_id in (("first", 9), ("second", 10)):
        scheduler.update_state_after_alloc(
            SimpleNamespace(request_id=request_id, num_tokens=6000), _blocks(tail=(block_id,)), 5999
        )
        before = base.clone()
        worker.start_fill_kv(scheduler.build_connector_meta(None))
        expected = before.clone()
        expected[block_id, :7, :8] = 0.0
        torch.testing.assert_close(base, expected, rtol=0, atol=0)
        # Mimic state written by an active request before the next admission.
        view[block_id].fill_(42.0)
    torch.testing.assert_close(attention[2:8], torch.full_like(attention[2:8], 0.015))


@pytest.mark.parametrize("block_id", [-1, 16])
def test_invalid_circular_block_ids_fail(block_id):
    worker = AscendDecodeBenchConnectorWorker(_config(), _cache_config())
    worker.register_kv_caches({"tail": [torch.ones(16, 7, 8)]})
    with pytest.raises(ValueError, match="circular block IDs"):
        worker._fill_blocks(1, [block_id], 5999)


def test_upstream_circular_groups_keep_upstream_implementations():
    config, cache = _config(), _cache_config(upstream=True)
    scheduler = AscendDecodeBenchConnector(config, KVConnectorRole.SCHEDULER, cache)
    worker = AscendDecodeBenchConnector(config, KVConnectorRole.WORKER, cache)
    assert type(scheduler.connector_scheduler) is DecodeBenchConnectorScheduler
    assert type(worker.connector_worker) is DecodeBenchConnectorWorker
    scheduler.update_state_after_alloc(SimpleNamespace(request_id="native", num_tokens=6000), _blocks(), 5999)
    assert scheduler.build_connector_meta(None).reqs_to_fill["native"][0][1] == [9]


def test_registry_loads_adapter_for_both_roles(monkeypatch):
    from vllm_ascend.distributed.kv_transfer import register_connector

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    config, cache = _config(), _cache_config()
    cls = KVConnectorFactory.get_connector_class(config.kv_transfer_config)
    assert cls is AscendDecodeBenchConnector
    scheduler = cls(config, KVConnectorRole.SCHEDULER, cache)
    worker = cls(config, KVConnectorRole.WORKER, cache)
    assert isinstance(scheduler.connector_scheduler, AscendDecodeBenchConnectorScheduler)
    assert isinstance(worker.connector_worker, AscendDecodeBenchConnectorWorker)
