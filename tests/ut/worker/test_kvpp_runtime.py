from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.worker.v2.kvpp as kvpp_module
from vllm_ascend.worker.v2.kvpp import (
    KVPPRuntime,
    KVPPScheduler,
    build_layer_cache_bundles,
    select_active_pages,
)


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _indexer(index: int) -> str:
    return f"model.layers.{index}.self_attn.indexer.k_cache"


def _scheduler(layer_count: int = 2) -> KVPPScheduler:
    owners = {_layer(index): 0 for index in range(layer_count)}
    transport = SimpleNamespace(initialize_transport=lambda _caches, _bundles, _max_pages: None)
    return KVPPScheduler(
        kvpp_group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owner_ranks=owners,
        kv_caches={layer_name: object() for layer_name in owners},
        num_physical_blocks=10,
        tokens_per_block=4,
        max_active_pages=10,
        transport=transport,
    )


def _begin(scheduler: KVPPScheduler) -> None:
    scheduler.schedule_forward(
        torch.tensor([[7, 2, -1], [2, 8, 12]], dtype=torch.int32),
        [5, 9],
    )


def test_kvpp_06_builds_sparse_main_and_indexer_cache_bundles():
    owners = {
        _layer(0): 0,
        _indexer(0): 0,
        _layer(1): 1,
        _indexer(1): 1,
    }

    layer_cache_bundles = build_layer_cache_bundles(owners, (_layer(0), _layer(1)))

    assert layer_cache_bundles == {
        _layer(0): (_layer(0), _indexer(0)),
        _layer(1): (_layer(1), _indexer(1)),
    }


def test_kvpp_07_active_pages_are_fixed_shape_deduplicated_and_masked():
    table = torch.tensor([[7, 2, -1, 0], [2, 8, 12, 0]], dtype=torch.int32)
    original = table.clone()

    pages = select_active_pages(table, [5, 9], tokens_per_block=4, num_physical_blocks=10)

    assert pages.physical_page_ids.shape == (table.numel(),)
    assert pages.physical_page_ids.tolist() == [2, 2, 7, 8, 10, 10, 10, 10]
    assert pages.valid_page_mask.tolist() == [True, False, True, True, False, False, False, False]
    assert pages.staging_page_indices.tolist() == [0, 0, 1, 2, 2, 2, 2, 2]
    assert torch.equal(table, original)


def test_kvpp_08_prefetch_starts_when_scheduled_and_advances_on_wait():
    scheduler = _scheduler()
    _begin(scheduler)
    assert scheduler._next_attention_layer_index == 0

    scheduler.wait_for_layer(_layer(0))
    assert scheduler._next_attention_layer_index == 1

    scheduler.wait_for_layer(_layer(1))
    assert scheduler._next_attention_layer_index == 2
    scheduler.complete_forward()
    assert scheduler._active_pages is None


def test_kvpp_runtime_disabled_preparation_is_noop():
    runtime = KVPPRuntime()
    runtime.prepare_forward((torch.zeros(1, 1, dtype=torch.int32),), [1])
    runtime.complete_forward()
    assert runtime.scheduler is None


def test_kvpp_09_runtime_binds_cache_and_attention_hook(monkeypatch):
    target = _layer(0)
    indexer = _indexer(0)
    hook = SimpleNamespace(layerwise_kv_cache_hook=None)
    target_cache = object()
    indexer_cache = object()
    context = {
        target: SimpleNamespace(impl=hook, kv_cache=target_cache),
        indexer: SimpleNamespace(kv_cache=indexer_cache),
    }
    owners = {target: 0, indexer: 0}
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    initialized_caches = {}

    class FakeTransport:
        def __init__(self, *_args):
            pass

        def initialize_transport(self, caches, _bundles, _max_pages):
            initialized_caches.update(caches)

    monkeypatch.setattr(
        kvpp_module.KVPPConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(size=2),
    )
    monkeypatch.setattr(kvpp_module, "map_kvpp_layers_to_owners", lambda *_args: owners)
    monkeypatch.setattr(kvpp_module, "get_kvpp_group", lambda: group)
    monkeypatch.setattr(kvpp_module, "MemFabricMTEKVPPTransport", FakeTransport)

    runtime = KVPPRuntime.create_from_kv_cache(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=(target, indexer))],
            num_blocks=8,
        ),
        block_tables=SimpleNamespace(
            blocks_per_kv_block=(2,),
            kernel_block_sizes=(128,),
        ),
        static_forward_context=context,
    )

    assert initialized_caches == {target: target_cache, indexer: indexer_cache}
    assert runtime.managed_cache_group_index == 0
    assert runtime.scheduler.layer_cache_bundles == {target: (target, indexer)}
    assert hook.layerwise_kv_cache_hook is runtime.scheduler


def test_kvpp_10_rejects_managed_layers_from_multiple_cache_groups(monkeypatch):
    target = _layer(0)
    indexer = _indexer(0)
    owners = {target: 0, indexer: 0}
    monkeypatch.setattr(
        kvpp_module.KVPPConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(size=2),
    )
    monkeypatch.setattr(kvpp_module, "map_kvpp_layers_to_owners", lambda *_args: owners)

    with pytest.raises(ValueError, match="must belong to one cache group"):
        KVPPRuntime.create_from_kv_cache(
            vllm_config=object(),
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(layer_names=(target,)),
                    SimpleNamespace(layer_names=(indexer,)),
                ]
            ),
            block_tables=object(),
            static_forward_context={},
        )
