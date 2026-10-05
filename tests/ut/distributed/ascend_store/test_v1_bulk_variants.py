"""Observable Worker traces for the explicit AscendStore Bulk variants."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    BulkProjectionBinder,
    ConsumerPipelineBulkProjection,
    HybridBulkProjection,
    OrdinaryBulkProjection,
    TPMismatchBulkProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupRequest,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import KVPoolRouteSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)

from .v1.helpers import FakeBackend, make_topology, make_worker


def _begin(worker, *, load=(), store=()) -> None:
    worker.begin_step(
        KVTransferStep(
            LoadCommandBatch(tuple(load)),
            StoreCommandBatch(tuple(store)),
        )
    )


def _store(worker, command: StoreCommand):
    _begin(worker, store=(command,))
    worker.finish_step()
    (completion,) = worker.fence_previous_store()
    worker.end_step()
    return completion


def _make_align_state_topology():
    base = make_topology(group_ids=(0, 1), physical_layers=(0,), tp_size=2)
    attention = KVPoolGroupTopology(
        0,
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
        (KVPoolLayerTopology(0, ("layers.0.attention",)),),
        replace(base.groups[0].key_metadata, cache_family="attention"),
    )
    state = KVPoolGroupTopology(
        1,
        MambaSpec(
            block_size=8,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        (KVPoolLayerTopology(1, ("layers.1.state",)),),
        replace(base.groups[1].key_metadata, cache_family="state"),
    )
    return replace(
        base,
        cache_transfer_granularity=8,
        hash_block_size=4,
        groups=(attention, state),
    )


def _make_sparse_group_topology():
    base = make_topology(group_ids=(1, 3), physical_layers=(0,))
    attention = KVPoolGroupTopology(
        1,
        FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
        (KVPoolLayerTopology(0, ("layers.0.attention",)),),
        replace(base.groups[1].key_metadata, cache_family="attention"),
    )
    sliding = KVPoolGroupTopology(
        3,
        SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=8,
        ),
        (KVPoolLayerTopology(1, ("layers.1.swa",)),),
        replace(base.groups[3].key_metadata, cache_family="swa"),
    )
    return replace(base, groups=(attention, sliding))


def _make_factory_config(
    *,
    speculative_config=None,
    additional_config=None,
    use_layerwise: bool = False,
):
    hf_config = SimpleNamespace(num_hidden_layers=2)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"use_layerwise": use_layerwise},
        ),
        parallel_config=SimpleNamespace(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            max_model_len=64,
            model="model",
            use_mla=False,
            hf_text_config=hf_config,
            hf_config=hf_config,
            get_total_num_kv_heads=lambda: 1,
            get_total_num_hidden_layers=lambda: 2,
        ),
        speculative_config=speculative_config,
        additional_config={} if additional_config is None else additional_config,
    )


def test_factory_selects_the_explicit_bulk_binder() -> None:
    topologies = (
        make_topology(),
        make_topology(tp_mismatch=True),
        make_topology(consumer_pipeline_partitions=(1, 1)),
        _make_sparse_group_topology(),
        _make_align_state_topology(),
    )
    for topology in topologies:
        spec = KVPoolRouteSpec(topology, "fake", 64)

        binder = vllm_adapter._compile_kv_pool_projection_binder(
            spec,
            SimpleNamespace(speculative_config=None),
            SimpleNamespace(),
        )

        assert isinstance(binder, BulkProjectionBinder), topology


def test_factories_reject_every_unverified_speculative_method(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", lambda *_args: None)
    for use_layerwise in (False, True):
        for method in ("mtp", "dspark", "eagle", "eagle3", "dflash"):
            config = _make_factory_config(
                speculative_config=SimpleNamespace(method=method),
                use_layerwise=use_layerwise,
            )
            for factory in ("scheduler", "worker"):
                with pytest.raises(ValueError, match=rf"speculative method '{method}'"):
                    if factory == "scheduler":
                        vllm_adapter.create_kv_pool_scheduler(config, SimpleNamespace(), "unused")
                    else:
                        vllm_adapter.resolve_kv_pool_route_spec(config, SimpleNamespace())


def test_layerwise_factories_reject_align_state_before_route_construction(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", lambda *_args: None)
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (8, 8))
    monkeypatch.setattr(vllm_adapter, "_uses_layerwise_buffer_reuse", lambda *_args: False)
    monkeypatch.setattr(vllm_adapter, "infer_cacheable_group_ids", lambda _groups: [0, 1])
    attention = FullAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    state = MambaSpec(
        block_size=8,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    cache_config = SimpleNamespace(
        transfer_group_ids=(0, 1),
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=attention),
            SimpleNamespace(kv_cache_spec=state),
        ],
        prefix_cache_retention_interval=None,
    )
    config = _make_factory_config(use_layerwise=True)

    for factory in ("scheduler", "worker"):
        with pytest.raises(ValueError, match="does not support Mamba align-state groups"):
            if factory == "scheduler":
                vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            else:
                vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)


def test_bulk_factories_reject_non_align_mamba_state(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", lambda *_args: None)
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (4, 4))
    monkeypatch.setattr(vllm_adapter, "infer_cacheable_group_ids", lambda _groups: [0])
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=MambaSpec(
                    block_size=4,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="none",
                )
            )
        ],
        prefix_cache_retention_interval=None,
    )
    config = _make_factory_config()

    for factory in ("scheduler", "worker"):
        with pytest.raises(ValueError, match="supports Mamba state only in mamba_cache_mode='align'"):
            if factory == "scheduler":
                vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            else:
                vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)


def test_private_groups_are_projected_out_of_scheduler_worker_and_backend(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", lambda *_args: None)
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (4, 4))
    monkeypatch.setattr(vllm_adapter, "resolve_dcp_kv_cache_spec", lambda spec, _size: spec)
    monkeypatch.setattr(vllm_adapter, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(vllm_adapter, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        vllm_adapter,
        "RemoteLookup",
        lambda _address: SimpleNamespace(close=lambda: None),
    )

    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    private = CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        head_size_v=0,
        dtype=torch.float32,
    )
    cache_config = SimpleNamespace(
        num_blocks=8,
        transfer_group_ids=(0, 1),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layers.0.attention"],
                kv_cache_spec=attention,
                is_eagle_group=False,
            ),
            SimpleNamespace(
                layer_names=["layers.1.private"],
                kv_cache_spec=private,
                is_eagle_group=False,
            ),
        ],
        prefix_cache_retention_interval=None,
    )
    config = _make_factory_config()

    scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
    assert scheduler._config.transfer_group_ids == (0,)
    assert scheduler._config.has_private_state
    scheduler.close()

    route_spec = vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)
    assert tuple(group.group_id for group in route_spec.topology.groups) == (0, 1)
    assert route_spec.topology.transfer_group_ids == (0,)
    binder = vllm_adapter._compile_kv_pool_projection_binder(route_spec, config, cache_config)
    assert isinstance(binder, BulkProjectionBinder)

    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=route_spec.topology)
    completion = _store(
        worker,
        RangeStoreCommand(
            "store",
            TokenRange(0, 4),
            ((7,), (99,)),
            (b"a",),
            4,
            17,
        ),
    )
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and "@group:0@" in put_call[1][0]
    assert put_call[2:] == (((1448,),), ((32,),))
    assert [item.source.group_id for item in completion.evidence.transfer_evidence] == [0]
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [7]

    worker.close()
    assert resources.closed


@pytest.mark.parametrize("kvpp_size", (1, 2))
def test_kvpp_authoritative_size_selects_noop_or_rejection(kvpp_size, monkeypatch) -> None:
    config = _make_factory_config(additional_config={"enable_kvpp": True})

    class FakeKVPPConfig:
        @classmethod
        def from_vllm_config(cls, observed_config):
            assert observed_config is config
            return SimpleNamespace(size=kvpp_size)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_config",
        SimpleNamespace(KVPPConfig=FakeKVPPConfig),
    )
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", lambda *_args: None)
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (4, 4))
    monkeypatch.setattr(vllm_adapter, "resolve_dcp_kv_cache_spec", lambda spec, _size: spec)
    monkeypatch.setattr(vllm_adapter, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(vllm_adapter, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        vllm_adapter,
        "RemoteLookup",
        lambda _address: SimpleNamespace(close=lambda: None),
    )
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        num_blocks=8,
        transfer_group_ids=(0,),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layers.0.attention"],
                kv_cache_spec=attention,
                is_eagle_group=False,
            )
        ],
        prefix_cache_retention_interval=None,
    )

    for factory in ("scheduler", "worker"):
        if kvpp_size > 1:
            with pytest.raises(ValueError, match="does not support active KVPP"):
                if factory == "scheduler":
                    vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
                else:
                    vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)
        elif factory == "scheduler":
            scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            scheduler.close()
        else:
            route_spec = vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)
            binder = vllm_adapter._compile_kv_pool_projection_binder(route_spec, config, cache_config)
            assert isinstance(binder, BulkProjectionBinder)


def test_ordinary_bulk_worker_owns_lookup_load_store_rows_and_ranges() -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend)
    assert isinstance(worker._bulk_projection, OrdinaryBulkProjection)

    backend.presence = [1, 0]
    lookup = worker.lookup(LookupRequest(TokenRange(0, 8), (0,), (b"a", b"b")))
    assert lookup.available_end_token == 4
    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert [key.endswith(suffix) for key, suffix in zip(exists_call[1], ("@61", "@62"), strict=True)] == [
        True,
        True,
    ]

    backend.get_result = [0, 0]
    load = LoadCommand("load", TokenRange(0, 6), ((2, 3),), (b"a", b"b"))
    _begin(worker, load=(load,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert get_call[2] == ((1128, 2128), (1192, 2192))
    assert get_call[3] == ((32, 32), (16, 16))

    completion = _store(
        worker,
        RangeStoreCommand("store", TokenRange(4, 6), ((2, 3),), (b"a", b"b"), 6, 17),
    )
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1][0].endswith("@62")
    assert put_call[2:] == (((1192, 2192),), ((16, 16),))
    (evidence,) = completion.evidence.transfer_evidence
    assert evidence.source.block_id == 3
    assert evidence.source.physical_layer_ids == (0, 1)

    worker.close()
    assert resources.closed


def test_sparse_hybrid_groups_keep_original_identity_through_public_worker() -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=_make_sparse_group_topology())
    assert isinstance(worker._bulk_projection, HybridBulkProjection)
    assert worker._bulk_projection.group_ids == (1, 3)
    assert [(group.dense_index, group.group_id) for group in worker._bulk_projection.groups] == [(0, 1), (1, 3)]

    lookup = worker.lookup(LookupRequest(TokenRange(0, 8), (1, 3), (b"a", b"b")))
    assert lookup.available_end_token == 8
    exists_calls = [call for call in backend.calls if call[0] == "exists"]
    assert [len(call[1]) for call in exists_calls] == [2, 2]
    assert all("@group:1@" in key for key in exists_calls[0][1])
    assert all("@group:3@" in key for key in exists_calls[1][1])

    backend.calls.clear()
    command = LoadCommand(
        "load",
        TokenRange(0, 8),
        ((), (11, 12), (), (31, 32)),
        (b"a", b"b"),
    )
    _begin(worker, load=(command,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert ["@group:1@" in key for key in get_call[1]] == [True, True, False, False]
    assert ["@group:3@" in key for key in get_call[1]] == [False, False, True, True]
    assert get_call[2] == ((11704,), (11768,), (33984,), (34048,))
    assert get_call[3] == ((32,), (32,), (32,), (32,))

    backend.calls.clear()
    completion = _store(
        worker,
        RangeStoreCommand(
            "store",
            TokenRange(0, 8),
            ((), (11, 12), (), (31, 32)),
            (b"a", b"b"),
            8,
            17,
        ),
    )
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1:] == (get_call[1], get_call[2], get_call[3])
    assert [item.source.group_id for item in completion.evidence.transfer_evidence] == [1, 1, 3, 3]
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [11, 12, 31, 32]
    assert [item.source.physical_layer_ids for item in completion.evidence.transfer_evidence] == [
        (0,),
        (0,),
        (1,),
        (1,),
    ]

    worker.close()
    assert resources.closed


def test_hybrid_bulk_worker_owns_fine_lookup_load_range_and_checkpoint_store(monkeypatch) -> None:
    backend = FakeBackend()
    backend.presence_results = [[1, 1], [1, 1, 1, 1]]
    worker, resources, _ = make_worker(backend, topology=_make_align_state_topology())
    assert isinstance(worker._bulk_projection, HybridBulkProjection)
    assert not hasattr(worker, "_layerwise_projection")

    lookup = worker.lookup(LookupRequest(TokenRange(0, 8), (0, 1), (b"a", b"b")))
    assert lookup.available_end_token == 8
    exists_calls = [call for call in backend.calls if call[0] == "exists"]
    assert [len(call[1]) for call in exists_calls] == [2, 4]
    assert all("@group:0@" in key for key in exists_calls[0][1])
    assert ["@head_or_tp_rank:0@" in key for key in exists_calls[1][1][::2]] == [True, False]
    assert ["@head_or_tp_rank:1@" in key for key in exists_calls[1][1][::2]] == [False, True]

    backend.calls.clear()
    monkeypatch.setattr(worker._bulk_projection.reachability, "select_for_load", lambda *_args: (None, None))
    load = LoadCommand(
        "load",
        TokenRange(0, 4),
        ((3,), (7,)),
        (b"a",),
        (TailKeyBoundary(0, 4), TailKeyBoundary(1, 4)),
    )
    _begin(worker, load=(load,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert ["@group:0@" in key for key in get_call[1]] == [True, False]
    assert ["@group:1@" in key for key in get_call[1]] == [False, True]
    assert get_call[2] == ((1192,), (12448,))
    assert get_call[3] == ((16,), (32,))

    backend.calls.clear()
    range_completion = _store(
        worker,
        RangeStoreCommand(
            "range",
            TokenRange(0, 8),
            ((3,), (7,)),
            (b"a", b"b"),
            8,
            17,
        ),
    )
    range_put = next(call for call in backend.calls if call[0] == "put")
    assert len(range_put[1]) == 1 and "@group:0@" in range_put[1][0]
    assert range_put[2:] == (((1192,),), ((32,),))
    assert [item.source.group_id for item in range_completion.evidence.transfer_evidence] == [0]

    backend.calls.clear()
    checkpoint_completion = _store(
        worker,
        CheckpointStoreCommand(
            "checkpoint",
            ((3,), (99,)),
            (b"a",),
            0,
            (StateCheckpointSource(1, 7, 4),),
            18,
        ),
    )
    checkpoint_put = next(call for call in backend.calls if call[0] == "put")
    assert ["@group:0@" in key for key in checkpoint_put[1]] == [True, False]
    assert ["@group:1@" in key for key in checkpoint_put[1]] == [False, True]
    assert checkpoint_put[2:] == (((1192,), (12448,)), ((32,), (32,)))
    assert [item.source.group_id for item in checkpoint_completion.evidence.transfer_evidence] == [0, 1]
    assert [item.source.physical_layer_ids for item in checkpoint_completion.evidence.transfer_evidence] == [
        (0,),
        (1,),
    ]

    worker.close()
    assert resources.closed


def test_hybrid_checkpoint_requires_unique_exact_align_state_sources() -> None:
    worker, resources, _ = make_worker(topology=_make_align_state_topology())

    duplicate = CheckpointStoreCommand(
        "duplicate",
        ((3,), (99,)),
        (b"a",),
        0,
        (
            StateCheckpointSource(1, 7, 4),
            StateCheckpointSource(1, 8, 4),
        ),
        17,
    )
    with pytest.raises(ValueError, match="duplicate cache-group sources"):
        worker._build_store_candidates((duplicate,))

    invalid = CheckpointStoreCommand(
        "invalid",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(0, 3, 4),),
        18,
    )
    with pytest.raises(ValueError, match="invalid source groups"):
        worker._build_store_candidates((invalid,))

    unaligned = CheckpointStoreCommand(
        "unaligned",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(1, 7, 3),),
        19,
    )
    with pytest.raises(ValueError, match="not aligned to the hash block size"):
        worker._build_store_candidates((unaligned,))

    null_source = CheckpointStoreCommand(
        "null",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(1, 0, 4),),
        20,
    )
    null_candidates = worker._build_store_candidates((null_source,))
    assert [group.row_count for group in null_candidates.groups] == [0, 0]

    aligned = CheckpointStoreCommand(
        "aligned",
        ((3,), (99,)),
        (b"a", b"b"),
        0,
        (StateCheckpointSource(1, 7, 8),),
        21,
    )
    aligned_candidates = worker._build_store_candidates((aligned,))
    assert [group.row_count for group in aligned_candidates.groups] == [0, 1]

    worker.close()
    assert resources.closed


def test_hybrid_bulk_load_failure_is_request_scoped() -> None:
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=make_topology(group_ids=(1, 3), physical_layers=(0,)))
    command = LoadCommand(
        "request",
        TokenRange(0, 4),
        ((), (1,), (), (2,)),
        (b"a",),
    )

    _begin(worker, load=(command,))
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed.*request"):
        worker.start_load()
    worker.end_step()

    worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    ("topology", "expected_ranks", "expected_addresses", "expected_sizes"),
    (
        (
            make_topology(
                physical_layers=(0,),
                tp_mismatch=True,
                tp_rank=1,
                tp_size=2,
                key_rank_count=4,
                key_slices_per_rank=2,
            ),
            (2, 3),
            ((1128, 1136, 1144), (1132, 1140, 1148)),
            ((4, 4, 4), (4, 4, 4)),
        ),
        (
            make_topology(
                physical_layers=(0,),
                tp_mismatch=True,
                tp_rank=2,
                tp_size=4,
                key_rank_count=4,
                key_slices_per_rank=1,
            ),
            (2,),
            ((1128,),),
            ((24,),),
        ),
    ),
)
def test_tp_mismatch_bulk_binds_effective_keys_to_exact_local_slices(
    topology,
    expected_ranks,
    expected_addresses,
    expected_sizes,
) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)
    assert isinstance(worker._bulk_projection, TPMismatchBulkProjection)

    completion = _store(
        worker,
        RangeStoreCommand("request", TokenRange(0, 3), ((2,),), (b"a",), 3, 17),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert tuple(f"@head_or_tp_rank:{rank}@" in key for key, rank in zip(put_call[1], expected_ranks, strict=True)) == (
        True,
    ) * len(expected_ranks)
    assert put_call[2] == expected_addresses
    assert put_call[3] == expected_sizes
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [2] * len(expected_ranks)

    worker.close()
    assert resources.closed


def test_tp_mismatch_bulk_preserves_axis_result_provenance() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_mismatch=True,
        tp_rank=1,
        tp_size=2,
        key_rank_count=4,
        key_slices_per_rank=2,
    )
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=topology, store=False)
    command = LoadCommand("request", TokenRange(0, 3), ((2,),), (b"a",))

    batch = worker._build_load_batch((command,))
    (completion,) = worker._execute_bulk_load(batch)

    assert [item.result_code for item in completion.transfer_evidence] == [0, -1]
    assert [item.source.block_id for item in completion.transfer_evidence] == [2, 2]
    assert [
        "@head_or_tp_rank:2@" in completion.transfer_evidence[0].source.key,
        "@head_or_tp_rank:3@" in completion.transfer_evidence[1].source.key,
    ] == [True, True]

    worker.close()
    assert resources.closed


def test_consumer_pipeline_store_keeps_key_memory_and_layer_provenance_atomic() -> None:
    topology = make_topology(
        physical_layers=(0, 1, 2, 3),
        consumer_pipeline_partitions=(2, 2),
    )
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(
        backend,
        topology=topology,
        requires_exists_before_put=True,
    )
    assert isinstance(worker._bulk_projection, ConsumerPipelineBulkProjection)

    completion = _store(
        worker,
        RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),
    )

    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert ["@pp_rank:0@" in exists_call[1][0], "@pp_rank:1@" in exists_call[1][1]] == [True, True]
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and "@pp_rank:1@" in put_call[1][0]
    assert put_call[2] == ((3064, 4064),)
    assert put_call[3] == ((32, 32),)
    (evidence,) = completion.evidence.transfer_evidence
    assert evidence.source.physical_layer_ids == (2, 3)
    assert "@pp_rank:1@" in evidence.source.key

    worker.close()
    assert resources.closed


def test_ordinary_bulk_partitions_writers_after_candidate_filtering() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_rank=1,
        tp_size=4,
        pcp_rank=1,
        pcp_size=2,
        put_step=2,
    )
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = _store(
        worker,
        RangeStoreCommand(
            "request",
            TokenRange(0, 16),
            ((1, 2, 3, 4),),
            (b"a", b"b", b"c", b"d"),
            16,
            17,
        ),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and put_call[1][0].endswith("@64")
    assert put_call[2] == ((1256,),)
    assert completion.evidence.transfer_evidence[0].source.block_id == 4

    worker.close()
    assert resources.closed


def test_dcp_bulk_keeps_every_row_of_its_distinct_key_shard() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_rank=1,
        tp_size=4,
        dcp_rank=1,
        dcp_size=2,
        put_step=2,
    )
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = _store(
        worker,
        RangeStoreCommand(
            "request",
            TokenRange(0, 8),
            ((1, 2),),
            (b"a", b"b"),
            8,
            17,
        ),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 2
    assert all("@dcp:1@" in key for key in put_call[1])
    assert put_call[2] == ((1064,), (1128,))
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [1, 2]

    worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    ("extra", "parallel", "model", "group_count", "message"),
    (
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"use_mla": True, "num_kv_heads": 1},
            1,
            "cannot represent the configured TP mismatch",
        ),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"num_kv_heads": 6},
            1,
            "cannot represent the configured TP mismatch",
        ),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"index_topk": 32, "num_kv_heads": 8},
            1,
            "does not support sparse KV layouts",
        ),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"num_kv_heads": 8},
            2,
            "requires one dense transferable KV cache group",
        ),
        (
            {"prefill_tp_size": 4, "consumer_is_to_put": True, "prefill_pp_size": 2},
            {"tensor_parallel_size": 2},
            {"num_kv_heads": 8},
            1,
            "cannot be composed with Consumer pipeline Store",
        ),
        (
            {},
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2, "decode_context_parallel_size": 2},
            {"num_kv_heads": 8},
            1,
            "does not support combined DCP and PCP",
        ),
        (
            {},
            {"tensor_parallel_size": 4, "decode_context_parallel_size": 2},
            {"num_kv_heads": 1},
            1,
            "remaining same-key TP writer replicas",
        ),
    ),
)
def test_factories_reject_unproven_bulk_compositions(
    extra,
    parallel,
    model,
    group_count,
    message,
    monkeypatch,
) -> None:
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", lambda *_args: None)
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (4, 4))
    monkeypatch.setattr(vllm_adapter, "infer_cacheable_group_ids", lambda _groups: [0])

    hf_values = {"num_hidden_layers": 2}
    if "index_topk" in model:
        hf_values["index_topk"] = model["index_topk"]
    hf_text_config = SimpleNamespace(**hf_values)
    model_config = SimpleNamespace(
        max_model_len=64,
        model="model",
        use_mla=model.get("use_mla", False),
        hf_text_config=hf_text_config,
        get_total_num_kv_heads=lambda: model["num_kv_heads"],
    )
    parallel_config = SimpleNamespace(
        world_size=parallel.get("tensor_parallel_size", 1),
        tensor_parallel_size=parallel.get("tensor_parallel_size", 1),
        pipeline_parallel_size=1,
        prefill_context_parallel_size=parallel.get("prefill_context_parallel_size", 1),
        decode_context_parallel_size=parallel.get("decode_context_parallel_size", 1),
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer", kv_connector_extra_config=extra),
        parallel_config=parallel_config,
        model_config=model_config,
        speculative_config=None,
    )
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for _ in range(group_count)],
        prefix_cache_retention_interval=None,
    )

    for factory in ("scheduler", "worker"):
        with pytest.raises(ValueError, match=message):
            if factory == "scheduler":
                vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            else:
                vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)
