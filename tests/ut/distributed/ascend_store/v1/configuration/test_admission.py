"""Supported cache families, parallel compositions, and peer configuration admission."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    make_worker,
    store_one,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    BulkProjectionBinder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    RangeStoreCommand,
)


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
    completion = store_one(
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


def test_v1_rejects_yuanrong_during_configuration(monkeypatch) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"backend": "yuanrong"}),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    kv_cache_config = SimpleNamespace(prefix_cache_retention_interval=None)
    monkeypatch.setattr(vllm_adapter, "_resolve_kv_pool_topology", lambda *_: object())

    with pytest.raises(ValueError, match="temporarily does not support the Yuanrong Backend"):
        vllm_adapter.resolve_kv_pool_route_spec(config, kv_cache_config)


def test_initialization_runs_layerwise_topology_validation_on_both_sides(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "_kvpp_size", lambda _config: 1)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: object())

    def reject_context_parallelism(_protocol, _parallel_config, use_layerwise):
        assert use_layerwise
        raise ValueError("Mooncake block-key layerwise does not support context parallelism")

    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", reject_context_parallelism)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"use_layerwise": True},
        ),
        parallel_config=SimpleNamespace(world_size=1),
        model_config=SimpleNamespace(max_model_len=64),
        speculative_config=None,
    )
    cache_config = SimpleNamespace(prefix_cache_retention_interval=None)

    for factory in ("scheduler", "worker"):
        with pytest.raises(ValueError, match="does not support context parallelism"):
            if factory == "scheduler":
                vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            else:
                vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)


@pytest.mark.parametrize("factory", ("scheduler", "worker"))
@pytest.mark.parametrize("kv_role, peer_role", (("kv_producer", "decode"), ("kv_consumer", "prefill")))
@pytest.mark.parametrize("context_axis", ("dcp", "pcp"))
def test_layerwise_factories_reject_peer_context_parallel_mismatch(factory, kv_role, peer_role, context_axis) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role,
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                f"{peer_role}_{context_axis}_size": 1 if context_axis == "dcp" else 2,
            },
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, prefill_context_parallel_size=1),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match="same dcp_size/pcp_size"):
        if factory == "scheduler":
            vllm_adapter.create_kv_pool_scheduler(config, SimpleNamespace(), "unused")
        else:
            vllm_adapter.create_kv_pool_worker(config, SimpleNamespace())


@pytest.mark.parametrize(
    "kv_role, extra_config, use_layerwise",
    (
        ("kv_producer", {"decode_dcp_size": 2, "decode_pcp_size": 1}, True),
        ("kv_consumer", {"prefill_dcp_size": 2, "prefill_pcp_size": 1}, True),
        ("kv_producer", {}, True),
        ("kv_consumer", {}, True),
        ("kv_both", {"prefill_dcp_size": 1, "decode_pcp_size": 2}, True),
        ("kv_producer", {"decode_dcp_size": 1}, False),
        ("kv_consumer", {"prefill_pcp_size": 2}, False),
    ),
)
def test_peer_context_parallel_check_preserves_production_scope(kv_role, extra_config, use_layerwise) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=kv_role, kv_connector_extra_config=extra_config),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, prefill_context_parallel_size=1),
        speculative_config=None,
    )

    vllm_adapter._validate_kv_pool_preflight(config, "memcache", use_layerwise=use_layerwise)


@pytest.mark.parametrize("private_only", (False, True))
def test_factory_rejects_private_state_layerwise_and_private_only(monkeypatch, private_only) -> None:
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    cache_config = SimpleNamespace(
        transfer_group_ids=() if private_only else (0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
        if private_only
        else [SimpleNamespace(kv_cache_spec=object()), SimpleNamespace(kv_cache_spec=object())],
    )
    if private_only:

        def reject_private_only(_groups):
            raise AssertionError("no cacheable groups")

        monkeypatch.setattr(
            vllm_adapter,
            "infer_cacheable_group_ids",
            reject_private_only,
        )
        expected_message = "at least one prefix-cacheable"
    else:
        monkeypatch.setattr(vllm_adapter, "infer_cacheable_group_ids", lambda _groups: [0])
        expected_message = "private KV state requires non-layerwise"
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"use_layerwise": True},
        ),
        parallel_config=SimpleNamespace(world_size=1),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match=expected_message):
        vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")


@pytest.mark.parametrize(
    "extra,force_reuse,expected_message",
    (
        (
            {"discard_partial_chunks": False},
            False,
            "does not support discard_partial_chunks=False",
        ),
        (
            {"use_layerwise": True, "discard_partial_chunks": False},
            False,
            "does not support discard_partial_chunks=False",
        ),
        (
            {"use_layerwise": True},
            True,
            "buffer reuse requires native partial-object support",
        ),
    ),
)
def test_factories_reject_unproven_partial_object_routes(
    monkeypatch,
    extra,
    force_reuse,
    expected_message,
) -> None:
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    monkeypatch.setattr(vllm_adapter, "_uses_layerwise_buffer_reuse", lambda *_: force_reuse)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        ),
        parallel_config=SimpleNamespace(world_size=1),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match=expected_message):
        vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
    with pytest.raises(ValueError, match=expected_message):
        vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)
