# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import MooncakeKVConnectorStats

from .helpers import make_full_spec, make_kv_cache_tensor, make_sfa_indexer_spec, make_sliding_spec


def test_transfer_max_workers() -> None:
    options: dict[str, int] = {}
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(get_from_extra_config=options.get))
    assert MooncakeBaseConnectorWorker._get_transfer_max_workers(config) == 32
    options["transfer_max_workers"] = 64
    assert MooncakeBaseConnectorWorker._get_transfer_max_workers(config) == 64
    for value in (0, True):
        options["transfer_max_workers"] = value
        with pytest.raises(ValueError, match="transfer_max_workers must be a positive integer"):
            MooncakeBaseConnectorWorker._get_transfer_max_workers(config)


@pytest.mark.parametrize(
    ("pcp_rank", "tp_rank", "target_index"),
    [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)],
)
def test_kvpp_publishes_owned_layers_and_mtp(monkeypatch, pcp_rank, tp_rank, target_index):
    from tests.ut.kvpp_utils import layer_name, make_kvpp_config

    names = [layer_name(i) for i in (9, 10, 11, 12, 17)]
    spec = make_full_spec()
    caches = {name: torch.zeros((2, 16, 1, 8), dtype=torch.float16) for name in names}
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = make_kvpp_config(tp=2, pcp=2)
    worker.vllm_config.parallel_config.cp_kv_cache_interleave_size = 16
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=4))
    worker.pcp_rank, worker.pcp_size = pcp_rank, 2
    worker.tp_rank, worker.tp_size = tp_rank, 2
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=cache.nbytes,
                layers=[name],
                layer_stride=cache.nbytes,
                block_stride=cache.stride(0) * cache.element_size(),
            )
            for name, cache in caches.items()
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
    )
    worker.engine_id, worker.te_rpc_port = "producer", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 16, "127.0.0.1", 5000
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    regions = MagicMock(wraps=base_worker.collect_configured_register_regions)
    monkeypatch.setattr(base_worker, "collect_configured_register_regions", regions)
    worker.register_kv_caches(caches)
    assert worker.xfer_handshake_metadata.layer_names == [names[target_index], names[4]]
    assert worker.xfer_handshake_metadata.kv_caches_base_addr == [
        [caches[names[target_index]].data_ptr()],
        [caches[names[4]].data_ptr()],
    ]
    assert worker.kv_caches is caches
    assert any(arg is caches for arg in regions.call_args.args)


def test_build_spec_mappings_expands_uniform_group_by_layer_spec() -> None:
    full = make_full_spec()
    sliding = make_sliding_spec()
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer.0": full, "layer.1": sliding},
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = MagicMock(
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0", "layer.1"], kv_cache_spec=uniform)]
    )

    worker._build_kv_cache_spec_mappings()

    assert worker.kv_cache_specs == [full, sliding]
    assert worker.layer_name_to_group_index == {"layer.0": 0, "layer.1": 0}
    assert worker.layer_name_to_spec_index == {"layer.0": 0, "layer.1": 1}


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("spec_factory", [make_full_spec, make_sliding_spec])
def test_register_kv_caches_uses_config_order_and_publishes_tensor_metadata(monkeypatch, layout, spec_factory) -> None:
    spec = spec_factory()
    shape = (4, 1, 16, 8) if layout == "HND" else (4, 16, 1, 8)
    k_cache = torch.empty(shape, dtype=torch.float16)
    v_cache = torch.empty(shape, dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=k_cache.nbytes + v_cache.nbytes,
                layers=["layer.0"],
                layer_stride=k_cache.nbytes + v_cache.nbytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = config
    worker.engine_id = "engine-d"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "10.0.0.1"
    worker.handshake_port = 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(
        base_worker,
        "global_te",
        transfer_engine,
    )
    monkeypatch.setattr(
        base_worker,
        "validate_register_region_count",
        MagicMock(),
    )

    worker.register_kv_caches({"layer.0": [k_cache, v_cache]})

    metadata = worker.xfer_handshake_metadata
    assert metadata is not None
    assert metadata.layer_names == ["layer.0"]
    assert metadata.layer_block_sizes == [spec.block_size]
    assert metadata.kv_caches_base_addr == [[k_cache.data_ptr(), v_cache.data_ptr()]]
    assert metadata.block_shapes == [[(1, 16, 8), (1, 16, 8)]]
    assert metadata.block_size_scales == [[2, 2]]
    assert metadata.block_strides == [[k_cache.stride(0) * 2, v_cache.stride(0) * 2]]
    assert metadata.block_lens == [[256, 256]]
    assert tuple(k_cache.shape) == shape
    transfer_engine.register_buffer.assert_called_once()


@pytest.mark.parametrize(
    ("heads", "shape", "step", "error"),
    [
        (2, (2, 16, 2, 8), 1, "exactly one KV head"),
        (16, (2, 16, 16, 8), 1, "cannot distinguish HND from NHD"),
        (1, (2, 16, 1, 16), 2, "contiguous bytes"),
    ],
)
def test_transfer_block_shape_rejects_unsupported_cache(heads, shape, step, error) -> None:
    cache = torch.empty(shape, dtype=torch.float16)[..., ::step]
    with pytest.raises(ValueError, match=error):
        MooncakeBaseConnectorWorker._get_transfer_block_shape(make_full_spec(num_kv_heads=heads), cache)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 2, 8, 512, 32),  # C8-MXFP K/V PA_NZ cache
        (4, 2, 32, 8, 16, 1),  # C8-MXFP K-scale cache
        (4, 2, 8, 16, 32, 1),  # C8-MXFP V-scale cache
    ],
)
def test_transfer_block_shape_preserves_packed_head_major_gqa(shape) -> None:
    cache = torch.empty(shape, dtype=torch.uint8)

    assert MooncakeBaseConnectorWorker._get_transfer_block_shape(make_full_spec(num_kv_heads=2), cache) == tuple(
        shape[1:]
    )


@pytest.mark.parametrize(
    ("shape", "step", "error"),
    [
        ((4, 1, 8, 512, 32), 1, "does not match"),
        ((4, 2, 8, 512, 32), 2, "contiguous bytes"),
    ],
)
def test_transfer_block_shape_rejects_invalid_packed_head_major_gqa(shape, step, error) -> None:
    cache = torch.empty(shape, dtype=torch.uint8)[..., ::step]
    with pytest.raises(ValueError, match=error):
        MooncakeBaseConnectorWorker._get_transfer_block_shape(make_full_spec(num_kv_heads=2), cache)


@pytest.mark.parametrize("spare", [0, 64, 256])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("page_stride", "k_offset", "scale_offset", "scale_len", "merged"),
    [
        pytest.param(40, 0, 32, 8, True, id="full-page"),
        pytest.param(64, 0, 32, 8, False, id="trailing-padding"),
        pytest.param(64, 16, 48, 8, False, id="shared-page-subregion"),
        pytest.param(64, 0, 40, 24, False, id="interior-gap"),
        pytest.param(64, 0, 16, 32, False, id="overlap-hides-gap"),
        pytest.param(64, 0, 24, 40, True, id="overlapping-full-page"),
    ],
)
def test_register_kv_caches_merges_only_full_pages(
    monkeypatch, spare, reverse, page_stride, k_offset, scale_offset, scale_len, merged
) -> None:
    spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16)
    # The same tensor views must publish the same number of descriptors even
    # when one peer's backing storage has spare capacity after the last block.
    required = 3 * page_stride + max(k_offset + 32, scale_offset + scale_len)
    raw_cache = torch.empty(required + spare, dtype=torch.float16)
    k_cache = torch.as_strided(raw_cache, size=(4, 32), stride=(page_stride, 1), storage_offset=k_offset)
    scale_cache = torch.as_strided(raw_cache, size=(4, scale_len), stride=(page_stride, 1), storage_offset=scale_offset)
    caches = [scale_cache, k_cache] if reverse else [k_cache, scale_cache]
    page_bytes = page_stride * raw_cache.element_size()
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=raw_cache.nbytes,
                layers=["layer.0"],
                layer_stride=raw_cache.nbytes,
                block_stride=page_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = config
    worker.engine_id = "engine-d"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "10.0.0.1"
    worker.handshake_port = 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(
        base_worker,
        "global_te",
        transfer_engine,
    )
    monkeypatch.setattr(
        base_worker,
        "validate_register_region_count",
        MagicMock(),
    )

    worker.register_kv_caches({"layer.0": caches})

    metadata = worker.xfer_handshake_metadata
    assert metadata is not None
    if merged:
        assert metadata.kv_caches_base_addr == [[raw_cache.data_ptr()]]
        assert metadata.block_strides == [[page_bytes]]
        assert metadata.block_lens == [[page_bytes]]
        assert metadata.block_shapes == [[(max(32, scale_len),)]]
        assert metadata.block_size_scales == [[1]]
    else:
        assert metadata.kv_caches_base_addr == [[cache.data_ptr() for cache in caches]]
        assert metadata.block_strides == [[page_bytes, page_bytes]]
        assert metadata.block_lens == [[cache[0].nbytes for cache in caches]]
        assert metadata.block_shapes == [[tuple(cache.shape[1:]) for cache in caches]]
        assert metadata.block_size_scales == [[1, 1]]


def test_shared_page_rejects_noncontiguous_blocks() -> None:
    raw_cache = torch.empty(4 * 64, dtype=torch.uint8)
    caches = (
        raw_cache.as_strided((4, 16), (64, 2)),
        raw_cache.as_strided((4, 48), (64, 1), 16),
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.num_blocks = 4

    assert worker._get_shared_page_metadata(caches) is None


def test_shared_page_handles_mixed_dtype_views() -> None:
    raw_cache = torch.empty(4 * 64, dtype=torch.uint8)
    k_cache = raw_cache.as_strided((4, 48), (64, 1))
    scale_cache = raw_cache.view(torch.float32).as_strided((4, 4), (16, 1), 12)
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.num_blocks = 4

    assert worker._get_shared_page_metadata((k_cache, scale_cache)) == (raw_cache.data_ptr(), 64, (48,), 1)


def test_register_kv_caches_publishes_raw_slot_for_component_major_mla_views(monkeypatch) -> None:
    spec = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.float16)
    raw_cache = torch.empty(4 * 128 * 576, dtype=torch.float16)
    slot_elems = 128 * 576
    nope_cache = torch.as_strided(
        raw_cache,
        size=(4, 128, 1, 512),
        stride=(slot_elems, 512, 512, 1),
        storage_offset=0,
    )
    rope_cache = torch.as_strided(
        raw_cache,
        size=(4, 128, 1, 64),
        stride=(slot_elems, 64, 64, 1),
        storage_offset=128 * 512,
    )
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=raw_cache.nbytes,
                layers=["layer.0"],
                layer_stride=raw_cache.nbytes,
                block_stride=slot_elems * raw_cache.element_size(),
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = config
    worker.engine_id = "engine-d"
    worker.te_rpc_port = 9000
    worker.block_size = 128
    worker.side_channel_host = "10.0.0.1"
    worker.handshake_port = 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(base_worker, "global_te", transfer_engine)
    monkeypatch.setattr(base_worker, "validate_register_region_count", MagicMock())

    worker.register_kv_caches({"layer.0": [nope_cache, rope_cache]})

    metadata = worker.xfer_handshake_metadata
    assert metadata is not None
    assert metadata.kv_caches_base_addr == [[raw_cache.data_ptr()]]
    assert metadata.block_strides == [[slot_elems * raw_cache.element_size()]]
    assert metadata.block_lens == [[slot_elems * raw_cache.element_size()]]
    assert metadata.block_shapes == [[(slot_elems,)]]
    assert metadata.indexer_replication_sizes == [0]
    assert metadata.cp_kv_cache_interleave_size == 16
    assert metadata.block_size_scales == [[1]]


def test_component_major_mla_binding_rejects_token_interleaved_parent() -> None:
    spec = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.float16)
    parent = torch.empty(4, 128, 1, 576, dtype=torch.float16)
    nope_cache = parent[..., :512]
    rope_cache = parent[..., 512:]

    assert not base_worker._is_component_major_mla_binding(spec, (nope_cache, rope_cache))


def test_shared_storage_is_not_enough_to_identify_a_packed_page() -> None:
    raw_cache = torch.empty(4 * 64, dtype=torch.float16)
    k_cache = torch.as_strided(raw_cache, size=(4, 32), stride=(32, 1), storage_offset=0)
    v_cache = torch.as_strided(raw_cache, size=(4, 32), stride=(32, 1), storage_offset=128)
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.num_blocks = 4

    assert worker._get_shared_page_metadata((k_cache, v_cache)) is None


@pytest.mark.parametrize("replication_size", [1, 2])
def test_register_kv_caches_publishes_sfa_indexer_virtual_block_size(
    monkeypatch,
    replication_size: int,
) -> None:
    spec = make_sfa_indexer_spec(block_size=16, replication_size=replication_size)
    cache = torch.empty((2 * replication_size, 16, 1, 8), dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=cache.nbytes,
                layers=["layer.0.indexer"],
                layer_stride=cache.nbytes,
                block_stride=cache.stride(0) * cache.element_size(),
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0.indexer"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = config
    worker.engine_id = "engine-d"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "10.0.0.1"
    worker.handshake_port = 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(
        base_worker,
        "global_te",
        transfer_engine,
    )
    monkeypatch.setattr(
        base_worker,
        "validate_register_region_count",
        MagicMock(),
    )

    worker.register_kv_caches({"layer.0.indexer": cache})

    metadata = worker.xfer_handshake_metadata
    assert metadata is not None
    assert metadata.indexer_replication_sizes == [replication_size]
    assert metadata.layer_block_sizes == [16 * replication_size]
    assert metadata.block_size_scales == [[replication_size]]
    assert metadata.layer_block_sizes[0] // metadata.block_size_scales[0][0] == spec.block_size


@pytest.fixture(params=[False, True], ids=["single-tensor", "kv-pair"])
def sharing_registration(monkeypatch, request):
    spec = make_full_spec()
    caches = {}
    for name in ("layer.1", "layer.3"):
        k_cache = torch.zeros((4, 1, 16, 8), dtype=torch.float16)
        caches[name] = [k_cache, torch.zeros_like(k_cache)] if request.param else k_cache
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer.4": SimpleNamespace(kv_sharing_target_layer_name="layer.1"),
                "layer.5": SimpleNamespace(kv_sharing_target_layer_name="layer.3"),
                # A declaration on another PP stage does not require a local binding.
                "layer.9": SimpleNamespace(kv_sharing_target_layer_name="layer.8"),
            }
        ),
    )
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=sum(cache.nbytes for cache in base_worker.as_kv_cache_tensors(binding)),
                layers=[name],
                layer_stride=sum(cache.nbytes for cache in base_worker.as_kv_cache_tensors(binding)),
                block_stride=spec.page_size_bytes,
            )
            for name, binding in caches.items()
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=list(caches), kv_cache_spec=spec)],
    )
    worker.engine_id, worker.te_rpc_port = "engine", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 16, "127.0.0.1", 5000
    transfer_engine = MagicMock()
    monkeypatch.setattr(base_worker, "global_te", transfer_engine)
    return worker, caches, transfer_engine


@pytest.mark.parametrize("new_views", [False, True])
def test_register_kv_sharing_keeps_owner_metadata_and_registration(sharing_registration, new_views):
    worker, caches, transfer_engine = sharing_registration
    worker.register_kv_caches(caches)
    owner_metadata = worker.transfer_metadata
    owner_registration = transfer_engine.register_buffer.call_args
    transfer_engine.reset_mock()

    def borrow(binding):
        if not new_views:
            return binding
        if isinstance(binding, torch.Tensor):
            return binding.view_as(binding)
        return tuple(cache.view_as(cache) for cache in binding)

    # Borrowers can precede owners in the runner's dictionary.
    shared_caches = {"layer.5": borrow(caches["layer.3"]), "layer.4": borrow(caches["layer.1"]), **caches}
    worker.register_kv_caches(shared_caches)

    assert worker.kv_caches is shared_caches
    assert worker.transfer_metadata.layer_names == ["layer.1", "layer.3"]
    assert worker.transfer_metadata == owner_metadata
    transfer_engine.register_buffer.assert_called_once()
    assert transfer_engine.register_buffer.call_args == owner_registration


def test_owner_descriptor_copy_is_visible_through_borrower(sharing_registration):
    worker, caches, _ = sharing_registration
    caches["layer.4"] = caches["layer.1"]
    worker.register_kv_caches(caches)
    source_block = torch.full((1, 16, 8), 7, dtype=torch.float16)
    metadata = worker.transfer_metadata

    # Emulate a transfer engine writing the published owner's first block.
    # No independent descriptor or second copy is needed for the borrower.
    assert "layer.4" not in metadata.layer_names
    for address, length in zip(metadata.kv_caches_base_addr[0], metadata.block_lens[0]):
        ctypes.memmove(address, source_block.data_ptr(), length)
    for cache in base_worker.as_kv_cache_tensors(caches["layer.4"]):
        torch.testing.assert_close(cache[0], source_block)
        assert torch.count_nonzero(cache[1:]) == 0


@pytest.mark.parametrize("target", [None, "missing", "layer.4", "layer.5"])
def test_register_kv_sharing_rejects_invalid_declarations(sharing_registration, target):
    worker, caches, transfer_engine = sharing_registration
    context = worker.vllm_config.compilation_config.static_forward_context
    context["layer.4"].kv_sharing_target_layer_name = target
    caches["layer.4"] = caches["layer.1"]

    with pytest.raises(ValueError, match="absent from kv_cache_tensors|must target a configured owner"):
        worker.register_kv_caches(caches)
    transfer_engine.register_buffer.assert_not_called()
    assert not hasattr(worker, "transfer_metadata")


@pytest.mark.parametrize("binding", ["copy", "other-owner", "shape", "stride", "dtype", "offset", "count", "empty"])
def test_register_kv_sharing_rejects_different_bindings(sharing_registration, binding):
    worker, caches, transfer_engine = sharing_registration
    owner_tensors = base_worker.as_kv_cache_tensors(caches["layer.1"])
    if binding == "copy":
        borrower = [cache.clone() for cache in owner_tensors]
    elif binding == "other-owner":
        borrower = caches["layer.3"]
    elif binding == "shape":
        borrower = [cache.view(4, 1, 8, 16) for cache in owner_tensors]
    elif binding == "stride":
        borrower = [cache.as_strided(cache.shape, (1, 512, 32, 4)) for cache in owner_tensors]
    elif binding == "dtype":
        borrower = [cache.view(torch.int16) for cache in owner_tensors]
    elif binding == "offset":
        borrower = [cache[1:] for cache in owner_tensors]
    elif binding == "count":
        borrower = list(owner_tensors) + [owner_tensors[0]]
    else:
        borrower = []
    caches["layer.4"] = borrower

    with pytest.raises(ValueError, match="must bind to the same cache"):
        worker.register_kv_caches(caches)
    transfer_engine.register_buffer.assert_not_called()


def test_register_kv_sharing_rejects_missing_owner(sharing_registration):
    worker, caches, transfer_engine = sharing_registration
    caches["layer.4"] = caches.pop("layer.1")

    with pytest.raises(ValueError, match="No KV cache was registered for configured layer 'layer.1'"):
        worker.register_kv_caches(caches)
    transfer_engine.register_buffer.assert_not_called()


def test_register_kv_sharing_rejects_kvpp_before_mapping(sharing_registration, monkeypatch):
    worker, caches, transfer_engine = sharing_registration
    worker.ascend_config.kvpp_config.size = 2
    caches["layer.4"] = caches["layer.1"]
    map_owners = MagicMock()
    monkeypatch.setattr(base_worker, "map_kvpp_layers_to_owners", map_owners)

    with pytest.raises(ValueError, match="does not support KV sharing with KV parallelism"):
        worker.register_kv_caches(caches)
    map_owners.assert_not_called()
    transfer_engine.register_buffer.assert_not_called()


def test_register_kv_caches_rejects_missing_and_unconfigured_layers() -> None:
    spec = make_full_spec()
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=64,
                layers=["layer.0"],
                layer_stride=64,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = config
    worker.engine_id = "engine"
    worker.te_rpc_port = 9000
    worker.block_size = 16
    worker.side_channel_host = "host"
    worker.handshake_port = 5000

    try:
        worker.register_kv_caches({})
    except ValueError as error:
        assert "No KV cache" in str(error)
    else:
        raise AssertionError("missing configured layer must fail")

    cache = torch.empty((2, 1, 16, 8), dtype=torch.float16)
    try:
        worker.register_kv_caches({"layer.0": cache, "unexpected": cache})
    except ValueError as error:
        assert "absent from kv_cache_tensors" in str(error)
    else:
        raise AssertionError("unexpected layer must fail")


def make_worker_config(
    *,
    is_consumer: bool = True,
    is_producer: bool = False,
    local_dp_rank: int | None = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=is_consumer,
            is_kv_producer=is_producer,
            kv_role="kv_consumer" if is_consumer else "kv_producer",
            kv_port=6000,
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            data_parallel_rank_local=local_dp_rank,
            data_parallel_size_local=3,
            data_parallel_rank=2,
            data_parallel_index=2,
        ),
    )


def patch_worker_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pcp_size: int = 1,
) -> tuple[MagicMock, object]:
    monkeypatch.setenv("ASCEND_TRANSFER_TIMEOUT", "original")
    tp_group = object()
    transfer_engine = MagicMock()
    transfer_engine.get_rpc_port.return_value = 9000
    global_te = MagicMock()
    global_te.get_transfer_engine.return_value = transfer_engine
    patches = {
        "init_ascend_config": MagicMock(),
        "get_ascend_config": MagicMock(return_value=object()),
        "get_transfer_timeout_value": MagicMock(return_value=30),
        "get_tensor_model_parallel_rank": MagicMock(return_value=2),
        "get_tp_group": MagicMock(return_value=tp_group),
        "get_pp_group": MagicMock(return_value=SimpleNamespace(rank_in_group=1)),
        "get_pcp_group": MagicMock(return_value=SimpleNamespace(rank_in_group=0, world_size=pcp_size)),
        "get_decode_context_model_parallel_world_size": MagicMock(return_value=2),
        "get_decode_context_model_parallel_rank": MagicMock(return_value=1),
        "get_ip": MagicMock(return_value="10.0.0.2"),
        "global_te": global_te,
    }
    for name, value in patches.items():
        monkeypatch.setattr(
            base_worker,
            name,
            value,
        )
    monkeypatch.setattr(torch.npu, "current_device", MagicMock(return_value=7))
    return global_te, tp_group


@pytest.mark.parametrize("independent_dp", [True, False], ids=["dense", "moe"])
def test_dp_workers_publish_distinct_handshake_ports(monkeypatch, independent_dp):
    patch_worker_runtime(monkeypatch)
    ports = []
    for dp_index in range(2):
        config = make_worker_config(local_dp_rank=dp_index)
        config.parallel_config.data_parallel_rank = 0 if independent_dp else dp_index
        config.parallel_config.data_parallel_index = dp_index
        config.parallel_config.data_parallel_size_local = 1 if independent_dp else 2
        worker = MooncakeBaseConnectorWorker(config, f"engine-{dp_index}", SimpleNamespace(num_blocks=1))
        ports.append(worker.handshake_port)
    assert len(set(ports)) == 2


def test_base_worker_initializes_parallel_ranks_and_handshake_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_worker_config()
    kv_cache_config = SimpleNamespace(num_blocks=32)
    global_te, tp_group = patch_worker_runtime(monkeypatch)

    worker = MooncakeBaseConnectorWorker(config, "engine-d", kv_cache_config)  # type: ignore[arg-type]

    assert worker.engine_id == "engine-d"
    assert worker.block_size == 16
    assert worker.num_blocks == 32
    assert worker.tp_rank == 2
    assert worker.tp_size == 4
    assert worker.tp_group is tp_group
    assert worker.pp_rank == 1
    assert worker.pp_size == 2
    assert worker.dp_rank == 1
    assert worker.dp_size == 3
    assert worker.pcp_rank == 0
    assert worker.pcp_size == 1
    assert worker.dcp_rank == 1
    assert worker.dcp_size == 2
    assert worker.max_device_id == 24
    assert worker.side_channel_host == "10.0.0.2"
    assert worker.side_channel_port == 6016
    assert worker.handshake_port == 6022
    assert worker.te_rpc_port == 9000
    assert worker.xfer_handshake_metadata is None
    assert worker.xfer_stats.is_empty()
    global_te.get_transfer_engine.assert_called_once_with("10.0.0.2", device_name="7")


@pytest.mark.parametrize(("is_consumer", "is_producer"), [(False, False), (True, True)])
def test_base_worker_rejects_invalid_transfer_roles(
    is_consumer: bool,
    is_producer: bool,
) -> None:
    config = make_worker_config(is_consumer=is_consumer, is_producer=is_producer)

    with pytest.raises(ValueError, match="exactly one KV transfer role"):
        MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]


def test_base_worker_requires_local_dp_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_worker_config(local_dp_rank=None)
    patch_worker_runtime(monkeypatch)

    with pytest.raises(ValueError, match="requires a local DP rank"):
        MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]


def test_base_worker_accepts_consumer_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_worker_config()
    patch_worker_runtime(monkeypatch, pcp_size=2)

    worker = MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]

    assert worker.pcp_size == 2


def test_base_worker_accepts_producer_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_worker_config(is_consumer=False, is_producer=True)
    patch_worker_runtime(monkeypatch, pcp_size=2)

    worker = MooncakeBaseConnectorWorker(config, "engine", SimpleNamespace(num_blocks=1))  # type: ignore[arg-type]

    assert worker.pcp_size == 2
    assert worker.max_device_id == 48
    assert worker.side_channel_port == 6032
    assert worker.handshake_port == 6042


def test_base_worker_stats_are_returned_and_reset() -> None:
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.xfer_stats = MooncakeKVConnectorStats(data={})

    assert worker.get_kv_connector_stats() is None

    worker.xfer_stats.record_transfer(0.1, 1024)
    result = worker.get_kv_connector_stats()

    assert result is not None
    assert result.data["transfer_duration"] == [0.1]
    assert worker.xfer_stats.is_empty()


def test_base_worker_abstract_contract() -> None:
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)

    with pytest.raises(NotImplementedError):
        worker.get_finished()
    with pytest.raises(NotImplementedError):
        worker.get_block_ids_with_load_errors()
    with pytest.raises(NotImplementedError):
        worker.start_load_kv(MagicMock())
