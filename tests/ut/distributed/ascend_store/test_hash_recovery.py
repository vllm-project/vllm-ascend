# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real tensor descriptors; requires Core's upstream boundary-state Store API."""

import ctypes
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from filelock import FileLock, Timeout

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data")

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import KeyMetadata
except ModuleNotFoundError as exc:
    if exc.name != "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout":
        raise
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import KeyMetadata

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_connector import (
    AscendHashStoreWorker,
    AscendStoreConnector,
    MooncakeStoreWorker,
    _align_store_eagle_policy,
    _AlignedStore,
    _engine_transfer_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_layout import AscendHashStoreLayout


def layout(*, state=False, hnd=False, page=16, unit=4):
    return AscendHashStoreLayout(KeyMetadata("test", 0, 0, 0, 0), page, unit, is_state=state, head_major=hnd)


@pytest.mark.parametrize("source", ["planner", "worker_context", "direct"])
def test_engine_policy_not_multiconnector_child_role(source):
    child = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role="kv_both"))
    parent = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role="kv_producer"))
    cache = SimpleNamespace()
    if source == "planner":
        cache.kv_transfer_config = parent.kv_transfer_config
    current = parent if source == "worker_context" else None
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_connector.get_current_vllm_config_or_none",
        return_value=current,
    ):
        resolved = _engine_transfer_config(child, cache)
    assert resolved is (child if source == "direct" else parent).kv_transfer_config
    assert child.kv_transfer_config.kv_role == "kv_both"


@pytest.mark.parametrize("role", ["producer", "consumer", "both"])
def test_store_checkpoint_policy_matches_existing_ascend_role_policy(role):
    group_type = namedtuple("Group", "use_eagle")
    groups = [group_type(True), group_type(False)]
    coord = SimpleNamespace(
        use_eagle=True, attention_groups=groups, eagle_group_ids={0}, eagle_proof_margin_by_group={0: 64}
    )
    config = SimpleNamespace(is_kv_producer=role in ("producer", "both"), is_kv_consumer=role in ("consumer", "both"))
    _align_store_eagle_policy(coord, config)
    if role == "producer":
        assert not coord.use_eagle
        assert not any(group.use_eagle for group in coord.attention_groups)
        assert not coord.eagle_group_ids and not coord.eagle_proof_margin_by_group
        assert groups[0].use_eagle  # Do not mutate Core's group/config objects.
    else:
        assert coord.use_eagle and coord.attention_groups is groups
        assert coord.eagle_group_ids == {0} and coord.eagle_proof_margin_by_group == {0: 64}


def test_pd_tail_uses_same_attention_and_mamba_hash_boundary():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import BoundaryPut
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import _partial_tail_non_mamba_puts

    coord = SimpleNamespace(
        use_eagle=True,
        attention_groups=[],
        mamba_group_ids={1},
        hash_block_size=64,
        num_hash_lookahead_tokens=0,
        lcm_block_size=1152,
        eagle_group_ids={1},
        eagle_proof_margin_by_group={},
    )
    request = SimpleNamespace(
        boundary_puts=[BoundaryPut(1, 9, 1536)],
        publish_partial_tail=True,
        num_prompt_tokens=1599,
        completed_token_len=1599,
        block_hashes=list(range(24)),
        block_ids=([3, 4], [8, 9]),
    )
    # The unadapted Core Store path reproduces the observed mismatched key.
    assert _partial_tail_non_mamba_puts(coord, request, [1152, 1152]) == [BoundaryPut(0, 4, 1472)]
    _align_store_eagle_policy(coord, SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    assert _partial_tail_non_mamba_puts(coord, request, [1152, 1152]) == [BoundaryPut(0, 4, 1536)]


@pytest.mark.parametrize("fail", [False, True])
def test_optional_store_bootstrap_lock_releases_on_success_and_failure(tmp_path, fail):
    path = str(tmp_path / "store-setup.lock")
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"npu_store_setup_lock": path})
    )

    def initialize(worker, *args):
        worker.coord = SimpleNamespace()
        with pytest.raises(Timeout), FileLock(path, timeout=0):
            pass
        if fail:
            raise RuntimeError("setup failed")

    with patch.object(MooncakeStoreWorker, "__init__", autospec=True, side_effect=initialize):
        if fail:
            with pytest.raises(RuntimeError, match="setup failed"):
                AscendHashStoreWorker(config, None)
        else:
            AscendHashStoreWorker(config, None)
    with FileLock(path, timeout=0):
        pass


def test_store_bootstrap_default_has_no_lock_and_rejects_relative_path():
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}))

    def initialize_worker(worker, *args):
        worker.coord = SimpleNamespace()

    with patch.object(MooncakeStoreWorker, "__init__", autospec=True, side_effect=initialize_worker) as initialize:
        AscendHashStoreWorker(config, None)
        assert initialize.call_args.args[1:] == (config, None)
    config.kv_transfer_config.kv_connector_extra_config["npu_store_setup_lock"] = "relative.lock"
    with pytest.raises(ValueError, match="absolute host-local"):
        AscendHashStoreWorker(config, None)


def payload(descriptor):
    addresses, sizes, _ = descriptor
    return b"".join(ctypes.string_at(address, size) for address, size in zip(addresses[0], sizes[0]))


def put_payload(descriptor, data):
    addresses, sizes, _ = descriptor
    cursor = 0
    for address, size in zip(addresses[0], sizes[0]):
        ctypes.memmove(address, data[cursor : cursor + size], size)
        cursor += size
    assert cursor == len(data)


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_tail_object_roundtrip(hnd, scale, dtype):
    # Core restores a complete logical block under a partial-boundary key.
    # Different physical padding, other blocks/layers remain guards.
    shape = (3, scale, 2, 16 // scale, 3) if hnd else (3, scale, 16 // scale, 2, 3)
    source_backing = torch.full((3, scale, 128), -9, dtype=dtype)
    target_backing = torch.full((3, scale, 160), -7, dtype=dtype)
    within = (16 // scale) * 6
    source = source_backing[..., :within].view(shape)
    target = target_backing[..., :within].view(shape)
    source.copy_(torch.arange(source.numel()).view(shape))
    before = target_backing.clone()
    expected = target.clone()
    token_axis = 3 if hnd else 2
    for kernel in range(scale):
        for token in range(16 // scale):
            logical = kernel * (16 // scale) + token
            if 0 <= logical < 16:
                index = [1, kernel, slice(None), slice(None), slice(None)]
                index[token_axis] = token
                expected[tuple(index)] = source[tuple(index)]
    src_layout, dst_layout = layout(hnd=hnd), layout(hnd=hnd)
    src_layout.register_kv_caches([source], 3)
    dst_layout.register_kv_caches([target], 3)
    src = src_layout.prepare_values([(4, 12)], [1], [0])
    dst = dst_layout.prepare_values([(4, 12)], [1], [0])
    put_payload(dst, payload(src))
    assert torch.equal(target, expected)
    assert torch.equal(target_backing[..., within:], before[..., within:])


def test_checkpoint_is_complete_not_proportional():
    backing = torch.full((3, 80), -1.0)
    conv = backing[:, :24].view(3, 3, 8)
    state = backing[:, 24:40].view(3, 4, 4)
    conv[1].fill_(2)
    state[1].fill_(3)
    cache = layout(state=True)
    cache.register_kv_caches([conv, state], 3)
    descriptor = cache.prepare_values([(0, 4)], [1], [0])
    assert sum(descriptor[1][0]) == 40 * 4
    assert payload(descriptor) == conv[1].numpy().tobytes() + state[1].numpy().tobytes()


def test_core_chunk_and_boundary_objects_share_payload_schema():
    worker = object.__new__(AscendHashStoreWorker)
    worker.hash_block_size, worker.tp_rank = 4, 0
    worker._group_tp_replication_factors = [1]
    worker._kv_cache_groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    database = worker._build_token_databases(KeyMetadata("test", 0, 0, 0, 0), None)[0]
    # The database must retain Core's physical chunk size. Hash granularity
    # belongs to keys/coordinator, not a second private object protocol.
    assert getattr(database, "chunk_size", getattr(database, "block_size", None)) == 16
    assert database.hash_block_size == 4
    chunks = list(database.process_tokens(20, [b"a", b"b", b"c", b"d", b"e"]))
    assert [(start, end) for start, end, _ in chunks] == [(0, 16), (16, 20)]
    views = [torch.arange(3 * 16 * 2 * 3).view(3, 16, 2, 3)]
    database.store_layout.register_kv_caches(views, 3)
    descriptor = database.store_layout.prepare_values([(16, 20)], [1, 2], [0])
    address, size = database.store_layout.prepare_value_for_block(2)
    assert descriptor[:2] == ([address], [size])
    assert payload(descriptor) == views[0][2].numpy().tobytes()


def test_bound_groups_ignore_legacy_flattened_representatives():
    attention = torch.arange(3 * 16 * 2 * 3).reshape(3, 16, 2, 3)
    conv, state = torch.ones(3, 3, 8), torch.full((3, 4, 4), 7.0)
    attn_layout, state_layout = layout(), layout(state=True)
    attn_layout.bind_group_views([attention])
    state_layout.bind_group_views([conv, state])
    # The legacy Core path mixes group types and omits the second Mamba
    # component. Each layout must keep its bound group, never this list.
    for cache in (attn_layout, state_layout):
        cache.register_kv_caches([attention, conv], 3)
    assert payload(attn_layout.prepare_values([(4, 12)], [1], [0])) == attention[1].numpy().tobytes()
    assert payload(state_layout.prepare_values([(0, 4)], [1], [0])) == (
        conv[1].numpy().tobytes() + state[1].numpy().tobytes()
    )


@pytest.mark.parametrize("state", [False, True])
def test_core_boundary_block_entrypoint_has_complete_nonempty_payload(state):
    # Core's boundary puts/get path bypasses prepare_values. Inheriting the
    # base implementation would return empty descriptors: this layout keeps
    # logical views rather than base._base_addrs / base._block_lens.
    backing = torch.full((3, 160), -7.0)
    views = (
        [backing[:, :24].view(3, 3, 8), backing[:, 24:40].view(3, 4, 4)]
        if state
        else [backing[:, :96].view(3, 16, 2, 3)]
    )
    for index, view in enumerate(views):
        view.copy_(torch.arange(view.numel()).view(view.shape) + index)
    cache = layout(state=state)
    cache.register_kv_caches(views, 3)
    addresses, sizes = cache.prepare_value_for_block(1)
    assert sizes and all(size > 0 for size in sizes)
    assert payload(([addresses], [sizes], [1])) == b"".join(view[1].numpy().tobytes() for view in views)
    assert sum(sizes) == sum(view[1].numel() * view.element_size() for view in views)
    for invalid in (-1, 0, 3):
        with pytest.raises(ValueError, match="null checkpoint"):
            cache.prepare_value_for_block(invalid)


@pytest.mark.parametrize("hnd", [False, True])
def test_ungrouped_attention_with_token_padding(hnd):
    shape = (3, 2, 16, 3) if hnd else (3, 16, 2, 3)
    strides = (256, 80, 5, 1) if hnd else (256, 8, 3, 1)
    backing = torch.full((3 * 256,), -1.0)
    view = torch.as_strided(backing, shape, strides)
    view.copy_(torch.arange(view.numel()).reshape(shape))
    cache = layout(hnd=hnd)
    cache.register_kv_caches([view], 3)
    descriptor = cache.prepare_values([(4, 12)], [1], [0])
    expected = view[1]
    assert payload(descriptor) == expected.contiguous().numpy().tobytes()


@pytest.mark.parametrize(
    "chunks,blocks,shards",
    [([(1, 4)], [1], [0]), ([(12, 20)], [1, 2], [0]), ([(0, 4)], [0], [0]), ([(0, 4)], [3], [0]), ([(0, 4)], [1], [1])],
)
def test_invalid_transfer_rejected(chunks, blocks, shards):
    cache = layout()
    cache.register_kv_caches([torch.zeros(3, 16, 2, 3)], 3)
    with pytest.raises(ValueError):
        cache.prepare_values(chunks, blocks, shards)


def test_overlapping_view_rejected():
    cache = layout()
    with pytest.raises(ValueError, match="Overlapping"):
        cache.register_kv_caches([torch.as_strided(torch.zeros(1000), (3, 16, 2, 3), (1, 6, 3, 1))], 3)


def test_aligned_registration_stays_inside_allocation():
    store = Mock()
    store.register_buffer.return_value = 0
    storage = SimpleNamespace(data_ptr=lambda: 1003, nbytes=lambda: 400)
    view = SimpleNamespace(
        untyped_storage=lambda: storage,
        numel=lambda: 10,
        data_ptr=lambda: 1040,
        shape=(2, 5),
        stride=lambda: (20, 1),
        element_size=lambda: 4,
    )
    wrapper = _AlignedStore(store, [view, view], alignment=16)
    wrapper.register_buffer(1003, 400)
    wrapper.register_buffer(1003, 400)
    store.register_buffer.assert_called_once_with(1008, 395)
    wrapper.unregister_buffer(1003)
    store.unregister_buffer.assert_called_once_with(1008)
    view.data_ptr = lambda: 1004
    with pytest.raises(ValueError, match="aligned registration"):
        _AlignedStore(store, [view], alignment=16)


def test_registration_failure_is_fatal():
    store = Mock()
    store.register_buffer.return_value = -1
    view = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 16, nbytes=lambda: 32),
        numel=lambda: 1,
        data_ptr=lambda: 16,
        shape=(1,),
        stride=lambda: (1,),
        element_size=lambda: 4,
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        _AlignedStore(store, [view], alignment=16).register_buffer(16, 32)


def test_mrv2_registration_uses_core_capacity_without_mutating_config():
    worker = object.__new__(AscendHashStoreWorker)
    worker._capacity_only = False
    original = worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    worker._kv_cache_config = SimpleNamespace(num_blocks=3, hisparse_host_num_blocks=None)
    worker._kv_cache_groups = []
    worker.token_dbs = []
    worker.store = Mock()
    with patch.object(MooncakeStoreWorker, "register_kv_caches") as register:
        worker.register_kv_caches({})
        register.assert_called_once_with({})
    assert worker.cache_config.num_gpu_blocks == 3
    assert original.num_gpu_blocks is None
    worker.cache_config.num_gpu_blocks = 4
    with pytest.raises(ValueError, match="capacity disagrees"):
        worker.register_kv_caches({})


def test_npu_event_shared_by_all_save_jobs():
    worker = object.__new__(AscendHashStoreWorker)
    worker._capacity_only, worker.can_put = False, True
    worker.kv_send_thread = Mock()
    requests = [SimpleNamespace(can_save=value) for value in (True, False, True)]
    with patch.object(torch, "npu", create=True) as npu:
        worker.wait_for_save(SimpleNamespace(requests=requests))
        npu.Event.assert_called_once()
        npu.Event.return_value.record.assert_called_once()
    assert worker.kv_send_thread.add_request.call_count == 2
    assert requests[0].current_event is requests[2].current_event


def test_worker_binds_all_components_before_upstream_registration():
    worker = object.__new__(AscendHashStoreWorker)
    worker._capacity_only = False
    worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    worker._kv_cache_config = SimpleNamespace(num_blocks=3, hisparse_host_num_blocks=None)
    worker._kv_cache_groups = [SimpleNamespace(layer_names=["attention"]), SimpleNamespace(layer_names=["mamba"])]
    attn_layout, state_layout = layout(), layout(state=True)
    worker.token_dbs = [SimpleNamespace(store_layout=attn_layout), SimpleNamespace(store_layout=state_layout)]
    worker.store = Mock()
    attention = (torch.ones(3, 16, 2, 3), torch.full((3, 16, 2, 3), 2.0))
    state = [torch.full((3, 3, 8), 3.0), torch.full((3, 4, 4), 4.0)]

    def legacy_registration(caches):
        for database in worker.token_dbs:
            database.store_layout.register_kv_caches([attention[0], state[0]], 3)

    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_connector._AlignedStore"),
        patch.object(MooncakeStoreWorker, "register_kv_caches", side_effect=legacy_registration),
    ):
        worker.register_kv_caches({"attention": attention, "mamba": state})
    assert len(attn_layout._views) == len(state_layout._views) == 2
    assert payload(state_layout.prepare_values([(0, 4)], [1], [0])) == (
        state[0][1].numpy().tobytes() + state[1][1].numpy().tobytes()
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"backend": "memcache"},
        {"use_layerwise": True},
        {"load_async": False},
        {"consumer_is_to_put": True},
        {"store_tp_size": 2},
        {"enable_store_tp_lcm": True},
        {"store_chunk_size": 16},
    ],
)
def test_hash_mode_rejects_unsupported_configuration(extra):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"pool_prefix_mode": "hash", **extra})
    )
    with pytest.raises(ValueError, match="rank-local Mooncake"):
        AscendStoreConnector(config, None)


def test_non_hash_entrypoint_rejected():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"pool_prefix_mode": "block_only"})
    )
    with pytest.raises(ValueError, match="original module"):
        AscendStoreConnector(config, None)


def test_uncovered_registration_span_rejected():
    view = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 16, nbytes=lambda: 32),
        numel=lambda: 9,
        data_ptr=lambda: 16,
        shape=(9,),
        stride=lambda: (1,),
        element_size=lambda: 4,
    )
    with pytest.raises(ValueError, match="exceeds its storage"):
        _AlignedStore(Mock(), [view], alignment=16)
