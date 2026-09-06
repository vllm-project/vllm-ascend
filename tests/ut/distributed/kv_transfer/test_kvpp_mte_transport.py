from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    MemFabricMTEKVPPTransport,
    MTEStagingRegion,
    _MTETransferRegion,
)


def _make_active_pages() -> KVPPActivePages:
    return KVPPActivePages(
        torch.tensor([2, 2, 7, 10], dtype=torch.int64),
        torch.tensor([True, False, True, False]),
        torch.tensor([0, 0, 1, 1], dtype=torch.int64),
    )


def _make_transport(group_rank: int, copy_pages_op=None):
    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=group_rank, world_size=2),
        10,
        copy_pages_op=copy_pages_op or (lambda *args: None),
    )
    transport._staging_shm_id = 31
    return transport


def _install_transfer_regions(transport):
    transport._transfer_regions_by_cache = {
        "main": (
            _MTETransferRegion(base_tensor=torch.empty(1), page_stride_bytes=32, page_length_bytes=16),
            _MTETransferRegion(base_tensor=torch.empty(1), page_stride_bytes=64, page_length_bytes=8),
        ),
        "indexer": (_MTETransferRegion(base_tensor=torch.empty(1), page_stride_bytes=16, page_length_bytes=8),),
    }


def test_staging_automatically_covers_all_physical_pages(monkeypatch):
    transport = _make_transport(0)
    transport._num_physical_pages = 100000
    transport._kvpp_group.cpu_group = object()
    _install_transfer_regions(transport)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda *_args, **_kwargs: None)
    # One complete bundle, not all layers and not limited by active requests.
    plans, required_bytes = transport._build_device_transfer_plans((("main", "indexer"), ("main",)), 4)
    transport._device_transfer_plans_by_cache_bundle = plans
    transport._check_staging_layout(required_bytes, required_bytes, 4)
    assert required_bytes == 4 << 20
    assert plans[("main", "indexer")].staging_region_offsets.tolist() == [0, 64, 96]


@pytest.mark.parametrize("peer_layout", [None, (4 << 20, 2 << 20), (2 << 20, 4 << 20)])
def test_invalid_staging_layout_never_allocates_shm(monkeypatch, peer_layout):
    transport = _make_transport(0)
    transport._kvpp_group.cpu_group = object()
    transport._kvpp_group.ranks = [0, 1]
    transport._shared_memory_backend = object()
    transport._planned_staging_capacity_bytes = 100 if peer_layout is None else 2 << 20
    _install_transfer_regions(transport)
    monkeypatch.setenv("MF_CONFIG_STORE_URL", "tcp://127.0.0.1:18291")
    monkeypatch.setattr(transport, "_build_transfer_regions", lambda _caches: transport._transfer_regions_by_cache)

    def gather(layouts, local, **_kwargs):
        if peer_layout is not None:
            required, capacity = peer_layout
            layouts[1] = (capacity, required, *local[2:])

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(
        transport, "_create_staging_memory", lambda *_args: pytest.fail("SHM allocated before validation")
    )
    with pytest.raises(ValueError, match="does not match"):
        transport.initialize_transport({}, (("main", "indexer"),), 10)


def test_staging_rejects_insufficient_plan_before_shm_allocation(monkeypatch):
    transport = _make_transport(0)
    transport._kvpp_group.cpu_group = object()
    transport._planned_staging_capacity_bytes = 100
    _install_transfer_regions(transport)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda *_args, **_kwargs: None)
    _, required_bytes = transport._build_device_transfer_plans((("main", "indexer"),), 10)
    with pytest.raises(ValueError, match="does not match"):
        transport._check_staging_layout(required_bytes, 100, 10)


def test_staging_rejects_asymmetric_rank_budgets(monkeypatch):
    transport = _make_transport(0)
    transport._kvpp_group.cpu_group = object()
    _install_transfer_regions(transport)

    def gather(layouts, local, **_kwargs):
        layouts[1] = (4 << 20, 2 << 20, *local[2:])

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    _, required_bytes = transport._build_device_transfer_plans((("main", "indexer"),), 10)
    with pytest.raises(ValueError, match="does not match"):
        transport._check_staging_layout(required_bytes, 2 << 20, 10)


@pytest.mark.parametrize("mismatch", ["lengths", "dtype", "page_count", "slots", "names"])
def test_equal_staging_capacity_does_not_hide_wire_layout_mismatch(monkeypatch, mismatch):
    transport = _make_transport(0)
    transport._kvpp_group.cpu_group = object()
    _install_transfer_regions(transport)
    transport._device_transfer_plans_by_cache_bundle, required = transport._build_device_transfer_plans(
        (("main", "indexer"),), 10
    )

    def gather(layouts, local, **_kwargs):
        peer = list(local)
        if mismatch == "page_count":
            peer[2] += 1
        elif mismatch == "slots":
            peer[3] -= 1
        else:
            names, regions = local[4][0]
            if mismatch == "lengths":
                # Same total payload, but indexer starts at the wrong offset.
                regions = ((8, torch.float32), (8, torch.float32), (16, torch.float32))
            elif mismatch == "dtype":
                regions = tuple((length, torch.int32) for length, _dtype in regions)
            else:
                names = tuple(reversed(names))
            peer[4] = ((names, regions),)
        layouts[1] = tuple(peer)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="wire layout"):
        transport._check_staging_layout(required, required, 10)


def test_rank_local_tensor_strides_need_not_match(monkeypatch):
    transports = [_make_transport(rank) for rank in range(2)]
    signatures = []
    for rank, transport in enumerate(transports):
        transport._kvpp_group.cpu_group = object()
        _install_transfer_regions(transport)
        if rank == 1:
            transport._transfer_regions_by_cache["main"] = tuple(
                _MTETransferRegion(torch.empty(1), region.page_stride_bytes * 2, region.page_length_bytes)
                for region in transport._transfer_regions_by_cache["main"]
            )
        transport._device_transfer_plans_by_cache_bundle, required = transport._build_device_transfer_plans(
            (("main", "indexer"),), 10
        )
        monkeypatch.setattr(
            torch.distributed, "all_gather_object", lambda _out, local, **_kwargs: signatures.append(local)
        )
        transport._check_staging_layout(required, required, 10)
    assert signatures[0] == signatures[1]


def test_mte_01_active_page_ordinals_are_fixed_shape():
    pages = _make_active_pages()

    assert pages.physical_page_ids.tolist() == [2, 2, 7, 10]
    assert pages.valid_page_mask.tolist() == [True, False, True, False]
    assert pages.staging_page_indices.tolist() == [0, 0, 1, 1]


def test_mte_02_bundle_staging_regions_are_disjoint():
    transport = _make_transport(0)
    _install_transfer_regions(transport)
    plans, required_bytes = transport._build_device_transfer_plans(
        (("main", "indexer"), ("main",), ("main", "indexer")),
        max_active_pages=10,
    )

    # Ten active-page slots: main's regions occupy [0, 240), while the
    # indexer occupies [240, 320).
    assert len(plans) == 2
    assert required_bytes == 2 << 20
    assert plans[("main",)].staging_region_offsets.tolist() == [0, 160]
    plan = plans[("main", "indexer")]
    assert plan.page_strides.tolist() == [32, 64, 16]
    assert plan.page_lengths.tolist() == [16, 8, 8]
    assert plan.staging_region_offsets.tolist() == [0, 160, 240]


def test_mte_03_owner_pushes_to_peer_and_consumer_receives_locally(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    calls = []

    def record_copy_pages(
        anchor,
        local_offsets,
        staging_offsets,
        lengths,
        staging_base,
        source_rank,
        destination_rank,
        shm_id,
    ):
        calls.append(
            (
                local_offsets.tolist(),
                staging_offsets.tolist(),
                lengths.tolist(),
                source_rank,
                destination_rank,
                staging_base,
            )
        )

    owner = _make_transport(0, copy_pages_op=record_copy_pages)
    _install_transfer_regions(owner)
    owner._local_staging_region = MTEStagingRegion(8000, 320, 0)
    owner._staging_regions_by_rank = [
        owner._local_staging_region,
        MTEStagingRegion(9000, 320, 1),
    ]
    owner._device_transfer_plans_by_cache_bundle, _ = owner._build_device_transfer_plans((("main", "indexer"),), 10)

    owner.copy_active_pages_to_staging(("main", "indexer"), _make_active_pages(), SimpleNamespace())
    owner_plan = owner._device_transfer_plans_by_cache_bundle[("main", "indexer")]
    base_offsets = owner_plan.local_base_offsets.tolist()
    assert calls == [
        (
            [
                base_offsets[0] + 64,
                base_offsets[0] + 64,
                base_offsets[0] + 224,
                base_offsets[0] + 320,
                base_offsets[1] + 128,
                base_offsets[1] + 128,
                base_offsets[1] + 448,
                base_offsets[1] + 640,
                base_offsets[2] + 32,
                base_offsets[2] + 32,
                base_offsets[2] + 112,
                base_offsets[2] + 160,
            ],
            [0, 0, 16, 16, 160, 160, 168, 168, 240, 240, 248, 248],
            [16, 0, 16, 0, 8, 0, 8, 0, 8, 0, 8, 0],
            -1,
            1,
            9000,
        )
    ]

    calls.clear()
    consumer = _make_transport(1, copy_pages_op=record_copy_pages)
    _install_transfer_regions(consumer)
    consumer._local_staging_region = MTEStagingRegion(9000, 320, 1)
    consumer._staging_regions_by_rank = [
        MTEStagingRegion(8000, 320, 0),
        consumer._local_staging_region,
    ]
    consumer._device_transfer_plans_by_cache_bundle, _ = consumer._build_device_transfer_plans(
        (("main", "indexer"),), 10
    )

    consumer.copy_active_pages_from_staging(("main", "indexer"), _make_active_pages(), SimpleNamespace())
    consumer_plan = consumer._device_transfer_plans_by_cache_bundle[("main", "indexer")]
    base_offsets = consumer_plan.local_base_offsets.tolist()
    assert calls == [
        (
            [
                base_offsets[0] + 64,
                base_offsets[0] + 64,
                base_offsets[0] + 224,
                base_offsets[0] + 320,
                base_offsets[1] + 128,
                base_offsets[1] + 128,
                base_offsets[1] + 448,
                base_offsets[1] + 640,
                base_offsets[2] + 32,
                base_offsets[2] + 32,
                base_offsets[2] + 112,
                base_offsets[2] + 160,
            ],
            [0, 0, 16, 16, 160, 160, 168, 168, 240, 240, 248, 248],
            [16, 0, 16, 0, 8, 0, 8, 0, 8, 0, 8, 0],
            1,
            -1,
            9000,
        )
    ]
