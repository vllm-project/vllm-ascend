from types import SimpleNamespace

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


def test_mte_01_active_page_ordinals_are_fixed_shape():
    pages = _make_active_pages()

    assert pages.physical_page_ids.tolist() == [2, 2, 7, 10]
    assert pages.valid_page_mask.tolist() == [True, False, True, False]
    assert pages.staging_page_indices.tolist() == [0, 0, 1, 1]


def test_mte_02_bundle_staging_regions_are_disjoint():
    transport = _make_transport(0)
    _install_transfer_regions(transport)
    transport._staging_regions_by_rank = [
        MTEStagingRegion(8000, 320, 0),
        MTEStagingRegion(9000, 320, 1),
    ]

    offsets_by_bundle = transport._build_staging_region_offsets(
        (("main", "indexer"),),
        max_active_pages=10,
    )

    # Ten active-page slots: main's regions occupy [0, 240), while the
    # indexer occupies [240, 320).
    assert offsets_by_bundle[("main", "indexer")] == {"main": (0, 160), "indexer": (240,)}

    transport._staging_region_offsets_by_cache_bundle = offsets_by_bundle
    plan = transport._build_device_transfer_plans((("main", "indexer"),))[("main", "indexer")]
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
    owner._staging_region_offsets_by_cache_bundle = owner._build_staging_region_offsets((("main", "indexer"),), 10)
    owner._device_transfer_plans_by_cache_bundle = owner._build_device_transfer_plans((("main", "indexer"),))

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
    consumer._staging_region_offsets_by_cache_bundle = consumer._build_staging_region_offsets(
        (("main", "indexer"),), 10
    )
    consumer._device_transfer_plans_by_cache_bundle = consumer._build_device_transfer_plans((("main", "indexer"),))

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
