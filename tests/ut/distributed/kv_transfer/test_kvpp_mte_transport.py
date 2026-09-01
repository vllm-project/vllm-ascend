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
    return MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=group_rank, world_size=2),
        10,
        copy_pages_op=copy_pages_op or (lambda *args: None),
    )


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


def test_mte_03_owner_pushes_to_peer_and_consumer_receives_locally(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    calls = []

    def record_copy_pages(
        base_tensor,
        physical_page_ids,
        valid_page_mask,
        staging_page_indices,
        page_stride_bytes,
        page_length_bytes,
        staging_region_offset_bytes,
        staging_base_address,
        staging_is_source,
        staging_group_rank,
        shm_id,
    ):
        calls.append(
            (
                page_stride_bytes,
                page_length_bytes,
                staging_region_offset_bytes,
                staging_is_source,
                staging_group_rank,
                staging_base_address,
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

    owner.copy_active_pages_to_staging(("main", "indexer"), _make_active_pages(), SimpleNamespace())
    assert calls == [
        (32, 16, 0, False, 1, 9000),
        (64, 8, 160, False, 1, 9000),
        (16, 8, 240, False, 1, 9000),
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

    consumer.copy_active_pages_from_staging(("main", "indexer"), _make_active_pages(), SimpleNamespace())
    assert calls == [
        (32, 16, 0, True, 1, 9000),
        (64, 8, 160, True, 1, 9000),
        (16, 8, 240, True, 1, 9000),
    ]
