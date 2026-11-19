# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owner-sharded PCP KV placement and bounded staging contracts.

These tests are transport-agnostic: they use per-owner in-process shards in
place of the VMM peer reader and compare the staged local view against the
replicated cache the consumer used before M2.
"""

import pytest
import torch

from vllm_ascend.attention.context_parallel import pcp_owner_kv as owner_kv


def _replicated_history(num_pages: int, page_rows: int, width: int, *, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_pages, page_rows, width, generator=generator)


def _owner_shards(history: torch.Tensor, layout: owner_kv.PcpOwnerLayout) -> list[torch.Tensor]:
    pages = torch.arange(history.shape[0])
    shards = []
    for rank in range(layout.pcp_size):
        owned = pages[layout.owner_of(pages) == rank]
        # Owner shards are ordered by the owner's compact local page index.
        order = torch.argsort(layout.local_page(owned))
        shard = history.new_zeros((layout.num_local_pages, *history.shape[1:]))
        shard[: owned.numel()] = history[owned[order]]
        shards.append(shard)
    return shards


def _owned_page_count(history: torch.Tensor, layout: owner_kv.PcpOwnerLayout, rank: int) -> int:
    pages = torch.arange(history.shape[0])
    return int((layout.owner_of(pages) == rank).sum())


@pytest.mark.parametrize("pcp_size", [2, 4, 16])
@pytest.mark.parametrize("num_pages", [17, 64, 256])
def test_owner_layout_round_trips_every_page(pcp_size: int, num_pages: int) -> None:
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=pcp_size)
    pages = torch.arange(num_pages)
    owners = layout.owner_of(pages)
    local = layout.local_page(pages)
    assert int(owners.min()) >= 0 and int(owners.max()) < pcp_size
    for rank in range(pcp_size):
        owned = pages[owners == rank]
        assert torch.equal(local[owners == rank], torch.arange(owned.numel()))
        assert layout.owned_mask(owned, rank).all()
    # Every rank's local page table covers its shard exactly once.
    assert layout.num_local_pages == (num_pages + pcp_size - 1) // pcp_size


@pytest.mark.parametrize("pcp_size", [2, 4])
@pytest.mark.parametrize("num_pages", [17, 64])
def test_contiguous_placement_round_trips(pcp_size: int, num_pages: int) -> None:
    layout = owner_kv.PcpOwnerLayout(
        num_pages=num_pages, pcp_size=pcp_size, placement=owner_kv.OWNER_PLACEMENT_CONTIGUOUS
    )
    pages = torch.arange(num_pages)
    for rank in range(pcp_size):
        owned = pages[layout.owner_of(pages) == rank]
        if owned.numel():
            assert torch.equal(layout.local_page(owned), torch.arange(owned.numel()))


@pytest.mark.parametrize("pcp_size", [2, 4])
@pytest.mark.parametrize("num_pages", [125, 129])
def test_chunk_placement_matches_the_mrv2_owner_map(pcp_size: int, num_pages: int) -> None:
    """Rank r owns chunk r and chunk 2P-1-r; pages never land on two ranks."""
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=pcp_size, placement=owner_kv.OWNER_PLACEMENT_CHUNK)
    chunk_pages = layout.resolved_chunk_pages
    pages = torch.arange(num_pages)
    owners = layout.owner_of(pages)

    expected = []
    for page in range(num_pages):
        chunk = page // chunk_pages
        expected.append(chunk if chunk < pcp_size else 2 * pcp_size - 1 - chunk)
    assert owners.tolist() == expected

    covered = torch.zeros(num_pages, dtype=torch.int64)
    for rank in range(pcp_size):
        owned = pages[layout.owned_mask(pages, rank)]
        local = layout.local_page(owned)
        assert owned.numel() > 0
        assert int(local.min()) >= 0
        assert int(local.max()) < layout.local_page_count(rank)
        torch.testing.assert_close(layout.logical_page(rank, local), owned)
        covered[owned] += 1
        # The returned compact index is a permutation of 0..count-1.
        assert sorted(local.tolist()) == list(range(layout.local_page_count(rank)))
    assert bool((covered == 1).all())


def test_chunk_placement_rounds_unaligned_query_len_up_to_a_page() -> None:
    # 33,000 tokens / 8 chunks = 4,125 tokens, rounded to 17 pages of 256.
    layout = owner_kv.PcpOwnerLayout.for_pcp_request(
        num_query_tokens=33000,
        tokens_per_page=256,
        pcp_size=4,
    )
    assert layout.placement == owner_kv.OWNER_PLACEMENT_CHUNK
    assert layout.replicated is False
    assert layout.num_pages == 129
    assert layout.resolved_chunk_pages == 17
    # 17 pages per chunk * 8 chunks covers every page and never splits one.
    assert layout.chunk_count * layout.resolved_chunk_pages >= layout.num_pages
    pages = torch.tensor([0, 16, 17, 33, 34, 118, 119, 128])
    assert layout.owner_of(pages).tolist() == [0, 0, 1, 1, 2, 1, 0, 0]


def test_short_prefill_and_decode_stay_replicated() -> None:
    layout = owner_kv.PcpOwnerLayout.for_pcp_request(
        num_query_tokens=64,
        tokens_per_page=256,
        pcp_size=4,
        replicated=True,
    )
    assert layout.replicated is True
    pages = torch.tensor([0, 1, 2])
    assert layout.owner_of(pages).tolist() == [-1, -1, -1]
    assert not bool(layout.owned_mask(pages, 0).any())
    assert layout.local_page_count(0) == layout.num_pages
    with pytest.raises(ValueError, match="replicated request"):
        layout.local_page(pages)
    with pytest.raises(ValueError, match="replicated request"):
        layout.logical_page(0, torch.tensor([0]))


def test_mixed_batch_keeps_chunked_and_replicated_requests_separate() -> None:
    """A batch may mix both kinds; each request applies its own layout."""
    chunked = owner_kv.PcpOwnerLayout.for_pcp_request(
        num_query_tokens=32768,
        tokens_per_page=256,
        pcp_size=4,
    )
    replicated = owner_kv.PcpOwnerLayout.for_pcp_request(
        num_query_tokens=64,
        tokens_per_page=256,
        pcp_size=4,
        replicated=True,
    )
    # Page ids are per request: apply each layout to that request's slice.
    assert chunked.owner_of(torch.tensor([0, 16, 127])).tolist() == [0, 1, 0]
    assert replicated.owner_of(torch.tensor([0])).tolist() == [-1]
    assert not bool(replicated.owned_mask(torch.tensor([0]), 0).any())
    assert bool(chunked.owned_mask(torch.arange(chunked.num_pages), 3).any())


@pytest.mark.parametrize("pcp_size", [2, 4])
def test_vmm_reader_reads_chunk_owned_pages(pcp_size: int) -> None:
    layout = owner_kv.PcpOwnerLayout(
        num_pages=125,
        pcp_size=pcp_size,
        placement=owner_kv.OWNER_PLACEMENT_CHUNK,
    )
    history = _replicated_history(125, 2, 3)
    reader = owner_kv.VmmPeerPageReader(history, layout)
    for rank in range(pcp_size):
        count = layout.local_page_count(rank)
        local = torch.arange(count)
        expected = layout.logical_page(rank, local)
        destination = torch.empty(count, 2, 3)
        reader.copy_pages(rank, local, destination)
        torch.testing.assert_close(destination, history.index_select(0, expected))


@pytest.mark.parametrize("pcp_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [8192, 32768])
def test_selective_page_materialization_matches_replicated(pcp_size: int, num_tokens: int) -> None:
    """The consumer sees exactly the replicated pages it selects.

    Page geometry follows the V4.1 C2 long-KV plane at a small width: 128
    source tokens per storage block and two source blocks per compressed page.
    """
    block_size = 128
    compressor_ratio = 2
    rows_per_page = block_size // compressor_ratio
    num_pages = (num_tokens + block_size - 1) // block_size // compressor_ratio
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=pcp_size)
    history = _replicated_history(num_pages, rows_per_page, 8)
    shards = _owner_shards(history, layout)
    reader = owner_kv.TensorPeerPageReader(shards)

    # Sparse attention selects a bounded top-k history, not every page.
    selected = torch.arange(0, num_pages, max(1, num_pages // 7))[:7].to(torch.int64)
    capacity = selected.numel() + 2
    for rank in range(pcp_size):
        stage = owner_kv.BoundedStageCache(page_shape=(rows_per_page, 8), capacity_pages=capacity, dtype=history.dtype)
        cache = owner_kv.PcpOwnerShardCache(
            persistent_cache=shards[rank],
            stage_cache=stage,
            layout=layout,
            rank=rank,
        )
        block_table = torch.tensor([[0, 1, -1], [selected[1].item(), -1, -1]], dtype=torch.int32)
        required = torch.cat((torch.tensor([0, 1]), selected))
        view = cache.materialize_for_attention(required, block_table=block_table, reader=reader)
        expected = history.index_select(0, view.staged_pages)
        torch.testing.assert_close(view.cache, expected)
        # The block table now addresses the compact stage instead of the full
        # logical cache, and each original page keeps its payload.
        remapped = owner_kv.remap_block_table_to_stage(block_table, view.staged_pages)
        assert bool((remapped >= 0)[block_table >= 0].all())
        torch.testing.assert_close(
            view.cache[remapped[block_table >= 0].long()],
            history[block_table[block_table >= 0].long()],
        )


@pytest.mark.parametrize("pcp_size", [2, 4])
def test_scatter_owned_matches_replicated_and_saves_capacity(pcp_size: int) -> None:
    num_pages = 33
    rows_per_page = 4
    width = 3
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=pcp_size)
    history = _replicated_history(num_pages, rows_per_page, width)
    shards = _owner_shards(history, layout)

    # Write the complete replicated history through the owner-shard path.
    pages = torch.arange(num_pages).repeat_interleave(rows_per_page)
    offsets = torch.arange(rows_per_page).repeat(num_pages)
    slot_mapping = torch.stack((pages, offsets), dim=1)
    values = history.flatten(0, 1)
    stage = owner_kv.BoundedStageCache(page_shape=(rows_per_page, width), capacity_pages=8, dtype=history.dtype)
    for rank in range(pcp_size):
        cache = owner_kv.PcpOwnerShardCache(
            persistent_cache=torch.zeros_like(shards[rank]),
            stage_cache=stage,
            layout=layout,
            rank=rank,
        )
        owned_count = _owned_page_count(history, layout, rank)
        written = cache.scatter_owned(slot_mapping, values)
        assert written == owned_count * rows_per_page
        torch.testing.assert_close(
            cache.persistent_cache[:owned_count],
            history[layout.owner_of(torch.arange(num_pages)) == rank],
        )

    # Persistent bytes are 1/pcp of the replicated cache (modulo rounding).
    replicated_bytes = history.numel() * history.element_size()
    owner_bytes = sum(shard.numel() * shard.element_size() for shard in shards)
    # Each rank keeps ceil(N/P) pages instead of N.  Padding can make the
    # global sum one page-group larger than one replicated copy, never more.
    assert max(shard.numel() * shard.element_size() for shard in shards) <= replicated_bytes
    assert owner_bytes <= replicated_bytes + pcp_size * rows_per_page * width * history.element_size()
    assert owner_bytes >= replicated_bytes


@pytest.mark.parametrize("pcp_size", [2, 4])
def test_vmm_peer_page_reader_reads_owner_compact_pages(pcp_size: int) -> None:
    """A VMM reader resolves compact owner pages through the full VA alias."""
    num_pages = 24
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=pcp_size)
    history = _replicated_history(num_pages, 3, 5)
    reader = owner_kv.VmmPeerPageReader(history, layout)

    for owner in range(pcp_size):
        plan = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=pcp_size)
        compact = torch.arange(plan.num_local_pages)
        logical = compact * pcp_size + owner
        keep = logical < num_pages
        compact = compact[keep]
        destination = torch.empty(compact.numel(), 3, 5)
        reader.copy_pages(owner, compact, destination)
        torch.testing.assert_close(destination, history.index_select(0, compact * pcp_size + owner))


def test_vmm_peer_page_reader_writes_owner_pages_in_place() -> None:
    num_pages = 9
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=3)
    history = _replicated_history(num_pages, 2, 4)
    reader = owner_kv.VmmPeerPageReader(history, layout)

    source = torch.full((3, 2, 4), -7.0)
    reader.write_pages(2, torch.tensor([0, 1, 2]), source)
    torch.testing.assert_close(history[2], source[0])
    torch.testing.assert_close(history[5], source[1])
    torch.testing.assert_close(history[8], source[2])


def test_vmm_peer_page_reader_contiguous_placement() -> None:
    layout = owner_kv.PcpOwnerLayout(num_pages=10, pcp_size=4, placement=owner_kv.OWNER_PLACEMENT_CONTIGUOUS)
    history = _replicated_history(10, 2, 2)
    reader = owner_kv.VmmPeerPageReader(history, layout)
    destination = torch.empty(2, 2, 2)
    # Rank 1 owns pages 3, 4, 5 (ceil(10 / 4) == 3 local pages).
    reader.copy_pages(1, torch.tensor([0, 1]), destination)
    torch.testing.assert_close(destination, history[3:5])


def test_vmm_peer_page_reader_rejects_bad_geometry() -> None:
    layout = owner_kv.PcpOwnerLayout(num_pages=8, pcp_size=4)
    history = _replicated_history(4, 2, 2)
    with pytest.raises(ValueError, match="needs 8"):
        owner_kv.VmmPeerPageReader(history, layout)
    reader = owner_kv.VmmPeerPageReader(_replicated_history(8, 2, 2), layout)
    destination = torch.empty(2, 2, 2)
    with pytest.raises(ValueError, match="outside rank"):
        reader.copy_pages(0, torch.tensor([0, 2]), destination)
    with pytest.raises(ValueError, match="non-negative"):
        reader.copy_pages(0, torch.tensor([-1]), destination)
    with pytest.raises(ValueError, match="outside PCP size"):
        reader.copy_pages(4, torch.tensor([0]), destination)
    with pytest.raises(ValueError, match="page count mismatch"):
        reader.copy_pages(0, torch.tensor([0, 1]), torch.empty(3, 2, 2))


def test_vmm_peer_page_reader_does_not_issue_collectives(monkeypatch) -> None:
    """The request-path reader must never reach a transport or a collective."""
    import torch.distributed as dist

    layout = owner_kv.PcpOwnerLayout(num_pages=8, pcp_size=4)
    history = _replicated_history(8, 2, 2)
    reader = owner_kv.VmmPeerPageReader(history, layout)

    def forbidden(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("VMM page read attempted a collective")

    for name in ("all_gather", "all_gather_into_tensor", "all_to_all", "all_to_all_single", "all_reduce"):
        monkeypatch.setattr(dist, name, forbidden)
    destination = torch.empty(2, 2, 2)
    reader.copy_pages(1, torch.tensor([0, 1]), destination)
    torch.testing.assert_close(destination, history[[1, 5]])


def test_stage_overflow_is_an_error_not_a_silent_replica() -> None:
    layout = owner_kv.PcpOwnerLayout(num_pages=64, pcp_size=4)
    history = _replicated_history(64, 2, 2)
    shards = _owner_shards(history, layout)
    reader = owner_kv.TensorPeerPageReader(shards)
    stage = owner_kv.BoundedStageCache(page_shape=(2, 2), capacity_pages=3, dtype=history.dtype)
    with pytest.raises(RuntimeError, match="bounded scratch"):
        stage.materialize(torch.arange(5), layout=layout, rank=0, reader=reader)


def test_sparse_selection_stays_bounded_at_32k_context() -> None:
    """Only top-k pages plus current rows are staged, never the block table.

    A 32K context at block size 128 with compressor ratio 2 holds 125 long-KV
    pages.  Sparse attention selects eight of them, so the scratch never sees
    more than ten pages even though the block table references all 125.
    """
    block_size, compress_ratio = 128, 2
    num_tokens = 32 * 1024
    num_pages = num_tokens // block_size // compress_ratio
    layout = owner_kv.PcpOwnerLayout(num_pages=num_pages, pcp_size=4)
    history = _replicated_history(num_pages, 2, 4)
    shards = _owner_shards(history, layout)
    reader = owner_kv.TensorPeerPageReader(shards)

    selected = torch.tensor([3, 17, 33, 49, 65, 81, 97, 113], dtype=torch.int64)
    slot_mapping = torch.tensor([[7, 0], [7, 1]], dtype=torch.int64)
    required = owner_kv.stage_pages_from_sparse_selection(selected, slot_mapping)
    assert required.tolist() == [3, 7, 17, 33, 49, 65, 81, 97, 113]

    capacity = required.numel() + 2
    for rank in range(4):
        stage = owner_kv.BoundedStageCache(page_shape=(2, 4), capacity_pages=capacity, dtype=history.dtype)
        cache = owner_kv.PcpOwnerShardCache(shards[rank], stage, layout, rank)
        view = cache.materialize_for_attention(required, reader=reader)
        assert view.staged_pages.tolist() == required.tolist()
        assert stage.counters.peak_pages == required.numel()
        torch.testing.assert_close(view.cache, history.index_select(0, required))
        remapped = owner_kv.remap_indices_to_stage(selected.reshape(1, -1), view.staged_pages)
        assert remapped.tolist() == [[0, 2, 3, 4, 5, 6, 7, 8]]

        # The full block table is what would overflow, not the sparse set.
        with pytest.raises(RuntimeError, match="bounded scratch"):
            stage.materialize(
                torch.arange(num_pages),
                layout=layout,
                rank=rank,
                reader=reader,
            )


def test_sparse_selection_deduplicates_and_drops_padding() -> None:
    selected = torch.tensor([[5, -1, 2], [2, 5, 7]])
    required = owner_kv.stage_pages_from_sparse_selection(selected, torch.tensor([[7, 0], [-1, -1]]))
    assert required.tolist() == [2, 5, 7]
    with pytest.raises(ValueError, match="1-D or 2-D"):
        owner_kv.stage_pages_from_sparse_selection(torch.zeros(1, 1, 1, dtype=torch.int64))


def test_block_table_remap_rejects_an_unstaged_page() -> None:
    staged = torch.tensor([1, 4, 9])
    block_table = torch.tensor([[1, 4, 7]])
    with pytest.raises(ValueError, match="absent from the staged set"):
        owner_kv.remap_block_table_to_stage(block_table, staged)


def test_sparse_index_remap_keeps_padding_and_unselected_sentinels() -> None:
    staged = torch.tensor([1, 4, 9])
    indices = torch.tensor([[9, -1, 3, 1]])
    remapped = owner_kv.remap_indices_to_stage(indices, staged)
    assert remapped.tolist() == [[2, -1, -1, 0]]


def test_full_va_staging_reads_logical_pages_without_owner_split() -> None:
    """A full-VA slab stages local and peer pages with one logical gather."""
    num_pages, rows, width = 125, 2, 3
    layout = owner_kv.PcpOwnerLayout(
        num_pages=num_pages,
        pcp_size=4,
        placement=owner_kv.OWNER_PLACEMENT_CHUNK,
    )
    slab = _replicated_history(num_pages, rows, width)
    rank = 1
    stage = owner_kv.BoundedStageCache(page_shape=(rows, width), capacity_pages=8, dtype=slab.dtype)

    class ForbiddenReader:
        def copy_pages(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("full-VA staging must not use per-owner reads")

    cache = owner_kv.PcpOwnerShardCache(
        persistent_cache=slab,
        stage_cache=stage,
        layout=layout,
        rank=rank,
        full_va=True,
    )

    # Rank 1 owns chunk 1 (pages 16..31) and chunk 6 (pages 96..111).
    owned_pages = torch.tensor([16, 17, 20, 96])
    slot_mapping = torch.stack(
        (owned_pages.repeat_interleave(rows), torch.arange(rows).repeat(owned_pages.numel())),
        dim=1,
    )
    values = torch.full((slot_mapping.shape[0], width), 123.0)
    assert cache.scatter_owned(slot_mapping, values) == slot_mapping.shape[0]
    torch.testing.assert_close(slab[16], values[:rows])
    torch.testing.assert_close(slab[96], values[-rows:])

    # Mix peer pages (0, 40, 112) with local pages in one sparse selection.
    required = torch.tensor([0, 16, 40, 96, 112])
    view = cache.materialize_for_attention(required, reader=ForbiddenReader())
    assert view.staged_pages.tolist() == required.tolist()
    torch.testing.assert_close(view.cache, slab.index_select(0, required))
    assert view.counters.peer_pages_materialized == 3
    assert view.counters.local_pages_materialized == 2


def test_full_va_scatter_never_writes_a_peer_page() -> None:
    layout = owner_kv.PcpOwnerLayout(
        num_pages=32,
        pcp_size=4,
        placement=owner_kv.OWNER_PLACEMENT_CHUNK,
    )
    slab = torch.zeros(32, 2, 2)
    stage = owner_kv.BoundedStageCache(page_shape=(2, 2), capacity_pages=4, dtype=slab.dtype)
    cache = owner_kv.PcpOwnerShardCache(slab, stage, layout, rank=2, full_va=True)
    slot_mapping = torch.tensor([[0, 0], [8, 0], [23, 1], [31, 0]])
    values = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    # 32 pages / 8 chunks = 4 pages per chunk; rank 2 owns chunks 2 and 5.
    assert layout.resolved_chunk_pages == 4
    assert layout.owner_of(torch.tensor([8, 23])).tolist() == [2, 2]
    assert cache.scatter_owned(slot_mapping, values) == 2
    torch.testing.assert_close(slab[8, 0], values[1])
    torch.testing.assert_close(slab[23, 1], values[2])
    torch.testing.assert_close(slab[0], torch.zeros(2, 2))
    torch.testing.assert_close(slab[31], torch.zeros(2, 2))


def test_current_row_overlay_matches_replicated_rows() -> None:
    layout = owner_kv.PcpOwnerLayout(num_pages=16, pcp_size=4)
    history = _replicated_history(16, 4, 5)
    shards = _owner_shards(history, layout)
    reader = owner_kv.TensorPeerPageReader(shards)
    rank = 1
    stage = owner_kv.BoundedStageCache(page_shape=(4, 5), capacity_pages=8, dtype=history.dtype)
    cache = owner_kv.PcpOwnerShardCache(persistent_cache=shards[rank], stage_cache=stage, layout=layout, rank=rank)
    required = torch.tensor([2, 5, 11])
    view = cache.materialize_for_attention(required, reader=reader)

    current = torch.full((2, 5), 99.0)
    slot_mapping = torch.tensor([[5, 1], [11, 3]])
    overlaid = stage.overlay_current_rows(view.staged_pages, slot_mapping, current)
    assert overlaid == 2

    expected = history.index_select(0, view.staged_pages).clone()
    expected[1, 1] = current[0]
    expected[2, 3] = current[1]
    torch.testing.assert_close(view.cache, expected)


def test_slot_mapping_page_extraction_supports_both_layouts() -> None:
    flat = torch.tensor([0, 127, 128, 255, -1])
    assert owner_kv.page_ids_from_slot_mapping(flat, 128).tolist() == [0, 0, 1, 1, -1]
    two_d = torch.tensor([[0, 3], [2, 1], [-1, -1]])
    assert owner_kv.page_ids_from_slot_mapping(two_d, 128).tolist() == [0, 2, -1]


def test_owner_sharing_mode_flag_defaults_to_replicated(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_PCP_KV_SHARING", raising=False)
    assert owner_kv.get_pcp_kv_sharing_mode() == owner_kv.PCP_KV_SHARING_REPLICATED
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "owner")
    assert owner_kv.owner_sharding_enabled()
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "invalid")
    with pytest.raises(ValueError, match="must be one of"):
        owner_kv.get_pcp_kv_sharing_mode()


def _config(*, pcp=4, dcp=1, tp=1, kv_transfer=None, additional_config=None):
    return type(
        "Config",
        (),
        {
            "parallel_config": type(
                "Parallel",
                (),
                {
                    "prefill_context_parallel_size": pcp,
                    "decode_context_parallel_size": dcp,
                    "tensor_parallel_size": tp,
                },
            )(),
            "kv_transfer_config": kv_transfer,
            "additional_config": additional_config or {},
        },
    )()


def test_additional_config_takes_precedence_over_the_env_flag(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "replicated")
    config = _config(additional_config={"pcp_kv_sharing": "owner"})
    assert owner_kv.pcp_kv_sharing_mode_from_config(config) == owner_kv.PCP_KV_SHARING_OWNER
    assert owner_kv.owner_sharding_enabled(config)
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "owner")
    config = _config(additional_config={"pcp_kv_sharing": "replicated"})
    assert owner_kv.pcp_kv_sharing_mode_from_config(config) == owner_kv.PCP_KV_SHARING_REPLICATED


def test_validate_owner_sharing_rejects_incompatible_topologies() -> None:
    assert owner_kv.validate_owner_sharing(_config(pcp=1))[0] is False
    assert owner_kv.validate_owner_sharing(_config(dcp=2))[0] is False
    assert owner_kv.validate_owner_sharing(_config(tp=2))[0] is False
    assert owner_kv.validate_owner_sharing(_config(kv_transfer=object()))[0] is False
    assert owner_kv.validate_owner_sharing(_config())[0] is True


def test_owner_sharing_fails_closed_without_the_runner_runtime(monkeypatch) -> None:
    from vllm_ascend.attention.context_parallel import pcp_owner_vmm

    monkeypatch.setattr(owner_kv, "_owner_kv_runtime", None)
    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (False, "test backend absent"))
    supported, reason = owner_kv.resolve_owner_sharing(_config())
    assert supported is False
    assert "not installed" in reason
    marker = object()
    owner_kv.register_owner_kv_runtime(marker)
    try:
        assert owner_kv.resolve_owner_sharing(_config()) == (True, "")
    finally:
        owner_kv.register_owner_kv_runtime(None)


def test_owner_sharing_uses_the_device_backend_probe(monkeypatch) -> None:
    from vllm_ascend.attention.context_parallel import pcp_owner_vmm

    monkeypatch.setattr(owner_kv, "_owner_kv_runtime", None)
    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (True, ""))
    assert owner_kv.resolve_owner_sharing(_config()) == (True, "")
    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (False, "missing symbol"))
    assert owner_kv.resolve_owner_sharing(_config())[0] is False


def test_scatter_owned_never_writes_a_peer_page() -> None:
    layout = owner_kv.PcpOwnerLayout(num_pages=8, pcp_size=4)
    persistent = torch.zeros(2, 2, 2)
    stage = owner_kv.BoundedStageCache(page_shape=(2, 2), capacity_pages=4, dtype=persistent.dtype)
    cache = owner_kv.PcpOwnerShardCache(persistent, stage, layout, rank=1)
    slot_mapping = torch.tensor([[0, 0], [1, 0], [5, 0], [7, 0]])
    values = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    written = cache.scatter_owned(slot_mapping, values)
    assert written == 2  # pages 1 and 5 belong to rank 1
    torch.testing.assert_close(cache.persistent_cache[0, 0], values[1])
    torch.testing.assert_close(cache.persistent_cache[1, 0], values[2])
    assert torch.count_nonzero(cache.persistent_cache) == written * values[0].numel()
