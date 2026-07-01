"""Hardened correctness suite for the QUEST sparse-decode kernels.

Two custom operators are exercised here:

* ``npu_quest_prefill_metadata`` -- refreshes the per-page per-channel ``min``
  and ``max`` of the key cache (the QUEST page summaries), in place, for the
  *complete* 128-token pages inside a per-request refresh range.
* ``npu_quest_block_select_paged`` -- scores every page of a request with the
  QUEST upper bound ``sum_d max(q_d * max_d, q_d * min_d)`` and returns the
  top-k page indices, optionally force-including page 0 (attention sink) and the
  last page (recent tail) as fixed "anchors".

The suite is intentionally strict.  It pins the *intended contract* (the same
contract the CPU reference and the Python integration assume), so any test that
fails points at a concrete kernel defect rather than at a fuzzy tolerance.

Oracle strategy
---------------
Selection is only unique when page scores are distinct, and the NPU ``Sort``
tie-break is not guaranteed to match any CPU reference.  We therefore use two
complementary checks:

* **Exact** (``assert_selection_exact``) -- used only with *structured* inputs
  whose page scores are distinct and well separated, so the ranking is
  unambiguous.  Anchors are compared as an unordered pair (their relative order
  is implementation-defined); interior pages are compared in order.
* **Invariant** (``assert_selection_valid``) -- tie-agnostic, used for random
  fuzz.  It enforces validity, uniqueness, anchor presence and the fundamental
  top-k property ``min(score of selected) >= max(score of unselected)``.  This
  cannot false-fail on a legitimate tie-break choice.

Metadata is checked with ``rtol=0, atol=0``: every reduced value is an exact
element of the (already low-precision) key cache, so the result is bit-exact.

NOTE: these are NPU tests; they must run on Ascend A2/A3 hardware.
"""

import gc

import pytest
import torch
import torch_npu  # noqa: F401  (registers the ``npu`` device / ``torch.npu``)

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BLOCK_SIZE = 128
HEAD_DIM = 128
# A "metadata block" stores the summaries of this many pages.  It happens to
# equal BLOCK_SIZE in the current layout, but it is a *distinct* concept and is
# named separately on purpose (the kernels conflate the two today).
PAGES_PER_METADATA_BLOCK = 128
# The selection kernel requires k (selected pages) to be a multiple of 8.
QUEST_SELECTED_BLOCKS_ALIGNMENT = 8


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _valid_page_count(seq_len: int) -> int:
    return _ceil_div(seq_len, BLOCK_SIZE) if seq_len > 0 else 0


def _meta_blocks_for(max_vpc: int) -> int:
    return max(1, _ceil_div(max_vpc, PAGES_PER_METADATA_BLOCK))


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _identity_metadata_block_tables(batch_size: int, meta_blocks: int) -> torch.Tensor:
    """Each request owns a contiguous, disjoint slice of physical metadata blocks.

    This mirrors ``QuestDecodeMetadataManager`` which hands each batch row a fixed
    ``arange`` of metadata block ids.
    """
    total = batch_size * meta_blocks
    return torch.arange(total, dtype=torch.int32).view(batch_size, meta_blocks)


# ===========================================================================
# CPU golden references
# ===========================================================================
def cpu_quest_prefill_metadata(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    refresh_start_seq_lens: torch.Tensor,
    refresh_end_seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for ``npu_quest_prefill_metadata``.

    Only *complete* pages inside ``[start, end)`` are refreshed; the partial tail
    page is intentionally left untouched (it is covered by the decode selector's
    fixed anchor).  When the first metadata block is only partially populated the
    trailing rows are zeroed.  Untouched regions keep their prior value, so the
    full output tensors can be compared bit-for-bit.
    """
    expected_max = maxblocks.clone()
    expected_min = minblocks.clone()
    batch_size = refresh_end_seq_lens.shape[0]

    for b in range(batch_size):
        start_len = int(refresh_start_seq_lens[b].item())
        end_len = int(refresh_end_seq_lens[b].item())
        if end_len <= start_len:
            continue
        assert start_len >= 0, "refresh_start_seq_lens must be non-negative"

        start_page = start_len // BLOCK_SIZE
        end_page = end_len // BLOCK_SIZE  # floor -> complete pages only
        start_meta = start_page // PAGES_PER_METADATA_BLOCK
        end_meta = _ceil_div(end_page, PAGES_PER_METADATA_BLOCK)

        for meta_block in range(start_meta, end_meta):
            meta_start_page = meta_block * PAGES_PER_METADATA_BLOCK
            first_page = max(start_page - meta_start_page, 0)
            last_page = min(end_page - meta_start_page, PAGES_PER_METADATA_BLOCK)
            if last_page - first_page <= 0:
                continue

            metadata_block_id = int(metadata_block_tables[b, meta_block].item())
            for page_offset in range(first_page, last_page):
                logical_page = meta_start_page + page_offset
                kv_block_id = int(block_tables[b, logical_page].item())
                # [block_size, num_kv_heads, head_dim] -> reduce over tokens
                page = k_cache[kv_block_id].to(torch.float32)
                expected_max[metadata_block_id, page_offset] = page.max(dim=0).values.to(maxblocks.dtype)
                expected_min[metadata_block_id, page_offset] = page.min(dim=0).values.to(minblocks.dtype)

            if first_page == 0 and last_page < PAGES_PER_METADATA_BLOCK:
                expected_max[metadata_block_id, last_page:] = 0
                expected_min[metadata_block_id, last_page:] = 0

    return expected_max, expected_min


def cpu_page_scores(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """QUEST page scores ``sum_d max(q_d * max_d, q_d * min_d)`` in fp32.

    Returns ``[batch, num_heads, max_pages]`` with ``-inf`` for invalid pages.
    Computed the same way the kernel computes it (fp32 over exact upcast inputs).
    """
    batch_size, num_heads, _ = query.shape
    num_kv_heads = maxblocks.shape[2]
    query_heads_per_kv_head = num_heads // num_kv_heads
    max_pages = int(metadata_block_tables.shape[1]) * PAGES_PER_METADATA_BLOCK

    scores = torch.full((batch_size, num_heads, max_pages), float("-inf"), dtype=torch.float32)
    for b in range(batch_size):
        vpc = _valid_page_count(int(seq_lens[b].item()))
        if vpc <= 0:
            continue
        pages = torch.arange(vpc)
        meta_block = pages // PAGES_PER_METADATA_BLOCK
        page_offset = pages % PAGES_PER_METADATA_BLOCK
        # Index tensors must be int64 (long) for advanced indexing below.
        metadata_block_id = metadata_block_tables[b][meta_block].long()
        # [vpc, num_kv_heads, head_dim]
        max_meta = maxblocks[metadata_block_id, page_offset].to(torch.float32)
        min_meta = minblocks[metadata_block_id, page_offset].to(torch.float32)
        for h in range(num_heads):
            kv_head = h // query_heads_per_kv_head
            q = query[b, h].to(torch.float32)
            upper_bound = torch.maximum(q * max_meta[:, kv_head, :], q * min_meta[:, kv_head, :])
            scores[b, h, :vpc] = upper_bound.sum(dim=-1)
    return scores


def cpu_select_pages(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    use_fixed_anchors: bool,
) -> torch.Tensor:
    """Reference page selection (the *intended* contract).

    * ``valid_page_count <= 0`` -> all-zero row.
    * ``k >= valid_page_count`` -> sequential ``arange`` (select every page),
      regardless of anchors, with zero padding.  This is the contract the Python
      side and the CPU reference assume.
    * otherwise -> top-k by score; with anchors, page 0 and the last page are
      pinned (score = +inf) and therefore always selected.
    """
    batch_size, num_heads, _ = scores.shape
    out = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
    for b in range(batch_size):
        vpc = _valid_page_count(int(seq_lens[b].item()))
        if vpc <= 0:
            continue
        for h in range(num_heads):
            if k >= vpc:
                out[b, h, :vpc] = torch.arange(vpc, dtype=torch.int32)
                continue
            s = scores[b, h, :vpc].clone()
            if use_fixed_anchors:
                s[0] = float("inf")
                s[vpc - 1] = float("inf")
            # stable descending sort -> ties broken by ascending page index
            order = torch.sort(s, descending=True, stable=True).indices
            out[b, h, :] = order[:k].to(torch.int32)
    return out


# ===========================================================================
# NPU execution wrappers
# ===========================================================================
def ascendc_prefill_metadata_exec(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    refresh_start_seq_lens: torch.Tensor,
    refresh_end_seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_npu = maxblocks.npu()
    min_npu = minblocks.npu()
    torch.ops._C_ascend.npu_quest_prefill_metadata(
        k_cache.npu(),
        block_tables.npu(),
        refresh_start_seq_lens.npu(),
        refresh_end_seq_lens.npu(),
        metadata_block_tables.npu(),
        max_npu,
        min_npu,
    )
    torch.npu.synchronize()
    return max_npu.cpu(), min_npu.cpu()


def ascendc_block_select_exec(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int,
) -> torch.Tensor:
    """The functional (allocating) variant -- returns ``[B, H, k]``."""
    out = torch.ops._C_ascend.npu_quest_block_select_paged(
        query.npu(),
        maxblocks.npu(),
        minblocks.npu(),
        metadata_block_tables.npu(),
        seq_lens.npu(),
        k,
        tokens_since_metadata_update,
    )
    torch.npu.synchronize()
    return out.cpu()


def ascendc_block_select_out_exec(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int,
) -> torch.Tensor:
    """The ``_out`` variant. ``k`` is a multiple of 8, so the output is allocated directly."""
    query_npu = query.npu()
    out = torch.empty(
        (query.size(0), query.size(1), k),
        dtype=torch.int32,
        device=query_npu.device,
    )
    torch.ops._C_ascend.npu_quest_block_select_paged_out(
        query_npu,
        maxblocks.npu(),
        minblocks.npu(),
        metadata_block_tables.npu(),
        seq_lens.npu(),
        out,
        tokens_since_metadata_update,
    )
    torch.npu.synchronize()
    return out.cpu()


# ===========================================================================
# assertions
# ===========================================================================
def assert_metadata_equal(actual_max, actual_min, expected_max, expected_min):
    # Reduced values are exact elements of the key cache -> bit-exact expected.
    torch.testing.assert_close(actual_max, expected_max, rtol=0, atol=0)
    torch.testing.assert_close(actual_min, expected_min, rtol=0, atol=0)


def _anchors_for(vpc: int, use_fixed_anchors: bool) -> set:
    if not use_fixed_anchors or vpc <= 0:
        return set()
    return {0, vpc - 1}


def assert_selection_exact(actual, expected, seq_lens, k, use_fixed_anchors):
    """Strict, order-sensitive comparison; valid only for distinct scores.

    Anchors are compared as an unordered set (their relative order in the output
    is implementation-defined).  Interior (non-anchor) pages are compared in
    order, which uniquely identifies the ranking when scores are distinct.
    """
    batch_size, num_heads, _ = actual.shape
    for b in range(batch_size):
        vpc = _valid_page_count(int(seq_lens[b].item()))
        for h in range(num_heads):
            a = actual[b, h].tolist()
            e = expected[b, h].tolist()
            if vpc <= 0:
                continue  # no pages consumed; the whole row is unused tail
            if k >= vpc:
                # Only [0, vpc) is consumed by paged_select_attention (it bounds
                # its loop by the kv seq length); the trailing slots are unused
                # tail and not necessarily zero. The consumed prefix is the ramp.
                assert a[:vpc] == e[:vpc], f"k>=vpc prefix mismatch (b={b}, h={h}): {a[:vpc]} != {e[:vpc]}"
                continue
            a_sel, e_sel = a[:k], e[:k]
            anchors = _anchors_for(vpc, use_fixed_anchors)
            for anchor in anchors:
                assert anchor in a_sel, f"missing anchor {anchor} (b={b}, h={h}): {a_sel}"
            a_interior = [p for p in a_sel if p not in anchors]
            e_interior = [p for p in e_sel if p not in anchors]
            assert a_interior == e_interior, f"interior order mismatch (b={b}, h={h}): {a_interior} != {e_interior}"


def assert_selection_valid(actual, scores, seq_lens, k, use_fixed_anchors, tol=1e-2):
    """Tie-agnostic top-k correctness invariants (safe for random scores)."""
    batch_size, num_heads, _ = actual.shape
    for b in range(batch_size):
        vpc = _valid_page_count(int(seq_lens[b].item()))
        for h in range(num_heads):
            row = actual[b, h].tolist()
            if vpc <= 0:
                continue
            n_real = min(k, vpc)
            selected = row[:n_real]

            for idx in selected:
                assert 0 <= idx < vpc, f"index {idx} out of range [0,{vpc}) (b={b}, h={h})"
            assert len(set(selected)) == len(selected), f"duplicate selected pages (b={b}, h={h}): {selected}"

            if k >= vpc:
                assert set(selected) == set(range(vpc)), f"k>=vpc must select every page (b={b}, h={h}): {selected}"
                continue

            anchors = _anchors_for(vpc, use_fixed_anchors)
            for anchor in anchors:
                assert anchor in selected, f"missing anchor {anchor} (b={b}, h={h}): {selected}"

            sel_interior = [p for p in selected if p not in anchors]
            unsel_interior = [p for p in range(vpc) if p not in anchors and p not in selected]
            if sel_interior and unsel_interior:
                row_scores = scores[b, h]
                min_selected = min(row_scores[p].item() for p in sel_interior)
                max_unselected = max(row_scores[p].item() for p in unsel_interior)
                assert min_selected >= max_unselected - tol, (
                    f"top-k invariant violated (b={b}, h={h}): "
                    f"min(selected)={min_selected:.4f} < max(unselected)={max_unselected:.4f}"
                )


# ===========================================================================
# input builders
# ===========================================================================
def _make_prefill_case(dtype, seed=0):
    """A multi-request prefill case exercising every refresh shape at once.

    * request 0: full refresh from scratch, spanning >1 metadata block, partial
      tail page (not refreshed), trailing zero-fill of the last metadata block.
    * request 1: incremental refresh starting mid-page and crossing a metadata
      block boundary.
    * request 2: no-op (end <= start).
    * request 3: refresh confined to a single page.
    Physical kv-block ids and metadata-block ids are shuffled / disjoint to catch
    indexing bugs, and metadata buffers are pre-filled with random sentinels so
    any stray write is detected.
    """
    g = torch.Generator().manual_seed(seed)
    batch_size = 4
    num_kv_heads = 2
    max_pages = 200  # block_tables width
    num_meta_blocks_per_req = 2  # supports up to 256 pages / request
    num_kv_blocks = 1024
    num_metadata_blocks = batch_size * num_meta_blocks_per_req

    k_cache = torch.randn((num_kv_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM), generator=g, dtype=dtype)

    # Disjoint, shuffled physical kv-block ids per request.
    block_tables = torch.empty((batch_size, max_pages), dtype=torch.int32)
    perm = torch.randperm(num_kv_blocks, generator=g)[: batch_size * max_pages]
    block_tables.copy_(perm.view(batch_size, max_pages).to(torch.int32))

    metadata_block_tables = _identity_metadata_block_tables(batch_size, num_meta_blocks_per_req)

    refresh_start_seq_lens = torch.tensor([0, 3 * BLOCK_SIZE + 40, 7 * BLOCK_SIZE, 9 * BLOCK_SIZE], dtype=torch.int32)
    refresh_end_seq_lens = torch.tensor(
        [130 * BLOCK_SIZE + 41, 132 * BLOCK_SIZE + 5, 7 * BLOCK_SIZE, 10 * BLOCK_SIZE],
        dtype=torch.int32,
    )

    metadata_shape = (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    maxblocks = torch.randn(metadata_shape, generator=g, dtype=dtype)
    minblocks = torch.randn(metadata_shape, generator=g, dtype=dtype)
    return (
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )


# A single rich batch covering: empty, 1 page, 2 pages, small (< k), a
# mid-size request (vpc=14), and a large multi-metadata-block request (vpc=131).
_RICH_SEQ_LENS = [0, 100, 2 * BLOCK_SIZE, 5 * BLOCK_SIZE - 3, 14 * BLOCK_SIZE - 3, 131 * BLOCK_SIZE - 5]


def _make_block_select_structured(dtype, seq_lens_list, num_heads, num_kv_heads, seed=0, negative_q=False):
    """Structured case with distinct, well-separated, dtype-exact page scores.

    Query is one-hot on channel 0, and only channel 0 of the metadata carries
    signal, so page score == an exactly-representable multiple of 64.  Scores are
    a random permutation per (request, kv-head) so different kv-heads rank pages
    differently -- this catches a wrong query-head -> kv-head mapping.  With
    ``negative_q`` the signal is carried by the ``min`` bound (q < 0), which
    catches a kernel that forgets the lower bound.
    """
    g = torch.Generator().manual_seed(seed)
    batch_size = len(seq_lens_list)
    max_vpc = max((_valid_page_count(s) for s in seq_lens_list), default=1)
    meta_blocks = _meta_blocks_for(max_vpc)
    num_metadata_blocks = batch_size * meta_blocks

    metadata_shape = (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    maxblocks = torch.zeros(metadata_shape, dtype=dtype)
    minblocks = torch.zeros(metadata_shape, dtype=dtype)

    query = torch.zeros((batch_size, num_heads, HEAD_DIM), dtype=dtype)
    query[:, :, 0] = -1 if negative_q else 1

    metadata_block_tables = _identity_metadata_block_tables(batch_size, meta_blocks)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)

    for b, seq_len in enumerate(seq_lens_list):
        vpc = _valid_page_count(seq_len)
        if vpc <= 0:
            continue
        for kv_head in range(num_kv_heads):
            ranks = torch.randperm(vpc, generator=g)
            for page in range(vpc):
                score = float(int(ranks[page].item()) + 1) * 64.0  # distinct, exact in fp16/bf16
                meta_block = page // PAGES_PER_METADATA_BLOCK
                page_offset = page % PAGES_PER_METADATA_BLOCK
                mid = int(metadata_block_tables[b, meta_block].item())
                if negative_q:
                    # q=-1: score = max(-max0, -min0); make min0 the discriminator.
                    maxblocks[mid, page_offset, kv_head, 0] = 10**4
                    minblocks[mid, page_offset, kv_head, 0] = -score
                else:
                    maxblocks[mid, page_offset, kv_head, 0] = score
                    minblocks[mid, page_offset, kv_head, 0] = -score
    return query, maxblocks, minblocks, metadata_block_tables, seq_lens


def _make_block_select_random(dtype, seq_lens_list, num_heads, num_kv_heads, seed):
    g = torch.Generator().manual_seed(seed)
    batch_size = len(seq_lens_list)
    max_vpc = max((_valid_page_count(s) for s in seq_lens_list), default=1)
    meta_blocks = _meta_blocks_for(max_vpc)
    num_metadata_blocks = batch_size * meta_blocks

    metadata_shape = (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    center = torch.randn(metadata_shape, generator=g, dtype=torch.float32)
    half_width = torch.rand(metadata_shape, generator=g, dtype=torch.float32)  # >= 0
    maxblocks = (center + half_width).to(dtype)
    minblocks = (center - half_width).to(dtype)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=g, dtype=torch.float32).to(dtype)

    metadata_block_tables = _identity_metadata_block_tables(batch_size, meta_blocks)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    return query, maxblocks, minblocks, metadata_block_tables, seq_lens


# ===========================================================================
# Tests: prefill metadata
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_prefill_metadata_matches_reference(dtype):
    case = _make_prefill_case(dtype)
    expected_max, expected_min = cpu_quest_prefill_metadata(*case)
    try:
        actual_max, actual_min = ascendc_prefill_metadata_exec(*case)
        assert_metadata_equal(actual_max, actual_min, expected_max, expected_min)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_prefill_metadata_is_deterministic(dtype):
    """Same inputs -> bit-identical outputs (guards against races / stale UB)."""
    case = _make_prefill_case(dtype)
    try:
        first_max, first_min = ascendc_prefill_metadata_exec(*case)
        second_max, second_min = ascendc_prefill_metadata_exec(*case)
        torch.testing.assert_close(first_max, second_max, rtol=0, atol=0)
        torch.testing.assert_close(first_min, second_min, rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_prefill_metadata_does_not_touch_inputs(dtype):
    """Only ``maxblocks``/``minblocks`` may change; everything else is read-only."""
    (
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    ) = _make_prefill_case(dtype)
    read_only = {
        "k_cache": k_cache.clone(),
        "block_tables": block_tables.clone(),
        "refresh_start_seq_lens": refresh_start_seq_lens.clone(),
        "refresh_end_seq_lens": refresh_end_seq_lens.clone(),
        "metadata_block_tables": metadata_block_tables.clone(),
    }
    try:
        k_npu = k_cache.npu()
        bt_npu = block_tables.npu()
        rss_npu = refresh_start_seq_lens.npu()
        res_npu = refresh_end_seq_lens.npu()
        mbt_npu = metadata_block_tables.npu()
        torch.ops._C_ascend.npu_quest_prefill_metadata(
            k_npu, bt_npu, rss_npu, res_npu, mbt_npu, maxblocks.npu(), minblocks.npu()
        )
        torch.npu.synchronize()
        torch.testing.assert_close(k_npu.cpu(), read_only["k_cache"], rtol=0, atol=0)
        torch.testing.assert_close(bt_npu.cpu(), read_only["block_tables"], rtol=0, atol=0)
        torch.testing.assert_close(rss_npu.cpu(), read_only["refresh_start_seq_lens"], rtol=0, atol=0)
        torch.testing.assert_close(res_npu.cpu(), read_only["refresh_end_seq_lens"], rtol=0, atol=0)
        torch.testing.assert_close(mbt_npu.cpu(), read_only["metadata_block_tables"], rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_prefill_metadata_noop_when_end_le_start(dtype):
    """A non-advancing refresh range must leave the metadata buffers untouched."""
    g = torch.Generator().manual_seed(7)
    batch_size, num_kv_heads = 3, 2
    k_cache = torch.randn((64, BLOCK_SIZE, num_kv_heads, HEAD_DIM), generator=g, dtype=dtype)
    block_tables = torch.arange(batch_size * 8, dtype=torch.int32).view(batch_size, 8)
    metadata_block_tables = _identity_metadata_block_tables(batch_size, 1)
    # end <= start for every request (equal, shrunk, zero-length).
    refresh_start_seq_lens = torch.tensor([5 * BLOCK_SIZE, 4 * BLOCK_SIZE, 0], dtype=torch.int32)
    refresh_end_seq_lens = torch.tensor([5 * BLOCK_SIZE, 2 * BLOCK_SIZE, 0], dtype=torch.int32)
    shape = (batch_size, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    maxblocks = torch.randn(shape, generator=g, dtype=dtype)
    minblocks = torch.randn(shape, generator=g, dtype=dtype)
    try:
        actual_max, actual_min = ascendc_prefill_metadata_exec(
            k_cache,
            block_tables,
            refresh_start_seq_lens,
            refresh_end_seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
        )
        torch.testing.assert_close(actual_max, maxblocks, rtol=0, atol=0)
        torch.testing.assert_close(actual_min, minblocks, rtol=0, atol=0)
    finally:
        _cleanup_npu()


# ===========================================================================
# Tests: block selection
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [8, 16, 24])
@pytest.mark.parametrize("tokens_since_metadata_update", [-1, 0])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (4, 2), (8, 2)])
@torch.inference_mode()
def test_block_select_structured_exact(dtype, k, tokens_since_metadata_update, num_heads, num_kv_heads):
    """Exact selection on distinct, well-separated scores across many vpc cases."""
    use_fixed_anchors = tokens_since_metadata_update >= 0
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_structured(
        dtype, _RICH_SEQ_LENS, num_heads, num_kv_heads, seed=k
    )
    scores = cpu_page_scores(query, maxblocks, minblocks, metadata_block_tables, seq_lens)
    expected = cpu_select_pages(scores, seq_lens, k, use_fixed_anchors)
    try:
        for exec_fn in (ascendc_block_select_exec, ascendc_block_select_out_exec):
            actual = exec_fn(
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, tokens_since_metadata_update
            )
            assert actual.shape == (len(_RICH_SEQ_LENS), num_heads, k)
            assert_selection_exact(actual, expected, seq_lens, k, use_fixed_anchors)
            assert_selection_valid(actual, scores, seq_lens, k, use_fixed_anchors)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens_since_metadata_update", [-1, 0])
@torch.inference_mode()
def test_block_select_uses_min_bound_for_negative_query(dtype, tokens_since_metadata_update):
    """With q < 0 the discriminating signal lives in the ``min`` bound.

    A kernel that drops the ``min`` term (scoring only ``q * max``) would see a
    constant score and fail this test.
    """
    use_fixed_anchors = tokens_since_metadata_update >= 0
    k = 8
    seq_lens_list = [10 * BLOCK_SIZE, 20 * BLOCK_SIZE]
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_structured(
        dtype, seq_lens_list, num_heads=4, num_kv_heads=2, seed=1, negative_q=True
    )
    scores = cpu_page_scores(query, maxblocks, minblocks, metadata_block_tables, seq_lens)
    expected = cpu_select_pages(scores, seq_lens, k, use_fixed_anchors)
    try:
        actual = ascendc_block_select_exec(
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, tokens_since_metadata_update
        )
        assert_selection_exact(actual, expected, seq_lens, k, use_fixed_anchors)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [8, 16])
@pytest.mark.parametrize("tokens_since_metadata_update", [-1, 0])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@torch.inference_mode()
def test_block_select_random_invariants(dtype, k, tokens_since_metadata_update, seed):
    """Fuzz with random scores; check tie-agnostic top-k invariants."""
    use_fixed_anchors = tokens_since_metadata_update >= 0
    # Mix of vpc regimes including 0, < k, ~ k and a large multi-block request.
    seq_lens_list = [0, 3 * BLOCK_SIZE, 9 * BLOCK_SIZE, 17 * BLOCK_SIZE, 200 * BLOCK_SIZE]
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_random(
        dtype, seq_lens_list, num_heads=8, num_kv_heads=2, seed=seed
    )
    scores = cpu_page_scores(query, maxblocks, minblocks, metadata_block_tables, seq_lens)
    try:
        actual = ascendc_block_select_exec(
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, tokens_since_metadata_update
        )
        assert_selection_valid(actual, scores, seq_lens, k, use_fixed_anchors)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens_since_metadata_update", [-1, 0])
@torch.inference_mode()
def test_block_select_tied_scores_return_distinct_pages(dtype, tokens_since_metadata_update):
    """Uniform metadata -> every page ties on score.

    The ``Sort`` tie-break must still yield ``k`` *distinct*, valid pages (plus
    the anchors when enabled).  A colliding tie-break would resurface the
    duplicate-page bug this suite exists to catch; random scores essentially
    never produce an exact tie, so this is the case that pins it.
    """
    use_fixed_anchors = tokens_since_metadata_update >= 0
    k = 8
    seq_lens_list = [20 * BLOCK_SIZE, 9 * BLOCK_SIZE]  # vpc = 20, 9 (both > k)
    batch_size = len(seq_lens_list)
    num_heads, num_kv_heads = 4, 2
    meta_blocks = _meta_blocks_for(max(_valid_page_count(s) for s in seq_lens_list))
    metadata_shape = (batch_size * meta_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    # Identical metadata for every page and channel -> identical QUEST score per
    # (request, head), i.e. an exact tie across all pages.
    maxblocks = torch.ones(metadata_shape, dtype=dtype)
    minblocks = torch.ones(metadata_shape, dtype=dtype)
    g = torch.Generator().manual_seed(4)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=g, dtype=torch.float32).to(dtype)
    metadata_block_tables = _identity_metadata_block_tables(batch_size, meta_blocks)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    scores = cpu_page_scores(query, maxblocks, minblocks, metadata_block_tables, seq_lens)
    try:
        actual = ascendc_block_select_exec(
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, tokens_since_metadata_update
        )
        assert_selection_valid(actual, scores, seq_lens, k, use_fixed_anchors)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_block_select_is_deterministic(dtype):
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_random(
        dtype, _RICH_SEQ_LENS, num_heads=4, num_kv_heads=2, seed=99
    )
    try:
        first = ascendc_block_select_exec(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 8, 0)
        second = ascendc_block_select_exec(query, maxblocks, minblocks, metadata_block_tables, seq_lens, 8, 0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_block_select_does_not_touch_inputs(dtype):
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_random(
        dtype, _RICH_SEQ_LENS, num_heads=4, num_kv_heads=2, seed=5
    )
    saved = (query.clone(), maxblocks.clone(), minblocks.clone(), metadata_block_tables.clone(), seq_lens.clone())
    try:
        q_npu, max_npu, min_npu = query.npu(), maxblocks.npu(), minblocks.npu()
        mbt_npu, sl_npu = metadata_block_tables.npu(), seq_lens.npu()
        _ = torch.ops._C_ascend.npu_quest_block_select_paged(q_npu, max_npu, min_npu, mbt_npu, sl_npu, 8, 0)
        torch.npu.synchronize()
        torch.testing.assert_close(q_npu.cpu(), saved[0], rtol=0, atol=0)
        torch.testing.assert_close(max_npu.cpu(), saved[1], rtol=0, atol=0)
        torch.testing.assert_close(min_npu.cpu(), saved[2], rtol=0, atol=0)
        torch.testing.assert_close(mbt_npu.cpu(), saved[3], rtol=0, atol=0)
        torch.testing.assert_close(sl_npu.cpu(), saved[4], rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_block_select_out_matches_functional(dtype):
    """The ``_out`` and functional variants must agree across k values."""
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_structured(
        dtype, _RICH_SEQ_LENS, num_heads=4, num_kv_heads=2, seed=3
    )
    try:
        for k in (8, 16, 24):
            functional = ascendc_block_select_exec(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, 0)
            out_variant = ascendc_block_select_out_exec(
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, 0
            )
            torch.testing.assert_close(out_variant, functional, rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 2, 9, 13])
@torch.inference_mode()
def test_block_select_rejects_unaligned_k(dtype, k):
    """k must be a positive multiple of 8; the op rejects anything else."""
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_random(
        dtype, [3 * BLOCK_SIZE], num_heads=4, num_kv_heads=2, seed=0
    )
    try:
        with pytest.raises(RuntimeError):
            ascendc_block_select_exec(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, 0)
    finally:
        _cleanup_npu()


# ===========================================================================
# Tests: end-to-end (prefill metadata -> block selection)
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens_since_metadata_update", [-1, 0])
@torch.inference_mode()
def test_prefill_then_select_end_to_end(dtype, tokens_since_metadata_update):
    """Chain both kernels and compare against the chained CPU reference.

    Sequence lengths are whole multiples of the page size so every page has
    fresh metadata, making the pure top-k comparison (anchors off) valid too.
    """
    use_fixed_anchors = tokens_since_metadata_update >= 0
    g = torch.Generator().manual_seed(11)
    num_kv_heads, num_heads = 2, 4
    seq_lens_list = [3 * BLOCK_SIZE, 6 * BLOCK_SIZE, 130 * BLOCK_SIZE]
    batch_size = len(seq_lens_list)
    max_pages = 200
    meta_blocks = _meta_blocks_for(max(_valid_page_count(s) for s in seq_lens_list))
    num_kv_blocks = 1024
    k = 8

    k_cache = torch.randn((num_kv_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM), generator=g, dtype=dtype)
    block_tables = torch.empty((batch_size, max_pages), dtype=torch.int32)
    perm = torch.randperm(num_kv_blocks, generator=g)[: batch_size * max_pages]
    block_tables.copy_(perm.view(batch_size, max_pages).to(torch.int32))
    metadata_block_tables = _identity_metadata_block_tables(batch_size, meta_blocks)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    refresh_start_seq_lens = torch.zeros(batch_size, dtype=torch.int32)
    refresh_end_seq_lens = seq_lens.clone()

    metadata_shape = (batch_size * meta_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    maxblocks = torch.zeros(metadata_shape, dtype=dtype)
    minblocks = torch.zeros(metadata_shape, dtype=dtype)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=g, dtype=torch.float32).to(dtype)

    # CPU golden pipeline.
    exp_max, exp_min = cpu_quest_prefill_metadata(
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )
    cpu_scores = cpu_page_scores(query, exp_max, exp_min, metadata_block_tables, seq_lens)
    try:
        # NPU pipeline: metadata kernel feeds the selection kernel.
        npu_max, npu_min = ascendc_prefill_metadata_exec(
            k_cache,
            block_tables,
            refresh_start_seq_lens,
            refresh_end_seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
        )
        assert_metadata_equal(npu_max, npu_min, exp_max, exp_min)
        actual = ascendc_block_select_exec(
            query, npu_max, npu_min, metadata_block_tables, seq_lens, k, tokens_since_metadata_update
        )
        # Random query -> use the tie-agnostic invariant against the CPU scores.
        assert_selection_valid(actual, cpu_scores, seq_lens, k, use_fixed_anchors)
    finally:
        _cleanup_npu()
