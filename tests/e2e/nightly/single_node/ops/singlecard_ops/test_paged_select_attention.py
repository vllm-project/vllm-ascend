"""Correctness suite for the ``npu_paged_select_attention`` operator.

``npu_paged_select_attention`` is the second half of the QUEST sparse-decode
path: given the per-(request, query-head) list of selected KV pages produced by
``npu_quest_block_select_paged``, it runs flash attention (online softmax) over
*only those pages*.

Contract (from the host tiling + kernel, pure decode):

* ``query``               -- ``[batch, num_heads, head_dim]`` (TND, 1 token/req)
* ``key`` / ``value``     -- paged cache ``[num_blocks, block_size, num_kv_heads * head_dim]``
* ``block_table``         -- ``[batch, max_blocks_per_batch]`` int32, logical->physical
* ``selected_kv_indices`` -- ``[batch, num_heads, k]`` int32, *logical* page ids
* ``actual_seq_lengths``    -- cumulative q lengths ``[1, 2, ..., batch]`` (q_len==1/req)
* ``actual_seq_lengths_kv`` -- per-request absolute kv length
* ``output``              -- ``empty_like(query)``

Semantics:

* ``scores = scale_value * (q . k)``  (scale is *multiplied* in, not 1/sqrt).
* GQA: query head ``h`` reads kv head ``h // (num_heads // num_kv_heads)``.
* Only the first ``min(k, ceil(kv_seqlen / block_size))`` selected entries per
  (request, head) are read; the trailing slots are *unused padding*.
* The final logical page is partial: only its first ``kv_seqlen % block_size``
  tokens are attended (the rest of that page is masked out).  No causal mask.

Oracle strategy
---------------
Flash attention in fp16/bf16 is not bit-exact against an fp32 reference, so the
broad ``matches_reference`` test is tolerance-based.  It is backed by *tight,
predictable* structural tests whose outputs are exact regardless of softmax
precision:

* single valid token  -> output is exactly that token's value (tail masking),
* uniform keys        -> output is the plain mean of the page's values (and the
                         per-head mean pins the GQA mapping),
* strict page subset  -> output reflects only the selected pages, and is provably
                         far from full attention (sparsity is real),
* extra padding slots  -> bit-identical output (the kernel ignores them).

NOTE: these are NPU tests; they must run on Ascend A2/A3 hardware.
"""

import gc
import math

import pytest
import torch
import torch_npu  # noqa: F401  (registers the ``npu`` device / ``torch.npu``)

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BLOCK_SIZE = 128
HEAD_DIM = 128

# fp16/bf16 flash attention vs an fp32 reference: tolerance is dominated by the
# output dtype rounding plus matmul accumulation over the attended tokens.
_TOL = {
    torch.float16: dict(rtol=1e-2, atol=2e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=4e-2),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _valid_page_count(kv_seq_len: int) -> int:
    return _ceil_div(kv_seq_len, BLOCK_SIZE) if kv_seq_len > 0 else 0


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _cumulative_q_lengths(batch_size: int) -> list[int]:
    """Pure decode: 1 token per request, given to the op as cumulative lengths."""
    return list(range(1, batch_size + 1))


def _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen):
    key = torch.randn((num_kv_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM), generator=gen, dtype=torch.float32)
    value = torch.randn((num_kv_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM), generator=gen, dtype=torch.float32)
    return key.to(dtype), value.to(dtype)


def _disjoint_block_table(batch_size, width, num_kv_blocks, gen) -> torch.Tensor:
    """Each request owns a disjoint, shuffled slice of physical block ids.

    Shuffling (rather than identity) catches logical<->physical mapping bugs.
    """
    assert batch_size * width <= num_kv_blocks
    perm = torch.randperm(num_kv_blocks, generator=gen)[: batch_size * width]
    return perm.view(batch_size, width).to(torch.int32)


# ===========================================================================
# CPU golden reference
# ===========================================================================
def cpu_paged_select_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths_kv: list[int],
    block_table: torch.Tensor,
    selected_kv_indices: torch.Tensor,
    num_heads: int,
    scale_value: float,
    num_kv_heads: int,
    block_size: int,
) -> torch.Tensor:
    """fp32 reference for ``npu_paged_select_attention``.

    ``key``/``value`` are the un-flattened paged cache
    ``[num_blocks, block_size, num_kv_heads, head_dim]``.  Returns
    ``[batch, num_heads, head_dim]`` in fp32.
    """
    batch_size, _, head_dim = query.shape
    k = selected_kv_indices.shape[2]
    group_size = num_heads // num_kv_heads

    out = torch.zeros((batch_size, num_heads, head_dim), dtype=torch.float32)
    q32 = query.to(torch.float32)
    key32 = key.to(torch.float32)
    value32 = value.to(torch.float32)

    for b in range(batch_size):
        kv_seq = int(actual_seq_lengths_kv[b])
        vpc = _valid_page_count(kv_seq)
        effective = min(k, vpc)
        if effective == 0:
            continue  # no kv -> output row is left untouched (unused in decode)
        for h in range(num_heads):
            kv_head = h // group_size
            q = q32[b, h]  # [head_dim]
            keys, values = [], []
            for i in range(effective):
                logical = int(selected_kv_indices[b, h, i].item())
                assert 0 <= logical < vpc, f"selected page {logical} out of [0,{vpc})"
                phys = int(block_table[b, logical].item())
                # The final logical page is partial; earlier pages are full.
                n_valid = min(block_size, kv_seq - logical * block_size)
                keys.append(key32[phys, :n_valid, kv_head, :])
                values.append(value32[phys, :n_valid, kv_head, :])
            k_sel = torch.cat(keys, dim=0)  # [N, head_dim]
            v_sel = torch.cat(values, dim=0)  # [N, head_dim]
            scores = scale_value * (k_sel @ q)  # [N]
            weights = torch.softmax(scores, dim=0)  # [N]
            out[b, h] = weights @ v_sel  # [head_dim]
    return out


def cpu_dense_attention(
    query, key, value, actual_seq_lengths_kv, block_table, num_heads, scale_value, num_kv_heads, block_size
) -> torch.Tensor:
    """Reference that attends *every* valid page (no sparsity).

    Used only to prove the sparse test is meaningful: the sparse output must be
    far from this when the selection is a strict subset.
    """
    batch_size, _, _ = query.shape
    vpc_list = [_valid_page_count(int(s)) for s in actual_seq_lengths_kv]
    max_vpc = max(vpc_list, default=1)
    all_pages = torch.zeros((batch_size, num_heads, max_vpc), dtype=torch.int32)
    for b, vpc in enumerate(vpc_list):
        all_pages[b, :, :vpc] = torch.arange(vpc, dtype=torch.int32)
    return cpu_paged_select_attention(
        query,
        key,
        value,
        actual_seq_lengths_kv,
        block_table,
        all_pages,
        num_heads,
        scale_value,
        num_kv_heads,
        block_size,
    )


# ===========================================================================
# NPU execution wrappers
# ===========================================================================
def _as_3d_cache(cache_4d: torch.Tensor) -> torch.Tensor:
    """``[num_blocks, block_size, num_kv_heads, head_dim]`` -> kernel's rank-3 view."""
    nb, bs, _, _ = cache_4d.shape
    return cache_4d.reshape(nb, bs, -1)


def ascendc_paged_select_attention_exec(
    query,
    key,
    value,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_table,
    selected_kv_indices,
    num_heads,
    scale_value,
    num_kv_heads,
    block_size,
) -> torch.Tensor:
    """Allocating variant -> returns ``[batch, num_heads, head_dim]``."""
    out = torch.ops._C_ascend.npu_paged_select_attention(
        query.npu(),
        _as_3d_cache(key).npu(),
        _as_3d_cache(value).npu(),
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table.npu(),
        selected_kv_indices.npu(),
        num_heads,
        scale_value,
        num_kv_heads,
        block_size,
    )
    torch.npu.synchronize()
    return out.cpu()


def ascendc_paged_select_attention_out_exec(
    query,
    key,
    value,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_table,
    selected_kv_indices,
    num_heads,
    scale_value,
    num_kv_heads,
    block_size,
) -> torch.Tensor:
    """``_out`` variant -> writes into a pre-allocated tensor."""
    query_npu = query.npu()
    out = torch.empty_like(query_npu)
    torch.ops._C_ascend.npu_paged_select_attention_out(
        query_npu,
        _as_3d_cache(key).npu(),
        _as_3d_cache(value).npu(),
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table.npu(),
        selected_kv_indices.npu(),
        num_heads,
        scale_value,
        num_kv_heads,
        block_size,
        out,
    )
    torch.npu.synchronize()
    return out.cpu()


# ===========================================================================
# input builders
# ===========================================================================
def _make_random_case(dtype, kv_seq_lens, num_heads, num_kv_heads, k, seed):
    """A random decode batch with a distinct selected-page subset per (req, head).

    ``k`` is chosen <= every request's valid page count, so every selected slot
    is read (no padding) and the comparison exercises the full attended set.
    Selected pages always include page 0 and the last page (the QUEST anchors)
    plus a random interior subset, mirroring real selector output.
    """
    gen = torch.Generator().manual_seed(seed)
    batch_size = len(kv_seq_lens)
    vpc_list = [_valid_page_count(s) for s in kv_seq_lens]
    assert all(k <= vpc for vpc in vpc_list), "this builder needs k <= vpc for every request"
    width = max(vpc_list)
    num_kv_blocks = batch_size * width + 16

    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, width, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)

    selected = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
    for b, vpc in enumerate(vpc_list):
        for h in range(num_heads):
            anchors = [0, vpc - 1]
            interior_pool = [p for p in range(1, vpc - 1)]
            need = k - len({a for a in anchors if 0 <= a < vpc})
            perm = torch.randperm(len(interior_pool), generator=gen)[:need].tolist()
            interior = [interior_pool[i] for i in perm]
            pages = sorted(set(anchors) | set(interior))[:k]
            # top up if dedup shrank it (only possible for tiny vpc, excluded here)
            while len(pages) < k:
                pages.append(pages[-1])
            selected[b, h] = torch.tensor(pages[:k], dtype=torch.int32)
    return query, key, value, block_table, selected, vpc_list


# ===========================================================================
# Tests: broad correctness vs fp32 reference
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2), (4, 1)])
@pytest.mark.parametrize("k", [8, 16])
@pytest.mark.parametrize("scale", [1.0 / math.sqrt(HEAD_DIM), 0.5])
@torch.inference_mode()
def test_matches_reference(dtype, num_heads, num_kv_heads, k, scale):
    """Random multi-request decode vs the fp32 sparse-attention reference.

    Includes a partial final page (kv_seqlen not a multiple of the page size) so
    the tail-token masking path is exercised.
    """
    # vpc = 16, 24, 33 (last is partial: 33*128 - 7 tokens) -- all >= k.
    kv_seq_lens = [16 * BLOCK_SIZE, 24 * BLOCK_SIZE, 33 * BLOCK_SIZE - 7]
    query, key, value, block_table, selected, _ = _make_random_case(
        dtype, kv_seq_lens, num_heads, num_kv_heads, k, seed=0
    )
    asl_q = _cumulative_q_lengths(len(kv_seq_lens))

    expected = cpu_paged_select_attention(
        query, key, value, kv_seq_lens, block_table, selected, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    try:
        actual = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            selected,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        assert actual.shape == query.shape
        torch.testing.assert_close(actual.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


# ===========================================================================
# Tests: tight structural properties (exact, softmax-precision-independent)
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_single_token_returns_its_value(dtype):
    """kv_seqlen == 1 -> the only attended token is token 0 of its page.

    Softmax over a single element is exactly 1, so the output must equal that
    token's value vector.  A kernel that fails to mask the rest of the (128-slot)
    page would instead average in 127 garbage tokens and miss badly.
    """
    gen = torch.Generator().manual_seed(1)
    batch_size, num_heads, num_kv_heads, k = 2, 4, 2, 8
    kv_seq_lens = [1, 1]
    num_kv_blocks = 32
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, 1, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)
    # Only logical page 0 exists; pad the rest of the row with the same id (unread).
    selected = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)

    expected = torch.empty((batch_size, num_heads, HEAD_DIM), dtype=torch.float32)
    group_size = num_heads // num_kv_heads
    for b in range(batch_size):
        phys = int(block_table[b, 0].item())
        for h in range(num_heads):
            expected[b, h] = value[phys, 0, h // group_size, :].to(torch.float32)
    try:
        actual = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            _cumulative_q_lengths(batch_size),
            kv_seq_lens,
            block_table,
            selected,
            num_heads,
            0.1,
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(actual.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_uniform_keys_average_values_respecting_gqa(dtype):
    """Identical keys -> uniform softmax -> output is the plain mean of values.

    With GQA (num_heads > num_kv_heads) each query head must average *its* kv
    head's values; a wrong head mapping would pull the wrong page.
    """
    gen = torch.Generator().manual_seed(2)
    batch_size, num_heads, num_kv_heads, k = 2, 4, 2, 8
    group_size = num_heads // num_kv_heads
    kv_seq_lens = [BLOCK_SIZE, BLOCK_SIZE]  # exactly one full page per request
    num_kv_blocks = 32
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, 1, num_kv_blocks, gen)
    # Make every token in each (block, kv_head) share token 0's key -> equal scores.
    for b in range(batch_size):
        phys = int(block_table[b, 0].item())
        key[phys, :, :, :] = key[phys, 0:1, :, :]
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)
    selected = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)  # page 0 for all

    expected = torch.empty((batch_size, num_heads, HEAD_DIM), dtype=torch.float32)
    for b in range(batch_size):
        phys = int(block_table[b, 0].item())
        for h in range(num_heads):
            expected[b, h] = value[phys, :, h // group_size, :].to(torch.float32).mean(dim=0)
    try:
        actual = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            _cumulative_q_lengths(batch_size),
            kv_seq_lens,
            block_table,
            selected,
            num_heads,
            1.0 / math.sqrt(HEAD_DIM),
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(actual.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_attends_only_selected_pages(dtype):
    """A strict subset selection must ignore the unselected pages.

    Even pages carry a large-positive value signal, odd pages a large-negative
    one.  Selecting only even pages must yield a positive output, provably far
    from dense attention over all pages (which would be ~0).
    """
    gen = torch.Generator().manual_seed(3)
    batch_size, num_heads, num_kv_heads = 1, 2, 1
    vpc, k = 16, 8
    kv_seq_lens = [vpc * BLOCK_SIZE]
    num_kv_blocks = 64
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, vpc, num_kv_blocks, gen)
    # Even logical pages -> +8, odd logical pages -> -8 (large, distinct signal).
    for logical in range(vpc):
        phys = int(block_table[0, logical].item())
        value[phys, :, :, :] = 8.0 if logical % 2 == 0 else -8.0
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)
    even_pages = list(range(0, vpc, 2))[:k]
    selected = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
    for h in range(num_heads):
        selected[0, h] = torch.tensor(even_pages, dtype=torch.int32)

    # Small scale -> near-uniform softmax, so dense attention over balanced
    # +8/-8 pages averages to ~0 deterministically (independent of the random
    # query), keeping the "sparse is far from dense" sanity check robust. The
    # sparse output is +8 for any scale since every selected page carries +8.
    scale = 0.01
    expected_sparse = cpu_paged_select_attention(
        query, key, value, kv_seq_lens, block_table, selected, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    expected_dense = cpu_dense_attention(
        query, key, value, kv_seq_lens, block_table, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    # Sanity: the sparse and dense references must differ a lot, else the test is vacuous.
    assert (expected_sparse - expected_dense).abs().max() > 1.0
    assert expected_sparse.min() > 4.0  # selecting only the +8 pages
    try:
        actual = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            _cumulative_q_lengths(batch_size),
            kv_seq_lens,
            block_table,
            selected,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(actual.to(torch.float32), expected_sparse, **_TOL[dtype])
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_ignores_padding_beyond_effective_count(dtype):
    """Slots past ``min(k, vpc)`` are unused padding and must not affect output.

    With ``k > vpc`` the effective count is ``vpc``.  Two runs whose only
    difference is the *padding* tail of ``selected_kv_indices`` (filled with two
    different valid page sets) must produce *bit-identical* output -- proving the
    kernel never reads the padding -- and must match the fp32 reference.
    """
    gen = torch.Generator().manual_seed(4)
    batch_size, num_heads, num_kv_heads = 2, 4, 2
    vpc, k = 4, 8  # k > vpc -> slots [vpc:k] are padding
    kv_seq_lens = [vpc * BLOCK_SIZE, vpc * BLOCK_SIZE]
    num_kv_blocks = 64
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, vpc, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)

    # Effective prefix: every page, in order. Padding tails differ between runs.
    base = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
    base[:, :, :vpc] = torch.arange(vpc, dtype=torch.int32)
    sel_a = base.clone()
    sel_b = base.clone()
    sel_a[:, :, vpc:] = 0  # padding -> page 0 repeated
    sel_b[:, :, vpc:] = vpc - 1  # padding -> last page repeated (different set, both valid)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    expected = cpu_paged_select_attention(
        query, key, value, kv_seq_lens, block_table, sel_a, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    asl_q = _cumulative_q_lengths(batch_size)
    try:
        out_a = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            sel_a,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        out_b = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            sel_b,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        # Padding is never read -> identical effective input -> identical output.
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        torch.testing.assert_close(out_a.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


# ===========================================================================
# Tests: hygiene (determinism, purity, _out parity)
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_is_deterministic(dtype):
    """Same inputs -> bit-identical output (guards against races / stale UB)."""
    kv_seq_lens = [12 * BLOCK_SIZE, 20 * BLOCK_SIZE]
    query, key, value, block_table, selected, _ = _make_random_case(
        dtype, kv_seq_lens, num_heads=8, num_kv_heads=2, k=8, seed=7
    )
    asl_q = _cumulative_q_lengths(len(kv_seq_lens))
    args = (query, key, value, asl_q, kv_seq_lens, block_table, selected, 8, 0.1, 2, BLOCK_SIZE)
    try:
        first = ascendc_paged_select_attention_exec(*args)
        second = ascendc_paged_select_attention_exec(*args)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_does_not_touch_inputs(dtype):
    """Only the output may change; every input tensor is read-only."""
    kv_seq_lens = [12 * BLOCK_SIZE, 20 * BLOCK_SIZE]
    query, key, value, block_table, selected, _ = _make_random_case(
        dtype, kv_seq_lens, num_heads=4, num_kv_heads=2, k=8, seed=8
    )
    asl_q = _cumulative_q_lengths(len(kv_seq_lens))
    key3d, value3d = _as_3d_cache(key), _as_3d_cache(value)
    saved = {
        "query": query.clone(),
        "key": key3d.clone(),
        "value": value3d.clone(),
        "block_table": block_table.clone(),
        "selected": selected.clone(),
    }
    try:
        q_npu = query.npu()
        k_npu = key3d.npu()
        v_npu = value3d.npu()
        bt_npu = block_table.npu()
        sel_npu = selected.npu()
        _ = torch.ops._C_ascend.npu_paged_select_attention(
            q_npu, k_npu, v_npu, asl_q, kv_seq_lens, bt_npu, sel_npu, 4, 0.1, 2, BLOCK_SIZE
        )
        torch.npu.synchronize()
        torch.testing.assert_close(q_npu.cpu(), saved["query"], rtol=0, atol=0)
        torch.testing.assert_close(k_npu.cpu(), saved["key"], rtol=0, atol=0)
        torch.testing.assert_close(v_npu.cpu(), saved["value"], rtol=0, atol=0)
        torch.testing.assert_close(bt_npu.cpu(), saved["block_table"], rtol=0, atol=0)
        torch.testing.assert_close(sel_npu.cpu(), saved["selected"], rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_out_matches_functional(dtype):
    """The allocating and ``_out`` variants must agree bit-for-bit."""
    kv_seq_lens = [16 * BLOCK_SIZE, 9 * BLOCK_SIZE]
    query, key, value, block_table, selected, _ = _make_random_case(
        dtype, kv_seq_lens, num_heads=8, num_kv_heads=2, k=8, seed=9
    )
    asl_q = _cumulative_q_lengths(len(kv_seq_lens))
    args = (query, key, value, asl_q, kv_seq_lens, block_table, selected, 8, 0.1, 2, BLOCK_SIZE)
    try:
        functional = ascendc_paged_select_attention_exec(*args)
        out_variant = ascendc_paged_select_attention_out_exec(*args)
        torch.testing.assert_close(out_variant, functional, rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_ignores_selection_order_and_masks_correct_tail_page(dtype):
    """Selection order must not matter, and the partial page must be masked by id.

    Every other test feeds pages in ascending order with the partial final page
    last.  Here the same page *set* is fed in two orders, once with the partial
    tail page (logical id ``vpc-1``) placed *first*.  Both must match the
    reference: the kernel identifies the tail page by value (``kv_seqlen``), not
    by its slot, and attention over a fixed set is order-independent.
    """
    gen = torch.Generator().manual_seed(21)
    batch_size, num_heads, num_kv_heads = 1, 2, 1
    vpc = 5
    kv_seq_lens = [vpc * BLOCK_SIZE - 30]  # partial tail page: 98 valid tokens
    num_kv_blocks = 32
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, vpc, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)
    k = 8  # > vpc -> padding slots; effective count == vpc == 5

    def _sel(order):
        s = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
        for h in range(num_heads):
            s[0, h, :vpc] = torch.tensor(order, dtype=torch.int32)
        return s

    ascending = _sel([0, 1, 2, 3, 4])  # tail page (4) last
    shuffled = _sel([4, 2, 0, 3, 1])  # tail page (4) first, interior reordered

    scale = 1.0 / math.sqrt(HEAD_DIM)
    expected = cpu_paged_select_attention(
        query, key, value, kv_seq_lens, block_table, ascending, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    asl_q = _cumulative_q_lengths(batch_size)
    try:
        out_asc = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            ascending,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        out_shuf = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            shuffled,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(out_asc.to(torch.float32), expected, **_TOL[dtype])
        torch.testing.assert_close(out_shuf.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_full_page_selection_order_within_precision(dtype):
    """Control for the tail-masking test: with no partial page, reordering the
    selection must leave the output unchanged up to the flash-attention precision
    floor.  This isolates finite-precision reorder noise (benign) from the
    partial-tail masking the ordering test above exercises.
    """
    gen = torch.Generator().manual_seed(24)
    batch_size, num_heads, num_kv_heads = 1, 2, 1
    vpc = 5
    kv_seq_lens = [vpc * BLOCK_SIZE]  # exact multiple -> every page full, no tail mask
    num_kv_blocks = 32
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, vpc, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)
    k = 8

    def _sel(order):
        s = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
        for h in range(num_heads):
            s[0, h, :vpc] = torch.tensor(order, dtype=torch.int32)
        return s

    ascending = _sel([0, 1, 2, 3, 4])
    shuffled = _sel([4, 2, 0, 3, 1])
    scale = 1.0 / math.sqrt(HEAD_DIM)
    expected = cpu_paged_select_attention(
        query, key, value, kv_seq_lens, block_table, ascending, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    asl_q = _cumulative_q_lengths(batch_size)
    try:
        out_asc = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            ascending,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        out_shuf = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            shuffled,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(out_asc.to(torch.float32), expected, **_TOL[dtype])
        torch.testing.assert_close(out_shuf.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mixed_effective_count_regimes_in_one_batch(dtype):
    """One call mixing a ``vpc < k`` request with a ``vpc >= k`` request.

    Request 0 (``vpc = 3 < k``) must be bounded by ``vpc`` and ignore its
    padding slots; request 1 (``vpc = 12 >= k``) is bounded by ``k``.  This
    stresses the per-(batch, head) ``effectiveBlockCount = min(k, vpc)`` -- a
    kernel using a single batch-wide bound would over-read request 0's padding.
    """
    gen = torch.Generator().manual_seed(22)
    num_heads, num_kv_heads, k = 4, 2, 8
    kv_seq_lens = [3 * BLOCK_SIZE, 12 * BLOCK_SIZE]  # vpc = 3 (< k) and 12 (>= k)
    batch_size = len(kv_seq_lens)
    width = max(_valid_page_count(s) for s in kv_seq_lens)
    num_kv_blocks = batch_size * width + 16
    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, width, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)

    selected = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
    # Request 0: pages [0, 1, 2]; slots [3:8] stay 0 -> if wrongly read, page 0
    # would be over-weighted and the output would diverge from the reference.
    selected[0, :, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    # Request 1: 8 distinct pages out of 12.
    for h in range(num_heads):
        selected[1, h] = torch.tensor([0, 2, 3, 5, 7, 8, 10, 11], dtype=torch.int32)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    expected = cpu_paged_select_attention(
        query, key, value, kv_seq_lens, block_table, selected, num_heads, scale, num_kv_heads, BLOCK_SIZE
    )
    asl_q = _cumulative_q_lengths(batch_size)
    try:
        actual = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            selected,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(actual.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


# ===========================================================================
# Test: end-to-end integration (metadata -> selection -> sparse attention)
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_end_to_end_select_then_attention(dtype):
    """Chain all three QUEST decode kernels through the real shared contract.

    ``quest_prefill_metadata`` -> ``quest_block_select_paged`` -> this kernel.
    The pages the selection kernel actually picks are fed straight into
    attention *and* into the reference, so this validates that the selection
    output (``[B, num_heads, k]`` int32 logical page ids) is consumed correctly
    -- a layout/dtype/semantics drift between the kernels shows up here even
    though the isolated tests would still pass.
    """
    gen = torch.Generator().manual_seed(123)
    batch_size, num_heads, num_kv_heads, k = 2, 4, 2, 8
    # vpc > k so selection runs its real top-k (sort) path; page-size multiples
    # so every page is complete and gets fresh metadata.
    kv_seq_lens = [10 * BLOCK_SIZE, 16 * BLOCK_SIZE]
    width = max(_valid_page_count(s) for s in kv_seq_lens)
    num_kv_blocks = batch_size * width + 16

    key, value = _build_paged_cache(num_kv_blocks, num_kv_heads, dtype, gen)
    block_table = _disjoint_block_table(batch_size, width, num_kv_blocks, gen)
    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(dtype)

    # One metadata block per request (vpc <= 128).
    metadata_block_tables = torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1)
    metadata_shape = (batch_size, BLOCK_SIZE, num_kv_heads, HEAD_DIM)
    maxblocks = torch.zeros(metadata_shape, dtype=dtype)
    minblocks = torch.zeros(metadata_shape, dtype=dtype)
    seq_lens = torch.tensor(kv_seq_lens, dtype=torch.int32)
    refresh_start = torch.zeros(batch_size, dtype=torch.int32)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    asl_q = _cumulative_q_lengths(batch_size)
    try:
        # 1) refresh QUEST metadata (per-page min/max of the key cache).
        max_npu, min_npu = maxblocks.npu(), minblocks.npu()
        torch.ops._C_ascend.npu_quest_prefill_metadata(
            key.npu(),
            block_table.npu(),
            refresh_start.npu(),
            seq_lens.npu(),
            metadata_block_tables.npu(),
            max_npu,
            min_npu,
        )
        # 2) select the top-k pages per (request, query head).
        selected = torch.ops._C_ascend.npu_quest_block_select_paged(
            query.npu(), max_npu, min_npu, metadata_block_tables.npu(), seq_lens.npu(), k, 0
        )
        torch.npu.synchronize()
        selected_cpu = selected.cpu()
        assert selected_cpu.dtype == torch.int32
        # 3) sparse attention over exactly those pages.
        out = ascendc_paged_select_attention_exec(
            query,
            key,
            value,
            asl_q,
            kv_seq_lens,
            block_table,
            selected_cpu,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        # Reference attends the SAME pages the selection kernel chose.
        expected = cpu_paged_select_attention(
            query,
            key,
            value,
            kv_seq_lens,
            block_table,
            selected_cpu,
            num_heads,
            scale,
            num_kv_heads,
            BLOCK_SIZE,
        )
        torch.testing.assert_close(out.to(torch.float32), expected, **_TOL[dtype])
    finally:
        _cleanup_npu()


# ===========================================================================
# Tests: host-side input validation (the wrapper's TORCH_CHECK contract)
# ===========================================================================
# The kernel-facing rank-3 cache layout: [num_blocks, block_size, num_kv_heads * head_dim].
def _valid_validation_inputs() -> dict:
    """A minimal, fully-valid call.  Negative tests mutate exactly one field."""
    gen = torch.Generator().manual_seed(0)
    batch, num_heads, num_kv_heads, k, vpc = 2, 4, 2, 8, 8
    num_kv_blocks = 32
    key4d, value4d = _build_paged_cache(num_kv_blocks, num_kv_heads, torch.float16, gen)
    block_table = _disjoint_block_table(batch, vpc, num_kv_blocks, gen)
    query = torch.randn((batch, num_heads, HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.float16)
    selected = torch.zeros((batch, num_heads, k), dtype=torch.int32)
    selected[:, :] = torch.arange(k, dtype=torch.int32)  # pages 0..k-1 < vpc
    return dict(
        query=query,
        key=_as_3d_cache(key4d),
        value=_as_3d_cache(value4d),
        asl_q=_cumulative_q_lengths(batch),
        asl_kv=[vpc * BLOCK_SIZE] * batch,
        block_table=block_table,
        selected=selected,
        num_heads=num_heads,
        scale=0.1,
        num_kv_heads=num_kv_heads,
        block_size=BLOCK_SIZE,
    )


def _call_raw(d: dict) -> torch.Tensor:
    out = torch.ops._C_ascend.npu_paged_select_attention(
        d["query"].npu(),
        d["key"].npu(),
        d["value"].npu(),
        d["asl_q"],
        d["asl_kv"],
        d["block_table"].npu(),
        d["selected"].npu(),
        d["num_heads"],
        d["scale"],
        d["num_kv_heads"],
        d["block_size"],
    )
    torch.npu.synchronize()
    return out.cpu()


@torch.inference_mode()
def test_validation_accepts_valid_baseline():
    """The baseline the negative tests mutate must itself be accepted and run."""
    d = _valid_validation_inputs()
    try:
        out = _call_raw(d)
        assert out.shape == d["query"].shape
    finally:
        _cleanup_npu()


@torch.inference_mode()
def test_rejects_non_divisible_head_counts():
    d = _valid_validation_inputs()
    d["num_kv_heads"] = 3  # 4 % 3 != 0
    try:
        with pytest.raises(RuntimeError):
            _call_raw(d)
    finally:
        _cleanup_npu()


@torch.inference_mode()
def test_rejects_head_count_mismatch():
    d = _valid_validation_inputs()
    d["num_heads"] = 8  # query/selected carry 4 heads
    try:
        with pytest.raises(RuntimeError):
            _call_raw(d)
    finally:
        _cleanup_npu()


@torch.inference_mode()
def test_rejects_dtype_mismatch():
    d = _valid_validation_inputs()
    d["key"] = d["key"].to(torch.bfloat16)  # query stays fp16
    try:
        with pytest.raises(RuntimeError):
            _call_raw(d)
    finally:
        _cleanup_npu()


@torch.inference_mode()
def test_rejects_non_int32_selected_indices():
    d = _valid_validation_inputs()
    d["selected"] = d["selected"].to(torch.int64)
    try:
        with pytest.raises(RuntimeError):
            _call_raw(d)
    finally:
        _cleanup_npu()


@torch.inference_mode()
def test_rejects_wrong_query_rank():
    d = _valid_validation_inputs()
    q = d["query"]
    d["query"] = q.reshape(q.shape[0], q.shape[1] * q.shape[2])  # rank-2
    try:
        with pytest.raises(RuntimeError):
            _call_raw(d)
    finally:
        _cleanup_npu()


@torch.inference_mode()
def test_rejects_mismatched_kv_width():
    d = _valid_validation_inputs()
    # key/value width must be num_kv_heads * head_dim; truncate one head_dim slab.
    d["key"] = d["key"][:, :, :-HEAD_DIM].contiguous()
    try:
        with pytest.raises(RuntimeError):
            _call_raw(d)
    finally:
        _cleanup_npu()
