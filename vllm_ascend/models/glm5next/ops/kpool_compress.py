# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend replacements for the GLM-5.3-Flash kpool indexer's fused kernels.

The upstream implementation fuses the Hadamard-128 rotation with the block-128
ue8m0 FP8 quantization into a single Triton kernel that keeps the rotated tensor
in registers. That kernel is CUDA-only, so the NPU path expresses the same
numerics in torch: the rotation becomes a matmul against a cached normalized
Hadamard matrix, and the quantization stays elementwise. Every step is
device-side, so the sequence remains ACL-graph safe.

The same applies to the two pieces around it. Compressing a pool of tokens into
one cache entry and expanding selected pools back into token ids are both fused
Triton kernels upstream; here they are torch expressions chosen to reproduce the
kernels' arithmetic step for step, including where the kernels round through
bfloat16. Neither touches the KV cache or attention metadata, so both can be
compared against a reference on CPU.
"""

import torch

# The indexer query is quantized against the e4m3 maximum, and the resulting
# scale is restricted to a power of two (ue8m0), matching the cached K basis.
FP8_E4M3_MAX = 448.0

# Guards rows whose rotated vector is all but zero, so log2 stays finite.
_MIN_ABSMAX = 1e-4

_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _normalized_hadamard(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached ``dim x dim`` Hadamard matrix scaled by ``dim ** -0.5``.

    Folding the normalization into the matrix keeps the rotation a single
    matmul. The matrix is symmetric, so it is used without a transpose.
    """
    key = (dim, device, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from scipy.linalg import hadamard  # type: ignore[import-untyped]
    except ImportError as err:
        raise ImportError(
            "The GLM-5.3-Flash kpool indexer requires SciPy for the Hadamard transform. Please install scipy."
        ) from err

    matrix = torch.tensor(hadamard(dim, dtype=float), dtype=dtype, device=device) * (dim**-0.5)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each 128-wide row by the Hadamard-128 transform, then FP8-quant.

    Args:
        q: ``[rows, 128]`` bf16 -- one head vector per row.

    Returns:
        (q_fp8 ``[rows, 128]`` float8_e4m3fn, scale ``[rows, 1]`` float32).
    """
    assert q.ndim == 2 and q.shape[1] == 128, q.shape

    rows, dim = q.shape
    if rows == 0:
        return (
            torch.empty((0, dim), dtype=torch.float8_e4m3fn, device=q.device),
            torch.empty((0, 1), dtype=torch.float32, device=q.device),
        )

    hadamard = _normalized_hadamard(dim, q.device, torch.float32)
    rotated = q.float() @ hadamard
    # The upstream kernel materializes bf16 between the rotation and the quant,
    # so the fp8 operand carries the same rounding on both backends.
    rotated = rotated.to(torch.bfloat16).to(torch.float32)

    absmax = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(_MIN_ABSMAX)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_E4M3_MAX)))
    q_fp8 = (rotated / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q_fp8, scale


def compress_pool(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Collapse each pool's ``pool_size`` token vectors down to a single vector.

    The mixing weights are a softmax over the pool's slots taken *per
    dimension*: dimension ``d`` of the result mixes the slots by
    ``softmax_s(slot_score[s, d] + ape[s, d])``. Every dimension gets its own
    mixture over the pooled tokens, which is what lets one cached key stand in
    for a whole run of them. ``ape`` is the learned per-slot position bias, so a
    slot's weight depends on where it sits inside the pool as well as on its
    gate score.

    Args:
        slot_k: ``[..., pool_size, head_dim]`` -- raw per-token indexer K.
        slot_score: ``[..., pool_size, head_dim]`` -- per-token gate score.
        ape: ``[pool_size, head_dim]`` -- per-slot position bias.

    Returns:
        ``[..., head_dim]`` in ``slot_k``'s dtype.
    """
    assert slot_score.shape == slot_k.shape, (slot_score.shape, slot_k.shape)
    assert ape.shape == slot_k.shape[-2:], (ape.shape, slot_k.shape)

    score = slot_score.float() + ape.float()
    # Subtracting the per-dimension max keeps the exponent finite; it cancels
    # against the denominator, so the weights themselves are unchanged.
    prob = torch.exp(score - score.amax(dim=-2, keepdim=True))
    pooled = (slot_k.float() * prob).sum(dim=-2) / prob.sum(dim=-2)
    return pooled.to(slot_k.dtype)


def kpool_compress_k(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress whole pools of indexer keys into one FP8 vector each.

    Args:
        slot_k: ``[n_pools, pool_size, 128]`` bf16 -- raw per-token indexer K.
        slot_score: ``[n_pools, pool_size, 128]`` -- per-token gate score.
        ape: ``[pool_size, 128]`` -- per-slot position bias.

    Returns:
        (k_fp8 ``[n_pools, 128]`` float8_e4m3fn, scale ``[n_pools, 1]`` float32),
        ready to be written to the pool-granular index-K cache.
    """
    assert slot_k.ndim == 3, slot_k.shape

    # The fused kernel rounds to bfloat16 here before rotating, and the FP8
    # operand only matches across backends if that rounding happens on both.
    return fwht128_quant_fp8(compress_pool(slot_k, slot_score, ape).to(torch.bfloat16))


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    *,
    selectable_pools: torch.Tensor | None = None,
    tail_width: int | None = None,
) -> torch.Tensor:
    """Turn selected pool ids into the token ids the sparse attention reads.

    Top-k runs at pool granularity, so it returns ``topk_tokens // pool_size``
    pool ids per query; each stands for the ``pool_size`` consecutive tokens it
    was compressed from. Those get expanded back out, and then the tokens after
    the selectable prefix are appended: they were never compressed into the
    index-K cache and so cannot be selected -- but they are the newest tokens,
    which must always be attended to.

    By default the appended tail is just the request's trailing incomplete
    pool. ``selectable_pools`` widens it, which is what makes a shared
    selection boundary usable: the operator applies one key length to every
    query row it is given, so a batch whose rows sit at different positions has
    to select over the prefix visible to *all* of them and cover the rest here.

    Args:
        pool_ids: ``[rows, topk_tokens // pool_size]`` -- selected pools per
            query row. Slots past what the row was offered hold whatever top-k
            padded with, which is not assumed to be negative.
        seq_lens: ``[rows]`` -- token-granular sequence length per query row.
        pool_size: tokens per pool (the checkpoint's ``index_kpool``).
        selectable_pools: ``[rows]`` -- how many leading pools were offered to
            top-k. Defaults to each row's own complete pool count, which is the
            widest prefix a single row can select from.
        tail_width: output width of the appended tail. Defaults to
            ``pool_size - 1``, which is exactly enough for the default
            ``selectable_pools``.

    Returns:
        ``[rows, topk_tokens + tail_width]`` int32 token ids, relative to the
        start of each request and padded with ``-1``.
    """
    assert pool_ids.ndim == 2, pool_ids.shape
    assert seq_lens.ndim == 1, seq_lens.shape
    assert seq_lens.shape[0] == pool_ids.shape[0], (seq_lens.shape, pool_ids.shape)

    rows, num_groups = pool_ids.shape
    device = pool_ids.device

    seq_lens = seq_lens.to(torch.int64).unsqueeze(-1)
    offered = seq_lens // pool_size if selectable_pools is None else selectable_pools.to(torch.int64).unsqueeze(-1)

    pool_ids = pool_ids.to(torch.int64).unsqueeze(-1)
    slot_offsets = torch.arange(pool_size, device=device)
    history = pool_ids * pool_size + slot_offsets
    # Top-k fills the slots beyond the offered prefix itself, and what it fills
    # them with is not part of the contract. Bound the ids by what the request
    # was actually allowed to select instead of trusting a negative pad: an id
    # at or past that count names a pool the request never had, and expanding
    # it points the attention at tokens whose KV was never written.
    in_range = (pool_ids >= 0) & (pool_ids < offered.unsqueeze(-1))
    history = torch.where(in_range, history, -1).reshape(rows, num_groups * pool_size)

    if tail_width is None:
        tail_width = pool_size - 1
    if tail_width == 0:
        # Every token is its own pool and each row selects its own prefix, so
        # there is nothing left over to append.
        return history.to(torch.int32)

    tail_start = offered * pool_size
    tail_offsets = torch.arange(tail_width, device=device)
    tail = torch.where(tail_offsets < seq_lens - tail_start, tail_start + tail_offsets, -1)

    return torch.cat((history, tail), dim=-1).to(torch.int32)


def shared_pool_prefix(seq_lens: torch.Tensor, query_lens: torch.Tensor, pool_size: int) -> torch.Tensor:
    """How many leading pools every query row of a request may select from.

    ``npu_lightning_indexer`` takes one key length per request, and its
    ``sparse_mode=3`` mask moves the boundary by one *key* per query row --
    which with pooled keys is ``pool_size`` tokens, four times too fast. So the
    selection runs unmasked over the prefix that is causally valid for the
    earliest row in the request, and the tail expansion covers the rest.

    Args:
        seq_lens: ``[num_requests]`` -- tokens known after this step, i.e. the
            last query row's sequence length.
        query_lens: ``[num_requests]`` -- query rows contributed this step.
        pool_size: tokens per pool.

    Returns:
        ``[num_requests]`` complete pool counts, at least zero.
    """
    return ((seq_lens - query_lens + 1).clamp_min(0) // pool_size).to(torch.int32)
