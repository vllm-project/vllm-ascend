import torch

from vllm_ascend.ops.gbsa import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)


GBSA_COMPRESS_RATIO = 4
GBSA_RESIDUAL_TOKENS = GBSA_COMPRESS_RATIO - 1
GBSA_PAGE_SIZE = 128


def is_qsa_gbsa_cache_compatible(cache: torch.Tensor) -> bool:
    """Return whether a cache can be exposed as zero-copy GBSA pages.

    GBSA requires every physical page to occupy a non-overlapping, uniformly
    strided region. A storage-block stride that is not an integer multiple of
    one 128-token page cannot be represented by a single PA_BBND tensor plus a
    remapped block table without copying.
    """
    if cache.ndim != 4 or min(cache.shape) == 0:
        return False
    if cache.shape[1] != GBSA_PAGE_SIZE and cache.shape[2] == GBSA_PAGE_SIZE:
        cache = cache.transpose(1, 2)
    _, storage_page_size, num_heads, head_dim = cache.shape
    if storage_page_size % GBSA_PAGE_SIZE:
        return False
    row_elements = num_heads * head_dim
    storage_block_elements = storage_page_size * row_elements
    subpage_elements = GBSA_PAGE_SIZE * row_elements
    return bool(
        cache.stride(3) == 1
        and (num_heads == 1 or cache.stride(2) == head_dim)
        and cache.stride(1) == row_elements
        and cache.stride(0) >= storage_block_elements
        and cache.stride(0) % subpage_elements == 0
    )


def _as_gbsa_page_pool(
    cache: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    """Expose a large-page cache as a zero-copy, 128-token physical pool.

    The cache may have padding between storage blocks (a non-dense dim-0
    stride), but every storage block must be dense internally.  Padding is
    represented by unreferenced 128-token virtual page ids.  The caller remaps
    the block table with separate block and subpage strides, so no data is
    copied; the cache planner guarantees that padding is a whole-page multiple.
    """
    if cache.ndim != 4:
        raise ValueError(f"QSA GBSA requires a 4-D KV cache, got {cache.ndim}-D")

    num_blocks, storage_page_size, num_heads, head_dim = cache.shape
    if min(cache.shape) == 0:
        raise ValueError("QSA GBSA requires a nonempty KV cache")
    if storage_page_size % GBSA_PAGE_SIZE:
        raise ValueError(
            "QSA GBSA requires storage pages divisible into 128-token pages, "
            f"got shape={tuple(cache.shape)}"
        )

    row_elements = num_heads * head_dim
    storage_block_elements = storage_page_size * row_elements
    subpage_elements = GBSA_PAGE_SIZE * row_elements
    subpages = storage_page_size // GBSA_PAGE_SIZE

    if cache.stride(3) != 1:
        raise ValueError("QSA GBSA requires a contiguous KV head dimension")
    if num_heads > 1 and cache.stride(2) != head_dim:
        raise ValueError("QSA GBSA requires dense KV heads inside each token")
    if cache.stride(1) != row_elements:
        raise ValueError(
            "QSA GBSA requires dense tokens inside each storage page, "
            f"got shape={tuple(cache.shape)} stride={cache.stride()} "
            f"storage_offset={cache.storage_offset()}"
        )
    if cache.stride(0) < storage_block_elements:
        raise ValueError("QSA GBSA does not support overlapping KV storage pages")
    if cache.stride(0) % subpage_elements:
        raise ValueError(
            "QSA GBSA cannot represent this non-contiguous cache as "
            "non-overlapping 128-token pages without copying, "
            f"got shape={tuple(cache.shape)} stride={cache.stride()} "
            f"storage_offset={cache.storage_offset()}"
        )
    virtual_page_elements = subpage_elements
    physical_block_stride = cache.stride(0) // subpage_elements
    physical_subpage_stride = 1
    physical_page_count = (
        (num_blocks - 1) * physical_block_stride
        + (subpages - 1) * physical_subpage_stride
        + 1
    )
    page_pool = cache.as_strided(
        size=(physical_page_count, GBSA_PAGE_SIZE, num_heads, head_dim),
        stride=(virtual_page_elements, row_elements, head_dim, 1),
    )
    return (
        page_pool,
        physical_block_stride,
        physical_subpage_stride,
        subpages,
    )


def qsa_gbsa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA GBSA requires 4-D K/V caches")
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA GBSA requires matching K/V cache shapes")
    if key_cache.shape[1] != GBSA_PAGE_SIZE and key_cache.shape[2] == GBSA_PAGE_SIZE:
        key_cache = key_cache.transpose(1, 2)
        value_cache = value_cache.transpose(1, 2)

    num_blocks, storage_page_size, _, _ = key_cache.shape
    logical_max_seqlen_kv = num_blocks * storage_page_size
    storage_page_size = key_cache.shape[1]
    if storage_page_size % GBSA_PAGE_SIZE:
        raise ValueError(
            "QSA GBSA requires storage pages divisible into 128-token pages, "
            f"got shape={tuple(key_cache.shape)}"
        )
    if storage_page_size != GBSA_PAGE_SIZE:
        key_cache, block_stride, subpage_stride, subpages = _as_gbsa_page_pool(
            key_cache
        )
        value_cache, value_block_stride, value_subpage_stride, value_subpages = (
            _as_gbsa_page_pool(value_cache)
        )
        if (block_stride, subpage_stride, subpages) != (
            value_block_stride,
            value_subpage_stride,
            value_subpages,
        ):
            raise ValueError("QSA GBSA requires matching K/V cache strides")
        subpage_offsets = torch.arange(
            subpages,
            dtype=block_table.dtype,
            device=block_table.device,
        ) * subpage_stride
        expanded_table = block_table.unsqueeze(-1) * block_stride + subpage_offsets
        block_table = torch.where(
            block_table.unsqueeze(-1) >= 0,
            expanded_table,
            torch.full_like(expanded_table, -1),
        ).flatten(1)
    num_kv_heads = key_cache.shape[2]
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QSA GBSA requires FP16/BF16 query")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("QSA GBSA requires matching Q/K/V dtypes")
    if compact_indices.shape[1] <= GBSA_RESIDUAL_TOKENS:
        raise ValueError("QSA GBSA requires compact LI-v2 indices")
    if min(key_cache.shape[:3]) == 0 or min(block_table.shape) == 0:
        raise ValueError("QSA GBSA requires nonempty KV cache and block table")
    if key_cache.shape != value_cache.shape or key_cache.shape[-1] != query.shape[-1]:
        raise ValueError("QSA GBSA requires matching KV shapes and Q/K head dimensions")
    if key_cache.shape[1] != GBSA_PAGE_SIZE:
        raise ValueError("QSA GBSA requires 128-token paged KV cache")
    if query.shape[1] % num_kv_heads:
        raise ValueError("QSA GBSA requires query heads divisible by KV heads")
    if seq_lens.shape != (block_table.shape[0],):
        raise ValueError("QSA GBSA sequence lengths must match block table rows")
    if token_to_req.shape != (query.shape[0],):
        raise ValueError("QSA GBSA request mapping must match query rows")
    if logical_positions.shape != (query.shape[0],):
        raise ValueError("QSA GBSA logical positions must match query rows")

    sparse_width = compact_indices.shape[1] - GBSA_RESIDUAL_TOKENS
    row_requests = token_to_req.long()
    row_request_lens = seq_lens[row_requests].to(torch.int32)
    row_visible_lens = torch.minimum(
        logical_positions.to(torch.int32).add(1).clamp_min(0),
        row_request_lens,
    )
    sparse_idx = compact_indices[:, :sparse_width].to(torch.int32)
    full_blocks = torch.div(
        row_visible_lens,
        GBSA_COMPRESS_RATIO,
        rounding_mode="floor",
    )
    valid = (sparse_idx >= 0) & (sparse_idx < full_blocks.unsqueeze(1))
    order = torch.argsort((~valid).to(torch.int32), dim=1, stable=True)
    sparse_idx = sparse_idx.gather(1, order)
    sparse_count = valid.sum(dim=1).to(torch.int32)
    sparse_idx = sparse_idx.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
    sparse_count = sparse_count.unsqueeze(0).expand(num_kv_heads, -1).contiguous()

    cu_q = torch.arange(query.shape[0] + 1, dtype=torch.int64, device=query.device)
    row_block_table = block_table[row_requests].contiguous()
    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=row_visible_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=1,
        is_consistent_topk=False,
    )
    metadata = generic_block_sparse_attention_metadata(
        sparse_idx,
        sparse_count,
        query.shape[1],
        num_kv_heads,
        query.shape[-1],
        [1, GBSA_COMPRESS_RATIO],
        max_seqlen_q=1,
        max_seqlen_kv=int(logical_max_seqlen_kv),
        **common,
    )
    result, _ = generic_block_sparse_attention(
        query.contiguous(),
        key_cache,
        value_cache,
        sparse_idx,
        sparse_count,
        [1, GBSA_COMPRESS_RATIO],
        metadata=metadata,
        block_table=row_block_table,
        softmax_scale=query.shape[-1] ** -0.5,
        **common,
    )
    output.copy_(result)
    return output
