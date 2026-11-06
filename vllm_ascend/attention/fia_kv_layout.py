"""Zero-copy FIA views of block-strided paged K/V caches."""

import torch


def fia_kv_cache_views(key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Expose gaps as unused physical blocks; return the block-ID multiplier.

    For interleaved K/V pages, K addresses blocks 0, 2, 4, ... in a
    contiguous view and V uses the same indices from its own storage offset.
    The views alias the original allocation, including its page ownership.
    Unsupported layouts retain the existing FIA conversion path.
    """
    if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape or key.numel() == 0:
        return key, value, 1
    num_blocks, block_size, hidden_size = key.shape
    page_elements = block_size * hidden_size
    if (
        key.stride() != value.stride()
        or key.stride()[1:] != (hidden_size, 1)
        or key.stride(0) < page_elements
        or key.stride(0) % page_elements
    ):
        return key, value, 1
    block_stride = key.stride(0) // page_elements
    if block_stride == 1:
        return key, value, 1
    # End at the last logical page, not after its trailing gap. This keeps
    # both aliases within their backing even for the last layer and V page.
    shape = ((num_blocks - 1) * block_stride + 1, block_size, hidden_size)
    strides = (page_elements, hidden_size, 1)
    return key.as_strided(shape, strides), value.as_strided(shape, strides), block_stride
