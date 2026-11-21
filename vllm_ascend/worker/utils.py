from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    prepare_kernel_block_sizes,
    select_common_block_size,
)

from vllm_ascend.utils import vllm_version_is


def _copy_dense_pages_async(block_views: Sequence[torch.Tensor], copies: Sequence[KVCacheBlockCopy]) -> None:
    """Submit consecutive dense cache segments in one stream-ordered batch."""
    count = block_views[0].shape[0]
    pairs = []
    for copy in copies:
        src, dst = copy.src_block_id, copy.dst_block_id
        if not (-count <= src < count and -count <= dst < count):
            raise IndexError("KV cache copy block index out of range")
        pairs.append((src % count, dst % count))
    overlaps = not {src for src, _ in pairs}.isdisjoint(dst for _, dst in pairs)
    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    sizes: list[int] = []
    device = block_views[0].device
    with torch.npu.device(device):
        stream = torch.npu.current_stream()
        if overlaps:
            # Each segment's scatter is queued before the next gather.
            # Only the largest segment needs scratch space.
            max_page_size = max(view.numel() * view.element_size() for view in block_views) // count
            scratch = torch.empty(len(pairs) * max_page_size, dtype=torch.uint8, device=device)
            scratch.record_stream(stream)
            scratch_base = scratch.data_ptr()
        for blocks in block_views:
            element_size = blocks.element_size()
            size = blocks.numel() // count * element_size
            if size == 0:
                continue
            stride = blocks.stride(0) * element_size
            blocks.record_stream(stream)
            base = blocks.data_ptr()
            sources = [base + src * stride for src, _ in pairs]
            destinations = [base + dst * stride for _, dst in pairs]
            if overlaps:
                # Snapshot each segment before writing it. Segments may alias,
                # so preserve their original gather/scatter order on the stream.
                scratch_ptrs = [scratch_base + i * size for i in range(len(pairs))]
                src_ptrs.extend(sources + scratch_ptrs)
                dst_ptrs.extend(scratch_ptrs + destinations)
                sizes.extend([size] * (2 * len(pairs)))
            else:
                src_ptrs.extend(sources)
                dst_ptrs.extend(destinations)
                sizes.extend([size] * len(pairs))
        if not sizes:
            return
        torch.ops._C_ascend.swap_blocks_batch(
            torch.tensor(src_ptrs, dtype=torch.int64, device="cpu"),
            torch.tensor(dst_ptrs, dtype=torch.int64, device="cpu"),
            torch.tensor(sizes, dtype=torch.int64, device="cpu"),
            2,
        )
        torch.autograd.graph.increment_version(block_view for block_view in block_views if block_view.numel())


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | Sequence[torch.Tensor | None] | None],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    """Copy logical cache blocks for Ascend's segmented cache layout.

    Unlike the upstream block-major allocation, an Ascend cache allocation can
    contain multiple block-indexed tensor segments. For example, Mamba stores
    all convolution states before all SSM states in the same storage. Treating
    that complete storage as ``[num_blocks, page_size]`` therefore copies the
    wrong byte ranges. Copy every tensor segment as ``num_blocks`` complete
    physical pages instead. A page may span multiple kernel-level cache blocks.
    """
    if not kv_cache_block_copies:
        return

    cache_tensors: list[torch.Tensor] = []
    seen_tensors: set[tuple[torch.device, int] | tuple[torch.device, int, tuple[int, ...], tuple[int, ...], int]] = (
        set()
    )
    for entry in kv_caches:
        if entry is None:
            continue
        if isinstance(entry, torch.Tensor):
            tensors = entry.unbind(0) if entry.ndim == 5 and entry.shape[0] == 2 else (entry,)
        else:
            tensors = entry
        for tensor in tensors:
            if tensor is None:
                continue
            data_ptr = tensor.data_ptr()
            # Hybrid models can overlay layers of different kinds on one
            # allocation, so views at the same address can differ in geometry.
            # Main keys on that geometry; the release lane keys on (device, address).
            key = (
                (tensor.device, data_ptr)
                if vllm_version_is("0.33.0")
                else (tensor.device, data_ptr, tuple(tensor.shape), tuple(tensor.stride()), tensor.element_size())
            )
            if key in seen_tensors:
                continue
            seen_tensors.add(key)
            cache_tensors.append(tensor)

    if not cache_tensors:
        return

    # A paired hybrid slot exposes a contiguous main-page view and strided
    # recurrent component views of the same storage. Copying both can replay a
    # cyclic copy over already-updated source bytes. Keep the enclosing view
    # once; independent indexer key/scale planes still need their own copies.
    contiguous_regions = [
        (
            i,
            tensor.untyped_storage().data_ptr(),
            tensor.data_ptr(),
            tensor.data_ptr() + tensor.numel() * tensor.element_size(),
        )
        for i, tensor in enumerate(cache_tensors)
        if tensor.is_contiguous()
    ]
    selected = []
    for i, tensor in enumerate(cache_tensors):
        storage = tensor.untyped_storage().data_ptr()
        start = tensor.data_ptr()
        span = (
            1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        ) * tensor.element_size()
        end = start + span
        enclosed = any(
            storage == parent_storage
            and parent_start <= start
            and end <= parent_end
            and (parent_end - parent_start > span or parent_index < i)
            for parent_index, parent_storage, parent_start, parent_end in contiguous_regions
            if parent_index != i
        )
        if not enclosed:
            selected.append(tensor)
    cache_tensors = selected

    device = cache_tensors[0].device
    use_native_copy = device.type == "npu" and hasattr(torch.ops._C_ascend, "swap_blocks_batch")
    dense_block_views: list[torch.Tensor] = []
    for tensor in cache_tensors:
        assert tensor.device == device
        assert tensor.shape[0] % num_blocks == 0
        # FP8 is storage here. NPU stack/index operators need byte views;
        # same-size bitcasts preserve the non-contiguous page geometry.
        if tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.int8)
        kernel_blocks_per_block = tensor.shape[0] // num_blocks
        # Page-strided MLA views are non-contiguous, so a flat ``view`` would
        # either fail or materialize a copy. Split only dim 0 to preserve the
        # original strided storage while copying every kernel block in a page.
        blocks = torch.as_strided(
            tensor,
            size=(num_blocks, kernel_blocks_per_block, *tensor.shape[1:]),
            stride=(tensor.stride(0) * kernel_blocks_per_block, tensor.stride(0), *tensor.stride()[1:]),
            storage_offset=tensor.storage_offset(),
        )
        if use_native_copy and not blocks.requires_grad:
            page = blocks[0]
            if page.is_contiguous() and blocks.stride(0) >= page.numel():
                dense_block_views.append(blocks)
                continue
        if dense_block_views:
            _copy_dense_pages_async(dense_block_views, kv_cache_block_copies)
            dense_block_views.clear()
        source_blocks = torch.stack([blocks[copy.src_block_id] for copy in kv_cache_block_copies])
        for index, copy in enumerate(kv_cache_block_copies):
            blocks[copy.dst_block_id].copy_(source_blocks[index])
    if dense_block_views:
        _copy_dense_pages_async(dense_block_views, kv_cache_block_copies)


def _group_tensor_strides(
    kv_cache_config: KVCacheConfig,
    layer_names: Sequence[str],
) -> list[tuple[int, int]]:
    """``(layer_stride, block_stride)`` of every tensor holding ``layer_names``."""
    return [
        (tensor.layer_stride, tensor.block_stride)
        for tensor in kv_cache_config.kv_cache_tensors
        if not set(tensor.layers).isdisjoint(layer_names)
    ]


def prepare_ascend_kernel_block_sizes(
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list[AttentionGroup]],
) -> list[int]:
    """Upstream ``prepare_kernel_block_sizes`` with an alias-aware packing check.

    vLLM #57169 refuses to split a cache group whose KV tensors are
    block-outermost packed, detected as ``block_stride > layer_stride``. Ascend
    layerwise reuse aliases several layers inside one physical slot, so those
    tensors carry ``layer_stride == 0`` and trip that test although their pages
    stay dense and contiguous. The group then keeps the manager block size,
    leaving NoPE SFA addressing 256-token blocks while its backend, metadata
    builder and block table all address 128-token kernel blocks. Re-split the
    groups whose only evidence of packing is aliasing.

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    # Upstream drops only encoder-only groups, so this index tracks its output.
    index = 0
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        if isinstance(spec, EncoderOnlyAttentionSpec):
            continue
        if index >= len(kernel_block_sizes):
            break
        manager_block_size = group.kv_cache_spec.block_size
        strides = _group_tensor_strides(kv_cache_config, group.layer_names)
        block_outermost_packed = any(0 < layer_stride < block_stride for layer_stride, block_stride in strides)
        aliases_layers = any(layer_stride == 0 and block_stride > 0 for layer_stride, block_stride in strides)
        if (
            spec.has_layer_views
            and isinstance(spec, AttentionSpec)
            and aliases_layers
            and not block_outermost_packed
            and kernel_block_sizes[index] == manager_block_size
            and group_id < len(attn_groups)
            and attn_groups[group_id]
        ):
            kernel_block_sizes[index] = select_common_block_size(
                manager_block_size,
                [attn_group.backend for attn_group in attn_groups[group_id]],
                [attn_group.kv_cache_spec for attn_group in attn_groups[group_id]],
            )
        index += 1
    return kernel_block_sizes


def get_single_raw_mla_backing(raw_cache: object) -> torch.Tensor | None:
    """Extract one raw MLA backing from either allocator representation.

    Pure MLA allocations use a one-element tuple so generic K/V unpacking does
    not mistake the backing for a `(key, value)` pair. Hybrid/shared pools can
    instead store a bare tensor slice for an MLA layer.
    """
    if isinstance(raw_cache, torch.Tensor):
        return raw_cache
    if isinstance(raw_cache, tuple) and len(raw_cache) == 1 and isinstance(raw_cache[0], torch.Tensor):
        return raw_cache[0]
    return None


def row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Return row-major strides for a shape without allocating tensor storage."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


def make_page_strided_cache_view(
    raw_tensor: torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    page_size_bytes: int,
    offset_bytes: int = 0,
) -> torch.Tensor:
    """Create a first-axis page-strided view over a raw cache allocation."""
    dtype_size = get_dtype_size(dtype)
    strides = row_major_strides(tuple(shape))
    storage_offset_bytes = raw_tensor.storage_offset() * raw_tensor.element_size() + offset_bytes
    assert storage_offset_bytes % dtype_size == 0
    return torch.as_strided(
        raw_tensor.view(dtype),
        size=tuple(shape),
        stride=(page_size_bytes // dtype_size, *strides[1:]),
        storage_offset=storage_offset_bytes // dtype_size,
    )


@contextmanager
def disable_compilation(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily disable compilation for all nested compiled modules."""
    # Multimodal wrappers can nest the compiled backbone below language_model.
    # Snapshot before changing flags; modules() also deduplicates shared modules.
    previous_states = [
        (module, module.do_not_compile) for module in model.modules() if hasattr(module, "do_not_compile")
    ]
    try:
        for module, _ in previous_states:
            module.do_not_compile = True
        yield
    finally:
        for module, previous in reversed(previous_states):
            module.do_not_compile = previous
