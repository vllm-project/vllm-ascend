# mypy: ignore-errors


import vllm
from vllm.triton_utils import tl, triton


@triton.jit
def batch_memcpy_kernel(
    src_ptrs,
    dst_ptrs,
    sizes,
    BLOCK_SIZE: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr,  # bytes per load element (4=uint32, 8=uint64)
):
    # 2D grid: axis-0 = batch index, axis-1 = chunk index within that tensor
    batch_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    # Load pointers and size for this batch entry
    src_base = tl.load(src_ptrs + batch_id)
    dst_base = tl.load(dst_ptrs + batch_id)
    size_bytes = tl.load(sizes + batch_id).to(tl.int64)  # enforce int64

    # Work in units of ELEMENT_SIZE bytes for wider memory transactions
    chunk_bytes = BLOCK_SIZE * ELEMENT_SIZE
    start_byte = chunk_id * chunk_bytes

    # Early exit if this chunk is entirely out of range
    if start_byte >= size_bytes:
        return

    # Cast base pointers once, outside any inner loop
    src_ptr = src_base.to(tl.pointer_type(tl.uint32 if ELEMENT_SIZE == 4 else tl.uint64))
    dst_ptr = dst_base.to(tl.pointer_type(tl.uint32 if ELEMENT_SIZE == 4 else tl.uint64))

    # Element-level offsets within this chunk
    offsets = tl.arange(0, BLOCK_SIZE)
    start_elem = start_byte // ELEMENT_SIZE
    size_elems = (size_bytes + ELEMENT_SIZE - 1) // ELEMENT_SIZE  # ceil div

    elem_offsets = start_elem + offsets
    mask = elem_offsets < size_elems

    # Wide load → store with streaming cache hint to avoid polluting L1
    data = tl.load(
        src_ptr + elem_offsets,
        mask=mask,
        cache_modifier=".cg",  # cache-global: bypass L1 for streaming data
    )
    tl.store(
        dst_ptr + elem_offsets,
        data,
        mask=mask,
        cache_modifier=".cg",
    )


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------
def batch_memcpy(src_ptrs, dst_ptrs, sizes, max_size: int | None = None):
    """
    Copy each src_ptrs[i] → dst_ptrs[i] for sizes[i] bytes.

    Args:
        src_ptrs:  1-D int64 tensor of source pointers, shape (B,)
        dst_ptrs:  1-D int64 tensor of destination pointers, shape (B,)
        sizes:     1-D int64 tensor of byte counts, shape (B,)
        max_size:  optional override for the largest tensor size (bytes).
                   If None, computed from sizes.max() — requires a device sync.
    """
    batch = src_ptrs.shape[0]

    if dst_ptrs.shape[0] != batch:
        raise ValueError(f"dst_ptrs batch dim {dst_ptrs.shape[0]} != {batch}")
    if sizes.shape[0] != batch:
        raise ValueError(f"sizes batch dim {sizes.shape[0]} != {batch}")

    # Determine the maximum number of chunks needed across all tensors.
    # Using a caller-supplied max_size avoids a device→host sync on the hot path.
    if max_size is None:
        max_size = int(sizes.max().item())

    # ELEMENT_SIZE=8 (uint64) gives the widest loads; adjust if alignment differs.
    ELEMENT_SIZE = 8
    # Compute max chunks conservatively using the largest possible BLOCK_SIZE.
    BLOCK_SIZE = 4096
    max_chunks = (max_size + BLOCK_SIZE * ELEMENT_SIZE - 1) // (BLOCK_SIZE * ELEMENT_SIZE)
    max_chunks = max(max_chunks, 1)

    grid = (batch, max_chunks)
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE, ELEMENT_SIZE)


vllm.v1.worker.mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
vllm.v1.worker.mamba_utils.batch_memcpy = batch_memcpy
