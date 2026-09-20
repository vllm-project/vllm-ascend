"""Generate decode KV copy descriptors without reading device values on the host."""

import torch
import triton
import triton.language as tl

_DESCRIPTOR_BLOCK_SIZE = 32
_MAX_TOKEN_CAPACITY = 1 << 31


@triton.jit(
    do_not_specialize=[
        "slots_ptr",
        "keys_ptr",
        "ropes_ptr",
        "sources_ptr",
        "destinations_ptr",
        "lengths_ptr",
        "count_ptr",
        "rows",
        "capacity",
        "key_base",
        "rope_base",
    ]
)
def descriptor_kernel(
    slots_ptr,
    keys_ptr,
    ropes_ptr,
    sources_ptr,
    destinations_ptr,
    lengths_ptr,
    count_ptr,
    rows,
    capacity,
    key_base,
    rope_base,
    KEY_BYTES: tl.constexpr,
    ROPE_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = indices < rows
    slots = tl.load(slots_ptr + indices, active, other=-1).to(tl.int64)
    valid = (slots >= 0) & (slots < capacity)
    safe = tl.minimum(tl.maximum(slots, 0), capacity - 1)
    offsets = indices.to(tl.int64)
    tl.store(sources_ptr + indices, keys_ptr.to(tl.int64) + offsets * KEY_BYTES, active)
    tl.store(sources_ptr + rows + indices, ropes_ptr.to(tl.int64) + offsets * ROPE_BYTES, active)
    tl.store(destinations_ptr + indices, key_base.to(tl.int64) + safe * KEY_BYTES, active)
    tl.store(destinations_ptr + rows + indices, rope_base.to(tl.int64) + safe * ROPE_BYTES, active)
    tl.store(lengths_ptr + indices, tl.where(valid, KEY_BYTES, 0), active)
    tl.store(lengths_ptr + rows + indices, tl.where(valid, ROPE_BYTES, 0), active)
    if tl.program_id(0) == 0:
        tl.store(count_ptr, rows * 2)


def build_writeback_descriptors(
    slots,
    keys,
    ropes,
    sources,
    destinations,
    lengths,
    count,
    capacity,
    key_base,
    rope_base,
    key_bytes,
    rope_bytes,
):
    rows = slots.numel()
    if slots.dtype != torch.int64 or keys.dtype != torch.bfloat16 or ropes.dtype != torch.bfloat16:
        raise ValueError("Writeback descriptors require int64 slots and BF16 KV tensors")
    if rows <= 0 or capacity <= 0 or capacity >= _MAX_TOKEN_CAPACITY or rows * 2 > sources.numel():
        raise ValueError("Writeback descriptor rows/capacity exceed supported buffers")
    descriptor_kernel[(triton.cdiv(rows, _DESCRIPTOR_BLOCK_SIZE),)](
        slots,
        keys,
        ropes,
        sources,
        destinations,
        lengths,
        count,
        rows,
        capacity,
        key_base,
        rope_base,
        KEY_BYTES=key_bytes,
        ROPE_BYTES=rope_bytes,
        BLOCK=_DESCRIPTOR_BLOCK_SIZE,
    )


def warmup_writeback_descriptors(manager):
    """Compile before graph capture without submitting a MemFabric copy."""
    device = manager.d2h_src_ptrs_npu.device
    slots = torch.zeros(1, dtype=torch.int64, device=device)
    keys = torch.empty(1, dtype=torch.bfloat16, device=device)
    ropes = torch.empty_like(keys)
    build_writeback_descriptors(
        slots,
        keys,
        ropes,
        manager.d2h_src_ptrs_npu,
        manager.d2h_dst_ptrs_npu,
        manager.d2h_lengths_npu,
        manager.d2h_size_npu,
        1,
        keys.data_ptr(),
        ropes.data_ptr(),
        manager.token_size_bytes_k,
        manager.token_size_bytes_v,
    )
    torch.npu.synchronize()
