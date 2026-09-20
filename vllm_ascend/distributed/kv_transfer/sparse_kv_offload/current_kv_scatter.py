"""Write current K/Rope together using valid, distinct planner-owned slots."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "slots_ptr",
        "keys_ptr",
        "ropes_ptr",
        "cache_keys_ptr",
        "cache_ropes_ptr",
        "capacity",
    ]
)
def paired_scatter_kernel(
    slots_ptr,
    keys_ptr,
    ropes_ptr,
    cache_keys_ptr,
    cache_ropes_ptr,
    capacity,
    KEY_ELEMENTS: tl.constexpr,
    ROPE_ELEMENTS: tl.constexpr,
    KEY_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (slot >= 0) & (slot < capacity)
    key_offsets = tl.arange(0, KEY_BLOCK)
    rope_offsets = tl.arange(0, ROPE_BLOCK)
    keys = tl.load(keys_ptr + row * KEY_ELEMENTS + key_offsets, key_offsets < KEY_ELEMENTS, 0)
    ropes = tl.load(ropes_ptr + row * ROPE_ELEMENTS + rope_offsets, rope_offsets < ROPE_ELEMENTS, 0)
    tl.store(cache_keys_ptr + slot * KEY_ELEMENTS + key_offsets, keys, valid & (key_offsets < KEY_ELEMENTS))
    tl.store(cache_ropes_ptr + slot * ROPE_ELEMENTS + rope_offsets, ropes, valid & (rope_offsets < ROPE_ELEMENTS))


def try_paired_current_scatter(slots, keys, ropes, cache_keys, cache_ropes):
    """Return False for layouts that require the original scatter implementation."""
    tensors = (slots, keys, ropes, cache_keys, cache_ropes)
    if (
        slots.ndim != 1
        or slots.dtype != torch.int32
        or slots.numel() == 0
        or any(tensor.device != keys.device or not tensor.is_contiguous() for tensor in tensors)
        or keys.device.type != "npu"
        or any(tensor.ndim != 2 or tensor.dtype != torch.bfloat16 for tensor in tensors[1:])
        or keys.shape[0] != slots.numel()
        or ropes.shape[0] != slots.numel()
        or keys.shape[1] != cache_keys.shape[1]
        or ropes.shape[1] != cache_ropes.shape[1]
        or cache_keys.shape[0] != cache_ropes.shape[0]
        or cache_keys.shape[0] == 0
        or keys.shape[1] == 0
        or ropes.shape[1] == 0
    ):
        return False
    paired_scatter_kernel[(slots.numel(),)](
        slots,
        keys,
        ropes,
        cache_keys,
        cache_ropes,
        cache_keys.shape[0],
        KEY_ELEMENTS=keys.shape[1],
        ROPE_ELEMENTS=ropes.shape[1],
        KEY_BLOCK=triton.next_power_of_2(keys.shape[1]),
        ROPE_BLOCK=triton.next_power_of_2(ropes.shape[1]),
    )
    return True


def warmup_current_scatter(manager):
    device = manager.fused_plan_current_linear_slots_npu.device
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    keys = torch.zeros(1, manager.token_size_bytes_k // 2, dtype=torch.bfloat16, device=device)
    ropes = torch.zeros(1, manager.token_size_bytes_v // 2, dtype=torch.bfloat16, device=device)
    if not try_paired_current_scatter(slots, keys, ropes, torch.empty_like(keys), torch.empty_like(ropes)):
        raise RuntimeError("Unsupported FSA current-KV scatter warmup layout")
    torch.npu.synchronize()
