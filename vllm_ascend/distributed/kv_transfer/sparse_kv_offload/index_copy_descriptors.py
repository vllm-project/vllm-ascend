"""Build graph-stable Mooncake index_copy destinations on the NPU."""

import torch
from vllm.triton_utils import tl, triton

_DESCRIPTOR_BLOCK_SIZE = 32


@triton.jit(
    do_not_specialize=[
        "slots_ptr",
        "destinations_ptr",
        "rows",
        "host_capacity",
    ]
)
def sanitize_index_copy_destinations_kernel(
    slots_ptr,
    destinations_ptr,
    rows,
    host_capacity,
    BLOCK: tl.constexpr,
):
    """Map invalid logical slots to the private Host-KV sink row."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < rows
    slots = tl.load(slots_ptr + offsets, mask=active, other=-1).to(tl.int64)
    valid = (slots >= 0) & (slots < host_capacity)
    destinations = tl.where(valid, slots, host_capacity)
    tl.store(destinations_ptr + offsets, destinations, mask=active)


def build_index_copy_destinations(
    slots: torch.Tensor,
    destinations: torch.Tensor,
    host_capacity: int,
) -> None:
    """Build destinations without compaction, padding, or a D2H round trip."""
    rows = slots.numel()
    if slots.dtype != torch.int64 or destinations.dtype != torch.int64:
        raise ValueError("Mooncake index_copy slots and destinations must use int64")
    if slots.device != destinations.device or not slots.is_contiguous() or not destinations.is_contiguous():
        raise ValueError("Mooncake index_copy descriptor tensors must be contiguous on one device")
    if rows <= 0 or rows > destinations.numel() or host_capacity <= 0:
        raise ValueError(
            "Mooncake index_copy descriptor dimensions are invalid: "
            f"rows={rows}, capacity={destinations.numel()}, "
            f"host_capacity={host_capacity}"
        )

    sanitize_index_copy_destinations_kernel[(triton.cdiv(rows, _DESCRIPTOR_BLOCK_SIZE),)](
        slots,
        destinations,
        rows,
        host_capacity,
        BLOCK=_DESCRIPTOR_BLOCK_SIZE,
    )


def warmup_index_copy_destinations(manager) -> None:
    """Compile the destination sanitizer before graph capture."""
    device = manager.d2h_dst_idx_npu.device
    slots = torch.zeros(1, dtype=torch.int64, device=device)
    build_index_copy_destinations(
        slots,
        manager.d2h_dst_idx_npu,
        1,
    )
    torch.npu.synchronize()
