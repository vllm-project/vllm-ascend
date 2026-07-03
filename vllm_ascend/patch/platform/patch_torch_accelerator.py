import torch


def _get_npu_memory_info(
    device: int | str | torch.device | None = None,
) -> tuple[int, int]:
    if device is None:
        device = torch.npu.current_device()
    elif isinstance(device, torch.device):
        if device.type != "npu":
            raise RuntimeError(f"Expected 'npu' device, got '{device.type}'")
        device = device.index if device.index is not None else torch.npu.current_device()
    elif isinstance(device, str):
        if not device.startswith("npu"):
            raise RuntimeError(f"Expected 'npu' device string, got '{device}'")
        parts = device.split(":")
        if len(parts) == 1:
            device = torch.npu.current_device()
        elif len(parts) == 2:
            device = int(parts[1])
        else:
            raise RuntimeError(f"Invalid device string format: '{device}'")

    return torch.npu.mem_get_info(device)


def patch_empty_cache() -> None:
    torch.npu.empty_cache()


torch.accelerator.empty_cache = patch_empty_cache

# Monkey-patch torch.accelerator memory APIs for NPU compatibility.
# Upstream vLLM (commit 747b068) replaced current_platform.memory_stats()
# with torch.accelerator.memory_stats(), but torch.accelerator does not
# properly delegate to NPU. We redirect to torch.npu.* equivalents.
torch.accelerator.memory_stats = torch.npu.memory_stats  # type: ignore[attr-defined]
torch.accelerator.get_memory_info = _get_npu_memory_info  # type: ignore[attr-defined]
torch.accelerator.memory_reserved = torch.npu.memory_reserved  # type: ignore[attr-defined]
torch.accelerator.memory_allocated = torch.npu.memory_allocated  # type: ignore[attr-defined]
torch.accelerator.reset_peak_memory_stats = torch.npu.reset_peak_memory_stats  # type: ignore[attr-defined]
# torch.accelerator.get_memory_info() routes through c10's
# CachingDeviceAllocator and asserts the backend allocator is a
# DeviceAllocator; NPU's caching allocator is not, so it crashes with
# "Allocator for npu is not a DeviceAllocator". Redirect to the
# NPU-native API. This is needed on v0.24.0+ where MemorySnapshot
# is constructed with an explicit device arg that triggers this path.
torch.accelerator.get_memory_info = torch.npu.mem_get_info  # type: ignore[attr-defined]
