import torch

from vllm_ascend.utils import vllm_version_is


def patch_empty_cache() -> None:
    torch.npu.empty_cache()


torch.accelerator.empty_cache = patch_empty_cache

# Monkey-patch torch.accelerator memory APIs for NPU compatibility.
# Upstream vLLM (commit 747b068) replaced current_platform.memory_stats()
# with torch.accelerator.memory_stats(), but torch.accelerator does not
# properly delegate to NPU. We redirect to torch.npu.* equivalents.
torch.accelerator.memory_stats = torch.npu.memory_stats  # type: ignore[attr-defined]
torch.accelerator.memory_reserved = torch.npu.memory_reserved  # type: ignore[attr-defined]
torch.accelerator.reset_peak_memory_stats = torch.npu.reset_peak_memory_stats  # type: ignore[attr-defined]
if not vllm_version_is("0.23.0"):
    # torch.accelerator.get_memory_info() routes through c10's
    # CachingDeviceAllocator and asserts the backend allocator is a
    # DeviceAllocator; NPU's caching allocator is not, so it crashes with
    # "Allocator for npu is not a DeviceAllocator". Redirect to the
    # NPU-native API. Only needed on v0.24.0+ where MemorySnapshot
    # is constructed with an explicit device arg that triggers this path.
    torch.accelerator.get_memory_info = torch.npu.mem_get_info  # type: ignore[attr-defined]
