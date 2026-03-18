import torch


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
