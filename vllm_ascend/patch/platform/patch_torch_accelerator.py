import torch


def patch_empty_cache() -> None:
    torch.npu.empty_cache()


torch.accelerator.empty_cache = patch_empty_cache
