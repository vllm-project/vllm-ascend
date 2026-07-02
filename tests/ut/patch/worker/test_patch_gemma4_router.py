import torch

from vllm_ascend.patch.worker.patch_gemma4_router import _cached_to_dtype


def test_cached_to_dtype_reuses_converted_tensor():
    module = torch.nn.Module()
    tensor = torch.ones(4, dtype=torch.float32)
    x = torch.empty(4, dtype=torch.float16)

    cached = _cached_to_dtype(module, "scale", tensor, x)
    cached_again = _cached_to_dtype(module, "scale", tensor, x)

    assert cached is cached_again
    assert cached.dtype == x.dtype
    assert cached.device == x.device


def test_cached_to_dtype_refreshes_after_tensor_update():
    module = torch.nn.Module()
    tensor = torch.ones(4, dtype=torch.float32)
    x = torch.empty(4, dtype=torch.float16)

    cached = _cached_to_dtype(module, "scale", tensor, x)
    tensor.add_(1.0)
    refreshed = _cached_to_dtype(module, "scale", tensor, x)

    assert refreshed is not cached
    assert torch.equal(refreshed, torch.full_like(refreshed, 2.0))
