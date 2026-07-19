import torch

from vllm_ascend.patch.worker.patch_gemma4_moe_topk import _is_precomputed_topk
from vllm_ascend.patch.worker.patch_gemma4_router import (
    _cached_to_dtype,
    _router_forward_topk,
)


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


def test_router_forward_topk_passes_persistent_scratch(monkeypatch):
    calls = []

    def fake_router_front(*args):
        calls.append(args)
        return torch.empty((1, 2)), torch.empty((1, 2), dtype=torch.int32)

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_dgemma_fused_router_front",
        fake_router_front,
        raising=False,
    )

    class DummyProj:
        weight = torch.empty((4, 8), dtype=torch.bfloat16)

    class DummyNorm:
        variance_epsilon = 1e-6

    module = torch.nn.Module()
    module.hidden_size = 8
    module.scale = torch.ones(8, dtype=torch.float32)
    module.proj = DummyProj()
    module.norm = DummyNorm()
    x = torch.empty((1, 8), dtype=torch.bfloat16)
    per_expert_scale = torch.ones(4, dtype=torch.float32)

    topk_weights, topk_ids = _router_forward_topk(module, x, per_expert_scale, top_k=2, sync_base=3)

    assert topk_weights.shape == (1, 2)
    assert topk_ids.dtype == torch.int32
    assert module._dgemma_router_norm_scratch.shape == (256, 8)
    assert module._dgemma_router_logits_scratch.shape == (256, 4)
    assert module._dgemma_router_sync_scratch.shape == (128,)
    assert calls[0][9] == 2
    assert calls[0][10] == 3


def test_is_precomputed_topk_requires_tensor_pair():
    weights = torch.empty((1, 2))
    ids = torch.empty((1, 2), dtype=torch.int32)

    assert _is_precomputed_topk((weights, ids))
    assert not _is_precomputed_topk(weights)
    assert not _is_precomputed_topk((weights, ids, ids))
