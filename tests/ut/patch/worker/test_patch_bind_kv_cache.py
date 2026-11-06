from typing import cast
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase

from vllm_ascend.patch.worker import patch_bind_kv_cache


class _TestMambaLayer(MambaBase):
    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((2, 3), (4,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.bfloat16, torch.float32)

    @property
    def mamba_type(self):
        return None


class _TensorOnlyAttentionLayer:
    """Models upstream attention hooks that only accept canonical tensors."""

    def __init__(self) -> None:
        self.kv_cache: object = torch.empty(0)
        self.bind_called = False

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.bind_called = True
        self.kv_cache = kv_cache.squeeze(1)


def test_bind_kv_cache_preserves_materialized_attention_tuple() -> None:
    """Tuple K/V views must not enter tensor-only upstream attention hooks."""
    layer_name = "model.layers.0.self_attn"
    layer = _TensorOnlyAttentionLayer()
    key_cache = torch.zeros(2, 16, 1, 8)
    value_cache = torch.zeros(2, 16, 1, 8)
    allocated = (key_cache, value_cache)

    patch_bind_kv_cache.bind_kv_cache_to_layers(
        {layer_name: allocated},
        {layer_name: cast(AttentionLayerBase, layer)},
    )

    assert layer.bind_called is False
    assert layer.kv_cache is allocated


@pytest.mark.parametrize("container_type", [list, tuple])
def test_bind_kv_cache_accepts_materialized_mamba_states(
    container_type,
) -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    conv_state = torch.zeros(5, 2, 3, dtype=torch.bfloat16)
    temporal_state = torch.zeros(5, 4, dtype=torch.float32)
    allocated = container_type((conv_state, temporal_state))
    runner_caches: list[object] = []

    patch_bind_kv_cache.bind_kv_cache(
        {layer_name: allocated},
        {layer_name: layer},
        runner_caches,
    )

    assert runner_caches == [allocated]
    assert layer.kv_cache is allocated
    assert layer.kv_cache[0].is_contiguous()
    assert layer.kv_cache[1].is_contiguous()
    assert layer.kv_cache[0].data_ptr() == conv_state.data_ptr()
    assert layer.kv_cache[1].data_ptr() == temporal_state.data_ptr()


@pytest.mark.parametrize("legacy_hook", [False, True])
@pytest.mark.parametrize("runner_binding", [False, True])
def test_binding_retains_tuple_layout_across_tracker_lifecycle(monkeypatch, legacy_hook, runner_binding):
    name = "model.layers.0.self_attn"
    layer = _TensorOnlyAttentionLayer()
    allocated = (torch.zeros(2, 16, 1, 8), torch.zeros(2, 16, 1, 8))
    context = {name: layer}
    groups = []
    hook = Mock()
    if legacy_hook:
        monkeypatch.setattr(patch_bind_kv_cache.utils, "share_replayssm_ring_trackers", hook, raising=False)
    else:
        monkeypatch.delattr(patch_bind_kv_cache.utils, "share_replayssm_ring_trackers", raising=False)

    runner_caches = []
    if runner_binding:
        patch_bind_kv_cache.bind_kv_cache({name: allocated}, context, runner_caches, kv_cache_groups=groups)
        assert runner_caches[0] is allocated
    else:
        patch_bind_kv_cache.bind_kv_cache_to_layers({name: allocated}, context, kv_cache_groups=groups)

    assert layer.kv_cache is allocated
    assert not layer.bind_called
    if legacy_hook:
        hook.assert_called_once_with([name], context, groups)
    else:
        hook.assert_not_called()


def test_binding_keeps_scheduler_storage_separate_from_kernel_view(monkeypatch):
    monkeypatch.delattr(patch_bind_kv_cache.utils, "share_replayssm_ring_trackers", raising=False)
    name = "model.layers.0.self_attn"
    storage = torch.arange(32).reshape(4, 8)
    kernel_view = storage.reshape(8, 4)
    layer = _TensorOnlyAttentionLayer()
    runner_caches = []

    patch_bind_kv_cache.bind_kv_cache(
        {name: storage}, {name: layer}, runner_caches, layer_kv_caches={name: kernel_view}
    )

    assert runner_caches[0] is storage
    assert layer.kv_cache is kernel_view
    assert runner_caches[0].shape == (4, 8)
    assert layer.kv_cache.shape == (8, 4)
    assert layer.kv_cache.data_ptr() == storage.data_ptr()
    storage[0, 0] = 99
    assert layer.kv_cache[0, 0] == 99
