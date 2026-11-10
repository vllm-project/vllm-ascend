from typing import cast

import pytest
import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase

from vllm_ascend.models.glm5next.kv_cache import Glm5NextTailCache
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


def test_glm_tail_binding_keeps_registered_tensor_and_wraps_model_cache():
    layer_name = "model.layers.1.indexer.tail_cache"
    tensor = torch.ones(4, 7, 8)
    registered = {layer_name: tensor}
    for binding in (patch_bind_kv_cache.bind_kv_cache, patch_bind_kv_cache.bind_kv_cache_to_layers):
        layer = Glm5NextTailCache.__new__(Glm5NextTailCache)
        torch.nn.Module.__init__(layer)
        if binding is patch_bind_kv_cache.bind_kv_cache:
            runner_caches: list[object] = []
            binding(registered, {layer_name: layer}, runner_caches)
            assert runner_caches[0] is tensor
        else:
            binding(registered, {layer_name: layer})
        assert registered[layer_name] is tensor
        assert isinstance(layer.kv_cache, list)
        assert layer.kv_cache[0] is tensor


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
