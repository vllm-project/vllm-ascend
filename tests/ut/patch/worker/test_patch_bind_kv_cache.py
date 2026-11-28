from typing import cast

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


def test_bind_kv_cache_to_layers_unpacks_raw_mamba_pages() -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    state_alignment = 16
    layer.state_alignment = state_alignment
    slab = torch.zeros(192, dtype=torch.int8)
    pages = slab.as_strided((3, 1, 1, 32), (64, 32, 32, 1), storage_offset=16)
    caches = {layer_name: pages}

    patch_bind_kv_cache.bind_kv_cache_to_layers(caches, {layer_name: layer})

    assert isinstance(layer.kv_cache, tuple)
    conv_state, temporal_state = layer.kv_cache
    assert conv_state.shape == (3, 2, 3) and conv_state.dtype == torch.bfloat16
    assert temporal_state.shape == (3, 4) and temporal_state.dtype == torch.float32
    conv_state[1].fill_(2)
    temporal_state[2].fill_(3)
    expected = torch.zeros_like(slab)
    expected[80:92].view(torch.bfloat16).fill_(2)
    temporal_start = pages.storage_offset() + 2 * pages.stride(0) + state_alignment
    expected[temporal_start : temporal_start + 16].view(torch.float32).fill_(3)
    torch.testing.assert_close(slab, expected)
    assert caches[layer_name] is pages


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [(torch.float32, (3, 1, 1, 32)), (torch.int8, (3, 32)), (torch.int8, (3, 2, 1, 32))],
)
def test_bind_kv_cache_to_layers_preserves_noncanonical_mamba_tensor(dtype, shape) -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    allocated = torch.empty(shape, dtype=dtype)

    patch_bind_kv_cache.bind_kv_cache_to_layers({layer_name: allocated}, {layer_name: layer})

    assert layer.kv_cache is allocated


@pytest.mark.parametrize("container_type", [list, tuple])
@pytest.mark.parametrize("runner_binding", [False, True])
def test_bind_kv_cache_accepts_materialized_mamba_states(
    container_type,
    runner_binding,
) -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    conv_state = torch.zeros(5, 2, 3, dtype=torch.bfloat16)
    temporal_state = torch.zeros(5, 4, dtype=torch.float32)
    allocated = container_type((conv_state, temporal_state))
    caches = {layer_name: allocated}
    context = {layer_name: layer}
    if runner_binding:
        runner_caches: list[object] = []
        patch_bind_kv_cache.bind_kv_cache(caches, context, runner_caches)
        assert runner_caches == [allocated]
    else:
        patch_bind_kv_cache.bind_kv_cache_to_layers(caches, context)

    assert layer.kv_cache is allocated
    assert layer.kv_cache[0].is_contiguous()
    assert layer.kv_cache[1].is_contiguous()
    assert layer.kv_cache[0].data_ptr() == conv_state.data_ptr()
    assert layer.kv_cache[1].data_ptr() == temporal_state.data_ptr()


@pytest.mark.parametrize("runner_binding", [False, True])
def test_binding_retains_tuple_layout(runner_binding):
    name = "model.layers.0.self_attn"
    layer = _TensorOnlyAttentionLayer()
    allocated = (torch.zeros(2, 16, 1, 8), torch.zeros(2, 16, 1, 8))
    context = {name: layer}

    runner_caches: list[object] = []
    if runner_binding:
        patch_bind_kv_cache.bind_kv_cache({name: allocated}, context, runner_caches)
        assert runner_caches[0] is allocated
    else:
        patch_bind_kv_cache.bind_kv_cache_to_layers({name: allocated}, context)

    assert layer.kv_cache is allocated
    assert not layer.bind_called


def test_binding_keeps_scheduler_storage_separate_from_kernel_view():
    name = "model.layers.0.self_attn"
    storage = torch.arange(32).reshape(4, 8)
    kernel_view = storage.reshape(8, 4)
    layer = _TensorOnlyAttentionLayer()
    runner_caches: list[torch.Tensor] = []

    patch_bind_kv_cache.bind_kv_cache(
        {name: storage}, {name: layer}, runner_caches, layer_kv_caches={name: kernel_view}
    )

    assert runner_caches[0] is storage
    assert layer.kv_cache is kernel_view
    bound_cache = cast(torch.Tensor, layer.kv_cache)
    assert runner_caches[0].shape == (4, 8)
    assert bound_cache.shape == (8, 4)
    assert bound_cache.data_ptr() == storage.data_ptr()
    storage[0, 0] = 99
    assert bound_cache[0, 0] == 99
