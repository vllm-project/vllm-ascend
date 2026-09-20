from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache

from vllm_ascend.patch.worker import patch_bind_kv_cache


class _TestMambaLayer(MambaBase):
    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((2, 3), (4,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.bfloat16, torch.float32)

    @property
    def mamba_type(self):
        return None


def test_bind_kv_cache_calls_qsa_layer_hook_and_builds_runtime_views(
    monkeypatch,
) -> None:
    context: dict[str, object] = {}
    cache_config = SimpleNamespace(block_size=768)
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context),
        num_speculative_tokens=3,
    )
    layer_name = "model.layers.0.self_attn.indexer.raw_key_cache"
    layer = QSAKeyStateCache(
        head_size=32,
        dtype=torch.bfloat16,
        cache_rope_positions=True,
        prefix=layer_name,
        cache_config=cache_config,
        compress_ratio=4,
        vllm_config=vllm_config,
    )
    assert layer.kv_cache.numel() == 0
    assert not hasattr(layer, "key_cache")

    # Main's canonical allocation is [blocks, heads, states, width]. The QSA
    # hook converts it to its runtime [blocks, states, heads, width] view and
    # splits the key and exact int64 MRoPE-position tail without a copy.
    allocated = torch.zeros(2, 1, 8, 44, dtype=torch.bfloat16)
    runner_caches: list[torch.Tensor] = []
    monkeypatch.setattr(patch_bind_kv_cache, "vllm_version_is", lambda _: True)
    patch_bind_kv_cache.bind_kv_cache(
        {layer_name: allocated},
        context,
        runner_caches,
    )

    assert len(runner_caches) == 1
    assert runner_caches[0] is allocated
    assert layer.kv_cache.shape == (2, 8, 1, 44)
    assert layer.key_cache.shape == (2, 8, 1, 32)
    assert layer.rope_position_cache is not None
    assert layer.rope_position_cache.shape == (2, 8, 1, 3)
    assert layer.key_cache.untyped_storage().data_ptr() == allocated.untyped_storage().data_ptr()
    assert layer.rope_position_cache.untyped_storage().data_ptr() == allocated.untyped_storage().data_ptr()
    assert layer.key_cache.storage_offset() == 0
    assert layer.rope_position_cache.storage_offset() == 8

    layer.key_cache.fill_(1)
    assert torch.count_nonzero(layer.rope_position_cache).item() == 0


def test_bind_kv_cache_calls_packed_mamba_layer_hook_and_preserves_state_dtypes(
    monkeypatch,
) -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    allocated = torch.zeros(2, 1, 1, 28, dtype=torch.int8)
    runner_caches: list[torch.Tensor] = []
    monkeypatch.setattr(patch_bind_kv_cache, "vllm_version_is", lambda _: True)

    patch_bind_kv_cache.bind_kv_cache(
        {layer_name: allocated},
        {layer_name: layer},
        runner_caches,
    )

    assert runner_caches == [allocated]
    assert len(layer.kv_cache) == 2
    conv_state, temporal_state = layer.kv_cache
    assert conv_state.shape == (2, 2, 3)
    assert conv_state.dtype == torch.bfloat16
    assert temporal_state.shape == (2, 4)
    assert temporal_state.dtype == torch.float32
    assert conv_state.untyped_storage().data_ptr() == allocated.untyped_storage().data_ptr()
    assert temporal_state.untyped_storage().data_ptr() == allocated.untyped_storage().data_ptr()
    assert conv_state.storage_offset() == 0
    assert temporal_state.storage_offset() == 3


@pytest.mark.parametrize("container_type", [list, tuple])
def test_bind_kv_cache_accepts_materialized_mamba_states(
    monkeypatch,
    container_type,
) -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    conv_state = torch.zeros(5, 2, 3, dtype=torch.bfloat16)
    temporal_state = torch.zeros(5, 4, dtype=torch.float32)
    allocated = container_type((conv_state, temporal_state))
    runner_caches: list[object] = []
    monkeypatch.setattr(patch_bind_kv_cache, "vllm_version_is", lambda _: True)

    patch_bind_kv_cache.bind_kv_cache(
        {layer_name: allocated},
        {layer_name: layer},
        runner_caches,
    )

    assert runner_caches == [allocated]
    assert layer.kv_cache == (conv_state, temporal_state)
    assert layer.kv_cache[0].is_contiguous()
    assert layer.kv_cache[1].is_contiguous()
    assert layer.kv_cache[0].data_ptr() == conv_state.data_ptr()
    assert layer.kv_cache[1].data_ptr() == temporal_state.data_ptr()


@pytest.mark.parametrize(
    ("states", "error", "match"),
    [
        (
            [torch.zeros(5, 2, 3, dtype=torch.bfloat16)],
            ValueError,
            "state count mismatch",
        ),
        (
            [
                torch.zeros(5, 2, 4, dtype=torch.bfloat16),
                torch.zeros(5, 4, dtype=torch.float32),
            ],
            ValueError,
            "state 0 shape mismatch",
        ),
        (
            [
                torch.zeros(5, 2, 3, dtype=torch.float32),
                torch.zeros(5, 4, dtype=torch.float32),
            ],
            TypeError,
            "state 0 dtype mismatch",
        ),
        (
            [
                torch.zeros(5, 2, 3, dtype=torch.bfloat16),
                torch.zeros(4, 4, dtype=torch.float32),
            ],
            ValueError,
            "different block counts",
        ),
        (
            [
                torch.zeros(5, 2, 3, dtype=torch.bfloat16),
                object(),
            ],
            TypeError,
            "states must be tensors",
        ),
    ],
)
def test_bind_kv_cache_rejects_invalid_materialized_mamba_states(
    monkeypatch,
    states,
    error,
    match,
) -> None:
    layer_name = "model.layers.1.linear_attn"
    layer = _TestMambaLayer()
    monkeypatch.setattr(patch_bind_kv_cache, "vllm_version_is", lambda _: True)

    with pytest.raises(error, match=match):
        patch_bind_kv_cache.bind_kv_cache(
            {layer_name: states},
            {layer_name: layer},
            [],
        )
