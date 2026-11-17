# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Level-2 sleep ownership for tensors created inside the ``weights`` mem-pool.

``NPUWorker.load_model()`` runs model construction, ``load_weights`` and
``process_weights_after_loading`` inside ``CaMemAllocator.use_memory_pool(
tag="weights")``. Level-2 sleep discards every allocation of that pool without a
CPU backup and ``wake_up()`` restores contents only for
``model.named_buffers()``. Runtime tensors created in that scope therefore come
back pointing at remapped, empty storage unless they are registered buffers,
and a guard such as ``if self._table is None`` keeps them from being rebuilt.

These tests pin the ownership contract: the tensors the forward paths read must
be discoverable through ``named_buffers()``, must stay non-persistent, and must
survive a deterministic level-2 discard/restore round trip.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.ops import rope_dsv4, rotary_embedding


def _level2_save(model: nn.Module) -> dict[str, torch.Tensor]:
    """Mirror ``NPUWorker.sleep(level=2)``."""
    return {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()}


def _level2_discard(model: nn.Module) -> None:
    """Mirror the level-2 unmap: pooled storage comes back undefined."""
    for _, buffer in model.named_buffers():
        buffer.fill_(float("nan"))


def _level2_restore(model: nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """Mirror ``NPUWorker.wake_up()``."""
    for name, buffer in model.named_buffers():
        if name in saved:
            buffer.data.copy_(saved[name].data)


def _buffer_is_visible(model: nn.Module, tensor: torch.Tensor) -> bool:
    """Whether the sleep backup path can reach ``tensor`` through the module."""
    return any(buffer is tensor for _, buffer in model.named_buffers())


# ---------------------------------------------------------------------------
# SFA indexer LI C8 Hadamard matrices
# ---------------------------------------------------------------------------


def _make_vllm_indexer() -> MagicMock:
    mock_indexer = MagicMock()
    mock_indexer.n_head = 64
    mock_indexer.head_dim = 128
    mock_indexer.topk_tokens = 2048
    mock_indexer.q_lora_rank = 1536
    mock_indexer.softmax_scale = 0.123
    mock_indexer.k_cache = MagicMock()
    mock_indexer.k_cache.prefix = "model.layers.0.indexer"
    return mock_indexer


@patch("vllm_ascend.attention.indexer.get_current_vllm_config")
@patch("vllm_ascend.attention.indexer.get_ascend_config")
def _make_indexer(
    mock_get_ascend_config: MagicMock,
    mock_get_vllm_config: MagicMock,
) -> AscendSFAIndexerBackend:
    mock_get_ascend_config.return_value.is_sparse_li_c8_layer.return_value = True
    mock_get_ascend_config.return_value.is_sparse_li_c4_layer.return_value = False
    # Keep the config a ``MagicMock`` so nested lookups such as
    # ``model_config.hf_config.model_type`` and ``attention_config.indexer_kv_dtype``
    # resolve without every level having to be spelled out.
    mock_get_vllm_config.return_value.model_config.hf_config.model_type = "deepseek_v32"
    mock_get_vllm_config.return_value.parallel_config.prefill_context_parallel_size = 1
    mock_get_vllm_config.return_value.attention_config.indexer_kv_dtype = "int8"
    with patch("vllm_ascend.attention.indexer.enable_sfa_dcp_sharded_indexer", return_value=False):
        indexer = AscendSFAIndexerBackend(_make_vllm_indexer(), qk_rope_head_dim=64)
    assert indexer.enable_sparse_li_c8
    return indexer


def _process_indexer_weights(indexer: AscendSFAIndexerBackend) -> None:
    """Run ``process_weights_after_loading`` without requiring an NPU device.

    The hook allocates its Hadamard matrices with ``device="npu"``. Resolve the
    real ``torch.tensor`` through the module at call time and drop the device
    argument so the allocator runs on CPU; the buffer-ownership contract under
    test is device agnostic.
    """
    real_tensor = importlib.import_module("torch").tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)

    with patch("vllm_ascend.attention.indexer.torch.tensor", side_effect=cpu_tensor):
        indexer.process_weights_after_loading()


def test_sfa_indexer_hadamard_is_a_non_persistent_buffer() -> None:
    indexer = _make_indexer()

    # The matrices only exist after the one-time post-load processing, but the
    # buffer slots must already belong to the module so the level-2 backup path
    # can find them.
    assert indexer.q_hadamard is None
    assert indexer.k_hadamard is None
    assert {"q_hadamard", "k_hadamard"} <= set(indexer._buffers)

    _process_indexer_weights(indexer)

    assert indexer.q_hadamard is not None
    assert indexer.k_hadamard is not None
    assert indexer._buffers["q_hadamard"] is indexer.q_hadamard
    assert indexer._buffers["k_hadamard"] is indexer.k_hadamard
    # ``persistent=False`` keeps runtime state out of checkpoints.
    assert "q_hadamard" not in indexer.state_dict()
    assert "k_hadamard" not in indexer.state_dict()

    # Re-running the hook (a weight reload) must keep the same tensor objects.
    q_hadamard = indexer.q_hadamard
    k_hadamard = indexer.k_hadamard
    _process_indexer_weights(indexer)
    assert indexer.q_hadamard is q_hadamard
    assert indexer.k_hadamard is k_hadamard


def test_sfa_indexer_hadamard_survives_level2_sleep() -> None:
    indexer = _make_indexer()
    _process_indexer_weights(indexer)
    assert indexer.q_hadamard is not None
    assert indexer.k_hadamard is not None

    model = nn.Module()
    model.impl = indexer
    expected_q = indexer.q_hadamard.clone()
    expected_k = indexer.k_hadamard.clone()
    assert _buffer_is_visible(model, indexer.q_hadamard)
    assert _buffer_is_visible(model, indexer.k_hadamard)

    saved = _level2_save(model)
    assert "impl.q_hadamard" in saved
    assert "impl.k_hadamard" in saved

    _level2_discard(model)
    assert not torch.equal(indexer.q_hadamard, expected_q)
    assert not torch.equal(indexer.k_hadamard, expected_k)

    _level2_restore(model, saved)
    assert torch.equal(indexer.q_hadamard, expected_q)
    assert torch.equal(indexer.k_hadamard, expected_k)


# ---------------------------------------------------------------------------
# DeepSeek-V4 DSA RoPE tables (process-global cache)
# ---------------------------------------------------------------------------

_ROPE_GROUP = "sleep_ownership_test"


def _make_rope_emb(layername: str, base: int) -> rope_dsv4.ComplexExpRotaryEmbedding:
    vllm_config = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    with patch.object(rope_dsv4, "current_platform", SimpleNamespace(device_type="cpu")):
        return rope_dsv4.ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=layername,
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=4,
            base=base,
            scaling_factor=1.0,
            beta_fast=7,
            beta_slow=2,
            rope_groups=[_ROPE_GROUP],
            is_neox_style=False,
        )


def test_dsv4_rope_tables_are_visible_to_the_sleep_backup() -> None:
    layername = "test.sleep.ownership.rope.0.attn"
    emb = _make_rope_emb(layername, base=12345)
    config_key, _ = rope_dsv4._ROPE_STATE.layer_info[layername]
    full_rope_cos, full_rope_sin = rope_dsv4._ROPE_STATE.full_rope_cache[config_key]

    model = nn.Module()
    model.rotary_emb = emb

    # The forward path reads the tables through the global cache, so the cached
    # tensors themselves have to be owned by the module tree.
    assert _buffer_is_visible(model, full_rope_cos)
    assert _buffer_is_visible(model, full_rope_sin)
    assert emb.get_buffer("full_rope_cos") is full_rope_cos
    assert emb.get_buffer("full_rope_sin") is full_rope_sin
    assert "full_rope_cos" not in emb.state_dict()
    assert "full_rope_sin" not in emb.state_dict()


def test_dsv4_rope_tables_survive_level2_sleep() -> None:
    base = 54321
    first_layer = "test.sleep.ownership.rope.1.attn"
    second_layer = "test.sleep.ownership.rope.2.attn"
    first = _make_rope_emb(first_layer, base=base)
    second = _make_rope_emb(second_layer, base=base)

    config_key, _ = rope_dsv4._ROPE_STATE.layer_info[first_layer]
    assert rope_dsv4._ROPE_STATE.layer_info[second_layer][0] == config_key
    full_rope_cos, full_rope_sin = rope_dsv4._ROPE_STATE.full_rope_cache[config_key]
    # Layers sharing one config key share the tables.
    assert second.get_buffer("full_rope_cos") is full_rope_cos

    model = nn.Module()
    model.rotary_emb = first
    model.rotary_emb_draft = second

    saved = _level2_save(model)
    # Sharing means the level-2 backup holds a single entry per table.
    assert sum(name.endswith("full_rope_cos") for name in saved) == 1
    assert sum(name.endswith("full_rope_sin") for name in saved) == 1

    expected_cos = full_rope_cos.clone()
    expected_sin = full_rope_sin.clone()
    _level2_discard(model)
    assert torch.isnan(full_rope_cos).all()
    _level2_restore(model, saved)

    assert torch.equal(full_rope_cos, expected_cos)
    assert torch.equal(full_rope_sin, expected_sin)


# ---------------------------------------------------------------------------
# Derived interleaved RoPE tables (module-level cache)
# ---------------------------------------------------------------------------


def _register_interleaved_rope_owner(cos_sin_cache: torch.Tensor) -> nn.Module:
    """Build the derived interleaved pair the way ``AscendRotaryEmbedding`` does."""
    owner = nn.Module()
    owner.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)
    # The recorder is first-module-wins, so start from a clean process state. The
    # globals are annotated as plain tensors because the forward paths only read
    # them after a rope module has been built.
    rotary_embedding._cos_cache = None  # type: ignore[assignment]
    rotary_embedding._sin_cache = None  # type: ignore[assignment]
    rotary_embedding._record_cos_and_sin_cache_interleaved(owner, cos_sin_cache)
    return owner


def test_interleaved_rope_tables_are_owned_non_persistent_buffers() -> None:
    cos_sin_cache = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    owner = _register_interleaved_rope_owner(cos_sin_cache)
    cos_cache = rotary_embedding._cos_cache
    sin_cache = rotary_embedding._sin_cache
    assert cos_cache is not None and sin_cache is not None

    # Every MLA/SFA rope lookup reads these through the module globals, so the
    # globals have to alias buffers the level-2 backup can reach.
    assert owner.get_buffer("_rope_derived_cos_cache") is cos_cache
    assert owner.get_buffer("_rope_derived_sin_cache") is sin_cache
    assert _buffer_is_visible(owner, cos_cache)
    assert _buffer_is_visible(owner, sin_cache)
    assert "_rope_derived_cos_cache" not in owner.state_dict()
    assert "_rope_derived_sin_cache" not in owner.state_dict()
    # The pair is the owner's rope table split into its cos and sin halves and
    # tiled back to full width, so the leading half of each derived table
    # reconstructs the module's table.
    hidden_dim = cos_sin_cache.shape[-1] // 2
    torch.testing.assert_close(
        torch.cat([cos_cache[:, :hidden_dim], sin_cache[:, :hidden_dim]], dim=-1),
        cos_sin_cache,
    )
    torch.testing.assert_close(cos_cache[:, hidden_dim:], cos_cache[:, :hidden_dim])
    torch.testing.assert_close(sin_cache[:, hidden_dim:], sin_cache[:, :hidden_dim])


def test_interleaved_rope_tables_survive_level2_sleep() -> None:
    cos_sin_cache = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 16)
    owner = _register_interleaved_rope_owner(cos_sin_cache)
    cos_cache = rotary_embedding._cos_cache
    sin_cache = rotary_embedding._sin_cache
    assert cos_cache is not None and sin_cache is not None
    expected_cos = cos_cache.clone()
    expected_sin = sin_cache.clone()

    model = nn.Module()
    model.rotary_emb = owner
    saved = _level2_save(model)
    assert sum(name.endswith("_rope_derived_cos_cache") for name in saved) == 1
    assert sum(name.endswith("_rope_derived_sin_cache") for name in saved) == 1

    _level2_discard(model)
    _level2_restore(model, saved)

    torch.testing.assert_close(cos_cache, expected_cos)
    torch.testing.assert_close(sin_cache, expected_sin)
    # The globals keep the very same objects, which is what keeps the addresses
    # baked into captured ACL graphs valid.
    assert rotary_embedding._cos_cache is cos_cache
    assert rotary_embedding._sin_cache is sin_cache
