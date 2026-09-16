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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.ops import rope_dsv4


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


@patch("vllm_ascend.attention.indexer.enable_dsa_cp", return_value=False)
@patch("vllm_ascend.attention.indexer.get_current_hardware_profile")
@patch("vllm_ascend.attention.indexer.get_current_vllm_config")
@patch("vllm_ascend.attention.indexer.get_ascend_config")
def _make_indexer(
    mock_get_ascend_config: MagicMock,
    mock_get_vllm_config: MagicMock,
    mock_hw_profile: MagicMock,
    _mock_enable_dsa_cp: MagicMock,
) -> AscendSFAIndexerBackend:
    mock_get_ascend_config.return_value.is_sparse_li_c8_layer.return_value = True
    mock_get_vllm_config.return_value = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v32")),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    mock_hw_profile.return_value.supports.return_value = True
    indexer = AscendSFAIndexerBackend(_make_vllm_indexer(), qk_rope_head_dim=64)
    assert indexer.enable_sparse_li_c8
    return indexer


def _process_indexer_weights(indexer: AscendSFAIndexerBackend) -> None:
    """Run ``process_weights_after_loading`` without requiring an NPU device."""
    real_tensor = torch.tensor

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
# DCP SFA sparse-index remap constants
# ---------------------------------------------------------------------------


def _make_dcp_impl(owner: nn.Module | None) -> AscendSFADCPImpl:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.dcp_size = 2
    impl.dcp_rank = 0
    impl._dcp_interleave_size = 2
    impl._dcp_index_topk = 8
    impl._remap_order = torch.arange(8, dtype=torch.float32)
    impl._remap_invalid_index = torch.tensor(-1.0)
    impl.o_proj = nn.Linear(4, 4)
    static_forward_context = {} if owner is None else {impl.layer_name: owner}
    compilation_config = SimpleNamespace(static_forward_context=static_forward_context)
    impl.vllm_config = SimpleNamespace(compilation_config=compilation_config)
    return impl


def test_dcp_remap_constants_are_owned_by_the_attention_layer() -> None:
    layer = nn.Module()
    impl = _make_dcp_impl(layer)

    impl._register_remap_buffers()

    assert layer._buffers["_dcp_sfa_remap_order"] is impl._remap_order
    assert layer._buffers["_dcp_sfa_remap_invalid_index"] is impl._remap_invalid_index
    assert "_dcp_sfa_remap_order" not in layer.state_dict()
    assert _buffer_is_visible(layer, impl._remap_order)
    assert _buffer_is_visible(layer, impl._remap_invalid_index)

    # Registration is idempotent across weight reloads.
    impl._register_remap_buffers()
    assert layer._buffers["_dcp_sfa_remap_order"] is impl._remap_order


def test_dcp_remap_constants_fall_back_to_the_projection_owner() -> None:
    impl = _make_dcp_impl(owner=None)

    impl._register_remap_buffers()

    assert impl.o_proj._buffers["_dcp_sfa_remap_order"] is impl._remap_order
    assert impl.o_proj._buffers["_dcp_sfa_remap_invalid_index"] is impl._remap_invalid_index


def test_dcp_remap_constants_survive_level2_sleep() -> None:
    layer = nn.Module()
    impl = _make_dcp_impl(layer)
    impl._register_remap_buffers()

    replicated_indices = torch.tensor([[0, 2, 1, 3, 4, 6, -1, 5]], dtype=torch.int32)
    expected = impl._remap_sparse_indices(replicated_indices.clone())

    saved = _level2_save(layer)
    assert saved
    _level2_discard(layer)
    _level2_restore(layer, saved)

    torch.testing.assert_close(impl._remap_sparse_indices(replicated_indices.clone()), expected)


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
