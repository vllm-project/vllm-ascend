# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp import qsa


def _plain_rotary_embedding(dtype: torch.dtype) -> SimpleNamespace:
    # Position 0 leaves the rotary channels alone, and position 1 rotates by pi/2.
    cache = torch.zeros((2, 64), dtype=dtype)
    cache[0, :32] = 1
    cache[1, 32:] = 1
    return SimpleNamespace(
        rotary_dim=64,
        is_neox_style=True,
        mrope_section=None,
        _match_cos_sin_cache_dtype=lambda tensor: cache.to(tensor),
    )


def _expected_rotary_rows(rows: torch.Tensor) -> torch.Tensor:
    expected = rows.clone()
    expected[1, ..., :32] = -rows[1, ..., 32:64]
    expected[1, ..., 32:64] = rows[1, ..., :32]
    return expected


@pytest.mark.parametrize("mrope_1d", [False, True])
@pytest.mark.parametrize("has_gate", [True, False])
def test_main_fallback_for_plain_rope_or_1d_mrope(monkeypatch, mrope_1d: bool, has_gate: bool) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.3.self_attn"
    layer.head_dim = 256
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.q_size = 512
    layer.kv_size = 256
    layer.attn_output_gate = has_gate
    layer.q_norm = torch.nn.RMSNorm(256, eps=1e-6, dtype=torch.bfloat16)
    layer.k_norm = torch.nn.RMSNorm(256, eps=1e-6, dtype=torch.bfloat16)
    layer.rotary_emb = _plain_rotary_embedding(torch.bfloat16)
    if mrope_1d:
        layer.rotary_emb.mrope_section = [11, 11, 10]
        layer.rotary_emb.mrope_interleaved = True
    fused = Mock(side_effect=AssertionError("ineligible inputs must not use fusion"))
    monkeypatch.setattr(torch.ops.vllm, "triton_split_qkv_rmsnorm_mrope", fused, raising=False)

    query = torch.arange(1024, dtype=torch.bfloat16).reshape(2, 2, 256)
    key = torch.arange(512, dtype=torch.bfloat16).reshape(2, 1, 256) + 1
    value = key + 100
    gate = query + 200
    q_gate = torch.cat((query, gate), dim=-1) if has_gate else query
    qkv = torch.cat((q_gate.flatten(1), key.flatten(1), value.flatten(1)), dim=-1)
    positions = torch.tensor([0, 1]) if mrope_1d else torch.tensor([[0, 1], [1, 0], [1, 0]])

    q, k, v, actual_gate = layer._project_qkv_gate(qkv, positions)

    torch.testing.assert_close(q, _expected_rotary_rows(layer.q_norm(query)).flatten(1))
    torch.testing.assert_close(k, _expected_rotary_rows(layer.k_norm(key)).flatten(1))
    torch.testing.assert_close(v, value.flatten(1))
    if has_gate:
        torch.testing.assert_close(actual_gate, gate.flatten(1))
    else:
        assert actual_gate is None
    fused.assert_not_called()


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("position_dims", [1, 2])
def test_draft_keeps_upstream_projection(monkeypatch, mrope: bool, position_dims: int) -> None:
    upstream = Mock(return_value=("q", "k", "v", "gate"))
    monkeypatch.setattr(qsa.upstream_qsa.Qwen4ExpQSAAttention, "_project_qkv_gate", upstream)
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "mtp.layers.48.self_attn.attn"
    layer.rotary_emb = SimpleNamespace(mrope_section=[11, 11, 10] if mrope else None)
    qkv = torch.empty((2, 8))
    positions = torch.arange(2)
    if position_dims == 2:
        positions = torch.stack((positions, positions + 1, positions + 2))
    assert layer._project_qkv_gate(qkv, positions) == ("q", "k", "v", "gate")
    expected_positions = positions[0] if position_dims == 2 and not mrope else positions
    upstream.assert_called_once()
    assert upstream.call_args.args[0] is qkv
    torch.testing.assert_close(upstream.call_args.args[1], expected_positions)


def test_indexer_falls_back_for_plain_rope() -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_head_dim = 128
    indexer.index_n_heads = 2
    indexer.index_kv_heads = 1
    indexer.rotary_emb = _plain_rotary_embedding(torch.bfloat16)
    indexer.q_layernorm = torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16)
    indexer.k_layernorm = torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16)
    query = torch.arange(512, dtype=torch.bfloat16).reshape(2, 2, 128)
    raw_key = torch.arange(256, dtype=torch.bfloat16).reshape(2, 1, 128) + 1
    indexer.index_qk_proj = Mock(return_value=(torch.cat((query.flatten(1), raw_key.flatten(1)), dim=-1), None))
    positions = torch.tensor([[0, 1], [1, 0], [1, 0]])

    actual_query, actual_raw_key = indexer.project_qk(torch.empty(2, 1), positions)
    compressed_key = indexer.normalize_compressed_keys(raw_key, positions.T)

    torch.testing.assert_close(actual_query, _expected_rotary_rows(indexer.q_layernorm(query)))
    torch.testing.assert_close(actual_raw_key, raw_key)
    torch.testing.assert_close(compressed_key, _expected_rotary_rows(indexer.k_layernorm(raw_key)))


def test_fused_prepare_upstream_abi_materializes_qkv_and_gate(monkeypatch) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.head_dim = 4
    layer.kv_cache = (torch.empty(1), torch.empty(1))
    layer.topk_indices_buffer = torch.empty((2, 3), dtype=torch.int32)

    main_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.tensor([0, 1]),
    )
    side_prefix = "model.layers.0.self_attn.indexer.raw_key_cache"
    side_metadata = SimpleNamespace(
        num_actual_tokens=2,
        token_to_req=torch.tensor([0, 0]),
        logical_positions=torch.tensor([0, 1]),
        seq_lens=torch.tensor([2]),
    )
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                layer.layer_name: main_metadata,
                side_prefix: side_metadata,
            }
        ),
    )

    selected = torch.zeros((2, 3), dtype=torch.int32)
    indexer = Mock(return_value=selected)
    indexer.output_width = 3
    indexer.raw_key_cache = SimpleNamespace(prefix=side_prefix)
    layer.indexer = indexer
    layer.impl = Mock()

    projected_q = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    projected_k = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    projected_v = projected_k + 10
    projected_gate = projected_q + 20
    layer._project_qkv_gate = Mock(return_value=(projected_q, projected_k, projected_v, projected_gate))

    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)
    qkv = torch.empty((2, 24))
    output = torch.empty((2, 2, 4))
    gate = torch.empty_like(output)

    layer._run_qsa(
        hidden_states,
        positions,
        None,
        None,
        None,
        output,
        qkv=qkv,
        gate=gate,
    )

    layer._project_qkv_gate.assert_called_once_with(qkv, positions)
    torch.testing.assert_close(gate, projected_gate.view_as(gate))
    layer.impl.do_kv_cache_update.assert_called_once()
    update_args = layer.impl.do_kv_cache_update.call_args.args
    torch.testing.assert_close(update_args[1], projected_k.view(2, 1, 4))
    torch.testing.assert_close(update_args[2], projected_v.view(2, 1, 4))
    layer.impl.forward_qsa.assert_called_once()
    forward_args = layer.impl.forward_qsa.call_args.args
    torch.testing.assert_close(forward_args[1], projected_q.view(2, 2, 4))


def test_fused_prepare_zeros_gate_without_runtime_metadata(monkeypatch) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    output = torch.ones((2, 2, 4))
    gate = torch.ones_like(output)

    layer._run_qsa(
        torch.empty((2, 8)),
        torch.arange(2),
        None,
        None,
        None,
        output,
        qkv=torch.empty((2, 24)),
        gate=gate,
    )

    assert not output.any()
    assert not gate.any()
