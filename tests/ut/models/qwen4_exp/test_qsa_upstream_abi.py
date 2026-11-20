# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp import qsa
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("layout", ["contiguous", "row_stride", "column_stride"])
def test_main_fused_norm_rope_requires_contiguous_qkv(layout) -> None:
    layer = SimpleNamespace(
        attn_output_gate=True,
        q_size=512,
        kv_size=256,
        num_heads=2,
        num_kv_heads=1,
        head_dim=256,
        rotary_emb=SimpleNamespace(
            mrope_section=[11, 11, 10],
            rotary_dim=64,
            is_neox_style=True,
            mrope_interleaved=True,
        ),
    )
    width = 2 * layer.q_size + 2 * layer.kv_size
    if layout == "row_stride":
        qkv = torch.empty((4, width), dtype=torch.bfloat16)[::2]
    elif layout == "column_stride":
        qkv = torch.empty((2, 2 * width), dtype=torch.bfloat16)[:, ::2]
    else:
        qkv = torch.empty((2, width), dtype=torch.bfloat16)

    eligible = qsa.AscendQwen4ExpQSAAttention._main_fused_norm_rope_eligible(layer, qkv, torch.arange(2))
    assert eligible == (layout == "contiguous")


def test_main_qkv_projection_falls_back_when_fused_abi_is_ineligible(
    monkeypatch,
) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    monkeypatch.setattr(
        layer,
        "_main_fused_norm_rope_eligible",
        Mock(return_value=False),
    )
    expected = (
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
    )
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(
        qsa.upstream_qsa.Qwen4ExpQSAAttention,
        "_project_qkv_gate",
        fallback,
    )
    qkv = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer._project_qkv_gate(qkv, positions) is expected
    fallback.assert_called_once_with(qkv, positions)


def test_main_falls_back_when_fused_projection_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value=("q", "k", "v", "gate"))
    monkeypatch.setattr(
        qsa.upstream_qsa.Qwen4ExpQSAAttention,
        "_project_qkv_gate",
        upstream_projection,
    )
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer._main_fused_norm_rope_eligible = Mock(return_value=False)
    qkv = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer._project_qkv_gate(qkv, positions) == ("q", "k", "v", "gate")
    upstream_projection.assert_called_once_with(qkv, positions)


def test_indexer_query_falls_back_when_fused_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value=("query", "key"))
    monkeypatch.setattr(
        qsa.upstream_indexer.QSAIndexer,
        "project_qk",
        upstream_projection,
    )
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_n_heads = 2
    layer.index_head_dim = 4
    layer.index_kv_heads = 1
    layer.index_qk_proj = Mock(return_value=(torch.empty((2, 12)), None))
    layer._fused_norm_rope_eligible = Mock(return_value=False)
    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer.project_qk(hidden_states, positions) == ("query", "key")
    upstream_projection.assert_called_once_with(hidden_states, positions)


def test_indexer_compressed_key_falls_back_when_fused_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value="keys")
    monkeypatch.setattr(
        qsa.upstream_indexer.QSAIndexer,
        "normalize_compressed_keys",
        upstream_projection,
    )
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_head_dim = 4
    layer.rotary_emb = SimpleNamespace(mrope_section=[1, 1, 0])
    layer._fused_norm_rope_eligible = Mock(return_value=False)
    pooled = torch.empty((2, 4))
    first_positions = torch.arange(6).reshape(2, 3)

    assert layer.normalize_compressed_keys(pooled, first_positions) == "keys"
    upstream_projection.assert_called_once_with(pooled, first_positions)


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
    # Main's indexer returns (selection, prepared main outputs).
    indexer = Mock(return_value=selected if vllm_version_is("0.33.0") else (selected, None))
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
    assert layer.impl.forward_qsa.call_count + layer.impl.forward.call_count == 1
    forward_call = layer.impl.forward_qsa if vllm_version_is("0.33.0") else layer.impl.forward
    forward_args = forward_call.call_args.args
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
