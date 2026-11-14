# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import torch

from vllm_ascend.models.qwen4_exp import qsa


def test_indexer_projection_runs_on_auxiliary_stream(monkeypatch) -> None:
    main_stream = Mock()
    aux_stream = Mock()
    ready = Mock()
    main_stream.record_event.return_value = ready
    monkeypatch.setattr(qsa.torch.npu, "current_stream", lambda: main_stream)
    monkeypatch.setattr(qsa, "_qsa_indexer_stream", lambda: aux_stream)
    monkeypatch.setattr(qsa, "npu_stream_switch", lambda stream: nullcontext())

    query = Mock()
    token_k = Mock()
    indexer = Mock()
    metadata = SimpleNamespace(num_actual_tokens=2)
    indexer._metadata.return_value = (metadata, metadata)
    indexer.project_qk.return_value = (query, token_k)
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.indexer = indexer
    hidden_states = MagicMock()
    positions = MagicMock()

    actual = layer._start_indexer_projection(hidden_states, positions)

    assert actual == (query, token_k)
    hidden_states.record_stream.assert_called_once_with(aux_stream)
    positions.record_stream.assert_called_once_with(aux_stream)
    aux_stream.wait_event.assert_called_once_with(ready)
    indexer.project_qk.assert_called_once_with(
        hidden_states[:2],
        positions[..., :2],
    )
    query.record_stream.assert_called_once_with(main_stream)
    token_k.record_stream.assert_called_once_with(main_stream)


def test_qsa_forward_side_effect_op_copies_into_stable_output(monkeypatch) -> None:
    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)
    expected = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    output = torch.empty_like(expected)
    layer = Mock()
    layer._forward_impl.return_value = expected
    monkeypatch.setattr(qsa, "_ascend_qsa_layer", lambda _name: layer)

    qsa.qwen4_exp_ascend_qsa_forward_with_output(
        hidden_states,
        positions,
        output,
        "layer",
    )

    layer._forward_impl.assert_called_once_with(positions, hidden_states)
    torch.testing.assert_close(output, expected)


def test_main_uses_ascend_projection_and_draft_uses_upstream(monkeypatch) -> None:
    assert "_project_qkv_gate" in qsa.AscendQwen4ExpQSAAttention.__dict__
    assert "_main_fused_norm_rope_eligible" in qsa.AscendQwen4ExpQSAAttention.__dict__

    upstream_projection = Mock(return_value=("q", "k", "v", "gate"))
    monkeypatch.setattr(
        qsa.upstream_qsa.Qwen4ExpQSAAttention,
        "_project_qkv_gate",
        upstream_projection,
    )
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(),
    )
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "mtp.layers.48.self_attn.attn"
    qkv = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer._project_qkv_gate(qkv, positions) == ("q", "k", "v", "gate")
    upstream_projection.assert_called_once_with(qkv, positions)


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
