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


def test_qsa_projections_pipeline_indexer_vector_before_main_cube() -> None:
    launches: list[str] = []
    qkv = Mock()
    index_query = Mock()
    index_token_k = Mock()

    def project_main(_hidden_states):
        launches.append("main_qkv")
        return qkv, None

    def project_indexer(_hidden_states, _positions):
        launches.append("indexer")
        return index_query, index_token_k

    layer = SimpleNamespace(
        qkv_proj=Mock(side_effect=project_main),
        _start_indexer_projection=Mock(side_effect=project_indexer),
    )
    hidden_states = Mock()
    positions = Mock()

    actual = qsa.AscendQwen4ExpQSAAttention._start_staggered_projections(
        layer,
        hidden_states,
        positions,
    )

    assert actual == (qkv, index_query, index_token_k)
    assert launches == ["indexer", "main_qkv"]


def test_qsa_core_side_effect_op_writes_graph_output(monkeypatch) -> None:
    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)
    output = torch.empty((2, 2, 4))
    layer = Mock()
    monkeypatch.setattr(qsa, "_ascend_qsa_layer", lambda _name: layer)

    qsa.qwen4_exp_ascend_qsa_core_with_output(
        hidden_states,
        positions,
        output,
        "layer",
    )

    layer._forward_core_with_output.assert_called_once_with(
        positions,
        hidden_states,
        output,
    )


def test_qsa_opaque_forward_runs_output_projection_after_core(monkeypatch) -> None:
    class OutputProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input: torch.Tensor | None = None

        def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, None]:
            self.input = tensor.clone()
            return tensor.add(1), None

    def core_with_output(
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        del hidden_states, positions, layer_name
        output.copy_(torch.arange(output.numel(), dtype=output.dtype).reshape_as(output))

    monkeypatch.setattr(qsa.current_platform, "opaque_attention_op", lambda: True)
    monkeypatch.setattr(
        qsa.torch.ops.vllm,
        "qwen4_exp_ascend_qsa_core_with_output",
        core_with_output,
    )
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.num_heads = 2
    layer.head_dim = 4
    projection = OutputProjection()
    layer.o_proj = projection
    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)

    result = layer.forward(positions, hidden_states)

    expected_core = torch.arange(16, dtype=hidden_states.dtype).reshape(2, 8)
    torch.testing.assert_close(projection.input, expected_core)
    torch.testing.assert_close(result, expected_core + 1)


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

    def project_into_gate(qkv_arg, positions_arg, gate_out=None):
        assert qkv_arg is qkv
        assert positions_arg is positions
        gate_out.copy_(projected_gate.view_as(gate_out))
        return projected_q, projected_k, projected_v, gate_out.view(2, 8)

    layer._project_qkv_gate = Mock(side_effect=project_into_gate)

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

    layer._project_qkv_gate.assert_called_once_with(qkv, positions, gate_out=gate)
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
