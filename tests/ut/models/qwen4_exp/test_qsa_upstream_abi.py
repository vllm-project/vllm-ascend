# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.models.qwen4_exp import qsa


def test_mtp_qkv_projection_uses_graph_safe_upstream_path(monkeypatch) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    qkv = torch.empty((2, 24))
    positions = torch.arange(2)
    expected = tuple(torch.empty(1) for _ in range(4))
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(is_draft_model=True),
    )

    with patch.object(
        qsa.upstream_qsa.Qwen4ExpQSAAttention,
        "_project_qkv_gate",
        return_value=expected,
    ) as upstream_project:
        actual = layer._project_qkv_gate(qkv, positions)

    assert actual == expected
    upstream_project.assert_called_once_with(qkv, positions)


def test_fused_prepare_upstream_abi_materializes_qkv_and_gate(monkeypatch) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.head_dim = 4
    layer.kv_cache = torch.empty(1)
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


@pytest.mark.parametrize("manager_block_size", [128, 384, 768])
def test_compressed_metadata_recovers_manager_pages_for_drafting(manager_block_size):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.utils import select_common_block_size

    spec = MLAAttentionSpec(
        block_size=manager_block_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, tokens_per_state=4
    )
    kernel_size = select_common_block_size(manager_block_size, [qsa.AscendQSABackend, qsa.qsa_cache.QSAStateBackend])
    factor = manager_block_size // kernel_size
    manager = torch.tensor([[2, 0, -1], [1, 3, -1]], dtype=torch.int32)
    expanded = (manager.unsqueeze(-1) * factor + torch.arange(factor)).flatten(1)
    expanded = torch.where(manager.repeat_interleave(factor, dim=1) >= 0, expanded, -1).to(torch.int32)
    locs = torch.tensor([0, 1, 2], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=locs,
        query_start_loc_cpu=locs,
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=expanded,
        slot_mapping=torch.tensor([0, 1]),
    )
    builder = qsa.AscendQSAStateMetadataBuilder.__new__(qsa.AscendQSAStateMetadataBuilder)
    builder.kv_cache_spec = spec
    builder.is_circular_buffer = False
    builder._manager_table = None
    builder._manager_table_rows = 2
    # Inspect the real common-metadata boundary before upstream Triton kernels.
    with patch.object(qsa.qsa_cache.QSAMetadataBuilder, "build", side_effect=lambda prefix, metadata, fast: metadata):
        first = builder.build(0, common)
        torch.testing.assert_close(first.block_table_tensor, manager)
        ptr = first.block_table_tensor.data_ptr()
        second = builder.build_for_drafting(common, 1)
        assert second.block_table_tensor.data_ptr() == ptr
        torch.testing.assert_close(second.block_table_tensor, manager)
        assert common.block_table_tensor is expanded
        builder.is_circular_buffer = True
        assert builder.build(0, common).block_table_tensor is expanded
