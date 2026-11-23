# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for QSA step preparation ownership."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


class _BaseBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.compress_ratio = kv_cache_spec.compress_ratio
        self.vllm_config = vllm_config

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return common_attn_metadata.result


def _load_builder(prepare_lengths, prepare_gbsa):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/qsa.py"
    module = ast.parse(source.read_text())
    definition = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendQSAIndexerMetadataBuilder"
    )
    namespace = {
        "Any": Any,
        "AttentionSpec": object,
        "CommonAttentionMetadata": object,
        "QSAIndexerMetadataBuilderBase": _BaseBuilder,
        "math": math,
        "prepare_qsa_gbsa_shared": prepare_gbsa,
        "prepare_qsa_lengths": prepare_lengths,
        "torch": torch,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["AscendQSAIndexerMetadataBuilder"]


def test_compressed_builder_owns_decode_step_preparation():
    calls = []

    def prepare_lengths(lengths, compressed, residual, requests, ratio):
        calls.append(("lengths", requests, ratio))
        compressed[:requests].copy_(lengths[:requests] // ratio)
        residual[:requests].copy_(lengths[:requests] % ratio)

    def prepare_gbsa(positions, lengths, mapping, table, visible, row_table, rows):
        calls.append(("gbsa", rows, table.data_ptr()))
        visible[:rows].copy_(positions[:rows] + 1)
        row_table[:rows].copy_(table[mapping[:rows].long()])

    builder_cls = _load_builder(prepare_lengths, prepare_gbsa)
    main_layer = SimpleNamespace(indexer=SimpleNamespace(index_n_heads=3, index_head_dim=128))
    context = {"model.layers.0.self_attn.attn": main_layer}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=8),
    )
    builder = builder_cls(
        SimpleNamespace(compress_ratio=4),
        ["model.layers.0.self_attn.indexer.compressed_key_cache"],
        config,
        torch.device("cpu"),
    )
    metadata = SimpleNamespace(
        num_prefills=0,
        num_actual_tokens=3,
        seq_lens=torch.tensor([5, 8], dtype=torch.int32),
        logical_positions=torch.tensor([4, 7, 0], dtype=torch.int64),
        token_to_req=torch.tensor([0, 1, 0], dtype=torch.int32),
    )
    table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    common = SimpleNamespace(result=metadata, block_table_tensor=table)

    result = builder.build(0, common)

    assert result is metadata
    assert calls == [("lengths", 2, 4), ("gbsa", 3, table.data_ptr())]
    torch.testing.assert_close(metadata.qsa_compressed_lengths[:2], torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(metadata.qsa_residual_lengths[:2], torch.tensor([1, 0], dtype=torch.int32))
    torch.testing.assert_close(metadata.qsa_gbsa_visible_lengths[:3], torch.tensor([5, 8, 1], dtype=torch.int32))
    torch.testing.assert_close(metadata.qsa_gbsa_row_block_table[:3], table[[0, 1, 0]])
    assert metadata.qsa_score_weights.shape == (8, 3)
    assert metadata.qsa_gbsa_cu_query.tolist() == list(range(9))
