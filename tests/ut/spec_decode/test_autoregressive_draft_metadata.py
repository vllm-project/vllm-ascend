# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU regressions for metadata prepared before draft graph replay."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

SPECULATOR_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"


@pytest.fixture
def initialize_draft_metadata():
    """Execute the production initializer without importing the NPU runtime."""
    tree = ast.parse(SPECULATOR_PATH.read_text(encoding="utf-8"))
    speculator_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendAutoRegressiveSpeculator"
    )
    method = next(
        node
        for node in speculator_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_decode_draft_attn_metadatas"
    )
    namespace = {
        "Any": Any,
        "copy": copy,
        "AscendAttentionState": SimpleNamespace(DecodeOnly="decode_only"),
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SPECULATOR_PATH), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize(("num_reqs", "num_reqs_padded"), [(1, 4), (5, 8), (4, 4)])
def test_gqa_rebuilds_padded_decode_metadata(initialize_draft_metadata, num_reqs, num_reqs_padded):
    full_block_table = [[index + 1, 0] for index in range(num_reqs)]
    full_block_table += [[0, 0] for _ in range(num_reqs_padded - num_reqs)]
    target_metadata = SimpleNamespace(
        block_tables=full_block_table[:num_reqs],
        seq_lens_cpu=[14] * num_reqs,
        max_query_len=14,
    )
    decode_metadata = SimpleNamespace(
        block_tables=full_block_table[:num_reqs_padded],
        max_query_len=1,
    )
    builder = Mock(return_value={"draft_layer": decode_metadata})
    speculator = SimpleNamespace(
        attn_architecture="GQA",
        input_batch=SimpleNamespace(num_reqs=num_reqs, seq_lens_cpu_upper_bound=[14] * num_reqs),
        input_buffers=SimpleNamespace(draft_seq_lens_cpus=[[0] * num_reqs_padded for _ in range(2)]),
        _build_draft_attn_metadata=builder,
    )

    steps = initialize_draft_metadata(speculator, {"draft_layer": target_metadata}, num_reqs_padded)

    assert len(steps) == 2
    for step in steps:
        metadata = step["draft_layer"]
        assert len(metadata.block_tables) == num_reqs_padded
        assert metadata.block_tables == full_block_table
        assert metadata.block_tables is decode_metadata.block_tables
        assert len(metadata.seq_lens_cpu) == num_reqs_padded
        assert metadata.max_query_len == 1
        assert metadata.attn_state == "decode_only"
    builder.assert_called_once_with(
        num_reqs=num_reqs,
        num_reqs_padded=num_reqs_padded,
        num_tokens_padded=num_reqs_padded,
        seq_lens_cpu_upper_bound=speculator.input_batch.seq_lens_cpu_upper_bound,
        step=1,
    )
    first, second = (step["draft_layer"] for step in steps)
    assert first is not second
    first.seq_lens_cpu[0] = 15
    assert second.seq_lens_cpu[0] == 0
    assert target_metadata.seq_lens_cpu == [14] * num_reqs
    assert len(target_metadata.block_tables) == num_reqs


def test_mla_keeps_separate_decode_metadata_per_step(initialize_draft_metadata):
    decode = SimpleNamespace(seq_lens_list=[14])
    metadata = SimpleNamespace(decode=decode)
    speculator = SimpleNamespace(
        attn_architecture="MLA",
        input_batch=SimpleNamespace(num_reqs=1, seq_lens_cpu_upper_bound=[14]),
        input_buffers=SimpleNamespace(draft_seq_lens_cpus=[[0] * 4 for _ in range(2)]),
        _build_draft_attn_metadata=Mock(return_value={"draft_layer": metadata}),
    )

    steps = initialize_draft_metadata(speculator, {"draft_layer": SimpleNamespace(decode=None)}, 4)

    first, second = (step["draft_layer"] for step in steps)
    assert first.decode is not second.decode
    assert first.decode is not decode
    first.decode.seq_lens_list = [15, 0, 0, 0]
    assert second.decode.seq_lens_list == [14]
    assert decode.seq_lens_list == [14]


@pytest.mark.parametrize("architecture", ["DSA", "SFA"])
def test_sparse_backends_skip_generic_metadata_initialization(initialize_draft_metadata, architecture):
    speculator = SimpleNamespace(attn_architecture=architecture, _build_draft_attn_metadata=Mock())

    assert initialize_draft_metadata(speculator, {"draft_layer": object()}, 4) == []
    speculator._build_draft_attn_metadata.assert_not_called()


def test_missing_metadata_skips_initialization(initialize_draft_metadata):
    speculator = SimpleNamespace(attn_architecture="GQA", _build_draft_attn_metadata=Mock())

    assert initialize_draft_metadata(speculator, None, 4) is None
    speculator._build_draft_attn_metadata.assert_not_called()
