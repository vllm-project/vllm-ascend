# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Regression for the block-table shape used by padded GQA draft graphs."""

import ast
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SPECULATOR_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"


def test_gqa_draft_block_table_matches_padded_batch():
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
    namespace: dict[str, Any] = {
        "Any": Any,
        "copy": copy,
        "AscendAttentionState": SimpleNamespace(DecodeOnly="decode_only"),
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SPECULATOR_PATH), "exec"), namespace)

    num_reqs = 1
    num_reqs_padded = 4
    max_blocks = 1000
    # The initializer only passes block tables through, so shape stubs suffice.
    target_metadata = SimpleNamespace(
        block_tables=SimpleNamespace(shape=(num_reqs, max_blocks)),
    )
    speculator = SimpleNamespace(
        attn_architecture="GQA",
        input_batch=SimpleNamespace(num_reqs=num_reqs, seq_lens_cpu_upper_bound=[14]),
        input_buffers=SimpleNamespace(draft_seq_lens_cpus=[[0] * num_reqs_padded]),
        _build_draft_attn_metadata=lambda **kwargs: {
            "draft_layer": SimpleNamespace(block_tables=SimpleNamespace(shape=(kwargs["num_reqs_padded"], max_blocks)))
        },
    )

    steps = namespace[method.name](speculator, {"draft_layer": target_metadata}, num_reqs_padded)

    assert steps[0]["draft_layer"].block_tables.shape == (num_reqs_padded, max_blocks)


def test_eager_draft_metadata_is_updated_before_use():
    """The metadata builder must apply the current draft step immediately."""
    tree = ast.parse(SPECULATOR_PATH.read_text(encoding="utf-8"))
    speculator_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendAutoRegressiveSpeculator"
    )
    method = next(
        node
        for node in speculator_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_draft_attn_metadata"
    )

    events: list[Any] = []
    metadata = SimpleNamespace(attn_state=None)

    class BaseSpeculator:
        def _build_draft_attn_metadata(self, *args, **kwargs):
            events.append("build")
            return {"draft_layer": metadata}

    namespace: dict[str, Any] = {
        "Any": Any,
        "BaseSpeculator": BaseSpeculator,
        "AscendAttentionState": SimpleNamespace(DecodeOnly="decode_only"),
        "build_draft_attn_metadata_factory": lambda *args: nullcontext(),
        "np": SimpleNamespace(ndarray=object),
        "torch": SimpleNamespace(Tensor=object, from_numpy=lambda value: value),
    }
    test_class = ast.ClassDef(
        name="TestSpeculator",
        bases=[ast.Name(id="BaseSpeculator", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(body=[test_class], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SPECULATOR_PATH), "exec"), namespace)

    speculator = namespace["TestSpeculator"]()
    speculator.input_batch = SimpleNamespace(is_prefilling_np=[False, False, False])
    speculator.input_buffers = SimpleNamespace(positions=[0, 0, 0])
    speculator._update_decode_attn_metadata = lambda built_metadata, step, num_reqs: events.append(
        ("update", built_metadata, step, num_reqs)
    )

    result = speculator._build_draft_attn_metadata(
        num_reqs=3,
        num_reqs_padded=4,
        num_tokens_padded=4,
        seq_lens_cpu_upper_bound=SimpleNamespace(),
        step=2,
    )

    assert result == {"draft_layer": metadata}
    assert metadata.attn_state == "decode_only"
    assert events == ["build", ("update", result, 2, 3)]
