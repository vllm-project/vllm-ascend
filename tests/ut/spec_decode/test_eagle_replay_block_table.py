# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Eagle3 GQA replay reuses the target block table unless PCP or DCP owns one."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SPECULATOR_PATH = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"
ACLGRAPH_PATH = ROOT / "vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py"


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    speculator_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(
        node for node in speculator_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _load_method(path: Path, class_name: str, method_name: str):
    method = _class_method(path, class_name, method_name)
    namespace: dict[str, Any] = {"Any": Any}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


class _BlockTable:
    def __init__(self, rows: int, cols: int):
        self.shape = (rows, cols)
        self.viewed = None

    def stride(self):
        return (self.shape[1], 1)

    def as_strided(self, size, stride):
        self.viewed = (size, stride)
        return self


def test_shared_layout_reuses_target_block_table():
    reuses = _load_method(SPECULATOR_PATH, "AscendAutoRegressiveSpeculator", "reuses_target_block_table")
    assert reuses(SimpleNamespace(replicated_pcp=False, use_dcp=False)) is True
    assert reuses(SimpleNamespace(replicated_pcp=True, use_dcp=False)) is False
    assert reuses(SimpleNamespace(replicated_pcp=False, use_dcp=True)) is False


def test_shared_replay_views_target_block_table_without_rebuilt_metadata():
    build_fia_params = _load_method(SPECULATOR_PATH, "AscendAutoRegressiveSpeculator", "build_fia_params")
    block_table = _BlockTable(rows=2, cols=8)
    speculator = SimpleNamespace(
        model_state=SimpleNamespace(
            attn_metadata={
                "target.0": SimpleNamespace(block_tables=_BlockTable(1, 1)),
                "draft.0": SimpleNamespace(
                    block_tables=block_table,
                    actual_seq_lengths_q=[1, 2],
                    seq_lens_list=[4, 5],
                ),
            }
        ),
        draft_attn_layer_names={"draft.0"},
        input_batch=SimpleNamespace(num_reqs=1, seq_lens_np=[10]),
        max_model_len=128,
        num_speculative_steps=3,
    )

    params = build_fia_params(speculator, 4, None, False)

    assert block_table.viewed[0] == (4, 8)
    assert [item["block_table"] for item in params] == [block_table, block_table]
    assert params[0]["actual_seq_lengths_kv"] == [11, 0, 0, 0]
    assert params[1]["actual_seq_lengths_kv"] == [12, 0, 0, 0]


def test_rebuilt_replay_keeps_the_draft_block_table():
    build_fia_params = _load_method(SPECULATOR_PATH, "AscendAutoRegressiveSpeculator", "build_fia_params")
    draft_table = _BlockTable(rows=4, cols=8)
    speculator = SimpleNamespace(
        draft_attn_layer_names={"draft.0"},
        input_batch=SimpleNamespace(num_reqs=1, seq_lens_np=[10]),
        max_model_len=128,
        num_speculative_steps=2,
    )

    params = build_fia_params(
        speculator,
        4,
        {"draft.0": SimpleNamespace(block_tables=draft_table)},
        False,
    )

    assert draft_table.viewed is None
    assert params[0]["block_table"] is draft_table


def test_updatable_replay_skips_metadata_rebuild_when_block_table_is_shared():
    method = _class_method(ACLGRAPH_PATH, "AutoRegressiveAclGraphManager", "run_fullgraph")
    early_return = next(node for node in method.body if isinstance(node, ast.If))
    test_source = ast.unparse(early_return.test)
    assert "reuses_target_block_table" in test_source
    assert "use_updatable_graph" in test_source
    returned = early_return.body[0]
    assert isinstance(returned, ast.Return)
    assert ast.unparse(returned.value) == "self._updatable_graph_replay(desc, None)"

    rebuild = next(
        node
        for node in method.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "draft_attn_metadatas"
    )
    assert method.body.index(rebuild) > method.body.index(early_return)
