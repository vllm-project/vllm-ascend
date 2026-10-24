# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Exercise the actual replay method without importing the NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def replay_class():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(node for node in module.body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "run_fullgraph"
    )
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None

    class Parent:
        def run_fullgraph(self, desc):
            self.parent_replays.append(desc)
            return self.expected_output

    definition.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    definition.body = [method]
    namespace = {"Parent": Parent, "logger": MagicMock(), "use_updatable_graph": MagicMock(return_value=False)}
    code = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)
    return namespace[definition.name], namespace["use_updatable_graph"]


@pytest.mark.parametrize("prefill", [False, True])
def test_qsa_preserves_parent_replay_without_fia_updates(replay_class, prefill):
    cls, use_updatable = replay_class
    manager = cls()
    manager.speculator = SimpleNamespace(attn_architecture="QSA", build_draft_attn_metadatas=MagicMock())
    manager.is_draft_model_prefill = prefill
    manager.update_stream = None
    manager.parent_replays = []
    manager.expected_output = object()
    descriptor = SimpleNamespace(num_tokens=8, num_reqs=2)
    assert manager.run_fullgraph(descriptor) is manager.expected_output
    assert manager.parent_replays == [descriptor]
    manager.speculator.build_draft_attn_metadatas.assert_not_called()
    use_updatable.assert_not_called()


@pytest.mark.parametrize("updatable", [False, True])
@pytest.mark.parametrize("architecture,replicated_pcp", [("GQA", False), ("QSA", True)])
def test_other_backends_keep_metadata_and_update_path(replay_class, updatable, architecture, replicated_pcp):
    cls, use_updatable = replay_class
    use_updatable.return_value = updatable
    manager = cls()
    manager.speculator = SimpleNamespace(
        attn_architecture=architecture,
        replicated_pcp=replicated_pcp,
        attn_backend=object(),
        draft_vllm_config=object(),
        build_draft_attn_metadatas=MagicMock(return_value=["metadata"]),
    )
    manager.is_draft_model_prefill = False
    manager.update_stream = object()
    manager._updatable_graph_replay = MagicMock(return_value="updatable")
    manager._graph_replay = MagicMock(return_value="legacy")
    descriptor = SimpleNamespace(num_tokens=8, num_reqs=2)
    assert manager.run_fullgraph(descriptor) == ("updatable" if updatable else "legacy")
    manager.speculator.build_draft_attn_metadatas.assert_called_once_with(2, 8, False)
    assert manager._updatable_graph_replay.call_count == int(updatable)
    assert manager._graph_replay.call_count == int(not updatable)
