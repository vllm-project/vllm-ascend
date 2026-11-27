# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for bypassing FIA host task updates on device-length backends."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("backend_kind", ["fa3", "cann", "device_mix", "fia", "mixed", "updatable"])
def test_device_length_graph_replay_preserves_fia_fallback(backend_kind):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DFlashAclGraphManager"
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) == "run_fullgraph"]

    class Parent:
        parent_replay: Mock

        def run_fullgraph(self, desc):
            self.parent_replay(desc)
            return "device replay"

    def device_backend(filename):
        backend_source = source.parents[4] / "attention" / filename
        backend_cls = next(
            node
            for node in ast.parse(backend_source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "AscendFlashAttentionBackend"
        )
        policy = next(
            node.value
            for node in backend_cls.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "requires_graph_task_update" for target in node.targets
            )
        )
        return type(filename, (), {"requires_graph_task_update": ast.literal_eval(policy)})

    cann = device_backend("fa_v1.py")
    fa3 = device_backend("flash_attention_v3.py")
    fia = type("FIA", (), {})
    namespace: dict[str, Any] = {
        "AscendGraphManagerMixin": type("GraphManagerMixin", (), {}),
        "DFlashCudaGraphManager": Parent,
        "use_updatable_graph": lambda _: backend_kind == "updatable",
    }
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    manager = namespace["DFlashAclGraphManager"]()
    backends = {
        "fa3": [fa3, fa3],
        "cann": [cann, cann],
        "device_mix": [fa3, cann],
        "fia": [fia, fia],
        "mixed": [fa3, fia],
        "updatable": [fia, fia],
    }[backend_kind]
    manager.speculator = SimpleNamespace(
        attn_backends=dict(enumerate(backends)),
        build_draft_attn_metadatas=Mock(return_value="metadata"),
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound="upper bound"),
    )
    manager.parent_replay = Mock()
    manager._graph_replay = Mock(return_value="FIA replay")
    manager._updatable_graph_replay = Mock(return_value="updatable replay")
    desc = SimpleNamespace(num_tokens=16, num_reqs=2)
    if backend_kind in ("fa3", "cann", "device_mix"):
        assert manager.run_fullgraph(desc) == "device replay"
        manager.parent_replay.assert_called_once_with(desc)
        manager.speculator.build_draft_attn_metadatas.assert_not_called()
        manager._graph_replay.assert_not_called()
    elif backend_kind == "updatable":
        assert manager.run_fullgraph(desc) == "updatable replay"
        manager.speculator.build_draft_attn_metadatas.assert_called_once_with(2, "upper bound")
        manager._updatable_graph_replay.assert_called_once_with(desc, "metadata")
        manager._graph_replay.assert_not_called()
    else:
        assert manager.run_fullgraph(desc) == "FIA replay"
        manager.speculator.build_draft_attn_metadatas.assert_called_once_with(2, "upper bound")
        manager._graph_replay.assert_called_once_with(desc, backends[0], 16, "metadata")
