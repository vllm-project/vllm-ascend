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


@pytest.mark.parametrize("backend_kind", ["fa3", "cann", "fia", "mixed"])
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

    cann = type("CANNFlash", (), {"full_cls_name": staticmethod(lambda: ("fa_v1", "Flash"))})
    fa3 = type(
        "FA3",
        (),
        {
            "full_cls_name": staticmethod(
                lambda: ("vllm_ascend.attention.flash_attention_v3", "AscendFlashAttentionBackend")
            )
        },
    )
    fia = type("FIA", (), {"full_cls_name": staticmethod(lambda: ("fia", "Attention"))})
    namespace: dict[str, Any] = {
        "AscendGraphManagerMixin": type("GraphManagerMixin", (), {}),
        "DFlashCudaGraphManager": Parent,
        "AscendFlashAttentionBackend": cann,
        "use_updatable_graph": lambda _: False,
    }
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    manager = namespace["DFlashAclGraphManager"]()
    backends = {"fa3": [fa3, fa3], "cann": [cann, cann], "fia": [fia, fia], "mixed": [fa3, fia]}[backend_kind]
    manager.speculator = SimpleNamespace(
        attn_backends=dict(enumerate(backends)),
        build_draft_attn_metadatas=Mock(return_value="metadata"),
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound="upper bound"),
    )
    manager.parent_replay = Mock()
    manager._graph_replay = Mock(return_value="FIA replay")
    desc = SimpleNamespace(num_tokens=16, num_reqs=2)
    if backend_kind in ("fa3", "cann"):
        assert manager.run_fullgraph(desc) == "device replay"
        manager.parent_replay.assert_called_once_with(desc)
        manager.speculator.build_draft_attn_metadatas.assert_not_called()
        manager._graph_replay.assert_not_called()
    else:
        assert manager.run_fullgraph(desc) == "FIA replay"
        manager.speculator.build_draft_attn_metadatas.assert_called_once_with(2, "upper bound")
        manager._graph_replay.assert_called_once_with(desc, backends[0], 16, "metadata")
