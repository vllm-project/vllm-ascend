# SPDX-License-Identifier: Apache-2.0
"""Exercise the production Indexer guard without importing model/NPU packages."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "enabled,c8,c4,capability,raises",
    [
        (False, True, False, False, False),
        (True, False, False, False, False),
        (True, True, False, False, True),
        (True, True, False, True, False),
        (True, False, True, True, True),
        (True, True, True, True, True),
        (True, False, True, False, True),
        (True, True, False, None, True),
    ],
)
def test_quantized_indexer_requires_selected_backend_capability(monkeypatch, enabled, c8, c4, capability, raises):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"
    tree = ast.parse(source.read_text())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "hisparse_config is not None and (cache_sparse_li_c8 or cache_sparse_li_c4)" in ast.unparse(node.test)
    )
    backend = SimpleNamespace() if capability is None else SimpleNamespace(supports_indexer_c8=capability)
    module = ModuleType("vllm.v1.hisparse.backend")
    module.get_hisparse_backend = lambda: backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = dict(
        vllm_config=SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=object() if enabled else None)),
        cache_sparse_li_c8=c8,
        cache_sparse_li_c4=c4,
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec")
    if raises:
        with pytest.raises(ValueError, match="native C8"):
            exec(code, namespace)
    else:
        exec(code, namespace)
