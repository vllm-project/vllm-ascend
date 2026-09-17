# SPDX-License-Identifier: Apache-2.0
"""The experimental Flash path defers CP compatibility to execution."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("flash", [True, False])
def test_attention_cp_precheck_follows_flash_flag(flash):
    path = Path(__file__).parents[4] / "vllm_ascend/patch/worker/patch_v2/patch_attn_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_attention_cp_compatibility"
    )
    upstream = Mock()
    namespace = {
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash),
        "_upstream_check_attention_cp_compatibility": upstream,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(function), str(path), "exec"), namespace)
    config = object()
    namespace["check_attention_cp_compatibility"](config)
    if flash:
        upstream.assert_not_called()
    else:
        upstream.assert_called_once_with(config)
