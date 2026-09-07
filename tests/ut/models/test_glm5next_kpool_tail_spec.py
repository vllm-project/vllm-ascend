# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kpool tail spec must be vLLM's class, not a redefinition.

vLLM decides GLM-5.3-Flash's KV cache layout by testing specs against its own
``KpoolTailSpec``. A same-named class defined here would fail that test, the
tail specs would be mistaken for attention specs, and the model would silently
fall back to the generic grouping path that cannot size GLM's pages.
"""

import ast
from pathlib import Path

import pytest
from vllm.v1.kv_cache_interface import KpoolTailSpec

VLLM_ASCEND_ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"
DUPLICATED_NAMES = ("KpoolTailSpec", "KpoolTailManager")


def _class_definitions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


@pytest.mark.parametrize("name", DUPLICATED_NAMES)
def test_no_module_redefines_the_upstream_class(name):
    offenders = [
        str(path.relative_to(VLLM_ASCEND_ROOT))
        for path in VLLM_ASCEND_ROOT.rglob("*.py")
        if name in _class_definitions(path)
    ]

    assert not offenders, (
        f"{name} is defined in {offenders}; import it from "
        "vllm.v1.kv_cache_interface instead so type checks in vLLM's "
        "GLM-5.3-Flash grouping recognise the spec."
    )


@pytest.mark.parametrize("name", DUPLICATED_NAMES)
def test_no_module_imports_the_class_from_a_local_path(name):
    pattern = f"import {name}"
    offenders = [
        str(path.relative_to(VLLM_ASCEND_ROOT))
        for path in VLLM_ASCEND_ROOT.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if pattern in line and "vllm_ascend" in line
    ]

    assert not offenders, f"{name} is imported from vllm_ascend in {offenders}"


def test_upstream_tail_spec_opts_out_of_prefix_caching():
    """``patch_kv_cache_coordinator`` reads this to skip the tail group."""
    spec = KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=None,
        sliding_window=4,
    )

    assert spec.prefix_cacheable is False
    assert spec.max_num_blocks_per_req(None, 65536) == 1
