# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_deepseek_v41_model_call_sites_use_compiled_operator_layouts():
    """Keep framework call sites aligned with the operator PR contracts."""
    source = ast.parse((REPO_ROOT / "vllm_ascend/attention/dsa_v41.py").read_text())
    calls = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("npu_sparse_flash_mla", "npu_sparse_flash_mla_metadata")
    ]
    assert len(calls) == 2
    for call in calls:
        layouts = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("layout_q", "layout_kv")}
        assert layouts == {"layout_q": "TND", "layout_kv": "PA_BBND"}

    source = ast.parse((REPO_ROOT / "vllm_ascend/models/deepseek_v41/indexer.py").read_text())
    common = next(
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "common" for target in node.targets)
    )
    assert isinstance(common, ast.Call)
    layouts = {kw.arg: ast.literal_eval(kw.value) for kw in common.keywords if kw.arg in ("layout_q", "layout_k")}
    assert layouts == {"layout_q": "TND", "layout_k": "PA_BBND"}


def test_compressor_call_sites_use_compiled_modes():
    for filename, count in (
        ("models/deepseek_v4/compressor.py", 1),
        ("attention/context_parallel/dsa_cp.py", 2),
    ):
        source = ast.parse((REPO_ROOT / "vllm_ascend" / filename).read_text())
        calls = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.ops._C_ascend.compressor"
        ]
        assert len(calls) == count
        for call in calls:
            modes = {
                kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("rotary_mode", "cache_mode")
            }
            assert modes == {"rotary_mode": 2, "cache_mode": 1}
