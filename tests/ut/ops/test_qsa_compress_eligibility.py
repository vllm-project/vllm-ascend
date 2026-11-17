# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "ratio, layout, norm_eligible, expected",
    [
        (4, "contiguous", True, True),
        (2, "contiguous", True, False),
        (8, "contiguous", True, False),
        (4, "strided", True, False),
        (4, "wrong_width", True, False),
        (4, "contiguous", False, False),
    ],
)
def test_compression_dispatch_respects_fused_contract(ratio, layout, norm_eligible, expected):
    # Execute the real metadata-only predicate without importing the NPU model.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/qsa.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendQSAIndexer")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_fused_compress_eligible"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    weight = torch.empty(256, dtype=torch.bfloat16)[::2] if layout == "strided" else torch.empty(128)
    if layout == "wrong_width":
        weight = weight[:64]
    layer = SimpleNamespace(
        compress_ratio=ratio,
        index_head_dim=128,
        k_layernorm=SimpleNamespace(weight=weight),
        _fused_norm_rope_eligible=lambda tensor, positions: norm_eligible,
    )
    assert namespace["_fused_compress_eligible"](layer, torch.empty(2, 1, 128), torch.arange(2)) == expected
