# SPDX-License-Identifier: Apache-2.0
"""Empty input must return before dividing by TopK or launching kernels."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("shape", [(0, 2048), (1, 0), (2, 3, 0)])
def test_empty_remap_preserves_shape_dtype_and_avoids_launch(shape):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "remap_sparse_indices_triton")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    value = torch.empty(shape, dtype=torch.int32)
    assert namespace["remap_sparse_indices_triton"](value, 8, 0, 128) is value
