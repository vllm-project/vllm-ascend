# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest
import torch


@pytest.fixture(params=["qsa_rmsnorm_mrope.py", "split_qkv_rmsnorm_mrope.py"])
def validate(request):
    # Load only the metadata validator so these tests need no Triton/NPU runtime.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/linearnorm" / request.param
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_rope_positions"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_validate_rope_positions"]


@pytest.mark.parametrize("rows", [0, 1, 32])
def test_strided_positions_are_valid(validate, rows):
    positions = torch.empty(3, 2 * rows + 17, dtype=torch.int64)[:, : 2 * rows : 2]
    tensor = (
        torch.empty(rows, 1536, dtype=torch.bfloat16)
        if "split_qkv" in validate.__code__.co_filename
        else torch.empty(rows, 4, 128, dtype=torch.bfloat16)
    )
    cache = torch.empty(64, 80, dtype=torch.bfloat16)[:, :64]
    validate(positions, cache, tensor, 64)
    validate(positions[0], cache, tensor, 64)


@pytest.mark.parametrize("case", ["axes", "rows", "float_positions", "empty_cache", "cache_width", "dtype"])
def test_invalid_direct_gather_contract(validate, case):
    tensor = torch.empty(32, 4, 128, dtype=torch.bfloat16)
    positions = torch.empty(
        (2 if case == "axes" else 3, 31 if case == "rows" else 32),
        dtype=torch.float32 if case == "float_positions" else torch.int64,
    )
    cache = torch.empty(
        0 if case == "empty_cache" else 64,
        32 if case == "cache_width" else 64,
        dtype=torch.float32 if case == "dtype" else torch.bfloat16,
    )
    with pytest.raises(ValueError):
        validate(positions, cache, tensor, 64)
