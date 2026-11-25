# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def validate():
    # This shape-only helper intentionally has no vLLM, Triton or NPU dependency.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/linearnorm/qsa_rope_utils.py"
    spec = importlib.util.spec_from_file_location("qsa_rope_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_qsa_rope_positions


@pytest.mark.parametrize("rows", [0, 1, 32])
def test_strided_positions_are_valid(validate, rows):
    positions = torch.empty(3, 2 * rows + 17, dtype=torch.int64)[:, : 2 * rows : 2]
    tensor = torch.empty(rows, 4, 128, dtype=torch.bfloat16)
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
