# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from tests.e2e.models.minimax_m31_lora.m31_observation import bf16_expected_delta, difference_summary


def test_factor_cast_is_distinct_from_ideal_fp64_and_reported():
    record = {"profile": {"kind": "linear"}, "input": [[1.0, -1.0]], "output": [[0.0]], "runtime_module": "o"}
    modules = [{"id": "o", "a_key": "a", "b_key": "b"}]
    mappings = [{"id": "o", "runtime_module": "o"}]
    tensors = {"a": torch.tensor([[1.00390625, 1.0]]), "b": torch.tensor([[8.0]])}
    observed = bf16_expected_delta(record, modules, mappings, tensors, 1)
    assert torch.equal(observed, torch.zeros(1, 1))
    ideal = torch.tensor([[0.03125]], dtype=torch.float64)
    report = difference_summary(observed, ideal, {"atol": 1 / 64, "rtol": 1 / 64})
    assert report == {"elements": 1, "max_abs_error": 0.03125, "violations": 1}


def test_intermediate_rounding_is_required_in_addition_to_final_rounding():
    record = {"profile": {"kind": "linear"}, "input": [[1.0, 1 / 256]], "output": [[0.0]], "runtime_module": "o"}
    modules = [{"id": "o", "a_key": "a", "b_key": "b"}]
    mappings = [{"id": "o", "runtime_module": "o"}]
    tensors = {"a": torch.ones(1, 2), "b": torch.tensor([[3.0]])}
    result = bf16_expected_delta(record, modules, mappings, tensors, 1)
    assert result.item() == 3.0
    assert torch.tensor(3 * (1 + 1 / 256)).bfloat16().item() == 3.015625


def test_qkv_reference_maps_scaled_factors_and_preserves_unadapted_columns():
    record = {"profile": {"kind": "qkv"}, "input": [[1.0, 2.0]], "output": [[0.0] * 6], "runtime_module": "qkv"}
    modules = [{"id": name, "a_key": name + "a", "b_key": name + "b"} for name in ("q", "v")]
    mappings = [
        {
            "id": name,
            "runtime_module": "qkv",
            "observer": {"slice_sizes": [2, 1, 1], "slice_index": index},
        }
        for name, index in (("q", 0), ("v", 2))
    ]
    tensors = {
        "qa": torch.tensor([[1.0, 0.0]]),
        "qb": torch.tensor([[1.0], [2.0]]),
        "va": torch.tensor([[0.0, 1.0]]),
        "vb": torch.tensor([[3.0]]),
    }
    expected = torch.tensor([[2.0, 4.0, 0.0, 12.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.equal(bf16_expected_delta(record, modules, mappings, tensors, 2), expected)
    del tensors["vb"]
    with pytest.raises(KeyError):
        bf16_expected_delta(record, modules, mappings, tensors, 2)
