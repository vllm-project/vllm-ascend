# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online quantization variants preserve their base model's configuration."""

import json
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    DEEPSEEK_V4_CASE,
    GLM51_CASE,
    QWEN35_CASE,
    _online_mx_case,
)


@pytest.mark.parametrize("base_case", (QWEN35_CASE, GLM51_CASE, DEEPSEEK_V4_CASE), ids=lambda case: case.id)
@pytest.mark.parametrize("bits", (4, 8))
def test_online_mx_case_preserves_base_model(base_case, bits):
    original = deepcopy(base_case)
    case = _online_mx_case(base_case, bits)
    assert case.id == f"{base_case.id}-online-{'mxfp8' if bits == 8 else 'w4a8-mxfp4'}"
    assert case.online_quantization
    changed_fields = {"id", "hf_overrides", "online_quantization", "extra_server_args"}
    for field in fields(base_case):
        if field.name not in changed_fields:
            assert getattr(case, field.name) == getattr(base_case, field.name), field.name
    overrides = deepcopy(case.hf_overrides)
    quantization = overrides.pop("quantization_config")
    assert overrides == base_case.hf_overrides
    assert quantization["config_groups"]["group_0"]["weights"]["num_bits"] == bits
    assert case.extra_server_args[: len(base_case.extra_server_args)] == base_case.extra_server_args
    # A derived variant must not share mutable model overrides with its base.
    case.hf_overrides.setdefault("text_config", {})["num_hidden_layers"] = 99
    assert base_case == original


def test_online_mx_case_merges_base_additional_config():
    base_case = replace(
        GLM51_CASE,
        extra_server_args=(
            "--max-num-seqs",
            "8",
            "--additional-config",
            '{"rl_config": {"enabled": true}, "weight_nz_mode": 2}',
        ),
    )
    case = _online_mx_case(base_case, 8)
    assert case.extra_server_args[:2] == ("--max-num-seqs", "8")
    assert case.extra_server_args.count("--additional-config") == 1
    config_index = case.extra_server_args.index("--additional-config") + 1
    assert json.loads(case.extra_server_args[config_index]) == {
        "rl_config": {"enabled": True},
        "weight_nz_mode": 0,
        "online_quantize_weights": True,
    }


def test_online_mx_case_rejects_unsupported_precision():
    with pytest.raises(ValueError, match="4 or 8 bits"):
        _online_mx_case(QWEN35_CASE, 16)
