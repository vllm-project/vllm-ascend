#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import copy

import pytest
import torch
import torch_npu
import utils

from .config import (
    PERSISTED_KEYS,
    prepare_consistency_params,
    resolve_consistency_config,
)
from .runner import DeterministicLevelGuard


def make_case(params=None, batch_size=3, token_count=5):
    return {
        "params": dict(params or {}),
        "metadata_input": {"batch_size": batch_size},
        "op_input": {
            "q": torch.zeros((batch_size, token_count, 1, 1)),
            "layout_q": "BSND",
            "layout_kv": "BSND",
        },
    }


def make_schema_case(operator, params=None):
    case = make_case(params)
    if operator == "qsmla":
        case["op_input"]["q_descale"] = None
    elif operator == "smla":
        case = {
            "params": dict(params or {}),
            "B": 3,
            "layout_q": "BSND",
            "layout_kv": "BSND",
            "metadata_input": {},
            "input": {
                "q": torch.zeros((3, 5, 1, 1)),
                "layout_q": "BSND",
                "layout_kv": "BSND",
            },
        }
    return case


def test_auto_without_parameters_is_disabled():
    assert resolve_consistency_config(make_case(), "auto") is None


def test_normal_excel_save_defaults_do_not_enable_consistency():
    params = utils.generate_param_combinations([{}], is_save_pt=True)[0]

    assert params["batch_consistency"] is False
    assert params["ori_sparse_indices_mode"] == "full"
    assert params["cmp_sparse_indices_mode"] == "full"
    assert not prepare_consistency_params(params, "auto")
    assert not any(key in params for key in PERSISTED_KEYS)


def test_batch_excel_save_values_keep_their_scalar_and_list_shapes():
    source = {
        "batch_consistency": True,
        "batch_consistency_seed": 7,
        "batch_consistency_order": [1, 0],
    }
    params = utils.generate_param_combinations([source], is_save_pt=True)[0]

    assert params["batch_consistency"] is True
    assert params["batch_consistency_seed"] == 7
    assert params["batch_consistency_order"] == [1, 0]


def test_explicit_parameters_are_preserved_and_persisted():
    case = make_case(
        {
            "batch_consistency": True,
            "batch_consistency_seed": 7,
            "batch_consistency_order": [2, 0, 1],
            "batch_consistency_batch_split": [1, 2],
            "batch_consistency_mode_batch": 1,
            "batch_consistency_token_split": [2, 3],
            "batch_consistency_shape_change": [3, 0],
        }
    )
    config = resolve_consistency_config(case, "auto", persist=True)
    assert config is not None
    assert config["npu_deterministic_level"] == 3
    assert all(case["params"][key] == config[key] for key in config)


def test_seed_makes_generated_parameters_reproducible():
    original = make_case({"batch_consistency": True, "batch_consistency_seed": 19})
    first = resolve_consistency_config(copy.deepcopy(original), "auto")
    second = resolve_consistency_config(copy.deepcopy(original), "auto")
    assert first == second


def test_off_filters_parameters_before_testcase_generation():
    params = {
        "batch_consistency": True,
        "batch_consistency_seed": 3,
        "npu_deterministic_level": 3,
    }
    assert not prepare_consistency_params(params, "off")
    assert not any(key in params for key in PERSISTED_KEYS)


def test_on_marks_testcase_enabled_before_generation():
    params = {}
    assert prepare_consistency_params(params, "on")
    assert params == {"batch_consistency": True}


def test_invalid_explicit_split_fails_instead_of_falling_back():
    case = make_case(
        {
            "batch_consistency": True,
            "batch_consistency_batch_split": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="CONFIG_ERROR"):
        resolve_consistency_config(case, "auto")


@pytest.mark.parametrize("operator", ["smla", "qsmla", "mqsmla"])
def test_persisted_config_survives_pt_round_trip(tmp_path, operator):
    case = make_schema_case(operator, {"batch_consistency": True})
    config = resolve_consistency_config(case, "auto", persist=True)
    case_path = tmp_path / f"{operator}.pt"
    torch.save(case, case_path)
    loaded = torch.load(case_path, map_location="cpu", weights_only=False)
    assert config is not None
    assert {key: loaded["params"][key] for key in PERSISTED_KEYS} == config


def test_deterministic_level_guard_sets_three_and_restores(monkeypatch):
    state = {"level": 1}
    monkeypatch.setattr(
        torch_npu.npu,
        "_get_deterministic_level",
        lambda: state["level"],
        raising=False,
    )
    monkeypatch.setattr(
        torch_npu.npu,
        "set_deterministic_level",
        lambda value: state.update(level=value),
        raising=False,
    )
    with DeterministicLevelGuard():
        assert state["level"] == 3
        with DeterministicLevelGuard():
            assert state["level"] == 3
        assert state["level"] == 3
    assert state["level"] == 1
