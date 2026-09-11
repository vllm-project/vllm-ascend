# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.dspark_aux import (
    DSparkAuxHiddenContract,
    DSparkAuxHiddenFormat,
    build_k3_mla_aux_hidden_contract,
)
from vllm_ascend.models.kimi_k3 import AscendKimiLinearModel


def _contract(**overrides) -> DSparkAuxHiddenContract:
    values = {
        "format": DSparkAuxHiddenFormat.RAW_PREFIX_SUM,
        "layer_ids": (1, 5),
        "capture_point": "post_layer_raw_prefix_sum",
        "target_hidden_size": 8,
        "dtype": torch.bfloat16,
    }
    values.update(overrides)
    return DSparkAuxHiddenContract(**values)


def test_build_k3_mla_contract_converts_checkpoint_layer_ids() -> None:
    config = SimpleNamespace(
        target_layer_ids=[0, 4],
        target_num_hidden_layers=5,
        num_target_layers=2,
        target_hidden_size=8,
    )

    contract = build_k3_mla_aux_hidden_contract(config, torch.bfloat16)

    assert contract.format == DSparkAuxHiddenFormat.RAW_PREFIX_SUM
    assert contract.layer_ids == (1, 5)
    assert contract.capture_point == "post_layer_raw_prefix_sum"
    assert contract.packed_hidden_size == 16


def test_build_k3_mla_contract_rejects_layer_count_mismatch() -> None:
    config = SimpleNamespace(
        target_layer_ids=[0, 4],
        num_target_layers=3,
        target_hidden_size=8,
    )

    with pytest.raises(ValueError, match="num_target_layers=3"):
        build_k3_mla_aux_hidden_contract(config, torch.bfloat16)


def test_build_k3_mla_contract_rejects_layer_past_declared_target() -> None:
    config = SimpleNamespace(
        target_layer_ids=[0, 4],
        target_num_hidden_layers=4,
        num_target_layers=2,
        target_hidden_size=8,
    )

    with pytest.raises(ValueError, match="exceed the declared 4 target layers"):
        build_k3_mla_aux_hidden_contract(config, torch.bfloat16)


def test_aux_contract_rejects_format_capture_point_mismatch() -> None:
    with pytest.raises(ValueError, match="must use capture point"):
        _contract(capture_point="pre_layer_materialized").validate_definition()


def test_kimi_target_applies_raw_aux_contract() -> None:
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=8, num_hidden_layers=5)
    model.aux_hidden_state_layers = (1, 5)
    model.dspark_aux_capture_materialized = True

    model.configure_dspark_aux_hidden_state_contract(_contract())

    assert not model.dspark_aux_capture_materialized


def test_kimi_target_rejects_aux_layer_mismatch() -> None:
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=8, num_hidden_layers=5)
    model.aux_hidden_state_layers = (1, 4)

    with pytest.raises(ValueError, match="do not match"):
        model.configure_dspark_aux_hidden_state_contract(_contract())


def test_kimi_target_rejects_aux_layer_past_model_boundary() -> None:
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=8, num_hidden_layers=4)
    model.aux_hidden_state_layers = (1, 5)

    with pytest.raises(ValueError, match="exceed the target's 4 layer boundaries"):
        model.configure_dspark_aux_hidden_state_contract(_contract())


def test_aux_contract_accepts_per_layer_and_packed_runtime_states() -> None:
    contract = _contract()
    per_layer = [
        torch.zeros(3, 8, dtype=torch.bfloat16),
        torch.zeros(3, 8, dtype=torch.bfloat16),
    ]
    packed = [torch.zeros(3, 16, dtype=torch.bfloat16)]

    contract.validate_runtime(per_layer, num_target_tokens=3, target_device=torch.device("cpu"))
    contract.validate_runtime(packed, num_target_tokens=3, target_device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("states", "match"),
    [
        (None, "requires raw_prefix_sum"),
        ([torch.zeros(2, 16, dtype=torch.bfloat16)], "3 are required"),
        ([torch.zeros(3, 15, dtype=torch.bfloat16)], "width is 15"),
        ([torch.zeros(3, 16, dtype=torch.float32)], "dtype"),
    ],
)
def test_aux_contract_rejects_invalid_runtime_states(states, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _contract().validate_runtime(
            states,
            num_target_tokens=3,
            target_device=torch.device("cpu"),
        )
