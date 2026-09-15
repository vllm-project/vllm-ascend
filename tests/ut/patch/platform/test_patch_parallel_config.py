# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
from vllm.config.parallel import ParallelConfig

import vllm_ascend.patch.platform.patch_parallel_config  # noqa: F401


def _make_parallel_config(**overrides) -> ParallelConfig:
    kwargs = {
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "enable_expert_parallel": True,
        "all2all_backend": "allgather_reducescatter",
    }
    kwargs.update(overrides)
    return ParallelConfig(**kwargs)


def test_sp_moe_enabled_with_single_dp_rank():
    config = _make_parallel_config(data_parallel_size=1)
    assert config.use_sequence_parallel_moe is True


def test_sp_moe_still_enabled_with_multiple_dp_ranks():
    config = _make_parallel_config(data_parallel_size=2)
    assert config.use_sequence_parallel_moe is True


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"all2all_backend": "flashinfer_all2allv"}, id="non-sp-backend"),
        pytest.param({"enable_expert_parallel": False}, id="ep-disabled"),
        pytest.param({"tensor_parallel_size": 1}, id="tp1"),
    ],
)
def test_sp_moe_off_conditions_unchanged(overrides):
    config = _make_parallel_config(data_parallel_size=1, **overrides)
    assert config.use_sequence_parallel_moe is False
