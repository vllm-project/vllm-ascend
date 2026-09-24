# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.eplb.state import EXPERT_MAPPING_EP_SIZE
from vllm_ascend.patch.platform.patch_eplb import (
    _build_distributed_initial_expert_map,
    _with_expert_mapping_ep_size,
)


def test_initial_replicas_are_spread_across_ep_ranks():
    layout = _build_distributed_initial_expert_map(5, 3, ep_size=2)
    assert layout == [0, 1, 2, 3, 2, 3, 4, 0]
    assert [layout[:4].count(0), layout[4:].count(0)] == [1, 1]


def test_initial_layout_preserves_v1_and_zero_redundancy():
    assert _build_distributed_initial_expert_map(5, 3, ep_size=1) == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
    ]
    assert _build_distributed_initial_expert_map(5, 0, ep_size=5) == list(range(5))
    with pytest.raises(ValueError, match="divisible"):
        _build_distributed_initial_expert_map(5, 2, ep_size=2)


def test_weight_mapping_ep_size_is_scoped_to_the_call():
    wrapped = _with_expert_mapping_ep_size(
        lambda _model: EXPERT_MAPPING_EP_SIZE.get(),
        lambda model: model.moe_config.ep_size if model._use_v2_model_runner else 1,
    )
    v2 = SimpleNamespace(moe_config=SimpleNamespace(ep_size=4), _use_v2_model_runner=True)
    v1 = SimpleNamespace(moe_config=SimpleNamespace(ep_size=4), _use_v2_model_runner=False)
    assert wrapped(v2) == 4
    assert wrapped(v1) == 1
    assert EXPERT_MAPPING_EP_SIZE.get() == 1
