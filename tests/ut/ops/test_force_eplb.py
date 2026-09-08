# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import force_eplb


def _make_moe_comm_method(*, ep_rank: int = 1):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            ep_size=2,
            ep_rank=ep_rank,
            experts_per_token=2,
            num_logical_experts=8,
        )
    )


def test_build_round_robin_topk_returns_expected_ids():
    result = force_eplb._build_round_robin_topk(
        num_tokens=2,
        top_k=2,
        num_logical_experts=8,
        ep_size=2,
        ep_rank=1,
        device=torch.device("cpu"),
        dtype=torch.int32,
    )

    torch.testing.assert_close(result, torch.tensor([[6, 3], [7, 0]], dtype=torch.int32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_tokens": 0}, "num_tokens must be positive"),
        ({"top_k": 0}, "top_k must be positive"),
        ({"ep_size": 0}, "ep_size must be positive"),
        ({"num_logical_experts": 7}, "num_logical_experts must be divisible"),
    ],
)
def test_build_round_robin_topk_rejects_invalid_shape(kwargs, message):
    params = {
        "num_tokens": 2,
        "top_k": 2,
        "num_logical_experts": 8,
        "ep_size": 2,
        "ep_rank": 0,
        "device": torch.device("cpu"),
        "dtype": torch.int32,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        force_eplb._build_round_robin_topk(**params)


def test_get_force_eplb_topk_reuses_cached_table():
    moe_comm_method = _make_moe_comm_method()
    context = SimpleNamespace(moe_comm_method=moe_comm_method)
    topk_ids = torch.empty((2, 2), dtype=torch.int32)

    with patch.object(force_eplb, "get_forward_context", return_value=context):
        first = force_eplb.get_force_eplb_topk(topk_ids, num_logical_experts=8)
        second = force_eplb.get_force_eplb_topk(topk_ids, num_logical_experts=8)

    assert first is second
    assert len(moe_comm_method._force_eplb_topk_cache) == 1


def test_get_force_eplb_topk_returns_none_without_comm_method():
    context = SimpleNamespace(moe_comm_method=None)

    with patch.object(force_eplb, "get_forward_context", return_value=context):
        result = force_eplb.get_force_eplb_topk(torch.empty((2, 2)), num_logical_experts=8)

    assert result is None


@pytest.mark.parametrize(
    ("configured_sizes", "expected_sizes"),
    [
        ([4, 8], {4, 8}),
        ([], {16}),
    ],
)
def test_build_force_eplb_topk_uses_configured_or_fallback_sizes(configured_sizes, expected_sizes):
    moe_comm_method = _make_moe_comm_method(ep_rank=0)
    context = SimpleNamespace(moe_comm_method=moe_comm_method)
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=configured_sizes))

    with (
        patch.object(force_eplb, "get_forward_context", return_value=context),
        patch.object(force_eplb, "get_current_vllm_config", return_value=vllm_config),
    ):
        force_eplb.build_force_eplb_topk(torch.device("cpu"), max_num_tokens=16)

    cached_sizes = {key[0] for key in moe_comm_method._force_eplb_topk_cache}
    assert cached_sizes == expected_sizes
    assert len(moe_comm_method._force_eplb_topk_cache) == 2 * len(expected_sizes)
