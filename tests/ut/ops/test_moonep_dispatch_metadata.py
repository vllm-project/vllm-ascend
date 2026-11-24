# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe import moonep_comm


@pytest.mark.parametrize("use_mxfp8", [False, True], ids=["bf16", "mxfp8"])
@pytest.mark.parametrize(
    ("capabilities", "weight_on_input", "send_weights"),
    [
        ({"planning_skips_source_dup": True}, False, False),
        ({"planning_skips_source_dup": True}, True, True),
        ({"planning_skips_source_dup": False}, False, True),
        ({}, False, True),
    ],
    ids=["no-dedup", "input-weighting", "dedup", "old-sdk"],
)
@pytest.mark.parametrize("planner_counts", [True, False], ids=["native-counts", "prefix-counts"])
def test_moonep_dispatch_metadata(monkeypatch, use_mxfp8, capabilities, weight_on_input, send_weights, planner_counts):
    expected_counts = torch.tensor([3, 0, 4], dtype=torch.int64)
    cu = torch.tensor([0, 0, 3, 3, 7], dtype=torch.int32)
    plan = SimpleNamespace(local_counts=expected_counts) if planner_counts else SimpleNamespace()
    rows = torch.zeros(8, 16, dtype=torch.bfloat16)
    scales = torch.zeros(8, 2, dtype=torch.uint8)
    dispatch = Mock(return_value=(rows, None, cu, plan))
    dispatch_mxfp8 = Mock(return_value=(rows, scales, None, cu, plan))
    combine = Mock(return_value=(rows, None, None))
    buffer = SimpleNamespace(
        _require_ctx=lambda: capabilities,
        dispatch=dispatch,
        dispatch_mxfp8=dispatch_mxfp8,
        combine_weighted=combine,
    )
    pads = {
        "hidden": torch.zeros(4, 16, dtype=torch.bfloat16),
        "topk": torch.zeros(4, 2, dtype=torch.int32),
        "combine_route": torch.zeros(4, 8, dtype=torch.float32),
    }
    monkeypatch.setattr(moonep_comm, "_SCRATCH", {})
    monkeypatch.setattr(moonep_comm, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    monkeypatch.setattr(moonep_comm, "_moonep_dispatch_transport", lambda *_: "mxfp8" if use_mxfp8 else "bf16")
    monkeypatch.setattr(moonep_comm, "get_moonep_tokens_capacity", lambda: 4)
    monkeypatch.setattr(moonep_comm, "_select_buffer_for_step", lambda *_: (buffer, 4))
    monkeypatch.setattr(moonep_comm, "_pads_for", lambda *_: pads)
    ep_group = Mock(return_value=SimpleNamespace(world_size=2, rank_in_group=1))
    monkeypatch.setattr(moonep_comm, "get_ep_group", ep_group)

    dispatcher = object.__new__(moonep_comm.TokenDispatcherWithMoonEP)
    dispatcher.num_experts = 4
    dispatcher.top_k = 2
    dispatcher._logged_transports = set()
    original_weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    topk = torch.tensor([[0, 1], [2, 3]])
    inputs = SimpleNamespace(
        hidden_states=torch.zeros(2, 16, dtype=torch.bfloat16),
        topk_weights=original_weights,
        topk_ids=topk,
        quant=SimpleNamespace(quant_type=object()),
        routing=SimpleNamespace(apply_router_weight_on_input=weight_on_input),
    )
    out = dispatcher.token_dispatch(inputs)
    selected, skipped = (dispatch_mxfp8, dispatch) if use_mxfp8 else (dispatch, dispatch_mxfp8)
    selected.assert_called_once()
    skipped.assert_not_called()
    _, dispatched_weights, expert_ids, _ = selected.call_args.args
    torch.testing.assert_close(expert_ids[:2], topk.to(torch.int32))
    if send_weights:
        torch.testing.assert_close(dispatched_weights[:2], original_weights)
    else:
        assert dispatched_weights is None
        assert "route" not in pads

    torch.testing.assert_close(out.group_list, expected_counts)
    if planner_counts:
        assert out.group_list is expected_counts
        ep_group.assert_not_called()
        assert "cu_pad" not in moonep_comm._SCRATCH[(4, 2, 16)]
    else:
        ep_group.assert_called_once()
    assert out.group_list_type == 1
    assert out.combine_metadata.group_list is out.group_list
    if use_mxfp8:
        torch.testing.assert_close(out.dynamic_scale, scales)
    else:
        assert out.dynamic_scale is None

    assert dispatcher.token_combine(rows, out.combine_metadata).shape == inputs.hidden_states.shape
    assert combine.call_args.kwargs["row_counts"] is out.group_list
    source_weights = combine.call_args.kwargs["source_topk_weights_sk"]
    assert source_weights.dtype == torch.float32
    torch.testing.assert_close(source_weights[:2, :2], original_weights)
    torch.testing.assert_close(source_weights[:, 2:], torch.zeros(4, 6))
