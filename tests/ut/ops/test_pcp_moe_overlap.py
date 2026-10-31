# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
import vllm_ascend.ops.fused_moe.routed_experts as routed_module
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather


@pytest.mark.parametrize("local_rows", [0, 1, 4])
@pytest.mark.parametrize("early,local_topk", [(True, False), (False, True), (True, True)])
def test_pcp_route_gather_preserves_rows_and_waits_before_consumption(monkeypatch, local_rows, early, local_topk):
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=1, pcp_size=8, is_sequence_parallel=False)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_pcp=4))
    events = []
    hidden = torch.arange(local_rows * 8, dtype=torch.float32).reshape(local_rows, 8)[:, ::2]
    logits = torch.arange(local_rows * 288, dtype=torch.float32).reshape(local_rows, 288)
    group = SimpleNamespace()

    def gather(value, dim):
        assert dim == 0
        events.append(("gather", value.shape[1]))
        return torch.cat([value] * 8)

    group.all_gather = gather
    group.reduce_scatter = lambda value, dim: value.chunk(8)[0]
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: group)

    def launch(value, actual_group):
        assert actual_group is group and value.is_contiguous()
        events.append(("launch", value.shape[1]))
        result = torch.full((32, 4), float("nan"))

        def wait():
            events.append(("wait", 4))
            result.copy_(torch.cat([value] * 8))

        return result, SimpleNamespace(wait=wait)

    monkeypatch.setattr(prepare_module, "all_gather_async", launch)

    def select(states, scores, force_balance):
        assert not force_balance and states.shape[0] == scores.shape[0] == 4
        events.append(("select", 288))
        # Large valid IDs exercise exact FP32 packing independently of routing.
        return torch.softmax(scores[:, -8:], dim=-1), torch.arange(280, 288).expand(4, -1).to(torch.int32)

    selector = Mock(side_effect=select)
    pending = layer.start_pcp_hidden_gather(hidden) if early else None
    if local_topk:
        output, weights, ids = layer.prepare_local_topk(hidden, logits, selector, pending=pending)
        expected_logits = torch.nn.functional.pad(logits, (0, 0, 0, 4 - local_rows))
        assert torch.equal(weights, torch.softmax(expected_logits[:, -8:], dim=-1).repeat(8, 1))
        assert ids.dtype == torch.int32
        assert torch.equal(ids, torch.arange(280, 288).expand(32, -1))
        selector.assert_called_once()
    else:
        output = pending(logits)
        selector.assert_not_called()
        expected_logits = torch.nn.functional.pad(logits, (0, 0, 0, 4 - local_rows)).repeat(8, 1)
        assert torch.equal(output.router_logits, expected_logits)
    padded = torch.nn.functional.pad(hidden, (0, 0, 0, 4 - local_rows))
    assert torch.equal(output.hidden_states, padded.repeat(8, 1))
    assert torch.equal(layer.finalize(output.hidden_states, reduce_results=False), hidden)
    if early:
        assert events[0] == ("launch", 4)
        assert events[-2:] == [("wait", 4), ("gather", 16 if local_topk else 288)]
        if local_topk:
            assert events[1] == ("select", 288)
    else:
        assert events == [("gather", 4), ("select", 288), ("gather", 16)]


@pytest.mark.parametrize(
    "max_rows,profiling,scoring,eplb",
    [
        (1024, False, "sigmoid", False),
        (1023, False, "sigmoid", False),
        (1024, True, "sigmoid", False),
        (1024, False, "softmax", False),
        (1024, False, "sigmoid", True),
    ],
)
def test_local_topk_selected_automatically_for_supported_batches(monkeypatch, max_rows, profiling, scoring, eplb):
    hidden = torch.zeros(1, 4)
    logits = torch.zeros(1, 288)
    prepared = SimpleNamespace(
        hidden_states=hidden.repeat(8, 1),
        router_logits=logits.repeat(8, 1),
        mc2_mask=None,
        padded_hidden_states_shape=None,
        pertoken_scale=None,
    )
    weights, ids = torch.ones(1, 8), torch.zeros(1, 8, dtype=torch.int32)
    comm = routed_module.AllGatherCommImpl.__new__(routed_module.AllGatherCommImpl)
    comm.prepare = Mock(return_value=prepared)
    comm.prepare_finalize = SimpleNamespace(prepare_local_topk=Mock(return_value=(prepared, weights, ids)))
    comm.finalize = Mock(return_value=hidden)
    state = Mock(spec=routed_module.AscendEplbLayerState) if eplb else None
    router = type("AscendFusedTopKRouter", (), {"top_k": 8, "scoring_func": scoring, "eplb_state": state})()
    layer = SimpleNamespace(
        enable_npugraph_ex_static_kernel=False,
        moe_config=SimpleNamespace(pcp_size=8, dp_size=1, is_sequence_parallel=False),
        dynamic_eplb=False,
        log2phy=None,
        mix_placement=False,
        router=router,
        quant_type=None,
        _select_experts=Mock(return_value=(weights, ids)),
        quant_method=SimpleNamespace(
            apply=Mock(return_value=SimpleNamespace(routed_out=hidden, expert_tokens=None, group_list_type=1))
        ),
        _use_v2_model_runner=True,
        return_with_event=False,
    )
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    monkeypatch.setattr(routed_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(routed_module, "get_ascend_config", lambda: SimpleNamespace(enable_force_eplb=False))
    monkeypatch.setattr(
        routed_module,
        "_EXTRA_CTX",
        SimpleNamespace(in_profile_run=profiling, max_tokens_across_pcp=max_rows, moe_comm_type=None),
    )
    result = routed_module.AscendRoutedExperts.forward_impl(layer, hidden_states=hidden, router_logits=logits)
    assert result is hidden
    optimized = max_rows >= 1024 and not profiling and scoring == "sigmoid" and not eplb
    assert comm.prepare_finalize.prepare_local_topk.call_count == int(optimized)
    assert comm.prepare.call_count == int(not optimized)
    assert layer._select_experts.call_count == int(not optimized)
    if eplb:
        state.record_load.assert_called_once()
        assert state.record_load.call_args.kwargs["num_source_tokens"] == hidden.shape[0]
        assert state.record_load.call_args.kwargs["prepare_finalize"] is comm.prepare_finalize
