# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
import vllm_ascend.ops.fused_moe.routed_experts as routed_module
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("local_rows", [0, 1, 4])
@pytest.mark.parametrize("dp,pcp", [(1, 2), (1, 8), (2, 1), (2, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("early,local_topk", [(True, False), (False, True), (True, True)])
def test_route_gather_preserves_rows_and_waits(monkeypatch, local_rows, dp, pcp, dtype, early, local_topk):
    events = []

    def group(size, name):
        def gather(value, dim):
            events.append((name, value.shape[1]))
            return torch.cat([value] * size)

        return SimpleNamespace(all_gather=gather, reduce_scatter=lambda value, dim: value.chunk(size)[0], size=size)

    dp_group, pcp_group = group(dp, "dp"), group(pcp, "pcp")
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=dp, pcp_size=pcp, dp_group=dp_group, is_sequence_parallel=False)
    pcp_rows = 4 * dp + 1 if dp > 1 else 4
    monkeypatch.setattr(
        prepare_module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=4, max_tokens_across_pcp=pcp_rows)
    )
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(prepare_module, "get_dp_group", lambda: dp_group)
    hidden = torch.arange(local_rows * 14, dtype=dtype).reshape(local_rows, 14)[:, ::2]
    logits = torch.randn(local_rows, 64)

    def launch(value, actual_group):
        assert actual_group is (dp_group if dp > 1 else pcp_group)
        assert value.is_contiguous()
        events.append(("launch", value.shape[1]))
        result = torch.full((value.shape[0] * actual_group.size, value.shape[1]), float("nan"), dtype=dtype)

        def wait():
            events.append(("wait", value.shape[1]))
            result.copy_(torch.cat([value] * actual_group.size))

        return result, SimpleNamespace(wait=wait)

    monkeypatch.setattr(prepare_module, "all_gather_async", launch)

    def select(states, scores, force_balance):
        assert not force_balance and states.shape[0] == scores.shape[0]
        events.append(("select", 64))
        # Exercise INT64 IDs beyond both FP32 exactness and INT32 range.
        ids = torch.tensor([2**24 + 1, 2**32 + 3]).expand(scores.shape[0], -1)
        return torch.softmax(scores[:, -2:], dim=-1).to(dtype), ids

    expected_hidden = layer.prepare(hidden, logits).hidden_states
    events.clear()
    pending = layer.start_pcp_hidden_gather(hidden) if early else None
    if local_topk:
        output, weights, ids = layer.prepare_local_topk(hidden, logits, select, pending=pending)
        expected_weights, expected_ids = select(hidden, logits, False)
        assert torch.equal(layer.finalize(weights, False), expected_weights)
        assert torch.equal(layer.finalize(ids, False), expected_ids)
        assert weights.dtype == dtype and ids.dtype == torch.int64
    else:
        output = pending(logits)
        reference = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
        reference.moe_config = layer.moe_config
        assert torch.equal(output.router_logits, reference.prepare(hidden, logits).router_logits)
    assert torch.equal(output.hidden_states, expected_hidden)
    assert torch.equal(layer.finalize(output.hidden_states, False), hidden)
    if early:
        assert events[0] == ("launch", 7)
        wait_index = events.index(("wait", 7))
        assert all(name not in ("dp", "pcp") for name, _ in events[:wait_index])
        if local_topk:
            assert events[1] == ("select", 64)


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("experts,k", [(16, 1), (64, 4), (256, 8), (288, 8)])
@pytest.mark.parametrize("grouped,bias,renormalize", [(False, False, False), (False, True, True), (True, True, True)])
@pytest.mark.parametrize("distribution", ["random", "ties", "saturated"])
def test_real_router_is_row_independent(scoring, experts, k, grouped, bias, renormalize, distribution):
    router = AscendGroupedTopKRouter(
        top_k=k,
        global_num_experts=experts,
        num_expert_group=4,
        topk_group=2,
        use_grouped_topk=grouped,
        scoring_func=scoring,
        renormalize=renormalize,
        routed_scaling_factor=1.7,
        e_score_correction_bias=torch.linspace(-0.2, 0.2, experts) if bias else None,
    )
    scores = torch.randn(13, experts)
    if distribution == "ties":
        scores.zero_()
    elif distribution == "saturated":
        scores *= 100
    hidden = torch.randn(13, 96)
    expected = router._select_experts(hidden, scores)
    shards = [router._select_experts(hidden[a:b], scores[a:b]) for a, b in [(0, 1), (1, 5), (5, 13)]]
    for index in range(2):
        assert torch.equal(torch.cat([shard[index] for shard in shards]), expected[index])
    assert router.supports_local_routing
    router.capture_fn = lambda ids: None
    assert not router.supports_local_routing


@pytest.mark.parametrize("router_cls", [AscendGroupedTopKRouter, AscendFusedTopKRouter])
def test_router_with_global_side_effects_keeps_global_routing(router_cls):
    router = router_cls(top_k=2, global_num_experts=32, num_expert_group=1, topk_group=1)
    assert router.supports_local_routing
    router.eplb_state = object()
    assert not router.supports_local_routing
    router.eplb_state = None
    if router_cls is AscendFusedTopKRouter:
        for attr in ["custom_routing_function", "tid2eid", "bias_vl"]:
            setattr(router, attr, object())
            assert not router.supports_local_routing
            setattr(router, attr, None)


@pytest.mark.parametrize(
    "quant_type", [QuantType.NONE, QuantType.W8A8, QuantType.W8A8MXFP, QuantType.W4A8MXFP, QuantType.W4A4MXFP]
)
def test_local_routing_reuses_sp_quantization_and_unpadding(monkeypatch, quant_type):
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(is_sequence_parallel=True)
    layer.dynamic_mx_quant_scale_alg = 0
    hidden, logits = torch.randn(5, 64), torch.randn(5, 32)
    scale = torch.randn(5, 1)
    quantized = quant_type != QuantType.NONE
    packed_width = 32 if quant_type == QuantType.W4A4MXFP else 64
    quantized_hidden = torch.randint(-127, 127, (5, packed_width), dtype=torch.int8)
    monkeypatch.setattr(
        prepare_module.torch_npu, "npu_dynamic_quant", Mock(return_value=(quantized_hidden, scale)), raising=False
    )
    monkeypatch.setattr(
        prepare_module.torch_npu, "npu_dynamic_mx_quant", Mock(return_value=(quantized_hidden, scale)), raising=False
    )

    # Simulate SP row ordering and removal of the last padding row.
    def gather(value):
        return torch.cat([value, value.flip(0)])[:-1]

    monkeypatch.setattr(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", gather, raising=False)
    router = AscendGroupedTopKRouter(top_k=2, global_num_experts=32, num_expert_group=1, topk_group=1)

    def select(states, scores, force):
        assert states.shape[-1] == (packed_width if quantized else hidden.shape[-1])
        return router._select_experts(states, scores)

    output, weights, ids = layer.prepare_local_topk(hidden, logits, select, quant_type=quant_type)
    expected_weights, expected_ids = router._select_experts(gather(hidden), gather(logits))
    assert torch.equal(weights, expected_weights) and torch.equal(ids, expected_ids)
    assert torch.equal(output.hidden_states, gather(quantized_hidden if quantized else hidden))
    if quantized:
        assert torch.equal(output.pertoken_scale, gather(scale))
    else:
        assert output.pertoken_scale is None


@pytest.mark.parametrize(
    "profiling,capability,experts,k",
    [(False, True, 64, 4), (True, True, 64, 4), (False, False, 64, 4), (False, True, 8, 8)],
)
def test_local_topk_uses_capability_and_payload_size(monkeypatch, profiling, capability, experts, k):
    hidden, logits = torch.zeros(1, 7), torch.zeros(1, experts)
    prepared = SimpleNamespace(
        hidden_states=hidden, router_logits=logits, mc2_mask=None, padded_hidden_states_shape=None, pertoken_scale=None
    )
    weights, ids = torch.ones(1, k), torch.zeros(1, k, dtype=torch.int32)
    comm = routed_module.AllGatherCommImpl.__new__(routed_module.AllGatherCommImpl)
    comm.prepare = Mock(return_value=prepared)
    comm.prepare_finalize = SimpleNamespace(
        prepare_local_topk=Mock(return_value=(prepared, weights, ids)),
        is_large_payload=lambda byte_count: byte_count > 0,
    )
    comm.finalize = Mock(return_value=hidden)
    layer = SimpleNamespace(
        enable_npugraph_ex_static_kernel=False,
        moe_config=SimpleNamespace(pcp_size=2, dp_size=2, is_sequence_parallel=False),
        dynamic_eplb=False,
        log2phy=None,
        mix_placement=False,
        router=SimpleNamespace(top_k=k, supports_local_routing=capability, eplb_state=None),
        quant_type=QuantType.NONE,
        _select_experts=Mock(return_value=(weights, ids)),
        quant_method=SimpleNamespace(
            apply=Mock(return_value=SimpleNamespace(routed_out=hidden, expert_tokens=None, group_list_type=1))
        ),
        _use_v2_model_runner=True,
        return_with_event=False,
    )
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    monkeypatch.setattr(routed_module, "get_ascend_config", lambda: SimpleNamespace(enable_force_eplb=False))
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(in_profile_run=profiling, moe_comm_type=None))
    assert routed_module.AscendRoutedExperts.forward_impl(layer, hidden_states=hidden, router_logits=logits) is hidden
    optimized = not profiling and capability and k * 12 < experts * 4
    assert comm.prepare_finalize.prepare_local_topk.call_count == int(optimized)
    assert comm.prepare.call_count == int(not optimized)


@pytest.mark.parametrize(
    "mode,sp", [(CUDAGraphMode.NONE, False), (CUDAGraphMode.FULL, False), (CUDAGraphMode.NONE, True)]
)
def test_early_hidden_uses_runtime_graph_mode_not_model_allowlist(monkeypatch, mode, sp):
    layer = SimpleNamespace(moe_config=SimpleNamespace(dp_size=2, pcp_size=2, is_sequence_parallel=sp))
    comm = routed_module.AllGatherCommImpl.__new__(routed_module.AllGatherCommImpl)
    comm.prepare_finalize = SimpleNamespace(
        start_pcp_hidden_gather=Mock(return_value="pending"), is_large_payload=lambda byte_count: True
    )
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=None))
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    result = routed_module.AscendRoutedExperts.start_pcp_hidden_gather(layer, torch.zeros(1, 7))
    assert result == ((comm, "pending") if mode == CUDAGraphMode.NONE and not sp else None)


@pytest.mark.parametrize("dp,pcp,tp,sp", [(1, 8, 1, False), (2, 2, 2, False), (2, 1, 4, True)])
@pytest.mark.parametrize(
    "rows,experts,k,enabled",
    [(128, 288, 8, False), (1024, 288, 8, True), (1024, 64, 4, False), (4096, 64, 4, True), (4096, 8, 8, False)],
)
def test_communication_budget_scales_with_payload_not_model(monkeypatch, dp, pcp, tp, sp, rows, experts, k, enabled):
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=dp, pcp_size=pcp, tp_size=tp, is_sequence_parallel=sp)
    # SP context describes tokens before TP sharding; DP describes its max
    # local batch. These values are identical on all participating ranks.
    monkeypatch.setattr(
        prepare_module,
        "_EXTRA_CTX",
        SimpleNamespace(max_tokens_across_dp=rows * tp if sp else rows, max_tokens_across_pcp=rows),
    )
    assert layer.is_large_payload(experts * 4 - k * 12) is enabled


def test_communication_budget_boundary(monkeypatch):
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=1, pcp_size=2, is_sequence_parallel=False)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_pcp=768))
    assert layer.is_large_payload(1024)
    assert not layer.is_large_payload(1023)
