# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3 as kimi_k3_module
from vllm_ascend.models.kimi_k3 import AscendKimiMoE
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.shared_experts import SharedExpertParallelMode
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture
def latent_overlap(monkeypatch):
    """Execute model/runner scheduling with stubbed kernels, transport and reductions."""
    order = []
    events = []
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner._gate = None
    runner.gate = None
    runner._shared_experts = object()
    runner.quant_type = QuantType.W4A8
    runner.routed_scaling_factor = 1.0
    runner.moe_config = SimpleNamespace(hidden_dim=2)

    def gather(x):
        order.append("AG")
        event = object()
        events.append(event)
        return PreparedSharedExpertInput(torch.cat((x, x)), is_gathered=True, ready_event=event)

    def down(x):
        order.append("Down")
        return x[:, :2].clone(), None

    def routed(**kwargs):
        order.append("routed")
        assert kwargs["hidden_states"].shape[-1] == 2
        assert kwargs["router_logits"].shape[0] == kwargs["hidden_states"].shape[0]
        return kwargs["hidden_states"] * 3, RoutedMoEMilestones()

    def shared(prepared, milestones, *, defer_output_wait):
        order.append("shared")
        assert prepared.is_gathered
        assert milestones.shared_input_ready is milestones.router_output_ready
        assert milestones.routed_finalize_done is not None
        assert defer_output_wait
        return prepared.hidden_states[: prepared.hidden_states.shape[0] // 2] * 2

    def up(x):
        order.append("Up")
        return x.repeat(1, 2)

    def wait(event):
        assert event is events[-1]
        order.append("wait_AG")

    runner.routed_input_transform = MagicMock(side_effect=down)
    runner.routed_output_transform = up
    runner.routed_experts = SimpleNamespace(
        forward_impl=MagicMock(side_effect=routed),
        quant_method=SimpleNamespace(has_unpadded_output=False),
    )
    runner.ascend_shared_experts = SimpleNamespace(
        multistream_overlap=True,
        parallel_mode=lambda: SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY,
        prepare_input_async=MagicMock(side_effect=gather),
        prepare_input_before_routed=MagicMock(side_effect=AssertionError("duplicate AG")),
        prepare_for_router_overlap=MagicMock(side_effect=AssertionError("early shared preparation")),
        enqueue_gate_up_after_router=MagicMock(side_effect=AssertionError("early Gate-Up")),
        forward=MagicMock(side_effect=shared),
        wait_for_output=MagicMock(side_effect=lambda: order.append("join")),
    )
    assert runner._defer_shared_submission
    runner._sequence_parallel_context = nullcontext
    runner._maybe_pad_hidden_states = lambda shared, x: (x, None, None)
    runner._encode_layer_name = lambda: "latent_overlap_test"
    runner._forward_entry = lambda x, logits, shared, ids, *_: runner._forward_impl(x, logits, shared, ids)
    runner._maybe_reduce_routed_output_before_transform = lambda x, reduced: (x, reduced)
    runner._maybe_reduce_shared_expert_output = lambda x, reduced: x
    runner._maybe_apply_routed_scale_to_output = lambda shared, routed: (shared, routed)
    runner._maybe_reduce_final_output = lambda x, *_: x
    runner._maybe_add_zero_expert_output = lambda x: x
    monkeypatch.setattr(AscendMoERunner, "_fused_output_is_reduced", property(lambda _: True))
    monkeypatch.setattr(fused_moe_module, "maybe_record_event", lambda: object())
    stream = SimpleNamespace(wait_event=MagicMock(side_effect=wait), record_event=lambda: object())
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)

    model = AscendKimiMoE.__new__(AscendKimiMoE)
    nn.Module.__init__(model)
    model.experts = runner

    def gate(x):
        order.append("Gate")
        assert x.shape[-1] == 4
        return x.sum(dim=-1, keepdim=True).expand(-1, 3), None

    model.gate = MagicMock(side_effect=gate, spec=[])
    return SimpleNamespace(model=model, runner=runner, order=order, stream=stream, events=events)


@pytest.mark.parametrize("ascend_runner", [False, True])
def test_kimi_constructor_uses_generic_runner(latent_overlap, monkeypatch, ascend_runner):
    runner = latent_overlap.runner if ascend_runner else nn.Module()
    runner.moe_config = SimpleNamespace(num_logical_experts=8, num_experts=8, num_local_experts=4)
    monkeypatch.setattr(kimi_k3_module, "FusedMoEFactory", lambda **kwargs: runner)
    monkeypatch.setattr(
        kimi_k3_module, "GateLinear", lambda input_size, output_size, **kwargs: nn.Linear(input_size, output_size)
    )
    monkeypatch.setattr(kimi_k3_module, "KimiMLP", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(
        kimi_k3_module, "ReplicatedLinear", lambda in_size, out_size, **kwargs: nn.Linear(in_size, out_size)
    )
    monkeypatch.setattr(kimi_k3_module, "AscendKimiRoutedOutputTransform", lambda norm, up: up)
    config = SimpleNamespace(
        hidden_size=4,
        moe_intermediate_size=8,
        num_experts=8,
        num_experts_per_token=2,
        routed_expert_hidden_size=2,
        latent_moe_use_norm=False,
        routed_scaling_factor=1.0,
        num_shared_experts=1,
        hidden_act="silu",
        moe_renormalize=True,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
    )
    model = AscendKimiMoE(
        config=config,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=False)),
        use_sequence_parallel=True,
    )
    assert not hasattr(model, "_overlap_shared_expert_input")
    assert model.n_local_physical_experts == 4
    if ascend_runner:
        assert runner._defer_shared_submission
        runner.ascend_shared_experts.prepare_input_async.assert_not_called()


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kimi_gate_ag_down_order_and_numerical_output(latent_overlap, monkeypatch, num_tokens, dtype):
    case = latent_overlap
    x = torch.arange(num_tokens * 4, dtype=dtype).reshape(num_tokens, 4)
    original = x.clone()
    foreach = MagicMock(side_effect=AssertionError("Latent SP overlap must use ordinary Add"))
    monkeypatch.setattr(torch, "_foreach_add", foreach)

    output = case.model(x)

    assert case.order == ["AG", "Gate", "Down", "wait_AG", "routed", "shared", "Up", "join"]
    torch.testing.assert_close(output, x * 2 + (x[:, :2] * 3).repeat(1, 2))
    torch.testing.assert_close(x, original)
    assert output.shape == x.shape and output.dtype == x.dtype
    case.runner.ascend_shared_experts.prepare_input_async.assert_called_once()
    case.runner.routed_input_transform.assert_called_once()
    case.stream.wait_event.assert_called_once_with(case.events[0])
    foreach.assert_not_called()


@pytest.mark.parametrize("quant_type", list(QuantType))
@pytest.mark.parametrize("gate_location", ["external", "internal", "internal_precast"])
def test_generic_latent_schedule_is_model_and_quantization_independent(
    latent_overlap, monkeypatch, quant_type, gate_location
):
    case = latent_overlap
    runner = case.runner
    runner.quant_type = quant_type
    x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    expected_logits = x.sum(dim=-1, keepdim=True).expand(-1, 3)
    if gate_location == "external":
        # Any external-gate model can use this runner API; no model opt-in.
        prepared = runner.prepare_shared_input(x)
        logits, _ = case.model.gate(x)
        output = runner(x, logits, prepared_shared_expert_input=prepared)
    elif gate_location == "internal":
        runner.gate = case.model.gate
        output = runner(x, x)
        runner.gate.assert_called_once_with(x)
    else:
        weight = torch.eye(3, 4)
        runner.gate = SimpleNamespace(weight_fp32=weight)
        linear = torch.nn.functional.linear

        def gate_linear(states, gate_weight):
            case.order.append("Gate")
            assert states.shape == (2, 4) and states.dtype == torch.float32
            return linear(states, gate_weight)

        monkeypatch.setattr(fused_moe_module.F, "linear", gate_linear)
        # Cover the caller's optional pre-cast FP32 router input as well.
        output = runner(x, x.float())
        expected_logits = linear(x.float(), weight)
    assert case.order == ["AG", "Gate", "Down", "wait_AG", "routed", "shared", "Up", "join"]
    actual_logits = runner.routed_experts.forward_impl.call_args.kwargs["router_logits"]
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(output, x * 2 + (x[:, :2] * 3).repeat(1, 2))
    runner.ascend_shared_experts.prepare_input_async.assert_called_once()


def test_internal_gate_consumes_prepared_input_without_duplicate_ag(latent_overlap):
    case = latent_overlap
    x = torch.ones(2, 4)
    case.runner.gate = case.model.gate
    prepared = case.runner.prepare_shared_input(x)
    output = case.runner(x, x, prepared_shared_expert_input=prepared)
    torch.testing.assert_close(output, torch.full_like(x, 5))
    assert case.order == ["AG", "Gate", "Down", "wait_AG", "routed", "shared", "Up", "join"]
    case.runner.ascend_shared_experts.prepare_input_async.assert_called_once()
    case.runner.gate.assert_called_once()


def test_internal_gate_with_pretransformed_input_uses_original_router_input(latent_overlap):
    case = latent_overlap
    x = torch.ones(2, 4)
    case.runner.gate = case.model.gate
    output = case.runner(x[:, :2], x, shared_experts_input=torch.cat((x, x)))
    torch.testing.assert_close(output, torch.full_like(x, 5))
    assert case.order == ["Gate", "routed", "shared", "Up", "join"]
    case.runner.gate.assert_called_once_with(x)
    case.runner.routed_input_transform.assert_not_called()
    case.runner.ascend_shared_experts.prepare_input_async.assert_not_called()


@pytest.mark.parametrize("failure", ["Gate", "Down", "routed"])
def test_kimi_failure_does_not_submit_shared_or_leak_prepared_input(latent_overlap, failure):
    case = latent_overlap
    operation = {
        "Gate": case.model.gate,
        "Down": case.runner.routed_input_transform,
        "routed": case.runner.routed_experts.forward_impl,
    }[failure]
    original = operation.side_effect
    operation.side_effect = RuntimeError("injected failure")
    x = torch.ones(2, 4)
    with pytest.raises(RuntimeError, match="injected failure"):
        case.model(x)
    case.runner.ascend_shared_experts.forward.assert_not_called()

    operation.side_effect = original
    case.order.clear()
    output = case.model(x)
    torch.testing.assert_close(output, torch.full_like(x, 5))
    assert len(case.events) == 2 and case.events[0] is not case.events[1]
    assert case.order == ["AG", "Gate", "Down", "wait_AG", "routed", "shared", "Up", "join"]


@pytest.mark.parametrize(
    "excluded",
    [None, "no_shared", "no_down", "no_up", "no_overlap", "shared_dp", "no_sp"],
)
def test_shared_schedule_requires_latent_sp_layout(latent_overlap, excluded):
    runner = latent_overlap.runner
    if excluded == "no_shared":
        runner.ascend_shared_experts = None
    elif excluded == "no_down":
        runner.routed_input_transform = None
    elif excluded == "no_up":
        runner.routed_output_transform = None
    elif excluded == "no_overlap":
        runner.ascend_shared_experts.multistream_overlap = False
    elif excluded == "shared_dp":
        runner.ascend_shared_experts.parallel_mode = lambda: SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP
    elif excluded == "no_sp":
        runner.ascend_shared_experts.parallel_mode = lambda: SharedExpertParallelMode.TENSOR_PARALLEL
    assert runner._defer_shared_submission is (excluded is None)
    if excluded is not None:
        assert runner.prepare_shared_input(torch.ones(2, 4)) is None


@pytest.mark.parametrize("invalid", ["legacy_input", "disabled", "not_gathered", "no_event"])
def test_prepared_shared_input_rejects_ambiguous_or_incomplete_contract(latent_overlap, invalid):
    runner = latent_overlap.runner
    x = torch.ones(2, 4)
    kwargs = {}
    if invalid == "legacy_input":
        kwargs["shared_experts_input"] = x
    if invalid == "disabled":
        runner.ascend_shared_experts.multistream_overlap = False
    prepared = PreparedSharedExpertInput(
        x, is_gathered=invalid != "not_gathered", ready_event=None if invalid == "no_event" else object()
    )
    with pytest.raises(ValueError):
        runner(x, x, prepared_shared_expert_input=prepared, **kwargs)
    runner.routed_input_transform.assert_not_called()
    runner.ascend_shared_experts.prepare_input_async.assert_not_called()


def test_legacy_pretransformed_input_still_skips_down_and_gather(latent_overlap):
    case = latent_overlap
    x = torch.ones(2, 4)
    output = case.runner(x[:, :2], x, shared_experts_input=torch.cat((x, x)))
    torch.testing.assert_close(output, torch.full_like(x, 5))
    case.runner.routed_input_transform.assert_not_called()
    case.runner.ascend_shared_experts.prepare_input_async.assert_not_called()
    assert case.order == ["routed", "shared", "Up", "join"]


def test_non_latent_schedule_retains_early_shared_submission_and_foreach(latent_overlap, monkeypatch):
    case = latent_overlap
    case.runner.routed_input_transform = None
    case.runner.routed_output_transform = None
    shared = case.runner.ascend_shared_experts
    # A non-latent SP path prepares its input before routed collectives and
    # joins inside the executor, as before this change.
    shared.prepare_input_before_routed.side_effect = shared.prepare_input_async.side_effect
    case.runner.routed_experts.forward_impl.side_effect = lambda **kwargs: (
        case.order.append("routed"),
        (kwargs["hidden_states"] * 3, RoutedMoEMilestones()),
    )[1]
    prepared_execution, gate_up = object(), object()
    shared.prepare_for_router_overlap.side_effect = lambda *args: (
        case.order.append("prepare_shared"),
        prepared_execution,
    )[1]
    shared.enqueue_gate_up_after_router.side_effect = lambda *args: (case.order.append("Gate-Up"), gate_up)[1]
    shared.forward.side_effect = lambda prepared, milestones, **kwargs: (
        case.order.append("shared_tail"),
        prepared.hidden_states[:2] * 2,
    )[1]
    foreach = MagicMock(side_effect=lambda a, b: [a[0] + b[0]])
    monkeypatch.setattr(torch, "_foreach_add", foreach)

    output = case.model(torch.ones(2, 4))

    torch.testing.assert_close(output, torch.full((2, 4), 5.0))
    assert case.order == [
        "Gate",
        "AG",
        "prepare_shared",
        "Gate-Up",
        "routed",
        "shared_tail",
    ]
    assert shared.forward.call_args.kwargs["submitted_gate_up"] is gate_up
    foreach.assert_called_once()


@pytest.mark.parametrize("internal_gate", [False, True])
def test_prepared_tensor_is_explicit_in_fake_graph_and_replay(latent_overlap, internal_gate):
    from torch.fx.experimental.proxy_tensor import make_fx

    case = latent_overlap
    runner = case.runner
    if internal_gate:
        runner.gate = case.model.gate
    library = torch.library.Library("latent_overlap_test", "FRAGMENT")
    library.define("experts(Tensor routed, Tensor logits, Tensor gathered) -> (Tensor, Tensor)")
    library.impl("experts", lambda routed, logits, gathered: (gathered[: routed.shape[0]] * 2, routed * 3), "CPU")

    def fake(routed, logits, gathered):
        return fused_moe_module._ascend_moe_forward_shared_sp_fake(routed, logits, gathered, None, "test", 0)

    torch.library.register_fake("latent_overlap_test::experts", fake, lib=library)
    runner._forward_entry = lambda x, logits, gathered, *_: torch.ops.latent_overlap_test.experts(x, logits, gathered)
    try:
        x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        forward = (lambda states: runner(states, states)) if internal_gate else case.model
        graph = make_fx(forward, tracing_mode="fake")(x)
        expert_node = next(
            node for node in graph.graph.nodes if node.target == torch.ops.latent_overlap_test.experts.default
        )
        assert expert_node.args[0].meta["val"].shape == (5, 2)
        assert expert_node.args[2].meta["val"].shape == (10, 4)
        assert expert_node.meta["val"][0].shape == (5, 4)
        for replay_input in (x, x + 1):
            torch.testing.assert_close(graph(replay_input), replay_input * 2 + (replay_input[:, :2] * 3).repeat(1, 2))
    finally:
        library._destroy()
