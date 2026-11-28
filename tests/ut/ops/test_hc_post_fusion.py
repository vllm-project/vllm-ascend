# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.fused_moe import fused_moe as module
from vllm_ascend.ops.triton.hc_post import can_fuse_hc_post_add, fused_hc_post_add


def _inputs(tokens=3, dim=16):
    torch.manual_seed(1024)
    return (
        torch.randn(tokens, dim, dtype=torch.bfloat16),
        torch.randn(tokens, dim, dtype=torch.bfloat16),
        torch.randn(tokens, 4, dim, dtype=torch.bfloat16),
        torch.randn(tokens, 4),
        torch.randn(tokens, 4, 4),
    )


def _post_reference(x, residual, post, comb):
    value = x.float().unsqueeze(1) * post.unsqueeze(-1)
    for source in range(4):
        value = value + residual[:, source, :].float().unsqueeze(1) * comb[:, source, :].unsqueeze(-1)
    return value.to(residual.dtype)


def test_opt_in_and_hardware_scope():
    offload_config = SimpleNamespace(enabled=False)
    assert AscendConfig(sparse_kv_offload_config=offload_config).enable_hc_post_fusion is False
    assert (
        AscendConfig(sparse_kv_offload_config=offload_config, enable_hc_post_fusion=True).enable_hc_post_fusion is True
    )
    for device in AscendDeviceType:
        assert get_hardware_profile(device).supports(HardwareCapability.HC_POST_ADD_SEPARATE_ROUNDING) == (
            device == AscendDeviceType.A3
        )


@pytest.mark.parametrize("tokens", [0, 1, 10, 12, 64])
def test_hc_post_metadata_accepts_decode_shapes(tokens):
    assert can_fuse_hc_post_add(*_inputs(tokens))


@pytest.mark.parametrize("case", ["no_shared", "fp16", "width", "branches", "coeff_dtype", "stride", "large_prefill"])
def test_hc_post_metadata_rejects_incompatible_inputs(case):
    shared, routed, residual, post, comb = _inputs()
    if case == "no_shared":
        shared = None
    elif case == "fp16":
        routed = routed.half()
    elif case == "width":
        shared = shared[:, :-1]
    elif case == "branches":
        residual = residual[:, :3]
    elif case == "coeff_dtype":
        post = post.bfloat16()
    elif case == "stride":
        residual = torch.randn(3, 4, 32, dtype=torch.bfloat16)[:, :, ::2]
    elif case == "large_prefill":
        shared, routed, residual, post, comb = _inputs(129)
    assert not can_fuse_hc_post_add(shared, routed, residual, post, comb)
    with pytest.raises(ValueError):
        fused_hc_post_add(shared, routed, residual, post, comb)


def test_hc_post_fake_is_pure_and_does_not_alias(monkeypatch):
    monkeypatch.setattr(module, "get_forward_context", MagicMock(side_effect=AssertionError("fake accessed context")))
    with FakeTensorMode():
        inputs = _inputs()
        result = torch.ops.vllm.ascend_moe_hc_post(*inputs, "missing.layer", 0)
        assert result.shape == inputs[2].shape
        assert result.dtype == inputs[2].dtype
        assert result is not inputs[2]
        assert result.is_contiguous()


@pytest.mark.parametrize(
    ("comm", "sp", "early_reduce", "expect_fused"),
    [
        (MoECommType.ALLGATHER, False, False, False),
        (MoECommType.MC2, False, False, True),
        (MoECommType.FUSED_MC2, False, False, True),
        (MoECommType.ALLTOALL, False, False, True),
        (MoECommType.ALLGATHER, True, False, True),
        (MoECommType.ALLGATHER, False, True, True),
    ],
)
def test_hc_post_respects_collective_boundary(monkeypatch, comm, sp, early_reduce, expect_fused):
    runner = module.AscendMoERunner.__new__(module.AscendMoERunner)
    runner.moe_config = SimpleNamespace(tp_size=2, ep_size=16, is_sequence_parallel=sp)
    runner.router = SimpleNamespace()
    runner.layer_name = "model.layers.0.mlp"
    runner._should_reduce_routed_before_combine = lambda: early_reduce
    context = SimpleNamespace(moe_comm_type=comm, no_compile_layers={runner.layer_name: runner})
    monkeypatch.setattr(module, "_EXTRA_CTX", context)
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    events = []

    def finish(shared, routed, trunc):
        events.append("finish_and_reduce")
        return (shared + routed) * 2

    def native(x, residual, post, comb):
        events.append("native_post")
        return _post_reference(x[0], residual[0], post[0], comb[0]).unsqueeze(0)

    def fused(shared, routed, residual, post, comb):
        events.append("fused_post")
        return _post_reference(shared + routed, residual, post, comb)

    runner._finish_output = finish
    monkeypatch.setattr(module, "fused_hc_post_add", fused)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_hc_post", native, raising=False)
    inputs = _inputs()
    original = [t.clone() for t in inputs]
    actual = module._ascend_moe_hc_post(*inputs, runner.layer_name, 0)
    expected_x = inputs[0] + inputs[1]
    if not expect_fused:
        expected_x = expected_x * 2
    assert torch.equal(actual, _post_reference(expected_x, *inputs[2:]))
    assert events == (["fused_post"] if expect_fused else ["finish_and_reduce", "native_post"])
    assert all(torch.equal(a, b) for a, b in zip(inputs, original))


@pytest.mark.parametrize("reason", ["zero_expert", "truncation", "no_shared"])
def test_hc_post_falls_back_for_additional_output_work(monkeypatch, reason):
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=2, ep_size=16, is_sequence_parallel=False),
        _should_reduce_routed_before_combine=lambda: False,
        _fused_output_is_reduced=True,
        router=SimpleNamespace(),
        _finish_output=MagicMock(return_value=torch.ones(3, 16, dtype=torch.bfloat16)),
    )
    inputs = list(_inputs())
    trunc = 0
    if reason == "zero_expert":
        runner.router = module.ZeroExpertRouter.__new__(module.ZeroExpertRouter)
        if isinstance(runner.router, torch.nn.Module):
            torch.nn.Module.__init__(runner.router)
        runner.router._zero_expert_output = torch.ones(3, 16)
    elif reason == "truncation":
        trunc = 8
    else:
        inputs[0] = None
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": runner}))
    fused = MagicMock(side_effect=AssertionError("ineligible fusion"))
    monkeypatch.setattr(module, "fused_hc_post_add", fused)
    native = MagicMock(return_value=torch.zeros(1, 3, 4, 16))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_hc_post", native, raising=False)
    module._ascend_moe_hc_post(*inputs, "layer", trunc)
    runner._finish_output.assert_called_once()
    native.assert_called_once()
    fused.assert_not_called()
    if reason == "zero_expert":
        assert runner.router._zero_expert_output is not None


def test_zero_expert_contribution_is_consumed_once_before_hc_post(monkeypatch):
    inputs = _inputs()
    zero = torch.ones_like(inputs[0])
    router = module.ZeroExpertRouter.__new__(module.ZeroExpertRouter)
    if isinstance(router, torch.nn.Module):
        torch.nn.Module.__init__(router)
    router._zero_expert_output = zero
    runner = module.AscendMoERunner.__new__(module.AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(tp_size=2, ep_size=16, is_sequence_parallel=False)
    runner.router = router
    runner._should_reduce_routed_before_combine = lambda: False
    runner._maybe_reduce_final_output = lambda x, size, reduced: x
    monkeypatch.setattr(module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.MC2))
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": runner}))
    monkeypatch.setattr(module, "fused_hc_post_add", MagicMock(side_effect=AssertionError("ineligible fusion")))
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_hc_post",
        lambda x, h, p, c: _post_reference(x[0], h[0], p[0], c[0]).unsqueeze(0),
        raising=False,
    )
    actual = module._ascend_moe_hc_post(*inputs, "layer", 0)
    expected = _post_reference((inputs[0] + inputs[1]) + zero, *inputs[2:])
    assert torch.equal(actual, expected)
    assert router._zero_expert_output is None


def test_precombined_mega_moe_output_keeps_native_hc_post(monkeypatch):
    shared, routed, residual, post, comb = _inputs()
    combined = shared + routed
    runner = module.AscendMoERunner.__new__(module.AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(tp_size=2, ep_size=16, is_sequence_parallel=False)
    runner.router = SimpleNamespace()
    runner._should_reduce_routed_before_combine = lambda: False
    runner._maybe_reduce_final_output = MagicMock(side_effect=lambda value, size, reduced: value)
    runner._maybe_add_zero_expert_output = lambda value: value
    monkeypatch.setattr(module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.MC2))
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": runner}))
    fused = MagicMock(side_effect=AssertionError("MegaMoE has already combined the expert outputs"))
    monkeypatch.setattr(module, "fused_hc_post_add", fused)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_hc_post",
        lambda value, h, p, c: _post_reference(value[0], h[0], p[0], c[0]).unsqueeze(0),
        raising=False,
    )

    actual = module._ascend_moe_hc_post(None, combined, residual, post, comb, "layer", 0)

    assert torch.equal(actual, _post_reference(combined, residual, post, comb))
    runner._maybe_reduce_final_output.assert_called_once_with(combined, None, True)
    fused.assert_not_called()


def test_reference_keeps_bf16_add_rounding():
    shared = torch.tensor([[1.0]], dtype=torch.bfloat16)
    routed = torch.tensor([[2**-8]], dtype=torch.bfloat16)
    residual = torch.zeros(1, 4, 1, dtype=torch.bfloat16)
    post = torch.full((1, 4), 1.5)
    comb = torch.zeros(1, 4, 4)
    correct = _post_reference(shared + routed, residual, post, comb)
    without_rounding = _post_reference(shared.float() + routed.float(), residual, post, comb)
    assert not torch.equal(correct, without_rounding)


def test_compiled_op_rechecks_runtime_reduction(monkeypatch):
    context = SimpleNamespace(moe_comm_type=MoECommType.MC2, no_compile_layers={})
    runner = module.AscendMoERunner.__new__(module.AscendMoERunner)
    runner.moe_config = SimpleNamespace(tp_size=2, ep_size=16, is_sequence_parallel=False)
    runner.router = SimpleNamespace()
    runner._should_reduce_routed_before_combine = lambda: False
    runner._finish_output = lambda shared, routed, trunc: (shared + routed) * 2
    context.no_compile_layers["layer"] = runner
    monkeypatch.setattr(module, "_EXTRA_CTX", context)
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(module, "fused_hc_post_add", lambda s, r, h, p, c: _post_reference(s + r, h, p, c))
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_hc_post",
        lambda x, h, p, c: _post_reference(x[0], h[0], p[0], c[0]).unsqueeze(0),
        raising=False,
    )
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    # Recent vLLM CPU registration already installs the production function.
    # Add a CPU implementation only for versions that register it on NPU alone.
    library = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::ascend_moe_hc_post", "CPU"):
        library = torch.library.Library("vllm", "IMPL", "CPU")
        library.impl("ascend_moe_hc_post", module._ascend_moe_hc_post)
    try:
        compiled = torch.compile(
            lambda s, r, h, p, c: torch.ops.vllm.ascend_moe_hc_post(s, r, h, p, c, "layer", 0),
            backend=backend,
            fullgraph=True,
        )
        inputs = _inputs()
        fused = compiled(*inputs)
        context.moe_comm_type = MoECommType.ALLGATHER
        fallback = compiled(*inputs)
        assert len(graphs) == 1
        assert torch.equal(fused, _post_reference(inputs[0] + inputs[1], *inputs[2:]))
        assert torch.equal(fallback, _post_reference((inputs[0] + inputs[1]) * 2, *inputs[2:]))
    finally:
        if library is not None:
            library._destroy()


def test_runner_passes_transformed_outputs_and_explicit_residual(monkeypatch):
    inputs = _inputs()
    shared, routed, residual, post, comb = inputs
    runner = module.AscendMoERunner.__new__(module.AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(hidden_dim_unpadded=0)
    runner.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(has_unpadded_output=False))
    runner.layer_name = "model.layers.0.mlp"
    runner.apply_routed_input_transform = lambda h: (h, h)
    runner._maybe_pad_hidden_states = lambda shared, h: (h, None, None)
    runner._forward_entry = lambda *args: (shared, routed)
    runner._encode_layer_name = lambda: runner.layer_name
    runner._maybe_reduce_routed_output_before_transform = lambda h, reduced: (h, reduced)
    runner._maybe_reduce_shared_expert_output = lambda s, reduced: s
    runner._maybe_apply_routed_scale_to_output = lambda s, r: (s * 2, r * 3)
    runner.apply_routed_output_transform = lambda h: h + 1
    monkeypatch.setattr(module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.MC2))
    finish = MagicMock(return_value=residual.clone())
    monkeypatch.setattr(torch.ops.vllm, "ascend_moe_hc_post", finish)
    result = runner(routed, routed, hc_post_inputs=(residual, post, comb))
    args = finish.call_args.args
    assert torch.equal(args[0], shared * 2)
    assert torch.equal(args[1], routed * 3 + 1)
    assert all(a is b for a, b in zip(args[2:5], (residual, post, comb)))
    assert result.shape == residual.shape
