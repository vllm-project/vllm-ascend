# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend import ascend_config
from vllm_ascend.ops.fused_moe import deepgemm_mega_moe as backend
from vllm_ascend.ops.fused_moe import moe_comm_method
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4


@pytest.fixture(autouse=True)
def clear_backend_cache():
    backend.load_deepgemm_mega_moe_ops.cache_clear()
    yield
    backend.load_deepgemm_mega_moe_ops.cache_clear()


@pytest.fixture
def native_module():
    return SimpleNamespace(
        __file__="/site-packages/deep_gemm/__init__.py",
        get_npu_arch=MagicMock(),
        transform_weights_for_mega_moe=MagicMock(),
        SymmBuffer=lambda group, experts, capacity, topk, hidden, intermediate, *, num_shared_experts: None,
        fp8_fp4_mega_moe=lambda output, l1, l2, buffer, *, activation_clamp: None,
    )


@pytest.fixture
def make_deepgemm_communicator(monkeypatch):
    dispatcher = object.__new__(TokenDispatcherWithMC2)
    dispatcher.max_num_tokens_per_rank = 8
    config = SimpleNamespace(enable_fused_mc2=1, enable_deepgemm_mega_moe=True)
    monkeypatch.setattr(moe_comm_method, "get_ascend_config", lambda: config)
    monkeypatch.setattr(FusedMC2CommImpl, "_get_token_dispatcher", lambda self: dispatcher)
    monkeypatch.setattr(FusedMC2CommImpl, "_get_prepare_finalize", lambda self: None)

    def make():
        moe_config = SimpleNamespace(
            activation=MoEActivation.SILU, num_local_experts=2, swiglu_limit=10.0, swiglu_alpha=None, swiglu_beta=None
        )
        with patch.object(torch, "zeros"):
            return FusedMC2CommImpl(moe_config)

    return make


def make_case(tokens=3):
    obj = object.__new__(backend.DeepGemmMegaMoE)
    obj.max_tokens = 8
    obj.config = SimpleNamespace(activation=MoEActivation.SILU, num_experts=4, experts_per_token=2, swiglu_limit=10.0)
    obj.ops = MagicMock()
    obj.buffer = SimpleNamespace(
        x=torch.zeros(8, 64),
        x_sf=torch.zeros(8, 1, dtype=torch.int16),
        topk_idx=torch.zeros(8, 2, dtype=torch.int64),
        topk_weights=torch.zeros(8, 2),
    )
    layer = SimpleNamespace(
        deepgemm_l1=(torch.zeros(2, 2, 256, 32), None),
        deepgemm_l2=(torch.zeros(2, 1, 256, 32), None),
        swiglu_limit=10.0,
    )
    inputs = SimpleNamespace(
        hidden_states=torch.ones(tokens, 64, dtype=torch.bfloat16),
        topk_ids=torch.ones(tokens, 2, dtype=torch.int64),
        topk_weights=torch.full((tokens, 2), 0.75),
        layer=layer,
        dynamic_eplb=False,
        routing=SimpleNamespace(apply_router_weight_on_input=False),
    )
    return obj, inputs


@pytest.mark.parametrize("tokens", [0, 1, 3, 8])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_stage_inputs_and_collective(tokens, scale_dtype):
    obj, inputs = make_case(tokens)
    scale_bytes = torch.tensor([127, 126], dtype=torch.uint8).expand(tokens, 2).contiguous()
    scales = scale_bytes.view(scale_dtype)
    with patch.object(
        backend.torch_npu, "npu_dynamic_mx_quant", return_value=(torch.ones(tokens, 64), scales)
    ) as quant:
        out = obj.apply(inputs)
    assert out.shape == inputs.hidden_states.shape
    assert out.dtype == torch.bfloat16
    assert quant.call_count == bool(tokens)
    assert obj.ops.fp8_fp4_mega_moe.call_count == 1
    assert obj.ops.fp8_fp4_mega_moe.call_args.kwargs["activation_clamp"] == 10.0
    assert "apply_router_weight_on_output" not in obj.ops.fp8_fp4_mega_moe.call_args.kwargs
    assert "use_cann_mxfp_quant" not in obj.ops.fp8_fp4_mega_moe.call_args.kwargs
    assert torch.equal(obj.buffer.topk_idx[:tokens], inputs.topk_ids)
    assert torch.equal(obj.buffer.x_sf[:tokens].view(torch.uint8), scale_bytes)


@pytest.mark.parametrize("invalid", ["overflow", "dtype", "input_weight", "eplb", "weights"])
def test_reject_invalid_input_before_launch(invalid):
    obj, inputs = make_case(9 if invalid == "overflow" else 1)
    if invalid == "dtype":
        inputs.hidden_states = inputs.hidden_states.float()
    if invalid == "input_weight":
        inputs.routing.apply_router_weight_on_input = True
    if invalid == "eplb":
        inputs.dynamic_eplb = True
    if invalid == "weights":
        del inputs.layer.deepgemm_l1
    with pytest.raises(ValueError):
        obj.apply(inputs)
    obj.ops.fp8_fp4_mega_moe.assert_not_called()


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_packed_weight_and_scale_bits_preserved(scale_dtype, noncontiguous, native_module):
    weights = torch.tensor([0, 127, 128, 255], dtype=torch.uint8).repeat(2, 2, 1)
    scale_bytes = torch.tensor([122, 128], dtype=torch.uint8).repeat(2, 2, 1)
    scales = scale_bytes.view(scale_dtype)
    if noncontiguous:
        weights = weights.transpose(0, 2)
        scales = scales.transpose(0, 2)
        scale_bytes = scale_bytes.transpose(0, 2)
    with patch.dict("sys.modules", {"deep_gemm": native_module}):
        backend.prepare_weights(weights, scales, weights, scales)
    args = native_module.transform_weights_for_mega_moe.call_args.args
    for weight, scale in args:
        assert weight.dtype == torch.int8 and scale.dtype == torch.int16
        assert torch.equal(weight.view(torch.uint8), weights)
        assert torch.equal(scale.view(torch.uint8), scale_bytes)


def test_teardown_refused_for_deepgemm():
    comm = object.__new__(FusedMC2CommImpl)
    comm.deepgemm_backend = object()
    with pytest.raises(RuntimeError, match="HCCL teardown"):
        comm.prepare_hccl_teardown()


def test_fused_experts_delegates_to_deepgemm():
    comm = object.__new__(FusedMC2CommImpl)
    comm.deepgemm_backend = MagicMock()
    _, inputs = make_case()
    output = torch.empty_like(inputs.hidden_states)
    comm.deepgemm_backend.apply.return_value = output

    result = comm.fused_experts(inputs)

    comm.deepgemm_backend.apply.assert_called_once_with(inputs)
    assert result.routed_out is output
    assert result.expert_tokens is None


def model_config():
    hf = SimpleNamespace(n_routed_experts=384, hidden_size=5120, moe_intermediate_size=2304)
    model = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        quantization="deepseek_v4_fp8",
        enforce_eager=True,
        hf_text_config=hf,
        hf_config=SimpleNamespace(quantization_config={"expert_dtype": "fp4", "scale_fmt": "ue8m0"}),
    )
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    return SimpleNamespace(model_config=model, parallel_config=parallel, speculative_config=None, lora_config=None)


def test_opt_in_requires_explicit_mega_mode():
    from vllm_ascend.ascend_config import AscendConfig

    with pytest.raises(ValueError, match="requires enable_fused_mc2=2"):
        AscendConfig(sparse_kv_offload_config={}, enable_deepgemm_mega_moe=True)


def test_config_defers_package_resolution_to_worker():
    with patch.object(ascend_config.importlib.util, "find_spec") as find_spec:
        config = ascend_config.AscendConfig(
            sparse_kv_offload_config={}, enable_fused_mc2=2, enable_deepgemm_mega_moe=True
        )
    assert config.enable_fused_mc2 == 1
    assert config._megamoe_supported
    find_spec.assert_not_called()


@pytest.mark.parametrize("invalid", ["valid", "eager", "tp", "ep", "fp4", "model", "placement", "lora", "loader"])
def test_config_constraints(invalid):
    from vllm_ascend.ascend_config import AscendConfig

    config = AscendConfig(sparse_kv_offload_config={})
    vc = model_config()
    if invalid == "loader":
        vc.model_config.quantization = "fp8"
    if invalid == "eager":
        vc.model_config.enforce_eager = False
    if invalid == "tp":
        vc.parallel_config.tensor_parallel_size = 2
    if invalid == "ep":
        vc.parallel_config.enable_expert_parallel = False
    if invalid == "fp4":
        vc.model_config.hf_config.quantization_config["expert_dtype"] = "fp8"
    if invalid == "model":
        vc.model_config.architectures = ["OtherModel"]
    if invalid == "placement":
        vc.parallel_config.expert_placement_strategy = "round_robin"
    if invalid == "lora":
        vc.lora_config = object()
    with patch("vllm_ascend.ascend_config.get_current_hardware_profile") as hw:
        hw.return_value.supports.return_value = True
        if invalid == "valid":
            config._validate_deepgemm_mega_moe(vc)
        else:
            with pytest.raises(ValueError):
                config._validate_deepgemm_mega_moe(vc)


def test_config_rejects_per_layer_swiglu_limits():
    from vllm_ascend.ascend_config import AscendConfig

    config = AscendConfig(sparse_kv_offload_config={})
    vc = model_config()
    vc.model_config.hf_text_config.expert_swiglu_limit_list = [0, 10]
    with (
        patch("vllm_ascend.ascend_config.get_current_hardware_profile") as hw,
        pytest.raises(ValueError, match="per-layer SwiGLU limits"),
    ):
        hw.return_value.supports.return_value = True
        config._validate_deepgemm_mega_moe(vc)


def test_config_rejects_rl_weight_reloading():
    from vllm_ascend.ascend_config import AscendConfig

    config = AscendConfig(sparse_kv_offload_config={})
    config.rl_config.enabled = True
    with (
        patch("vllm_ascend.ascend_config.get_current_hardware_profile") as hw,
        pytest.raises(ValueError, match="RL mode"),
    ):
        hw.return_value.supports.return_value = True
        config._validate_deepgemm_mega_moe(model_config())


def test_buffer_allocated_once_and_layer_clamp_used():
    obj, inputs = make_case(0)
    buffer = obj.buffer
    obj.buffer = None
    obj.ops.SymmBuffer.return_value = buffer
    inputs.layer.swiglu_limit = 7.0
    with patch.object(backend, "get_mc2_group", return_value=SimpleNamespace(device_group="test_group")):
        obj.apply(inputs)
        obj.apply(inputs)
    obj.ops.SymmBuffer.assert_called_once_with("test_group", 4, 8, 2, 64, 256, num_shared_experts=0)
    assert obj.ops.fp8_fp4_mega_moe.call_count == 2
    assert obj.ops.fp8_fp4_mega_moe.call_args.kwargs["activation_clamp"] == 7.0


@pytest.mark.parametrize("missing", ["fp8_fp4_mega_moe", "transform_weights_for_mega_moe", "SymmBuffer"])
def test_reject_incomplete_native_api(native_module, missing):
    delattr(native_module, missing)
    with (
        patch.dict("sys.modules", {"deep_gemm": native_module}),
        pytest.raises(RuntimeError, match=f"Missing required APIs.*{missing}"),
    ):
        backend.load_deepgemm_mega_moe_ops()


def test_communicator_rejects_installed_cuda_backend(native_module, make_deepgemm_communicator):
    del native_module.get_npu_arch
    with (
        patch.dict("sys.modules", {"deep_gemm": native_module}),
        pytest.raises(RuntimeError, match="Missing required APIs.*get_npu_arch.*Install DeepGEMM-Ascend"),
    ):
        make_deepgemm_communicator()
    native_module.transform_weights_for_mega_moe.assert_not_called()


def test_rejects_unsupported_activation_before_loading_backend():
    config = SimpleNamespace(activation=MoEActivation.GELU)
    with (
        patch.object(backend, "load_deepgemm_mega_moe_ops") as load_backend,
        pytest.raises(NotImplementedError, match="requires SwiGLU activation"),
    ):
        backend.DeepGemmMegaMoE(config, 8)
    load_backend.assert_not_called()


def test_communicators_and_weight_preparation_share_validated_backend(native_module, make_deepgemm_communicator):
    weights = torch.zeros(2, 2, dtype=torch.uint8)
    with (
        patch.dict("sys.modules", {"deep_gemm": native_module}),
        patch.object(backend, "import_module", wraps=backend.import_module) as import_package,
    ):
        first = make_deepgemm_communicator()
        backend.prepare_weights(weights, weights, weights, weights)
        second = make_deepgemm_communicator()

    assert first.deepgemm_backend.ops is second.deepgemm_backend.ops is native_module
    native_module.transform_weights_for_mega_moe.assert_called_once()
    import_package.assert_called_once_with("deep_gemm")


def test_native_api_accepts_empty_rank_without_patch_arguments(native_module):
    calls = []

    def native_moe(y, l1_weights, l2_weights, sym_buffer, *, activation_clamp=None):
        calls.append(activation_clamp)

    native_module.fp8_fp4_mega_moe = native_moe
    fixture, inputs = make_case(0)
    with patch.dict("sys.modules", {"deep_gemm": native_module}):
        obj = backend.DeepGemmMegaMoE(fixture.config, 8)
    obj.buffer = fixture.buffer
    assert obj.apply(inputs).shape == (0, 64)
    assert calls == [10.0]


def test_default_weight_preparation_delegates_to_parent():
    method = object.__new__(w4a8_mxfp4.AscendW4A8MXFPDSDynamicFusedMoEMethod)
    layer = SimpleNamespace()
    with (
        patch.object(w4a8_mxfp4, "get_ascend_config", return_value=SimpleNamespace(enable_deepgemm_mega_moe=False)),
        patch.object(w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod, "process_weights_after_loading") as prepare,
    ):
        method.process_weights_after_loading(layer)
    prepare.assert_called_once_with(layer)


def test_deepgemm_weight_preparation_releases_original_tensors():
    method = object.__new__(w4a8_mxfp4.AscendW4A8MXFPDSDynamicFusedMoEMethod)
    weights = [torch.zeros(2, 2, dtype=torch.uint8) for _ in range(4)]
    layer = SimpleNamespace(**dict(zip(("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"), weights)))
    converted = (object(), object())
    with (
        patch.object(w4a8_mxfp4, "get_ascend_config", return_value=SimpleNamespace(enable_deepgemm_mega_moe=True)),
        patch.object(backend, "prepare_weights", return_value=converted) as prepare,
        patch.object(w4a8_mxfp4, "dispose_tensor") as dispose,
    ):
        method.process_weights_after_loading(layer)
        prepare.assert_called_once_with(*weights)
        assert (layer.deepgemm_l1, layer.deepgemm_l2) == converted
        assert [call.args[0] is tensor for call, tensor in zip(dispose.call_args_list, weights)] == [True] * 4
        assert dispose.call_count == 4
        with pytest.raises(RuntimeError, match="weight reload"):
            method.process_weights_after_loading(layer)
        assert prepare.call_count == 1
