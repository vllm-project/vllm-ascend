# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import deepgemm_mega_moe as backend
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4


def make_case(tokens=3):
    obj = object.__new__(backend.DeepGemmMegaMoE)
    obj.max_tokens = 8
    obj.config = SimpleNamespace(num_experts=4, experts_per_token=2, swiglu_limit=10.0)
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
def test_stage_inputs_and_collective(tokens):
    obj, inputs = make_case(tokens)
    scales = torch.tensor([127, 126], dtype=torch.uint8).expand(tokens, 2).contiguous()
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
    assert torch.equal(obj.buffer.x_sf[:tokens], scales.view(torch.int16))


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


def test_packed_weight_and_scale_bits_preserved():
    weights = torch.tensor([0, 127, 128, 255], dtype=torch.uint8).repeat(2, 2, 1)
    scales = torch.tensor([122, 128], dtype=torch.uint8).repeat(2, 2, 1)
    module = MagicMock()
    with patch.dict("sys.modules", {"deep_gemm": module}):
        backend.prepare_weights(weights, scales, weights, scales)
    args = module.transform_weights_for_mega_moe.call_args.args
    for weight, scale in args:
        assert weight.dtype == torch.int8 and scale.dtype == torch.int16
        assert torch.equal(weight.view(torch.uint8), weights)
        assert torch.equal(scale.view(torch.uint8), scales)


def test_teardown_refused_for_deepgemm():
    comm = object.__new__(FusedMC2CommImpl)
    comm.deepgemm_backend = object()
    with pytest.raises(RuntimeError, match="HCCL teardown"):
        comm.prepare_hccl_teardown()


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
def test_reject_incomplete_native_api(missing):
    module = SimpleNamespace(
        fp8_fp4_mega_moe=lambda *args: None,
        transform_weights_for_mega_moe=lambda *args: None,
        SymmBuffer=lambda *args: None,
    )
    delattr(module, missing)
    with (
        patch.dict("sys.modules", {"deep_gemm": module}),
        pytest.raises(RuntimeError, match="native MegaMoE API"),
    ):
        backend.DeepGemmMegaMoE(SimpleNamespace(), 8)


def test_native_api_accepts_empty_rank_without_patch_arguments():
    calls = []

    def native_moe(y, l1_weights, l2_weights, sym_buffer, *, activation_clamp=None):
        calls.append(activation_clamp)

    module = SimpleNamespace(
        fp8_fp4_mega_moe=native_moe,
        transform_weights_for_mega_moe=lambda *args: None,
        SymmBuffer=lambda *args: None,
    )
    fixture, inputs = make_case(0)
    with patch.dict("sys.modules", {"deep_gemm": module}):
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
