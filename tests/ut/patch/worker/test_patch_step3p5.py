# SPDX-License-Identifier: Apache-2.0
"""Exercise Step3.5 patch loading without initializing vLLM or an NPU."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def step3p5_patch(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class Module:
        pass

    class Tensor:
        pass

    class GateLinear:
        def __init__(self, input_size, output_size, *, out_dtype, params_dtype, prefix):
            self.input_size = input_size
            self.output_size = output_size
            self.out_dtype = out_dtype
            self.params_dtype = params_dtype
            self.prefix = prefix

    torch = stub("torch", Tensor=Tensor, float32=object(), zeros=Mock())
    torch.nn = stub("torch.nn", Module=Module, Parameter=Mock())
    stub("vllm.config", VllmConfig=object)
    stub(
        "vllm.distributed",
        get_dp_group=Mock(),
        get_ep_group=lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1)),
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tp_group=Mock(),
        tensor_model_parallel_all_gather=Mock(),
        tensor_model_parallel_reduce_scatter=Mock(),
    )
    stub("vllm.logger", logger=Mock())
    factory = Mock()
    stub("vllm.model_executor.layers.fused_moe", FusedMoEFactory=factory, GateLinear=GateLinear)
    stub("vllm.model_executor.layers.layernorm", GemmaRMSNorm=Mock())
    # Current upstream exposes GateLinear, but no FP32ReplicatedLinear.
    upstream = stub(
        "vllm.model_executor.models.step3p5",
        **{
            name: type(name, (Module,), {"forward": Mock()})
            for name in ("FusedMoEBlock", "Step3p5Attention", "Step3p5DecoderLayer", "Step3p5Model")
        },
        Step3p5MLP=Mock(),
    )
    stub("vllm.model_executor.models.utils", extract_layer_index=lambda _: 0, sequence_parallel_chunk=Mock())
    stub("vllm_ascend.device.device_op", DeviceOperator=Mock())
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_step3p5.py"
    spec = importlib.util.spec_from_file_location("_step3p5_patch_under_test", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    return upstream, torch, factory


@pytest.mark.parametrize("is_sequence_parallel", [False, True])
def test_import_and_fp32_router_initialization(step3p5_patch, is_sequence_parallel):
    upstream, torch, factory = step3p5_patch
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                moe_num_experts=4,
                use_moe_router_bias=True,
                moe_router_scaling_factor=1.0,
                need_fp32_gate=True,
                swiglu_limits=[],
                share_expert_dim=32,
                moe_top_k=2,
                moe_intermediate_size=32,
                norm_expert_weight=True,
            )
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(enable_eplb=False, eplb_config=SimpleNamespace(num_redundant_experts=0)),
    )
    block = upstream.FusedMoEBlock(config, prefix="model.layers.0.moe", is_sequence_parallel=is_sequence_parallel)

    assert block.gate.input_size == 16
    assert block.gate.output_size == 4
    assert block.gate.out_dtype is torch.float32
    assert block.gate.params_dtype is torch.float32
    assert block.gate.prefix == "model.layers.0.moe.gate"
    assert factory.call_args.kwargs["gate"] is block.gate
    assert factory.call_args.kwargs["router_logits_dtype"] is torch.float32
    assert factory.call_args.kwargs["is_sequence_parallel"] is is_sequence_parallel
