# SPDX-License-Identifier: Apache-2.0
"""CPU helpers: exercise native method bodies with layout and ACL doubles."""

import ast
import ctypes
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import (
    ExpertOffloadManager,
    LayerOffloadBinding,
    MXFPExpertAdapter,
)
from vllm_ascend.quantization.quant_type import QuantType

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def load_hook_class():
    def load(path, class_name, methods, parent, namespace):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert len(selected) == len(methods)
        for method in selected:
            method.decorator_list = []
        fields = [node for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant)]
        module = ast.parse(f"from __future__ import annotations\nclass {class_name}(Parent):\n    pass\n")
        hook_class = module.body[1]
        assert isinstance(hook_class, ast.ClassDef)
        hook_class.body = [*fields, *selected]
        namespace = {**namespace, "Parent": parent}
        exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
        return namespace[class_name]

    return load


@pytest.fixture
def expert_factory(load_hook_class):
    def make(kind="mxfp4", *, experts=6, layer_name="model.layers.0.mlp.experts"):
        is_fp8 = kind == "mxfp8"
        filename, class_name = (
            ("w8a8/w8a8_mxfp8.py", "AscendW8A8MXFP8DynamicFusedMoEMethod")
            if is_fp8
            else ("w4a8/w4a8_mxfp4.py", "AscendW4A8MXFPDynamicFusedMoEMethod")
        )
        scheme = load_hook_class(
            "vllm_ascend/quantization/methods/" + filename,
            class_name,
            {"process_weights_after_loading"},
            object,
            {
                "torch": torch,
                "torch_npu": SimpleNamespace(
                    npu_format_cast=lambda tensor, *a, **k: tensor, float4_e2m1fn_x2=torch.uint8
                ),
                "MoEActivation": SimpleNamespace(SITU="situ"),
                "get_current_vllm_config": lambda: None,
                "use_cann_megamoe": lambda _: False,
                "ACL_FORMAT_FRACTAL_NZ": 29,
                "maybe_trans_nz_with_scale": lambda weight, scale, **k: (
                    weight.transpose(1, 2).contiguous(),
                    scale.transpose(1, 2).contiguous(),
                ),
            },
        )()
        scheme.group_size = 32
        scheme.quant_type = QuantType.W8A8MXFP if is_fp8 else QuantType.W4A8MXFP
        scheme.use_expert_weight_list = False
        layer = torch.nn.Module()
        layer.layer_name, layer.local_num_experts, layer.top_k, layer.activation = layer_name, experts, 2, "silu"
        layer.moe_config = SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False)
        for name in MXFPExpertAdapter.tensor_names:
            output = 128 if name.startswith("w13") else 64
            width = 2 if name.endswith("_scale") else (64 if is_fp8 else 32)
            dtype = torch.float8_e4m3fn if is_fp8 and name.endswith("_weight") else torch.uint8
            param = torch.nn.Parameter(torch.empty(experts, output, width, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, param)
        layer._expert_offload_binding = LayerOffloadBinding(
            layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2)
        )
        for name, param in layer.named_parameters():
            for expert_id in range(experts):
                value = 120 + expert_id if name.endswith("_scale") else expert_id + 1
                param.data[expert_id].fill_(value)
        return layer, scheme

    return make


@pytest.fixture
def manager_factory():
    def make(*layers):
        for layer in layers:
            layer._expert_offload_binding.process_weights_after_loading(layer)
        manager = ExpertOffloadManager()
        manager.finalize_loading(torch.nn.ModuleList(layers))
        manager.get_comm = Mock(return_value=None)
        return manager

    return make


@pytest.fixture
def native_runtime(monkeypatch):
    calls: list[str | tuple[int, int]] = []
    allocations = []
    stream = Mock(npu_stream=123, synchronize=lambda: calls.append("sync"))
    npu = SimpleNamespace(
        current_stream=lambda _: stream,
        Stream=Mock(return_value=stream),
        synchronize=lambda: calls.append("sync"),
        set_device=lambda _: None,
        stream=lambda _: nullcontext(),
        _launch_host_func=Mock(side_effect=lambda stream, callback, arg: callback(arg)),
        _subscribe_report=Mock(),
        _unsubscribe_report=Mock(),
    )
    runtime = SimpleNamespace(
        get_npu_format=lambda _: 29,
        get_storage_size=lambda tensor: tensor.untyped_storage().nbytes() // tensor.element_size(),
    )
    empty = torch.empty

    def pinned_empty(*args, pin_memory=False, **kwargs):
        if pin_memory:
            allocations.append(args[0])
        return empty(*args, **kwargs)

    def memcpy(dst, capacity, src, size, direction, *args):
        assert capacity == size
        calls.append((direction, size))
        ctypes.memmove(dst, src, size)
        return 0

    monkeypatch.setattr(torch, "empty", pinned_empty)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _: True)
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)
    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy=memcpy, memcpy_async=memcpy)))
    monkeypatch.setattr(
        "vllm_ascend.expert_offload.transfer.torch",
        SimpleNamespace(device=lambda _: SimpleNamespace(type="npu"), npu=npu, uint8=torch.uint8),
    )
    return SimpleNamespace(calls=calls, allocations=allocations, stream=stream, npu=npu)
