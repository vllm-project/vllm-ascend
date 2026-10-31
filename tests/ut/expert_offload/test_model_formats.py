# SPDX-License-Identifier: Apache-2.0
"""CPU loading/storage contracts for GLM5Next and Qwen4Exp expert formats.

Native post-processing bodies are exercised with CPU layout/quantizer doubles.
This does not replace checkpoint, NPU GEMM or hybrid-attention acceptance.
"""

import ast
import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, LayerOffloadBinding
from vllm_ascend.expert_offload.host_store import HostExpertStore
from vllm_ascend.expert_offload.quantization import (
    BlockFP8ExpertAdapter,
    MXFP8ExpertAdapter,
    W8A8ExpertAdapter,
    expert_adapter_type,
)
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.quantization.quant_type import QuantType

ROOT = Path(__file__).resolve().parents[3]


def cpu_mx_quant(weight, **_kwargs):
    """A deterministic group quantizer double with expert-dependent scales."""
    groups = weight.float().unflatten(-1, (-1, 32))
    exponent = groups.abs().amax(-1).clamp_min(1).log2().ceil()
    quantized = (groups / exponent.exp2().unsqueeze(-1)).flatten(-2).to(torch.float8_e4m3fn)
    scale = (exponent + 127).to(torch.uint8).unflatten(-1, (-1, 2))
    return quantized, scale


def native_scheme(kind):
    """Use actual native post-processing, replacing only NPU layout/quantize ops."""
    directory = ROOT / "vllm_ascend/quantization/methods/w8a8"
    filename, classname = {
        "mxfp8": ("w8a8_mxfp8.py", "AscendW8A8MXFP8DynamicFusedMoEMethod"),
        "int8": ("w8a8_dynamic.py", "AscendW8A8DynamicFusedMoEMethod"),
        "block_fp8": ("fp8_block.py", "AscendFp8BlockFusedMoEMethod"),
    }[kind]
    tree = ast.parse((directory / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    cls.bases = [ast.Name(id="object", ctx=ast.Load())]
    cls.decorator_list = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("process_weights_after_loading", "_requantize_experts_inplace")
    ]
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("resolve_block_scales", "_mx_quantize", "_supports_mx_regroup")
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *helpers, cls],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "QuantType": QuantType,
        "torch_npu": SimpleNamespace(
            npu_format_cast=lambda weight, *_a, **_k: weight,
            npu_dynamic_mx_quant=cpu_mx_quant,
        ),
        "get_current_vllm_config": lambda: None,
        "get_ascend_config": lambda: SimpleNamespace(enable_fused_mc2=0),
        "use_cann_megamoe": lambda _: False,
        "maybe_trans_nz_with_scale": lambda weight, scale, **_k: (
            weight.transpose(1, 2).contiguous(),
            scale.transpose(1, 2).contiguous(),
        ),
        "ACL_FORMAT_FRACTAL_NZ": 29,
        "BLOCK_FP8_WEIGHT_DTYPE": torch.float8_e4m3fn,
        "_ROWS_PER_DEQUANT_STEP": 1024,
        "cdiv": lambda n, d: (n + d - 1) // d,
    }
    exec(compile(ast.fix_missing_locations(module), str(directory / filename), "exec"), namespace)
    scheme = namespace[classname]()
    scheme.group_size = 32
    scheme.quant_type = {"mxfp8": QuantType.W8A8MXFP, "int8": QuantType.W8A8, "block_fp8": QuantType.NONE}[kind]
    scheme.use_expert_weight_list = False
    if kind == "block_fp8":
        scheme.block_n = scheme.block_k = scheme.group_size = 128
        scheme.model_dtype = torch.bfloat16
        scheme.mxfp8_method = native_scheme("mxfp8")
        scheme._mx_scale_alg = 0
    return scheme


def expert_layer(kind):
    scheme = native_scheme(kind)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"
    layer.local_num_experts, layer.top_k, layer.activation = 6, 2, "silu"
    layer.swiglu_limit = 10.0
    layer.moe_config = SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False)
    layer.weight_loader = shard_loader
    shapes = {"w13_weight": (6, 256, 128), "w2_weight": (6, 128, 128)}
    adapter = expert_adapter_type(scheme)
    for name in adapter.tensor_names:
        weight_name = name.split("_scale")[0].split("_offset")[0]
        e, n, k = shapes[weight_name]
        if name.endswith("_weight"):
            shape, dtype = (e, n, k), torch.int8 if kind == "int8" else torch.float8_e4m3fn
        elif kind == "block_fp8":
            shape, dtype = (e, n // scheme.block_n, k // scheme.block_k), torch.float32
        elif kind == "int8":
            shape, dtype = (e, n, 1), torch.bfloat16
        else:
            shape, dtype = (e, n, k // 32), torch.uint8
        layer.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))
    return layer, scheme


def shard_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
    expert = param.data[expert_id]
    if shard_id in ("w1", "w3"):
        half = expert.shape[0] // 2
        expert = expert[:half] if shard_id == "w1" else expert[half:]
    expert.copy_(loaded_weight.reshape_as(expert))
    return True if return_success else None


def load_experts(layer, names):
    for name in names:
        param = getattr(layer, name)
        for eid in range(layer.local_num_experts):
            shards = ("w1", "w3") if name.startswith("w13") else ("w2",)
            for shard in shards:
                shape = list(param.shape[1:])
                if name.startswith("w13"):
                    shape[0] //= 2
                if "scale" in name:
                    value = 120 + eid if param.dtype == torch.uint8 else (eid + 1) * 0.5
                else:
                    value = eid * 0.125 if "offset" in name else eid + 1
                tensor = torch.full(shape, value, dtype=param.dtype)
                # W8A8 per-channel checkpoints may store scale/offset as vectors.
                if tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
                param.weight_loader(param, tensor, name, shard, eid)


@pytest.mark.parametrize("kind", ["mxfp8", "int8", "block_fp8"])
def test_native_preparation_prefill_and_eviction_preserve_every_expert_tensor(kind):
    layer, scheme = expert_layer(kind)
    adapter_type = expert_adapter_type(scheme)
    binding = LayerOffloadBinding(layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2))
    load_experts(layer, adapter_type.tensor_names)
    reference = torch.nn.Module()
    for name in adapter_type.tensor_names:
        reference.register_parameter(
            name, torch.nn.Parameter(getattr(layer, name).detach().clone(), requires_grad=False)
        )
    reference.activation = "silu"
    native_scheme(kind).process_weights_after_loading(reference)
    binding.process_weights_after_loading(layer)
    layer._expert_offload_binding = binding
    manager = ExpertOffloadManager()
    manager.finalize_loading(layer)
    manager.get_comm = lambda *_a: None
    addresses = {name: tensor.data_ptr() for name, tensor in binding.resident.items()}
    for prefill, logical in ((True, [[4, 1]]), (False, [[3, 5]]), (False, [[0, 4]])):
        manager.begin_execution()
        manager.has_prefill = prefill
        ids = torch.tensor(logical, dtype=torch.int32)
        view, physical, _ = binding.prepare(layer, ids)
        assert view.swiglu_limit == 10.0
        for name in adapter_type.output_names:
            actual = getattr(view, name)[physical.long()].float()
            expected = getattr(reference, name)[ids.long()].float()
            torch.testing.assert_close(actual, expected)
        assert addresses == {name: tensor.data_ptr() for name, tensor in binding.resident.items()}
        manager.finish_execution(failed=False)
    if kind == "block_fp8":
        assert not hasattr(layer, "w13_weight_scale_inv")
        assert expert_adapter_type(scheme) is BlockFP8ExpertAdapter
    manager.close()


@pytest.mark.parametrize("kind", ["mxfp8", "int8", "block_fp8"])
def test_new_formats_reject_incomplete_checkpoint_and_invalid_tensor_before_execution(kind):
    layer, scheme = expert_layer(kind)
    adapter = expert_adapter_type(scheme)(layer, scheme)
    adapter.validate_layer(layer)
    last = adapter.tensor_names[-1]
    param = getattr(layer, last)
    setattr(layer, last, torch.nn.Parameter(param[:5], requires_grad=False))
    with pytest.raises(ValueError, match=last):
        adapter.validate_layer(layer)
    layer, scheme = expert_layer(kind)
    binding = LayerOffloadBinding(layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2))
    load_experts(layer, adapter.tensor_names[:-1])
    with pytest.raises(RuntimeError, match="incomplete expert checkpoint"):
        binding.process_weights_after_loading(layer)
    assert not binding.resident


@pytest.mark.parametrize("kind,adapter", [("mxfp8", MXFP8ExpertAdapter), ("int8", W8A8ExpertAdapter)])
def test_format_selected_from_quantization_instead_of_model_name(kind, adapter):
    layer, scheme = expert_layer(kind)
    for model_type in ("glm5_next_text", "qwen4_exp"):
        layer.model_type = model_type
        assert expert_adapter_type(scheme) is adapter
        adapter(layer, scheme).validate_layer(layer)


@pytest.mark.parametrize("kind", ["mxfp8", "int8", "block_fp8"])
def test_prepared_native_bytes_include_derived_scales_and_survive_raw_retirement(monkeypatch, kind):
    layer, scheme = expert_layer(kind)
    adapter = expert_adapter_type(scheme)(layer, scheme)
    store = HostExpertStore(layer, adapter.tensor_names)
    load_experts(layer, adapter.tensor_names)
    store.finalize(layer, pin_memory=False)
    destination = adapter.materialize(store, (0, 1), "cpu")
    expected = adapter.materialize(store, (4, 2), "cpu")
    npu = SimpleNamespace(
        get_npu_format=lambda tensor: 2,
        get_storage_size=lambda tensor: tensor.untyped_storage().nbytes() // tensor.element_size(),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    stream = SimpleNamespace(synchronize=lambda: None, npu_stream=123)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda _: stream), raising=False)
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **k: original_empty(*a, **k))

    def memcpy(dst, capacity, src, size, *_args):
        assert capacity == size
        ctypes.memmove(dst, src, size)
        return 0

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy=memcpy, memcpy_async=memcpy)))
    store.prepare_native(adapter, destination, "cpu")
    assert not store.tensors
    assert set(store.prepared) == set(adapter.output_names)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _: True)
    # Execute the native byte-copy branch without mapping an NPU into this test.
    monkeypatch.setattr(
        "vllm_ascend.expert_offload.transfer.torch",
        SimpleNamespace(device=lambda _: SimpleNamespace(type="npu"), npu=torch.npu, uint8=torch.uint8),
    )
    TorchExpertTransfer.copy_experts(adapter, store, destination, ((4, 0), (2, 1)), "cpu")
    for name in adapter.output_names:
        torch.testing.assert_close(destination[name].float(), expected[name].float())
    pool = adapter.allocate_pool(store, 6, "cpu")
    for name in adapter.output_names:
        assert pool[name].shape[0] == 6
        assert pool[name].stride() == destination[name].stride()
    store.close()
