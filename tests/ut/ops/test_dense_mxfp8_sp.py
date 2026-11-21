# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for dense MXFP8 communication fusion.

The collective doubles execute two-rank tensor mathematics and decode E8M0
scales; they do not establish NPU kernel accuracy or performance. This leaf
module can also be tested without a vLLM installation using pytest --noconftest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
GROUP_SIZE = 32


def quantize_reference(x):
    """A CPU MXFP8 quantizer with independently varying power-of-two scales."""
    blocks = x.float().reshape(x.shape[0], -1, GROUP_SIZE)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(2.0**-20) / 448)).clamp(-126, 127)
    factors = torch.pow(2.0, exponent)
    quantized = (blocks / factors.unsqueeze(-1)).reshape_as(x).to(torch.float8_e4m3fn)
    return quantized, (exponent + 127).to(torch.uint8)


def unpack_activation(q, scale):
    factors = torch.pow(2.0, scale.reshape(q.shape[0], -1).float() - 127)
    return q.float() * factors.repeat_interleave(GROUP_SIZE, dim=1)[:, : q.shape[1]]


def quantize_weight(weight):
    q, scale = quantize_reference(weight.t().contiguous())
    return q.t().contiguous(), scale.reshape(weight.shape[1], -1, 2).transpose(0, 1).contiguous()


def reference_mm(q, weight, activation_scale, weight_scale):
    x = unpack_activation(q, activation_scale)
    w = unpack_activation(weight.t(), weight_scale.transpose(0, 1)).t()
    return x @ w


class TwoRankKernels:
    """Known peer inputs/sums emulate rank-ordered collectives on CPU."""

    def __init__(self):
        self.rank = 0
        self.peer_input = None
        self.peer_partial = None
        self.events = []
        self.npu = ModuleType("torch_npu")
        self.npu.float8_e8m0fnu = 293
        self.npu.get_npu_format = Mock(return_value=2)
        self.npu.npu_dynamic_mx_quant = Mock(side_effect=self.quantize)
        self.npu.npu_quant_matmul = Mock(side_effect=self.matmul)
        self.npu.npu_all_gather_quant_mm = Mock(side_effect=self.all_gather_mm)
        self.npu.npu_quant_mm_reduce_scatter = Mock(side_effect=self.mm_reduce_scatter)
        self.gather = Mock(side_effect=self.all_gather)
        self.scatter = Mock(side_effect=self.reduce_scatter)

    def quantize(self, x, *, dst_type, scale_alg=0):
        assert dst_type == torch.float8_e4m3fn
        self.events.append("quantize")
        return quantize_reference(x)

    def matmul(self, q, weight, scale=None, *, pertoken_scale=None, output_dtype=None, **kwargs):
        self.events.append("matmul")
        assert kwargs["group_sizes"] == [1, 1, GROUP_SIZE]
        assert kwargs["scale_dtype"] == self.npu.float8_e8m0fnu
        assert kwargs["pertoken_scale_dtype"] == self.npu.float8_e8m0fnu
        return reference_mm(q, weight, pertoken_scale, scale).to(output_dtype)

    def _rank_order(self, local, peer):
        return [local, peer] if self.rank == 0 else [peer, local]

    def all_gather(self, local):
        self.events.append("all_gather")
        assert local.dtype == torch.bfloat16
        return torch.cat(self._rank_order(local, self.peer_input), dim=0)

    def reduce_scatter(self, partial):
        self.events.append("reduce_scatter")
        summed = partial.float() + self.peer_partial.float()
        return summed.chunk(2, dim=0)[self.rank].to(partial.dtype)

    def _check_fused(self, q, weight, hcom, world_size, kwargs):
        assert hcom == "test-hccl"
        assert world_size == 2
        assert q.dtype == weight.dtype == torch.float8_e4m3fn
        assert q.is_contiguous() and weight.is_contiguous()
        assert kwargs["group_sizes"] == [1, 1, GROUP_SIZE]
        assert kwargs["y_dtype"] == torch.bfloat16
        assert kwargs["x1_scale_dtype"] == kwargs["x2_scale_dtype"] == self.npu.float8_e8m0fnu
        assert kwargs["x1_scale"].shape == (q.shape[0], q.shape[1] // 64, 2)
        assert kwargs["x2_scale"].shape == (weight.shape[0] // 64, weight.shape[1], 2)
        assert kwargs["x1_scale"].dtype == kwargs["x2_scale"].dtype == torch.uint8
        assert kwargs["comm_mode"] == "ai_cpu"

    def all_gather_mm(self, q, weight, hcom, world_size, **kwargs):
        self.events.append("all_gather_mm")
        self._check_fused(q, weight, hcom, world_size, kwargs)
        assert kwargs["gather_output"] is False
        peer_q, peer_scale = quantize_reference(self.peer_input)
        # Concatenating float8 is unsupported on some CPU PyTorch versions.
        gathered = torch.cat([value.float() for value in self._rank_order(q, peer_q)]).to(q.dtype)
        gathered_scale = torch.cat(self._rank_order(kwargs["x1_scale"], peer_scale.reshape_as(kwargs["x1_scale"])))
        result = reference_mm(gathered, weight, gathered_scale, kwargs["x2_scale"])
        return result.to(kwargs["y_dtype"]), torch.empty(0), torch.empty(0)

    def mm_reduce_scatter(self, q, weight, hcom, world_size, **kwargs):
        self.events.append("mm_reduce_scatter")
        self._check_fused(q, weight, hcom, world_size, kwargs)
        local = reference_mm(q, weight, kwargs["x1_scale"], kwargs["x2_scale"])
        summed = local.to(torch.bfloat16).float() + self.peer_partial.float()
        result = summed.chunk(world_size, dim=0)[self.rank].to(kwargs["y_dtype"])
        return result, torch.empty(0)


@pytest.fixture
def runtime():
    """Load the production leaf with temporary device/import boundaries only."""
    kernels = TwoRankKernels()
    registrations = {}
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: "test-hccl")
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        device_group=SimpleNamespace(_get_backend=lambda device: backend),
    )
    modules = {}
    dependencies = {
        "torch_npu": kernels.npu,
        "vllm.distributed.parallel_state": SimpleNamespace(get_tp_group=lambda: group),
        "vllm_ascend.models.common.ops.sequence_parallel": SimpleNamespace(
            sp_all_gather=kernels.gather, sp_reduce_scatter=kernels.scatter
        ),
        "vllm.utils.torch_utils": SimpleNamespace(
            direct_register_custom_op=lambda **kwargs: registrations.update({kwargs["op_name"]: kwargs})
        ),
    }
    for name, dependency in dependencies.items():
        names = name.split(".")
        for depth in range(1, len(names)):
            parent = ".".join(names[:depth])
            if parent not in sys.modules and parent not in modules:
                modules[parent] = ModuleType(parent)
                modules[parent].__path__ = []
        modules[name] = dependency
    name = "_dense_mxfp8_sp_test_module"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_ascend/ops/dense_mxfp8_sp.py")
    module = importlib.util.module_from_spec(spec)
    modules[name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        yield SimpleNamespace(module=module, kernels=kernels, registrations=registrations, group=group)


def make_values(rows=256, k=256, n=256):
    generator = torch.Generator().manual_seed(317)
    x = torch.randn(rows, k, generator=generator)
    x *= torch.pow(2.0, (torch.arange(k) // GROUP_SIZE % 5 - 2).float())
    # The two ranks must have different scale distributions.
    x[rows // 2 :] *= 8
    weight = torch.randn(k, n, generator=generator) / k**0.5
    qweight, scale = quantize_weight(weight)
    return x.to(torch.bfloat16), qweight, scale


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fused", [False, True])
def test_all_gather_mm_matches_rank_order_and_group_scales(runtime, rank, fused):
    x, weight, scale = make_values(rows=256 if fused else 10)
    local, peer = (x.chunk(2)[rank], x.chunk(2)[1 - rank])
    runtime.kernels.rank = rank
    runtime.kernels.peer_input = peer
    q, activation_scale = quantize_reference(x)
    expected = reference_mm(q, weight, activation_scale, scale).to(x.dtype)

    output = runtime.module.dense_mxfp8_all_gather_mm(local, weight, scale, 0, 2)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert runtime.kernels.events == (["quantize", "all_gather_mm"] if fused else ["all_gather", "quantize", "matmul"])
    assert output.shape == (x.shape[0], weight.shape[1])


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fused", [False, True])
def test_reduce_scatter_sums_distinct_weight_partials(runtime, rank, fused):
    x, weight, scale = make_values(rows=256 if fused else 10)
    q, activation_scale = quantize_reference(x)
    partial = reference_mm(q, weight, activation_scale, scale).to(x.dtype)
    peer_weight, peer_scale = quantize_weight(weight.float().roll(1, 1) * 0.375)
    peer_partial = reference_mm(q, peer_weight, activation_scale, peer_scale).to(x.dtype)
    runtime.kernels.rank = rank
    runtime.kernels.peer_partial = peer_partial
    expected = (partial.float() + peer_partial.float()).chunk(2)[rank].to(x.dtype)

    output = runtime.module.dense_mxfp8_mm_reduce_scatter(x, weight, scale, 1, 2)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert runtime.kernels.events == (
        ["quantize", "mm_reduce_scatter"] if fused else ["quantize", "matmul", "reduce_scatter"]
    )
    assert runtime.kernels.npu.npu_dynamic_mx_quant.call_args.kwargs["scale_alg"] == 1


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter"])
def test_missing_kernel_uses_unfused_collectives(runtime, operation):
    x, weight, scale = make_values()
    runtime.kernels.peer_input = x[128:]
    runtime.kernels.peer_partial = torch.zeros(256, 256, dtype=x.dtype)
    if operation == "all_gather":
        del runtime.kernels.npu.npu_all_gather_quant_mm
        output = runtime.module.dense_mxfp8_all_gather_mm(x[:128], weight, scale, 0, 2)
        assert output.shape == (256, 256)
        runtime.kernels.gather.assert_called_once()
    else:
        del runtime.kernels.npu.npu_quant_mm_reduce_scatter
        output = runtime.module.dense_mxfp8_mm_reduce_scatter(x, weight, scale, 0, 2)
        assert output.shape == (128, 256)
        runtime.kernels.scatter.assert_called_once()
    runtime.kernels.npu.npu_quant_matmul.assert_called_once()
    assert not runtime.module.dense_mxfp8_sp_available()


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter"])
def test_collective_failure_is_propagated_without_second_collective(runtime, operation):
    x, weight, scale = make_values()
    failure = RuntimeError("simulated collective failure")
    if operation == "all_gather":
        runtime.kernels.npu.npu_all_gather_quant_mm.side_effect = failure
        function, value = runtime.module.dense_mxfp8_all_gather_mm, x[:128]
    else:
        runtime.kernels.npu.npu_quant_mm_reduce_scatter.side_effect = failure
        function, value = runtime.module.dense_mxfp8_mm_reduce_scatter, x
    with pytest.raises(RuntimeError, match="simulated collective failure"):
        function(value, weight, scale, 0, 2)
    runtime.kernels.gather.assert_not_called()
    runtime.kernels.scatter.assert_not_called()
    runtime.kernels.npu.npu_quant_matmul.assert_not_called()


@pytest.mark.parametrize("bad_input", ["rank", "k", "weight_scale"])
@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter"])
def test_bad_contract_fails_before_quantization_or_collective(runtime, bad_input, operation):
    x, weight, scale = make_values()
    if bad_input == "rank":
        x = x.unsqueeze(0)
    elif bad_input == "k":
        x = x[:, :-64]
    else:
        scale = scale[:, :, :1]
    function = getattr(
        runtime.module, f"dense_mxfp8_{'all_gather_mm' if operation == 'all_gather' else 'mm_reduce_scatter'}"
    )
    with pytest.raises(ValueError):
        function(x, weight, scale, 0, 2)
    assert runtime.kernels.events == []
    runtime.kernels.gather.assert_not_called()
    runtime.kernels.scatter.assert_not_called()


def test_mx_scale_normalization_preserves_e8m0_bytes(runtime):
    scale = torch.tensor([[0, 127, 128, 254], [1, 126, 129, 255]], dtype=torch.uint8)
    packed = runtime.module.normalize_mx_scale(scale, 2, 128)
    assert packed.shape == (2, 2, 2)
    assert packed.dtype == torch.uint8
    assert packed.data_ptr() == scale.data_ptr()
    assert torch.equal(packed.reshape_as(scale), scale)
    assert runtime.module.normalize_mx_scale(packed, 2, 128).data_ptr() == packed.data_ptr()
    with pytest.raises(ValueError):
        runtime.module.normalize_mx_scale(scale[:, :-1], 2, 128)


def test_custom_op_registration_and_fake_shape_are_collective_free(runtime):
    assert set(runtime.registrations) == {"dense_mxfp8_all_gather_mm", "dense_mxfp8_mm_reduce_scatter"}
    weight = torch.empty(256, 512, dtype=torch.float8_e4m3fn, device="meta")
    scale = torch.empty(4, 512, 2, dtype=torch.uint8, device="meta")
    for name, rows, expected_rows in [
        ("dense_mxfp8_all_gather_mm", 7, 14),
        ("dense_mxfp8_mm_reduce_scatter", 7, 4),
    ]:
        registration = runtime.registrations[name]
        assert registration["dispatch_key"] == "PrivateUse1"
        assert registration["mutates_args"] == []
        x = torch.empty(rows, 256, dtype=torch.bfloat16, device="meta")
        output = registration["fake_impl"](x, weight, scale, 0, 2)
        assert output.shape == (expected_rows, 512)
        assert output.dtype == torch.bfloat16
        assert output.device.type == "meta"
    assert runtime.kernels.events == []
