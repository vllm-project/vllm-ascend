# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A5 integration: standard CLI configuration, real kernels and graph replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm import SamplingParams
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8_mxfp8 import AscendMXFP8OnlineLinearMethod
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.mark.parametrize("quantization", [None, "ascend"])
@pytest.mark.parametrize("enforce_eager", [True, False])
@wait_until_npu_memory_free()
def test_standard_online_mxfp8_model_generation(quantization, enforce_eager):
    if quantization == "ascend" and get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("Online MXFP8 requires A5")
    with VllmRunner(
        "Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        quantization=quantization,
        hf_overrides={
            "quantization_config_dict_json": {
                "online_quantization": True,
                "model_quant_type": "W8A8_MXFP8",
                "group_size": 32,
                "ignore": ["lm_head"],
            }
        }
        if quantization
        else {},
        enforce_eager=enforce_eager,
        max_model_len=128,
        cudagraph_capture_sizes=[1, 2],
    ) as runner:
        results = runner.model.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
        assert results[0].outputs[0].token_ids


def _assert_mxfp8_numerics(actual, reference, *, dtype, factor):
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    difference = actual - reference
    reference_rms = reference.square().mean().sqrt()
    reference_peak = reference.abs().max()
    relative_rmse = (difference.square().mean().sqrt() / reference_rms.clamp_min(torch.finfo(torch.float32).eps)).item()
    max_abs_error = difference.abs().max()
    normalized_max_error = (max_abs_error / reference_peak.clamp_min(torch.finfo(torch.float32).eps)).item()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item()
    metrics = (
        f"dtype={dtype}, factor={factor}, relative_rmse={relative_rmse:.6f}, "
        f"max_abs_error={max_abs_error.item():.6f}, "
        f"normalized_max_error={normalized_max_error:.6f}, cosine={cosine:.6f}"
    )
    print(f"MXFP8 numerical metrics: {metrics}")
    assert torch.isfinite(actual).all(), metrics
    assert relative_rmse <= 0.10, metrics
    assert normalized_max_error <= 0.25, metrics
    assert cosine >= 0.99, metrics


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_online_mxfp8_native_reload_replays_original_graph(dtype):
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("Online MXFP8 requires A5")
    torch.manual_seed(42)
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))
    with set_current_vllm_config(vllm_config), torch.device("npu:0"):
        layer = torch.nn.Module()
        scheme = AscendMXFP8OnlineLinearMethod()
        adapter = AscendLinearMethod(scheme)
        layer.quant_method = adapter
        adapter.create_weights(layer, 96, [64], 96, 64, dtype, weight_loader=default_weight_loader)
        record_metadata_for_reloading(layer)
        layer.weight.data.normal_(std=0.05)
        adapter.process_weights_after_loading(layer)
        x = torch.randn(8, 96, dtype=dtype)
        for _ in range(3):
            scheme.apply(layer, x)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = scheme.apply(layer, x)
        original = {
            name: (id(param), param.data_ptr(), tuple(param.stride())) for name, param in layer.named_parameters()
        }
        for factor in (0.03, 0.06):
            initialize_layerwise_reload(layer)
            weights = torch.randn(64, 96, dtype=dtype) * factor
            layer.weight.weight_loader(layer.weight, weights)
            finalize_layerwise_reload(layer, SimpleNamespace(dtype=dtype))
            torch.npu.synchronize()
            current = {
                name: (id(param), param.data_ptr(), tuple(param.stride())) for name, param in layer.named_parameters()
            }
            assert current == original
            graph.replay()
            torch.npu.synchronize()
            eager = scheme.apply(layer, x)
            torch.testing.assert_close(output, eager)
            reference = torch.nn.functional.linear(x, weights)
            _assert_mxfp8_numerics(output, reference, dtype=dtype, factor=factor)
