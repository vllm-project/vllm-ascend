# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import pytest
import torch
import torch_npu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, get_gemma_rms_norm_weight


def make_norm(width, dtype):
    with set_current_vllm_config(VllmConfig()):
        norm = AscendGemmaRMSNorm(width).to(device="npu", dtype=dtype)
    norm.weight.weight_loader(norm.weight, torch.randn_like(norm.weight))
    return norm


def reference(norm, x, residual):
    if residual is None:
        return DeviceOperator.npu_gemma_rms_norm(x, norm.weight, norm.variance_epsilon)
    y, _, res = torch_npu.npu_add_rms_norm(x, residual, 1.0 + norm.weight, norm.variance_epsilon)
    return y, res


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [256, 2048])
@pytest.mark.parametrize("rows", [1, 4, 16, 128, 1024])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_gemma_norm_cached_weight(dtype, width, rows, with_residual):
    norm = make_norm(width, dtype)
    x = torch.randn(rows, width, device="npu", dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None
    expected = reference(norm, x.clone(), residual.clone() if with_residual else None)
    actual = norm.forward_oot(x, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(get_gemma_rms_norm_weight(norm), 1.0 + norm.weight, rtol=0, atol=0)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("layerwise", [False, True])
@torch.inference_mode()
def test_gemma_norm_graph_replay_after_reload(with_residual, layerwise):
    norm = make_norm(2048, torch.bfloat16)
    if layerwise:
        with torch.device(norm.weight.device):
            record_metadata_for_reloading(norm)
    x = torch.randn(16, 2048, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    for _ in range(3):
        norm.forward_oot(x, residual)
    torch.npu.synchronize()
    address = norm._effective_weight.data_ptr()
    weight_address = norm.weight.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = norm.forward_oot(x, residual)
    for _ in range(2):
        loaded = torch.randn_like(norm.weight)
        if layerwise:
            initialize_layerwise_reload(norm)
        norm.weight.weight_loader(norm.weight, loaded)
        if layerwise:
            finalize_layerwise_reload(norm, model_config=None)
        assert norm.weight.data_ptr() == weight_address
        assert norm._effective_weight.data_ptr() == address
        torch.testing.assert_close(get_gemma_rms_norm_weight(norm), 1.0 + loaded, rtol=0, atol=0)
        graph.replay()
        torch.npu.synchronize()
        expected = reference(norm, x.clone(), residual.clone() if with_residual else None)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_gemma_norm_repeated_residual_forward_has_no_weight_add():
    norm = make_norm(2048, torch.bfloat16)
    x = torch.randn(16, 2048, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm.forward_oot(x, residual)
    torch.npu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as baseline:
        reference(norm, x, residual)
        torch.npu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as cached:
        norm.forward_oot(x, residual)
        torch.npu.synchronize()
    assert any(event.key == "aten::add" for event in baseline.key_averages())
    assert not any(event.key == "aten::add" for event in cached.key_averages())
