# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_ascend.ops.kimi_pixel_norm import KimiPixelNormLut, build_kimi_pixel_norm
from vllm_ascend.ops.triton.kimi_pixel_norm import _kimi_pixel_norm_kernel


def _warmup(norm):
    registry = JitWarmupRegistry(None)
    with registry.activate():
        norm.register_warmup()
    registry.warmup()


def _reference():
    values = np.arange(256, dtype=np.float32)
    lut = (values / 255.0 - np.float32(0.5)) * (np.float32(1.0) / np.float32(0.5))
    return torch.from_numpy(lut).reshape(256, 1).expand(-1, 3).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pixel_lut_exhaustive_values_boundaries_and_graph(dtype):
    reference = _reference()
    device = torch.device("npu", torch.npu.current_device())
    norm = KimiPixelNormLut(reference, dtype=dtype, device=device)
    _warmup(norm)
    for patches in (0, 1, 3, 9, 334, 335, 4096, 4097, 33488):
        pixels = (torch.arange(patches * 588) % 256).to(torch.uint8).reshape(patches, 3, 14, 14)
        indices = pixels.long() * 3 + torch.arange(3).reshape(1, 3, 1, 1)
        expected = reference.flatten()[indices].to(device=device).to(dtype)
        inputs = pixels.to(device=device)
        actual = norm(inputs, dtype)
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = norm(inputs, dtype)
    graph.replay()
    assert torch.equal(captured.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    for value in (127, 255):
        inputs.fill_(value)
        graph.replay()
        expected = reference[value, 0].to(dtype).expand_as(captured.cpu())
        assert torch.equal(captured.cpu().view(torch.int16), expected.view(torch.int16))


def test_pixel_lut_rejects_invalid_layout_and_dtype():
    device = torch.device("npu", torch.npu.current_device())
    table = torch.zeros(256, dtype=torch.bfloat16, device=device)
    cores = torch.npu.get_device_properties(device).vector_core_num
    op = torch.ops.vllm.kimi_pixel_norm_lut
    with pytest.raises(ValueError, match="uint8"):
        op(torch.zeros((1, 588), dtype=torch.float32, device=device), table, cores)
    with pytest.raises(ValueError, match="contiguous"):
        op(torch.zeros((1, 3, 14, 14), dtype=torch.uint8, device=device).transpose(2, 3), table, cores)


def test_dynamic_shapes_share_one_torch_compile_graph():
    device = torch.device("npu", torch.npu.current_device())
    reference = _reference()
    norm = KimiPixelNormLut(reference, dtype=torch.bfloat16, device=device)
    _warmup(norm)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(norm, backend=backend, fullgraph=True, dynamic=True)
    for patches in (36, 335, 1152, 4097):
        pixels = torch.zeros((patches, 588), dtype=torch.uint8, device=device)
        actual = compiled(pixels, torch.bfloat16)
        assert actual.shape == pixels.shape
        assert torch.equal(actual.cpu(), norm(pixels, torch.bfloat16).cpu())
    assert len(graphs) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_registry_warmup_covers_runtime_shapes_without_new_compilation(dtype):
    device = torch.device("npu", torch.npu.current_device())
    reference = _reference()
    norm = KimiPixelNormLut(reference, dtype=dtype, device=device)
    _warmup(norm)
    with patch.object(_kimi_pixel_norm_kernel, "compile", side_effect=AssertionError("unexpected runtime compilation")):
        for patches in (1, 3, 36, 64, 334, 335, 1152, 4096, 4097, 33488):
            pixels = torch.zeros((patches, 588), dtype=torch.uint8, device=device)
            actual = norm(pixels, dtype)
            assert torch.equal(actual.cpu(), reference[0, 0].to(dtype).expand_as(actual.cpu()))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = norm(pixels, dtype)
        graph.replay()
        assert torch.equal(captured.cpu(), actual.cpu())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pixel_lut_restores_table_after_dtype_device_moves_and_loading(dtype):
    reference = _reference()
    device = torch.device("npu", torch.npu.current_device())
    norm = KimiPixelNormLut(reference, dtype=dtype, device=device)
    assert not norm.state_dict()
    norm.cpu().half().bfloat16().to(device=device, dtype=dtype)
    norm.lut.zero_()
    _warmup(norm)
    assert torch.equal(norm.lut.cpu().view(torch.int16), reference[:, 0].to(dtype).view(torch.int16))
    pixels = torch.arange(588).remainder(256).to(torch.uint8).reshape(1, 588).to(device)
    expected = reference[:, 0][pixels.cpu().long()].to(dtype)
    actual = norm(pixels, dtype)
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
    with pytest.raises(ValueError, match="BF16 or FP16"):
        norm.float()


@pytest.mark.parametrize(
    "model_type,architecture,dtype",
    [
        ("kimi_k3", "KimiK3ForConditionalGeneration", torch.float16),
        ("kimi_k31", "KimiK31ForConditionalGeneration", torch.bfloat16),
    ],
)
def test_kimi_module_build_registers_final_dtype_warmup(tmp_path, model_type, architecture, dtype):
    media = {
        "patch_size": 14,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "merge_kernel_size": 2,
        "temporal_merge_kernel_size": 4,
    }
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"processor_class": "KimiK3Processor", "media_proc_cfg": media})
    )
    device = torch.device("npu", torch.npu.current_device())
    projection = torch.nn.Conv2d(3, 8, 14, stride=14, bias=False, device=device, dtype=torch.bfloat16)
    tower = SimpleNamespace(patch_embed=SimpleNamespace(proj=projection))
    config = SimpleNamespace(
        multimodal_config=SimpleNamespace(mm_device_do_normalize=True),
        hf_config=SimpleNamespace(model_type=model_type, architectures=[architecture]),
        model=str(tmp_path),
        revision=None,
    )
    registry = JitWarmupRegistry(None)
    with registry.activate():
        norm = build_kimi_pixel_norm(config, tower)
    # Loading may move the module after registration. Expand the final dtype
    # only when the runner's shared startup registry compiles it.
    projection.to(dtype=dtype)
    norm.to(dtype=dtype)
    registry.warmup()
    pixels = torch.arange(588).remainder(256).to(torch.uint8).reshape(1, 588).to(device)
    with patch.object(_kimi_pixel_norm_kernel, "compile", side_effect=AssertionError("unexpected runtime compilation")):
        actual = norm(pixels, projection.weight.dtype)
    expected = _reference()[:, 0][pixels.cpu().long()].to(dtype)
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
