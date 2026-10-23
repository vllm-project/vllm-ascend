#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope_vnorm import qkv_rmsnorm_rope_vnorm_fits_ub

# Per-rank (head_dim, num_heads, num_kv_heads) of gemma-4-31B's two attention
# types. At tensor parallel size 4 both fit one vector core's unified buffer; at
# size 2 the full attention layers no longer do.
GEMMA4_TP4_SLIDING = (256, 8, 4)
GEMMA4_TP4_FULL = (512, 8, 1)
GEMMA4_TP2_SLIDING = (256, 16, 8)
GEMMA4_TP2_FULL = (512, 16, 2)
# A head_dim 128 shape that the q/k-only patterns register for, and that also
# fits the unified buffer.
QWEN3_8B_TP1 = (128, 32, 8)

EPSILONS = (1e-6, 1e-5)


@pytest.fixture(scope="module")
def pass_module():
    # base_pattern imports npugraph_ex (or torchair) at module level. Building
    # the pass does not need it, so stand in for it when neither is installed.
    stubbed = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
    if stubbed:
        sys.modules["npugraph_ex"] = MagicMock()
    try:
        return importlib.import_module("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass")
    finally:
        if stubbed:
            del sys.modules["npugraph_ex"]


def make_vllm_config(head_size=128, partial_rotary_factor=1.0):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            use_mla=False,
            get_head_size=lambda: head_size,
            hf_text_config=SimpleNamespace(head_dim=head_size, partial_rotary_factor=partial_rotary_factor),
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=SimpleNamespace(
            splitting_ops=[], use_inductor_graph_partition=False, pass_config=SimpleNamespace()
        ),
    )


@pytest.fixture
def traced_cpu_patterns(monkeypatch, pass_module):
    """Run real example construction and pattern bodies, checking device-op shapes."""
    patterns = []
    empty, ones = torch.empty, torch.ones

    def cpu_allocation(factory):
        def allocate(*args, **kwargs):
            if kwargs.get("device") == "npu":
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return allocate

    def rms_norm(hidden, weight, eps):
        assert hidden.shape[-1] == weight.shape[-1]
        return hidden, None

    def rotary_embedding(positions, query, key, cache, head_size, rotary_dim, is_neox_style):
        assert 0 < rotary_dim <= head_size, "example rotary dimension exceeds its attention head"
        assert cache.shape[-1] == rotary_dim, "example RoPE cache must have the rotary width"
        assert query.shape[-1] % head_size == key.shape[-1] % head_size == 0
        assert positions.shape[0] == query.shape[0] == key.shape[0]
        return query, key

    def trace_examples(pattern, _passes):
        inputs = pattern.get_inputs()
        pattern.get_pattern()(*inputs)
        patterns.append((pattern, inputs))

    monkeypatch.setattr(torch, "empty", cpu_allocation(empty))
    monkeypatch.setattr(torch, "ones", cpu_allocation(ones))
    monkeypatch.setattr(torch.ops.npu, "npu_rms_norm", rms_norm, raising=False)
    monkeypatch.setattr(torch.ops.vllm, "npu_rotary_embedding", rotary_embedding, raising=False)
    for pattern_class in (
        pass_module.QKNormRopeFusionPattern,
        pass_module.QKNormRopeFusionPatternWithBias,
        pass_module.QKVNormRopeFusionPattern,
    ):
        monkeypatch.setattr(pattern_class, "register", trace_examples)
    return patterns


def register_patterns(pass_module, layer_shapes, device_type=AscendDeviceType.A2):
    """Build the pass over attention layers of the given shapes on the given hardware.

    Returns the (head_dim, num_heads, num_kv_heads, eps) of every
    QKVNormRopeFusionPattern and of every q/k-only pattern it registered.
    """
    layers = {
        f"model.layers.{i}.self_attn.attn": SimpleNamespace(head_size=h, num_heads=n, num_kv_heads=kv)
        for i, (h, n, kv) in enumerate(layer_shapes)
    }
    with (
        patch.object(pass_module, "get_layers_from_vllm_config", return_value=layers),
        patch.object(pass_module, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(pass_module, "HAS_TRITON", True),
        patch.object(pass_module, "qkv_rmsnorm_rope_vnorm_fits_ub", qkv_rmsnorm_rope_vnorm_fits_ub, create=True),
        patch.object(pass_module.QKVNormRopeFusionPattern, "register", autospec=True) as qkv_register,
        patch.object(pass_module.QKNormRopeFusionPattern, "register", autospec=True) as qk_register,
        patch.object(pass_module.QKNormRopeFusionPatternWithBias, "register", autospec=True) as qk_bias_register,
    ):
        pass_module.QKNormRopeFusionPass(make_vllm_config())

    def shapes(register):
        return sorted(
            (p.head_dim, p.num_heads, p.num_kv_heads, p.eps) for p in (c.args[0] for c in register.call_args_list)
        )

    return shapes(qkv_register), sorted(shapes(qk_register) + shapes(qk_bias_register))


def test_pattern_key_is_distinct_per_shape(pass_module):
    """Both Gemma4 shapes must register; a shared key would drop the second."""
    vllm_config = make_vllm_config()
    keys = {
        pass_module.QKVNormRopeFusionPattern(
            vllm_config=vllm_config,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=1e-6,
        ).pattern_key()
        for head_dim, num_heads, num_kv_heads in (GEMMA4_TP4_SLIDING, GEMMA4_TP4_FULL)
    }
    assert len(keys) == 2, f"expected one key per shape, got {keys}"


def test_registers_one_pattern_per_attention_shape(pass_module):
    """Layers sharing a shape register it once, per epsilon."""
    layer_shapes = [GEMMA4_TP4_SLIDING, GEMMA4_TP4_SLIDING, GEMMA4_TP4_FULL, GEMMA4_TP4_SLIDING]

    qkv_shapes, qk_shapes = register_patterns(pass_module, layer_shapes)

    assert qkv_shapes == sorted((*shape, eps) for shape in (GEMMA4_TP4_SLIDING, GEMMA4_TP4_FULL) for eps in EPSILONS)
    # The q/k-only patterns stay behind their head_dim == 128 guard, so they
    # cannot claim the q/k half of a Gemma4 layer and leave its v norm unfused.
    assert qk_shapes == []


def test_skips_attention_shape_over_unified_buffer(pass_module):
    """A shape whose single token overflows the unified buffer is not registered."""
    head_dim, num_heads, num_kv_heads = GEMMA4_TP2_FULL
    assert not qkv_rmsnorm_rope_vnorm_fits_ub(
        q_hidden_size=num_heads * head_dim,
        kv_hidden_size=num_kv_heads * head_dim,
        head_dim=head_dim,
        rope_dim=head_dim,
    )

    qkv_shapes, _ = register_patterns(pass_module, [GEMMA4_TP2_SLIDING, GEMMA4_TP2_FULL])

    assert qkv_shapes == sorted((*GEMMA4_TP2_SLIDING, eps) for eps in EPSILONS)


def test_hardware_without_the_kernel_keeps_only_qk_patterns(pass_module):
    """Ascend 950 has no split_qkv_rmsnorm_rope_vnorm kernel, so only the q/k/v pattern is skipped.

    The q/k-only patterns still register there, because DeviceOperator routes
    them to the SIMT kernel on that hardware.
    """
    assert not get_hardware_profile(AscendDeviceType.A5).supports(HardwareCapability.GRAPH_QKV_NORM_ROPE_FUSION)

    qkv_shapes, qk_shapes = register_patterns(pass_module, [QWEN3_8B_TP1], AscendDeviceType.A5)

    assert qkv_shapes == []
    assert qk_shapes == sorted(2 * [(*QWEN3_8B_TP1, eps) for eps in EPSILONS])


def test_skips_qk_examples_when_target_rope_exceeds_discovered_head(pass_module, traced_cpu_patterns):
    # An explicit 256-wide target head can coexist with a 128-wide draft head.
    # Registration traces examples before graph matching can reject the shape.
    target_config = make_vllm_config(head_size=256)
    layers = {"draft.layers.0.attn": SimpleNamespace(head_size=128, num_heads=32, num_kv_heads=8)}
    with (
        patch.object(pass_module, "get_layers_from_vllm_config", return_value=layers),
        patch.object(
            pass_module, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A2)
        ),
        patch.object(pass_module, "HAS_TRITON", True),
    ):
        pass_module.QKNormRopeFusionPass(target_config)

    assert len(traced_cpu_patterns) == len(EPSILONS)
    for pattern, inputs in traced_cpu_patterns:
        assert type(pattern) is pass_module.QKVNormRopeFusionPattern
        assert pattern.head_dim == pattern.rope_dim == inputs[-2].shape[-1] == 128


@pytest.mark.parametrize("pattern_name", ["QKNormRopeFusionPattern", "QKNormRopeFusionPatternWithBias"])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5], ids=["full128", "partial64"])
def test_qk_examples_use_actual_rotary_cache_width(
    pass_module, traced_cpu_patterns, pattern_name, partial_rotary_factor
):
    pattern = getattr(pass_module, pattern_name)(
        make_vllm_config(partial_rotary_factor=partial_rotary_factor),
        head_dim=128,
        num_heads=32,
        num_kv_heads=8,
    )

    pattern.register(None)

    assert len(traced_cpu_patterns) == 1
    traced_pattern, inputs = traced_cpu_patterns[0]
    assert traced_pattern is pattern
    assert pattern.rope_dim == inputs[-2].shape[-1] == int(128 * partial_rotary_factor)
