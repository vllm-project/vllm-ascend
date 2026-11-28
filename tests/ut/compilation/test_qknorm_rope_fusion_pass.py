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
import inspect
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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
QKV_PATTERNS = ("QKNormRopeFusionPattern", "QKNormRopeFusionPatternWithBias", "QKVNormRopeFusionPattern")


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


def make_vllm_config():
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.compilation_config.splitting_ops = []
    vllm_config.compilation_config.use_inductor_graph_partition = False
    return vllm_config


def register_patterns(pass_module, layer_shapes, device_type=AscendDeviceType.A2, rope_dim=128):
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
        patch.object(pass_module, "get_rope_dim", return_value=rope_dim),
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


@pytest.mark.parametrize("pattern_name", QKV_PATTERNS)
def test_target_and_draft_layouts_have_distinct_global_pattern_keys(pass_module, pattern_name):
    cls = getattr(pass_module, pattern_name)
    config = make_vllm_config()
    with patch.object(pass_module, "get_rope_dim", return_value=128):
        keys = {cls(config, 128, heads, kv_heads).pattern_key() for heads, kv_heads in [(32, 8), (16, 8), (16, 16)]}
        config.model_config.dtype = torch.float16
        keys.add(cls(config, 128, 32, 8).pattern_key())
    assert len(keys) == 4


@pytest.mark.parametrize("pattern_name", QKV_PATTERNS)
def test_backend_registers_both_target_and_draft_patterns(pass_module, monkeypatch, pattern_name):
    base = importlib.import_module("vllm_ascend.compilation.passes.base_pattern")
    monkeypatch.setattr(base, "_registered_patterns", set())
    monkeypatch.setattr(base, "_registered_by_matcher", weakref.WeakKeyDictionary())
    native_register, backend_register = MagicMock(), MagicMock()
    monkeypatch.setattr(base.pm, "register_replacement", native_register)
    monkeypatch.setattr(base.nge, "register_replacement", backend_register)
    matcher = pass_module.PatternMatcherPass()
    cls = getattr(pass_module, pattern_name)
    with patch.object(pass_module, "get_rope_dim", return_value=128):
        for heads in (32, 16, 32):
            pattern = cls(make_vllm_config(), 128, heads, 8)
            monkeypatch.setattr(pattern, "get_inputs", lambda: [])
            pattern.register(matcher)
    assert native_register.call_count == 2
    assert backend_register.call_count == 2


@pytest.mark.parametrize("pattern_name", QKV_PATTERNS)
def test_foreign_qkv_width_is_a_trace_specialization_miss(pass_module, pattern_name):
    cls = getattr(pass_module, pattern_name)
    with patch.object(pass_module, "get_rope_dim", return_value=128):
        pattern = cls(make_vllm_config(), 128, 32, 8).get_pattern()
    with FakeTensorMode():
        draft_qkv = torch.empty(2, 4096, dtype=torch.bfloat16)
        unused_inputs = [None] * (len(inspect.signature(pattern).parameters) - 1)
        with pytest.raises(RuntimeError):
            pattern(draft_qkv, *unused_inputs)


@pytest.mark.parametrize("pattern_name", QKV_PATTERNS)
def test_npu_split_value_error_becomes_a_trace_miss(pass_module, pattern_name):
    cls = getattr(pass_module, pattern_name)
    with patch.object(pass_module, "get_rope_dim", return_value=128):
        pattern = cls(make_vllm_config(), 128, 32, 8).get_pattern()
    qkv = torch.empty(2, 4096, dtype=torch.bfloat16)
    unused_inputs = [None] * (len(inspect.signature(pattern).parameters) - 1)
    error = ValueError("NPU split sizes do not match the tensor width")
    with patch.object(torch.Tensor, "split", side_effect=error), pytest.raises(RuntimeError) as caught:
        pattern(qkv, *unused_inputs)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("pattern_name", QKV_PATTERNS)
@pytest.mark.parametrize("heads", [16, 32])
def test_matching_qkv_layout_keeps_its_split(pass_module, pattern_name, heads):
    cls = getattr(pass_module, pattern_name)
    with patch.object(pass_module, "get_rope_dim", return_value=128):
        pattern = cls(make_vllm_config(), 128, heads, 8)
    with FakeTensorMode():
        q, k, v = pattern._split_qkv(torch.empty(2, (heads + 16) * 128, dtype=torch.bfloat16))
        assert q.shape == (2, heads * 128)
        assert k.shape == v.shape == (2, 1024)


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
