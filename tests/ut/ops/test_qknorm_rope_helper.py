from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_ascend.ops.helper import qknorm_rope_fusion as fusion


class Rotary:
    def __init__(self, head_dim):
        self.head_size = head_dim
        self.rotary_dim = head_dim
        self.is_neox_style = True
        self.use_mtp = False
        self.cos_sin_cache = torch.empty(16, head_dim, dtype=torch.bfloat16)


class Norm:
    def __init__(self, head_dim):
        self.weight = torch.ones(head_dim, dtype=torch.bfloat16)
        self.bias = None
        self.bias_loaded = False
        self.variance_epsilon = 1e-6
        self.has_weight = True
        self.pass_weight = True
        self.variance_size_override = None


def _inputs(head_dim=128):
    q_size = 2 * head_dim
    kv_size = head_dim
    qkv = torch.empty(3, q_size + 2 * kv_size, dtype=torch.bfloat16)
    positions = torch.arange(3)
    return qkv, positions, Norm(head_dim), Norm(head_dim), Rotary(head_dim), q_size, kv_size


def _enable_fusion(monkeypatch):
    monkeypatch.setattr(fusion, "AscendRotaryEmbedding", Rotary)
    monkeypatch.setattr(fusion, "RMSNorm", Norm)
    enabled = SimpleNamespace(fuse_qknorm_rope=True)
    monkeypatch.setattr(fusion, "get_ascend_config", lambda: SimpleNamespace(ascend_compilation_config=enabled))
    return enabled


def test_qwen3_qk_fusion_dispatch(monkeypatch):
    _enable_fusion(monkeypatch)
    fused = (object(), object(), object())
    kernel = Mock(return_value=fused)
    monkeypatch.setattr(fusion.DeviceOperator, "split_qkv_rmsnorm_rope", kernel)
    qkv, positions, q_norm, k_norm, rope, q_size, kv_size = _inputs()
    q_norm.bias = torch.zeros(128, dtype=torch.bfloat16)
    k_norm.bias = torch.zeros(128, dtype=torch.bfloat16)

    assert fusion.maybe_fused_qknorm_rope(qkv, positions, q_norm, k_norm, rope, q_size, kv_size, 128) is fused
    assert kernel.call_args.kwargs["q_bias"] is None
    assert kernel.call_args.kwargs["k_bias"] is None


def test_qwen3_qk_fusion_passes_loaded_norm_biases(monkeypatch):
    _enable_fusion(monkeypatch)
    kernel = Mock(return_value=(object(), object(), object()))
    monkeypatch.setattr(fusion.DeviceOperator, "split_qkv_rmsnorm_rope", kernel)
    qkv, positions, q_norm, k_norm, rope, q_size, kv_size = _inputs()
    q_norm.bias = torch.full((128,), 0.25, dtype=torch.bfloat16)
    k_norm.bias = torch.full((128,), -0.5, dtype=torch.bfloat16)
    q_norm.bias_loaded = k_norm.bias_loaded = True

    assert (
        fusion.maybe_fused_qknorm_rope(qkv, positions, q_norm, k_norm, rope, q_size, kv_size, 128)
        is kernel.return_value
    )
    assert kernel.call_args.kwargs["q_bias"] is q_norm.bias
    assert kernel.call_args.kwargs["k_bias"] is k_norm.bias


def test_qwen3_qk_fusion_falls_back_for_single_loaded_bias(monkeypatch):
    _enable_fusion(monkeypatch)
    kernel = Mock()
    monkeypatch.setattr(fusion.DeviceOperator, "split_qkv_rmsnorm_rope", kernel)
    qkv, positions, q_norm, k_norm, rope, q_size, kv_size = _inputs()

    for biased_norm in (q_norm, k_norm):
        biased_norm.bias = torch.full((128,), 0.25, dtype=torch.bfloat16)
        biased_norm.bias_loaded = True
        assert fusion.maybe_fused_qknorm_rope(qkv, positions, q_norm, k_norm, rope, q_size, kv_size, 128) is None
        biased_norm.bias_loaded = False
    kernel.assert_not_called()


def test_qwen3_qk_fusion_falls_back_for_incompatible_bias(monkeypatch):
    _enable_fusion(monkeypatch)
    kernel = Mock()
    monkeypatch.setattr(fusion.DeviceOperator, "split_qkv_rmsnorm_rope", kernel)
    qkv, positions, q_norm, k_norm, rope, q_size, kv_size = _inputs()
    q_norm.bias = torch.ones(128, dtype=torch.bfloat16)
    k_norm.bias = torch.ones(128, dtype=torch.float32)
    q_norm.bias_loaded = k_norm.bias_loaded = True

    assert fusion.maybe_fused_qknorm_rope(qkv, positions, q_norm, k_norm, rope, q_size, kv_size, 128) is None
    kernel.assert_not_called()


def test_disabled_and_unsupported_inputs_fall_back(monkeypatch):
    enabled = _enable_fusion(monkeypatch)
    kernel = Mock()
    monkeypatch.setattr(fusion.DeviceOperator, "split_qkv_rmsnorm_rope", kernel)
    qkv, positions, q_norm, k_norm, rope, q_size, kv_size = _inputs()

    enabled.fuse_qknorm_rope = False
    assert fusion.maybe_fused_qknorm_rope(qkv, positions, q_norm, k_norm, rope, q_size, kv_size, 128) is None
    enabled.fuse_qknorm_rope = True
    rope.is_neox_style = False
    assert fusion.maybe_fused_qknorm_rope(qkv, positions, q_norm, k_norm, rope, q_size, kv_size, 128) is None
    rope.is_neox_style = True
    assert fusion.maybe_fused_qknorm_rope(qkv.float(), positions, q_norm, k_norm, rope, q_size, kv_size, 128) is None
    kernel.assert_not_called()
