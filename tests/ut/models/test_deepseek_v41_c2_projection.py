# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real FP32 projections and the model checkpoint-loading method."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend import envs
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor

PREFIX = "model.layers.0.self_attn.compressor"
WIDTH = 4
HIDDEN = 8


def _compressor(monkeypatch, ratio=2, fused=True, device="cpu"):
    monkeypatch.setenv("VLLM_ASCEND_FUSE_V41_C2_PROJ", "1" if fused else "0")
    config = SimpleNamespace(head_dim=WIDTH, hidden_size=HIDDEN, rms_norm_eps=1e-6)
    with torch.device(device):
        return DeepseekV41Compressor(config, ratio)


def _model(monkeypatch, compressor, alias=False):
    # Avoid constructing the full LLM or distributed process groups. Keep the
    # production loader, real parameter registration, and default weight loader.
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_module, "fused_moe_make_expert_params_mapping", lambda *a, **kw: [])
    monkeypatch.setattr(model_module, "is_pp_missing_parameter", lambda *a: False)
    model = nn.Module()
    model.config = SimpleNamespace(n_routed_experts=0, num_attention_heads=1)
    model.num_redundant_experts = 0
    model.model = nn.Module()
    if alias:
        model.model.canonical = nn.Module()
        model.model.canonical.compressor = compressor
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.compressor = compressor
    model.model.layers = nn.ModuleList([layer])
    return model


def _load(model, weights):
    return model_module.AscendDeepseekV41LLMForCausalLM._load_model_weights(model, weights)


def _weights(reverse=False, dtype=torch.float32):
    kv = torch.arange(WIDTH * HIDDEN, dtype=dtype).reshape(WIDTH, HIDDEN)
    gate = -kv - 1
    weights = [(f"{PREFIX}.wkv.weight", kv), (f"{PREFIX}.wgate.weight", gate)]
    return list(reversed(weights)) if reverse else weights


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_opt_in_preserves_c1(monkeypatch, ratio, fused):
    compressor = _compressor(monkeypatch, ratio, fused)
    enabled = ratio == 2 and fused
    assert compressor._fuse_c2_proj == enabled
    assert hasattr(compressor, "wkv_gate") == enabled
    assert hasattr(compressor, "wkv") != enabled
    if ratio == 1:
        assert compressor.wkv.weight.dtype == torch.bfloat16
        assert not hasattr(compressor, "wgate")


def test_default_is_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_FUSE_V41_C2_PROJ", raising=False)
    assert not envs.VLLM_ASCEND_FUSE_V41_C2_PROJ
    config = SimpleNamespace(head_dim=WIDTH, hidden_size=HIDDEN, rms_norm_eps=1e-6)
    assert not DeepseekV41Compressor(config, 2)._fuse_c2_proj


@pytest.mark.parametrize("tokens", [0, 1, 3, 32])
def test_projection_values_and_contiguous_layout(monkeypatch, tokens):
    unfused = _compressor(monkeypatch, fused=False)
    fused = _compressor(monkeypatch)
    with torch.no_grad():
        fused.wkv_gate.weight.copy_(torch.cat((unfused.wkv.weight, unfused.wgate.weight)))
    # Non-contiguous input also exercises the linear input contract.
    x = torch.randn(HIDDEN, tokens).t()
    expected = unfused.project_c2(x)
    actual = fused.project_c2(x)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
        assert result.shape == (tokens, WIDTH)
        assert result.dtype == torch.float32
        assert result.is_contiguous()
        if tokens > 1:
            assert result.stride() == (WIDTH, 1)
    monkeypatch.setenv("VLLM_ASCEND_FUSE_V41_C2_PROJ", "0")
    assert fused._fuse_c2_proj  # Construction fixes the choice for capture.


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_fused_checkpoint_order_and_registered_name(monkeypatch, reverse, alias):
    compressor = _compressor(monkeypatch)
    model = _model(monkeypatch, compressor, alias)
    param = compressor.wkv_gate.weight
    weights = _weights(reverse)
    loaded = _load(model, weights)
    canonical = "model.canonical.compressor" if alias else PREFIX
    assert loaded == {f"{canonical}.wkv_gate.weight"}
    assert compressor.wkv_gate.weight is param
    expected = dict(_weights())
    torch.testing.assert_close(param[:WIDTH], expected[f"{PREFIX}.wkv.weight"])
    torch.testing.assert_close(param[WIDTH:], expected[f"{PREFIX}.wgate.weight"])


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("alias", [False, True])
def test_c1_and_unfused_c2_checkpoint_loading(monkeypatch, ratio, alias):
    compressor = _compressor(monkeypatch, ratio=ratio, fused=False)
    model = _model(monkeypatch, compressor, alias)
    weights = _weights()[:ratio]
    loaded = _load(model, weights)
    canonical = "model.canonical.compressor" if alias else PREFIX
    assert loaded == {f"{canonical}.{name.rsplit('.', 2)[1]}.weight" for name, _ in weights}
    for name, value in weights:
        param = getattr(compressor, name.rsplit(".", 2)[1]).weight
        torch.testing.assert_close(param, value.to(param.dtype))


@pytest.mark.parametrize("reverse", [False, True])
def test_meta_rejected_then_explicitly_materialized(monkeypatch, reverse):
    compressor = _compressor(monkeypatch, device="meta")
    model = _model(monkeypatch, compressor)
    with pytest.raises(RuntimeError, match="Materialize the model on the configured load device"):
        _load(model, _weights(reverse))
    assert compressor.wkv_gate.weight.is_meta
    # Placement belongs to the caller. Allocate before the loader snapshots
    # named_parameters(); never replace param.data across device types.
    model.to_empty(device="cpu")
    param = compressor.wkv_gate.weight
    assert _load(model, _weights(reverse)) == {f"{PREFIX}.wkv_gate.weight"}
    assert compressor.wkv_gate.weight is param
    expected = dict(_weights())
    torch.testing.assert_close(param[:WIDTH], expected[f"{PREFIX}.wkv.weight"])
    torch.testing.assert_close(param[WIDTH:], expected[f"{PREFIX}.wgate.weight"])


@pytest.mark.parametrize("shape", [(WIDTH + 1, HIDDEN), (WIDTH, HIDDEN + 1), (WIDTH * HIDDEN,)])
def test_wrong_checkpoint_shape_fails(monkeypatch, shape):
    compressor = _compressor(monkeypatch)
    model = _model(monkeypatch, compressor)
    with pytest.raises(ValueError, match="expected"):
        _load(model, [(f"{PREFIX}.wkv.weight", torch.empty(shape))])


@pytest.mark.parametrize("shard", [0, 1])
def test_missing_shard_fails(monkeypatch, shard):
    model = _model(monkeypatch, _compressor(monkeypatch))
    with pytest.raises(ValueError, match="Incomplete C2 projection weights"):
        _load(model, [_weights()[shard]])


def test_pp_missing_layer_is_skipped(monkeypatch):
    model = _model(monkeypatch, _compressor(monkeypatch))
    monkeypatch.setattr(model_module, "is_pp_missing_parameter", lambda *a: True)
    assert _load(model, _weights()) == set()


def test_fused_forward_requires_c2_pipeline(monkeypatch):
    with pytest.raises(RuntimeError, match="project_c2"):
        _compressor(monkeypatch)(torch.randn(2, HIDDEN))
