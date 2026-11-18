# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""CPU checks for the integrated BF16 DSpark path, without importing vLLM."""

import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_hf_parser_sentinel_does_not_require_checkpoint_metadata(adapter_modules):
    sentinel = SimpleNamespace(model_type="dummy_minimax_m3_vl")
    assert adapter_modules.weights.draft_hf_override(sentinel) is sentinel


class _StubMiniMaxTextConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_model_module(monkeypatch):
    class DSparkLoader(torch.nn.Module):
        def load_weights(self, weights):
            self.received = dict(weights)
            return set(self.received)

    modules = {
        "vllm.transformers_utils.configs.minimax_m3": SimpleNamespace(MiniMaxM3TextConfig=_StubMiniMaxTextConfig),
        "vllm.model_executor.layers.layernorm": SimpleNamespace(GemmaRMSNorm=object, RMSNorm=object),
        "vllm.model_executor.models.qwen3_dflash": SimpleNamespace(
            DFlashQwen3Attention=object,
            DFlashQwen3DecoderLayer=torch.nn.Module,
            DFlashQwen3ForCausalLM=object,
            _resolve_layer_attention=Mock(),
        ),
        "vllm.model_executor.models.qwen3_dspark": SimpleNamespace(Qwen3DSparkModel=torch.nn.Module),
        "vllm_ascend.models.minimax_m3.minimax_m3": SimpleNamespace(MiniMaxM3MLP=object),
        "vllm_ascend.models.qwen3_dspark": SimpleNamespace(AscendQwen3DSparkForCausalLM=DSparkLoader),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    model_path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/minimax_m3/minimax_m3_dspark.py"
    spec = importlib.util.spec_from_file_location("minimax_dspark_model_under_test", model_path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    model_module = importlib.util.module_from_spec(spec)
    loader.exec_module(model_module)
    return model_module


@pytest.fixture
def adapter_modules(monkeypatch):
    model_module = _load_model_module(monkeypatch)
    yield SimpleNamespace(weights=model_module, model=model_module)
    sys.modules.pop("minimax_dspark_model_under_test", None)


@pytest.fixture
def checkpoint():
    config = {
        "hidden_size": 4,
        "head_dim": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 5,
        "vocab_size": 6,
        "dspark_ffn_hidden_size": 8,
        "dspark_markov_rank": 2,
        "dspark_target_layer_ids": [1, 3],
        "dspark_block_size": 4,
        "dspark_markov_head_type": "low_rank",
        "dspark_noise_token_id": 5,
        "dspark_hybrid_context_fusion": False,
        "rope_theta": 10000,
        "rotary_dim": 2,
    }
    shapes = {
        "fc.weight": (4, 8),
        "hidden_norm.weight": (4,),
        "final_norm.weight": (4,),
        "markov_head.markov_w1.weight": (6, 2),
        "markov_head.markov_w2.weight": (6, 2),
        "confidence_head.proj.weight": (1, 6),
        "confidence_head.proj.bias": (1,),
    }
    for layer in range(5):
        prefix = f"layers.{layer}.decoder_layer."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (4,),
                prefix + "post_attention_layernorm.weight": (4,),
                prefix + "self_attn.q_norm.weight": (2,),
                prefix + "self_attn.k_norm.weight": (2,),
                prefix + "self_attn.q_proj.weight": (4, 4),
                prefix + "self_attn.k_proj.weight": (2, 4),
                prefix + "self_attn.v_proj.weight": (2, 4),
                prefix + "self_attn.o_proj.weight": (4, 4),
                prefix + "mlp.gate_proj.weight": (8, 4),
                prefix + "mlp.up_proj.weight": (8, 4),
                prefix + "mlp.down_proj.weight": (4, 8),
            }
        )
    tensors = {
        "language_model.model.dspark." + name: torch.arange(math.prod(shape), dtype=torch.bfloat16).reshape(shape)
        for name, shape in shapes.items()
    }
    return config, tensors


def _header(tensors):
    return {name: {"shape": list(value.shape), "dtype": "BF16"} for name, value in tensors.items()}


def test_native_bf16_loading_preserves_tensor_identity_and_values(adapter_modules, checkpoint):
    config, tensors = checkpoint
    snapshots = {name: tensor.clone() for name, tensor in tensors.items()}
    loaded = dict(adapter_modules.weights.iter_draft_weights(iter(tensors.items()), config))
    assert len(loaded) == 62
    assert loaded["norm.weight"] is tensors["language_model.model.dspark.final_norm.weight"]
    assert (
        loaded["layers.0.self_attn.q_proj.weight"]
        is tensors["language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight"]
    )
    for name, tensor in tensors.items():
        normalized = adapter_modules.weights.map_weight_name(name)
        assert loaded[normalized] is tensor
        assert loaded[normalized].data_ptr() == tensor.data_ptr()
        torch.testing.assert_close(tensor, snapshots[name], rtol=0, atol=0)


@pytest.fixture
def adapter_model_module(adapter_modules):
    return adapter_modules.model


def test_model_loader_passes_bf16_weights_to_the_existing_dspark_loader(
    adapter_model_module, checkpoint, adapter_modules
):
    model_module = adapter_model_module
    model = model_module.MiniMaxFlashDSparkForCausalLM.__new__(model_module.MiniMaxFlashDSparkForCausalLM)
    torch.nn.Module.__init__(model)
    config, tensors = checkpoint
    model.config = SimpleNamespace(to_dict=lambda: config)
    loaded = model.load_weights(iter(tensors.items()))
    assert len(loaded) == 62
    for name, tensor in tensors.items():
        assert model.received[adapter_modules.weights.map_weight_name(name)] is tensor


def test_mxfp8_context_kv_uses_native_quantized_projection(adapter_model_module, monkeypatch):
    module = adapter_model_module
    model = module.MiniMaxFlashDSparkModel.__new__(module.MiniMaxFlashDSparkModel)
    torch.nn.Module.__init__(model)
    model.quant_config = object()
    model.hidden_norm = torch.nn.Identity()
    context = torch.ones(2, 4)
    projected = torch.arange(16).reshape(2, 8).float()
    update = Mock()
    attn = SimpleNamespace(
        qkv_proj=Mock(return_value=(projected, None)),
        q_size=4,
        kv_size=2,
        num_kv_heads=1,
        head_dim=2,
        k_norm=lambda x: x + 1,
        rotary_emb=lambda pos, keys, _: (keys + 10, None),
        attn=SimpleNamespace(impl=SimpleNamespace(do_kv_cache_update=update), kv_cache=object()),
    )
    model.layers = [SimpleNamespace(self_attn=attn)]
    monkeypatch.setattr(module.F, "linear", Mock(side_effect=AssertionError("Must not dequantize/fuse MXFP8 KV")))
    model._build_fused_kv_buffers()
    model.precompute_and_store_context_kv(context, torch.arange(2), torch.arange(2))
    attn.qkv_proj.assert_called_once_with(context)
    torch.testing.assert_close(update.call_args.args[1], (projected[:, 4:6] + 11).reshape(2, 1, 2))
    torch.testing.assert_close(update.call_args.args[2], projected[:, 6:].reshape(2, 1, 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bf16_draft_is_rejected_without_conversion(adapter_modules, checkpoint, dtype):
    config, tensors = checkpoint
    name = "language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight"
    tensors[name] = tensors[name].to(dtype)
    with pytest.raises(ValueError, match="Expected native BF16 DSpark tensor"):
        list(adapter_modules.weights.iter_draft_weights(tensors.items(), config))


def test_legacy_scale_tensors_are_rejected(adapter_modules, checkpoint):
    config, tensors = checkpoint
    name = "language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight_scale_inv"
    tensors[name] = torch.ones((4, 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Tensor schema mismatch"):
        list(adapter_modules.weights.iter_draft_weights(tensors.items(), config))


def _mxfp8_header(header):
    result = {name: dict(info) for name, info in header.items()}
    for name, info in list(result.items()):
        if ".layers." in name and len(info["shape"]) == 2:
            info["dtype"] = "F8_E4M3"
            rows, columns = info["shape"]
            result[name.removesuffix("weight") + "weight_scale_inv"] = {
                "dtype": "U8",
                "shape": [rows, math.ceil(columns / 32)],
            }
    return result


def test_mxfp8_loading_preserves_packed_weights_and_scale_bytes(adapter_modules, checkpoint):
    config, tensors = checkpoint
    for name, tensor in list(tensors.items()):
        if ".layers." in name and tensor.ndim == 2:
            tensors[name] = tensor.to(torch.float8_e4m3fn)
            tensors[name.removesuffix("weight") + "weight_scale_inv"] = torch.full(
                (tensor.shape[0], math.ceil(tensor.shape[1] / 32)), 127, dtype=torch.uint8
            )
    loaded = dict(adapter_modules.weights.iter_draft_weights(tensors.items(), config))
    assert len(loaded) == 97
    for name, tensor in tensors.items():
        normalized = adapter_modules.weights.map_weight_name(name).replace("weight_scale_inv", "weight_scale")
        assert loaded[normalized] is tensor
        assert loaded[normalized].dtype == tensor.dtype


@pytest.mark.parametrize("fault", ["missing", "duplicate", "mapped_duplicate", "shape", "prefix"])
def test_bf16_schema_still_rejects_invalid_checkpoints(adapter_modules, checkpoint, fault):
    config, tensors = checkpoint
    entries = list(tensors.items())
    if fault == "missing":
        entries.pop()
    elif fault == "duplicate":
        entries.append(entries[0])
    elif fault == "mapped_duplicate":
        entries.append(
            ("language_model.model.dspark.norm.weight", tensors["language_model.model.dspark.final_norm.weight"])
        )
    elif fault == "shape":
        entries[0] = (entries[0][0], torch.zeros(1, dtype=torch.bfloat16))
    else:
        entries[0] = ("other_model.fc.weight", entries[0][1])
    with pytest.raises(ValueError):
        list(adapter_modules.weights.iter_draft_weights(iter(entries), config))


def test_native_architecture_is_registered_by_the_standard_plugin(adapter_modules, monkeypatch):
    registry = SimpleNamespace(register_model=Mock())
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/__init__.py"
    spec = importlib.util.spec_from_file_location("ascend_model_registration", path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    registration = importlib.util.module_from_spec(spec)
    loader.exec_module(registration)
    registration.register_model()
    registry.register_model.assert_any_call(
        "DSparkMiniMaxDraftModel", "vllm_ascend.models.minimax_m3.minimax_m3_dspark:MiniMaxFlashDSparkForCausalLM"
    )
