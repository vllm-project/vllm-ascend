# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the INT2 KV cache quantization method (scheme registration
and class surgery)."""

from typing import Any
from unittest import mock

import pytest
import torch

from vllm_ascend.quantization.methods.kv_int2 import (
    AscendINT2KVCacheAttentionMethod,
    _int2_kv_scale_weight_loader,
)
from vllm_ascend.quantization.methods.registry import get_scheme_class


class _FakeLayer:
    kv_cache_torch_dtype: Any = None
    k_cache_scale: Any = None
    v_cache_scale: Any = None

    def __init__(self, with_impl=False):
        if with_impl:
            self.impl = type("Impl", (), {})()


class TestSchemeRegistration:
    def test_registered_under_int2_per_token_head(self):
        cls = get_scheme_class("INT2_PER_TOKEN_HEAD", "attention")
        assert cls is AscendINT2KVCacheAttentionMethod


class TestCreateWeights:
    def _method(self):
        with mock.patch("vllm_ascend.quantization.methods.kv_int2.get_current_vllm_config") as cfg:
            cfg.return_value = type("C", (), {"kv_transfer_config": None})()
            m = AscendINT2KVCacheAttentionMethod({"type": "int2"}, "prefix")
        return m

    def test_sets_packed_dtype_and_scales(self):
        m = self._method()
        layer = _FakeLayer()
        m.create_weights(layer)
        assert layer.kv_cache_torch_dtype == torch.uint8
        assert layer.k_cache_scale.numel() == 1
        assert layer.v_cache_scale.numel() == 1
        assert not layer.k_cache_scale.requires_grad

    def test_class_surgery_rebinds_impl(self):
        m = self._method()
        layer = _FakeLayer(with_impl=True)
        fake_backend = type("FakeINT2Backend", (), {})()
        import sys
        import types

        mod = types.ModuleType("vllm_ascend.attention.attention_int2")
        # dynamic module attribute; mypy cannot see the patched-in class
        mod.AscendINT2AttentionBackendImpl = fake_backend.__class__  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"vllm_ascend.attention.attention_int2": mod}):
            m.create_weights(layer)
        assert layer.impl.__class__ is fake_backend.__class__

    def test_apply_fails_closed(self):
        m = self._method()
        layer = _FakeLayer()
        m.create_weights(layer)
        with pytest.raises(RuntimeError, match="should not be called"):
            m.apply(layer, None, None, None, None, None, None, None, None)


class TestScaleWeightLoader:
    def test_scalar_fill(self):
        p = torch.nn.Parameter(torch.ones(1))
        _int2_kv_scale_weight_loader(p, torch.tensor([2.0]))
        assert p.data.item() == 2.0

    def test_tp_shard_per_head(self):
        # loaded [total_kv_heads, head_size] -> shard for this rank
        with (
            mock.patch("vllm_ascend.quantization.methods.kv_int2.get_tensor_model_parallel_rank", return_value=1),
            mock.patch("vllm_ascend.quantization.methods.kv_int2.get_tensor_model_parallel_world_size", return_value=2),
        ):
            p = torch.nn.Parameter(torch.zeros(2, 4))
            loaded = torch.arange(16, dtype=torch.float32).reshape(4, 4) * 1.0
            _int2_kv_scale_weight_loader(p, loaded)
            # rank 1 of 4 heads -> rows 2:4
            assert torch.equal(p.data, loaded[2:4])


class TestModelSlimInt2ScaleMapping:
    """PR-21 wiring: get_cache_scale must map INT2 scales (symmetric, no
    offsets) and keep the C8 mapping intact."""

    # main-port note: the per-name get_cache_scale helper became the
    # instance-level get_cache_scale_mapper(); test through its suffix map.
    def _suffixes(self, kv_type):
        from vllm_ascend.quantization.configs import modelslim_config as msc

        cfg = object.__new__(msc.AscendModelSlimConfig)
        cfg.quant_description = {"kv_cache_type": kv_type}
        cfg.enable_int2_quant = kv_type == "INT2"
        cfg.enable_c8_quant = kv_type == "C8"
        cfg.enable_fa_quant = False
        cfg.enable_indexer_quant = False
        # enable_mxfp_c8_quant / _has_mxfp_c8_recipe are read-only properties
        # that query the engine config and return False without one
        from vllm.model_executor.models.utils import WeightsMapper

        mapper = cfg.get_cache_scale_mapper()
        assert isinstance(mapper, WeightsMapper)
        return mapper.orig_to_new_suffix

    def test_int2_scales_map_and_offsets_are_none(self):
        m = self._suffixes("INT2")
        assert m[".k_proj.kv_cache_scale"] == ".attn.k_cache_scale"
        assert m[".v_proj.kv_cache_scale"] == ".attn.v_cache_scale"
        assert ".k_proj.kv_cache_offset" not in m and ".v_proj.kv_cache_offset" not in m

    def test_c8_mapping_unchanged(self):
        m = self._suffixes("C8")
        assert m[".k_proj.kv_cache_scale"] == ".attn.k_cache_scale"
        assert m[".k_proj.kv_cache_offset"] == ".attn.k_cache_offset"
        assert m[".v_proj.kv_cache_scale"] == ".attn.v_cache_scale"
        assert m[".v_proj.kv_cache_offset"] == ".attn.v_cache_offset"

    def test_other_types_or_names_return_none(self):
        m = self._suffixes("")
        assert ".k_proj.kv_cache_scale" not in m


class TestModelSlimInt2Selection:
    """PR-21 wiring: is_int2_quant_layer gate + get_quant_method selection."""

    def test_int2_layer_predicate(self):
        from types import SimpleNamespace

        from vllm_ascend.quantization.configs import modelslim_config as msc

        on = SimpleNamespace(enable_int2_quant=True, int2_quant_layers=[3])
        assert msc.AscendModelSlimConfig.is_int2_quant_layer(on, "model.layers.3.self_attn.q_proj")
        assert not msc.AscendModelSlimConfig.is_int2_quant_layer(on, "model.layers.4.self_attn.q_proj")
        off = SimpleNamespace(enable_int2_quant=False, int2_quant_layers=[3])
        assert not msc.AscendModelSlimConfig.is_int2_quant_layer(off, "model.layers.3.self_attn.q_proj")

    def test_get_quant_method_selects_int2_for_int2_layer(self):
        from types import SimpleNamespace

        from vllm_ascend.quantization.configs import modelslim_config as msc
        from vllm_ascend.quantization.methods.kv_int2 import (
            AscendINT2KVCacheAttentionMethod,
        )

        class _FakeBase(torch.nn.Module):
            pass

        ns = SimpleNamespace(
            quant_description={"kv_cache_type": "INT2"},
            packed_modules_mapping={},
            is_fa_quant_layer=lambda p: False,
            is_indexer_quant_layer=lambda p: False,
            is_c8_quant_layer=lambda p: False,
            is_int2_quant_layer=lambda p: True,
            quant_prefix_mapper=lambda mt, p: p,
            _update_packed_modules_mapping=lambda mt: None,
        )
        fake_vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="ut_fake_model"))
        )
        with (
            mock.patch.object(msc, "get_current_vllm_config", return_value=fake_vllm_cfg),
            mock.patch.object(msc, "get_quant_type_for_layer", return_value=None),
            mock.patch.object(msc, "LinearBase", type("_LB", (), {})),
            mock.patch.object(msc, "AttentionLayerBase", _FakeBase),
            mock.patch("vllm_ascend.quantization.method_adapters.AscendKVCacheMethod") as fake_kv,
            mock.patch(
                "vllm_ascend.quantization.methods.kv_int2.get_current_vllm_config",
                return_value=SimpleNamespace(kv_transfer_config=None),
            ),
        ):
            msc.AscendModelSlimConfig.get_quant_method(ns, _FakeBase(), "model.layers.3.self_attn")
        fake_kv.assert_called_once()
        assert isinstance(fake_kv.call_args.args[0], AscendINT2KVCacheAttentionMethod)
