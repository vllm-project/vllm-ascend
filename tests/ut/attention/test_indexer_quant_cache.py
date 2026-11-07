# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import indexer as module
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile


@pytest.mark.parametrize("family", list(AscendDeviceType))
def test_quant_cache_hardware_contract(family):
    assert get_hardware_profile(family).supports(HardwareCapability.LI_QUANT_CACHE) == (family == AscendDeviceType.A3)


@pytest.mark.parametrize(
    "enabled,family,pcp,subclass,cache_dtype,expected",
    [
        (True, AscendDeviceType.A3, 1, False, torch.int8, True),
        (False, AscendDeviceType.A3, 1, False, torch.int8, False),
        (True, AscendDeviceType.A2, 1, False, torch.int8, False),
        (True, AscendDeviceType.A5, 1, False, torch.int8, False),
        (True, AscendDeviceType.A3, 2, False, torch.int8, False),
        (True, AscendDeviceType.A3, 1, True, torch.int8, False),
        (True, AscendDeviceType.A3, 1, False, torch.float8_e4m3fn, False),
    ],
)
def test_constructor_scopes_opt_in_to_supported_backend(
    monkeypatch, enabled, family, pcp, subclass, cache_dtype, expected
):
    config = SimpleNamespace(
        enable_li_quant_cache_fusion=enabled,
        is_sparse_li_c8_layer=lambda prefix: True,
        is_sparse_li_c4_layer=lambda prefix: False,
    )
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(indexer_kv_dtype="int8"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="glm_moe_dsa")),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
    )
    monkeypatch.setattr(module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(module, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(module, "is_pcp_decode_sharding_enabled", lambda _: False)
    monkeypatch.setattr(module, "kv_cache_dtype_str_to_dtype", lambda *a: cache_dtype)
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    model = SimpleNamespace(
        n_head=1,
        head_dim=128,
        topk_tokens=4,
        q_lora_rank=128,
        wq_b=torch.nn.Identity(),
        wk_weights_proj=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        softmax_scale=1.0,
        k_cache=SimpleNamespace(prefix="layer.indexer.k_cache"),
    )

    class CustomCacheBackend(module.AscendSFAIndexerBackend):
        pass

    cls = CustomCacheBackend if subclass else module.AscendSFAIndexerBackend
    backend = cls(model, qk_rope_head_dim=64)
    assert backend._use_fused_quant_cache is expected


def make_pipeline(monkeypatch, tokens=6, enabled=True, decode=True, fused_store=True):
    values = torch.arange(tokens * 128).reshape(tokens, 128).to(torch.bfloat16)
    slots = torch.arange(tokens, dtype=torch.int32) + 5
    events = []
    key_cache = torch.full((2, 128, 1, 128), -7, dtype=torch.int8)
    scale_cache = torch.full((2, 128, 1, 1), -7, dtype=torch.float16)
    quant = torch.full((tokens, 128), 11, dtype=torch.int8)
    scale = torch.full((tokens,), 0.25, dtype=torch.float32)
    native = Mock(side_effect=lambda *a, **kw: (quant, scale))
    monkeypatch.setattr(module.torch_npu, "npu_dynamic_quant", native)
    monkeypatch.setattr(module, "HAS_TRITON", True)
    monkeypatch.setattr(module, "rope_forward_triton_siso", lambda x, *a, **kw: x)

    def gather(key, scales, metadata):
        events.append("gather")
        return key, scales, metadata.slot_mapping

    def store(key, scales, mapping, cache, scales_cache):
        events.append("store")
        cache.view(-1, 128)[mapping.long()] = key
        scales_cache.view(-1, 1)[mapping.long()] = scales.reshape(-1, 1).half()

    writer = Mock(side_effect=store)
    monkeypatch.setattr(module, "store_indexer_key_scale", writer)
    backend = SimpleNamespace(
        _use_fused_cache_store=fused_store,
        _use_fused_quant_cache=enabled,
        head_dim=128,
        qk_rope_head_dim=64,
        is_rope_neox_style=False,
        enable_sparse_li_quant=True,
        enable_sparse_li_c8=True,
        enable_sparse_li_c4=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
        wk_weights_proj=lambda x: (torch.cat((x, torch.ones(tokens, 8)), dim=-1).to(x.dtype), None),
        k_norm=lambda x: x,
        q_hadamard=torch.eye(128, dtype=torch.bfloat16),
        k_cache=SimpleNamespace(kv_cache=(key_cache, scale_cache)),
        _gather_cache_inputs=gather,
        write_cache=Mock(),
    )
    backend._quantize_li_tensor = lambda *a, **kw: module.AscendSFAIndexerBackend._quantize_li_tensor(backend, *a, **kw)
    backend.forward_k = lambda *a, **kw: module.AscendSFAIndexerBackend.forward_k(backend, *a, **kw)
    metadata = SimpleNamespace(
        cos=torch.empty(tokens, 64),
        sin=torch.empty(tokens, 64),
        num_actual_tokens=tokens,
        num_decode_tokens=tokens if decode else 0,
        slot_mapping=slots,
    )
    return backend, metadata, values, quant, scale, native, writer, events


def run_pipeline(backend, metadata, values):
    return module.AscendSFAIndexerBackend.forward(backend, values, values, values, metadata, compute_topk=False)


def test_six_row_decode_defers_quant_and_publishes_after_gather(monkeypatch):
    backend, metadata, values, quant, scale, native, writer, events = make_pipeline(monkeypatch)

    def fused(raw, slots, cache, scale_cache):
        events.append("fused")
        torch.testing.assert_close(raw, values, rtol=0, atol=0)
        cache.view(-1, 128)[slots.long()] = quant
        scale_cache.view(-1, 1)[slots.long()] = scale[:, None].half()
        return quant, scale[:, None].half()

    op = Mock(side_effect=fused)
    monkeypatch.setattr(module, "try_li_quant_cache", op)
    assert run_pipeline(backend, metadata, values) is None
    native.assert_not_called()
    writer.assert_not_called()
    backend.write_cache.assert_not_called()
    op.assert_called_once()
    assert events == ["gather", "fused"]
    torch.testing.assert_close(backend.k_cache.kv_cache[0].view(-1, 128)[5:11], quant, rtol=0, atol=0)


def test_unsupported_metadata_or_old_build_restores_native_path(monkeypatch):
    backend, metadata, values, quant, scale, native, writer, events = make_pipeline(monkeypatch)
    op = Mock(return_value=None)
    monkeypatch.setattr(module, "try_li_quant_cache", op)
    assert run_pipeline(backend, metadata, values) is None
    native.assert_called_once()
    writer.assert_called_once()
    # The current main baseline passes FP32 native scales to its fused store;
    # fallback must preserve that conversion point.
    assert writer.call_args.args[1].dtype == torch.float32
    assert events == ["gather", "store"]
    torch.testing.assert_close(backend.k_cache.kv_cache[0].view(-1, 128)[5:11], quant, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,enabled", [(1, True), (5, True), (7, True), (6, False)])
def test_other_shapes_and_disabled_flag_preserve_native_path(monkeypatch, tokens, enabled):
    backend, metadata, values, _, _, native, writer, _ = make_pipeline(monkeypatch, tokens=tokens, enabled=enabled)
    op = Mock()
    monkeypatch.setattr(module, "try_li_quant_cache", op)
    run_pipeline(backend, metadata, values)
    op.assert_not_called()
    native.assert_called_once()
    writer.assert_called_once()


@pytest.mark.parametrize("decode,fused_store", [(False, True), (True, False)])
def test_prefill_and_non_fused_layout_use_existing_cache_writer(monkeypatch, decode, fused_store):
    backend, metadata, values, _, _, native, writer, _ = make_pipeline(
        monkeypatch, decode=decode, fused_store=fused_store
    )
    op = Mock()
    monkeypatch.setattr(module, "try_li_quant_cache", op)
    run_pipeline(backend, metadata, values)
    op.assert_not_called()
    native.assert_called_once()
    writer.assert_not_called()
    backend.write_cache.assert_called_once()


def test_failed_operator_propagates_without_native_retry(monkeypatch):
    backend, metadata, values, _, _, native, writer, _ = make_pipeline(monkeypatch)
    op = Mock(side_effect=RuntimeError("kernel failed"))
    monkeypatch.setattr(module, "try_li_quant_cache", op)
    with pytest.raises(RuntimeError, match="kernel failed"):
        run_pipeline(backend, metadata, values)
    op.assert_called_once()
    native.assert_not_called()
    writer.assert_not_called()
