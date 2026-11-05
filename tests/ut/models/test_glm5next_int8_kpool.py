# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.indexer_kpool as backend_module
from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.models.glm5next.attention import Indexer
from vllm_ascend.models.glm5next.cache_views import view_compressed_indexer_cache


@pytest.mark.parametrize("dtype", ["mxfp4", "nvfp4"])
def test_unsupported_dtype_fails_before_layer_construction(dtype):
    config = SimpleNamespace(attention_config=SimpleNamespace(indexer_kv_dtype=dtype))
    with pytest.raises(ValueError, match="auto, bf16, fp8 or int8"):
        Indexer(config, None, 8, 8, None, None, None)


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
def test_int8_kpool_requires_a3_without_restricting_bf16(monkeypatch, device, dtype):
    profile = get_hardware_profile(device)
    assert profile.supports(HardwareCapability.POOL_KEY_INDEXER_INT8) == (device in (AscendDeviceType.A2, AscendDeviceType.A3))
    monkeypatch.setattr(backend_module, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(Glm5NextKPoolIndexerBackend, "_init_cann_ops", lambda self: None)
    indexer = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
        ),
        n_head=2,
        head_dim=128,
        topk_tokens=8,
        q_lora_rank=4,
        index_kpool=4,
        wq_b=SimpleNamespace(),
        wk_weights_proj=SimpleNamespace(),
        k_norm=SimpleNamespace(),
        softmax_scale=1.0,
        index_kpool_compress_ape=None,
        index_kpool_compress_gate=None,
        k_cache=SimpleNamespace(dtype=dtype),
        tail_cache=SimpleNamespace(),
        topk_indices_buffer=None,
    )
    if dtype == torch.int8 and device is not AscendDeviceType.A3:
        with pytest.raises(NotImplementedError, match="INT8.*Ascend A3"):
            Glm5NextKPoolIndexerBackend(indexer, qk_rope_head_dim=0)
    else:
        backend = Glm5NextKPoolIndexerBackend(indexer, qk_rope_head_dim=0)
        assert backend.kpool_int8 == (dtype == torch.int8)


@pytest.mark.parametrize("rows", [0, 1, 8])
def test_int8_quantization_zero_and_nonzero_rows(rows):
    torch.manual_seed(4)
    values = torch.randn(rows, 128, dtype=torch.bfloat16)
    if rows:
        values[0].zero_()
    keys, scales = Glm5NextKPoolIndexerBackend._symmetric_int8(values)
    assert keys.dtype == torch.int8 and scales.dtype == torch.float16
    assert scales.shape == (rows,)
    assert torch.isfinite(scales).all() and (scales > 0).all()
    restored = keys.float() * scales.float().unsqueeze(-1)
    assert torch.all((restored - values.float()).abs() <= scales.float().unsqueeze(-1) * 0.51)
    if rows:
        assert torch.count_nonzero(keys[0]) == 0


@pytest.mark.parametrize("blocks_per_page", [1, 4, 5])
def test_key_and_scale_share_transferable_pages(blocks_per_page):
    spec = AscendMLAAttentionSpec(
        block_size=128 * blocks_per_page,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        model_version="glm5_next",
        tokens_per_state=4,
        indexes_kv_by_block_stride=True,
        page_size_padded=8192 * blocks_per_page,
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8)
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, b, h, d: (n, b, h, d))
    keys, scales = view_compressed_indexer_cache("indexer", spec, raw, backend, 128)
    assert keys.shape == (2 * blocks_per_page, 32, 1, 128)
    assert scales.shape == (2 * blocks_per_page, 32, 1, 1)
    assert keys.stride(0) == scales.stride(0) * 2 == 8192
    keys.fill_(23)
    scales.fill_(0.125)
    # Simulate transporting a complete scheduler page, including all scales.
    destination = torch.zeros_like(raw)
    destination[: spec.page_size_bytes].copy_(raw[: spec.page_size_bytes])
    k2, s2 = view_compressed_indexer_cache("destination", spec, destination, backend, 128)
    torch.testing.assert_close(k2[:blocks_per_page], keys[:blocks_per_page])
    torch.testing.assert_close(s2[:blocks_per_page], scales[:blocks_per_page])
    assert torch.count_nonzero(k2[blocks_per_page:]) == 0
    assert torch.count_nonzero(s2[blocks_per_page:]) == 0
