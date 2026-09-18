# SPDX-License-Identifier: Apache-2.0
"""C8 cache bytes, manager/kernel page splitting, and runner view aliases."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheTensor, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.worker.flash_kv_cache import (
    create_flash_kernel_block_view,
    customize_flash_mla_c8_spec,
    split_flash_mla_c8_cache,
)


def _attention():
    return SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        impl=SimpleNamespace(dtype=torch.float8_e4m3fn, fa_quant_layer=True),
    )


def test_c8_spec_preserves_upstream_capabilities_and_padding():
    source = MLAAttentionSpec(
        block_size=768,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
        page_size_padded=768 * 1152,
    )
    spec = customize_flash_mla_c8_spec(source, _attention())
    assert type(spec) is type(source)
    assert spec.block_size == 768 and spec.head_size == 576
    assert spec.non_causal_multi_token_decode
    assert spec.state_content_size_bytes == 640
    assert spec.unpadded_page_size_bytes == 768 * 640
    assert spec.page_size_bytes == source.page_size_bytes
    assert source.dtype == torch.bfloat16 and source.state_content_size_bytes == 1152


@pytest.mark.parametrize("layout", [KVCacheLayout.BLHNC, KVCacheLayout.LBHNC])
@pytest.mark.parametrize("manager_size", [128, 768])
@pytest.mark.parametrize("dcp", [1, 8])
def test_mixed_dtype_kernel_views_preserve_layer_pages_and_guard_bytes(layout, manager_size, dcp):
    kernel_size, layers, prefix = 128, 2, 64
    blocks = 24 // dcp
    source = MLAAttentionSpec(block_size=manager_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    spec = customize_flash_mla_c8_spec(source, _attention())
    # DCP only changes how many local pages the manager allocates. The layout
    # and static layer scale must not multiply physical payload by DCP size.
    assert spec.state_content_size_bytes == 640
    page = spec.page_size_bytes
    pitch = layers * page if layout == KVCacheLayout.BLHNC else page
    layer_stride = page if layout == KVCacheLayout.BLHNC else blocks * page
    backing = torch.full((prefix + blocks * layers * page + 64,), -91, dtype=torch.int8)
    descriptor = KVCacheTensor(
        size=backing.numel(),
        layers=["mla0", "mla1"],
        offset=0,
        layer_stride=layer_stride,
        block_stride=pitch,
    )
    ratio = manager_size // kernel_size
    views = []
    for layer in range(layers):
        raw = backing.as_strided((blocks, page), (pitch, 1), prefix + layer * layer_stride)
        if ratio > 1:
            packed = create_flash_kernel_block_view(raw, spec, kernel_size, layout, descriptor, layer).squeeze(1)
        else:
            packed = raw.view(torch.float8_e4m3fn).view(blocks, kernel_size, 640)
        latent, rope = split_flash_mla_c8_cache(packed)
        assert latent.shape == (blocks * ratio, 128, 1, 512)
        assert rope.shape == (blocks * ratio, 128, 1, 64)
        assert latent.dtype == torch.float8_e4m3fn and rope.dtype == torch.bfloat16
        assert latent.stride(0) == pitch // ratio
        assert rope.stride(0) == pitch // ratio // 2
        assert latent.stride(1) == 512 and rope.stride(1) == 64
        assert rope.data_ptr() - latent.data_ptr() == 128 * 512
        assert latent.untyped_storage().data_ptr() == rope.untyped_storage().data_ptr() == backing.data_ptr()
        views.append((latent, rope))

    touched = torch.zeros_like(backing, dtype=torch.bool)
    for layer, (latent, rope) in enumerate(views):
        for page_id in (0, ratio, blocks * ratio - 1):
            for token in (0, 127):
                latent[page_id, token].fill_(layer + 1)
                rope[page_id, token].fill_(layer + 3)
                page_offset = latent.data_ptr() - backing.data_ptr() + page_id * latent.stride(0)
                latent_offset = page_offset + token * 512
                rope_offset = page_offset + 128 * 512 + token * 128
                touched[latent_offset : latent_offset + 512] = True
                touched[rope_offset : rope_offset + 128] = True
    for layer, (latent, rope) in enumerate(views):
        for page_id in (0, ratio, blocks * ratio - 1):
            for token in (0, 127):
                assert torch.all(latent[page_id, token].float() == layer + 1)
                assert torch.all(rope[page_id, token] == layer + 3)
    assert torch.all(backing[~touched] == -91)


def test_c8_views_reject_wrong_payload_and_misaligned_bf16_component():
    with pytest.raises(ValueError, match="640"):
        split_flash_mla_c8_cache(torch.empty(2, 128, 576, dtype=torch.float8_e4m3fn))
    storage = torch.empty(2 * 128 * 641, dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError):
        split_flash_mla_c8_cache(storage.as_strided((2, 128, 640), (128 * 641, 641, 1)))


@pytest.mark.parametrize("runner_version", [1, 2])
@pytest.mark.parametrize("manager_size", [128, 768])
def test_both_runner_reshape_paths_return_c8_alias_tuple(runner_version, manager_size):
    # Reuse the existing production-method harnesses without loading an NPU.
    from tests.ut.worker.test_flash_mla_upstream_cache import _MLAAttention, _runner_methods
    from tests.ut.worker.test_model_runner_v2_flash_cache_cpu import _cache_methods

    spec = customize_flash_mla_c8_spec(
        MLAAttentionSpec(block_size=manager_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16), _attention()
    )
    blocks, page, layers = 3, spec.page_size_bytes, ["mla0", "mla1"]
    group = SimpleNamespace(layer_names=layers, kv_cache_spec=spec, kv_cache_group_id=0, backend=None)
    config = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_groups=[group],
        kv_cache_tensors=[
            KVCacheTensor(size=blocks * 2 * page, layers=layers, offset=0, layer_stride=page, block_stride=2 * page)
        ],
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC),
    )
    helpers = {
        "create_flash_kernel_block_view": create_flash_kernel_block_view,
        "split_flash_mla_c8_cache": split_flash_mla_c8_cache,
    }
    if runner_version == 1:
        runner = _runner_methods()
        runner._reshape_kv_cache_tensors.__func__.__globals__.update(helpers)
        runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
        runner.vllm_config = vllm_config
        runner.device = torch.device("cpu")
        runner.compilation_config = SimpleNamespace(static_forward_context={name: _MLAAttention() for name in layers})
        runner.runner_only_attn_layers = set()
        runner.kernel_block_sizes = [[128]]
        runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])
        raw = runner._allocate_kv_cache_tensors(config)
        caches = runner._reshape_kv_cache_tensors(config, raw)
    else:
        namespace = _cache_methods()
        namespace.update(helpers, MLAAttentionSpec=MLAAttentionSpec, get_current_vllm_config=lambda: vllm_config)
        raw = namespace["_allocate_kv_cache"](config, {}, torch.device("cpu"))
        caches = namespace["_reshape_kv_cache_v2"]([group], raw, "auto", [128], {}, config)
    ratio = manager_size // 128
    for name in layers:
        assert isinstance(caches[name], tuple)
        latent, rope = caches[name]
        assert latent.shape == (blocks * ratio, 128, 1, 512)
        assert rope.shape == (blocks * ratio, 128, 1, 64)
        assert latent.untyped_storage().data_ptr() == raw[name].untyped_storage().data_ptr()
        assert rope.data_ptr() - latent.data_ptr() == 128 * 512
        assert rope.stride(0) * 2 == latent.stride(0) == 2 * page // ratio
