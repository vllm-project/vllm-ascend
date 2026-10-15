# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.worker.c8_hybrid_layout import (
    c8_contiguous_views,
    configure_contiguous_c8_hybrid_layout,
    mamba_contiguous_views,
    use_contiguous_c8_hybrid_layout,
)


@pytest.mark.parametrize("heads,width", [(8, 1536), (16, 3072)])
@pytest.mark.parametrize("conv_len", [3, 6])
def test_contiguous_sections_preserve_scales(heads, width, conv_len):
    conv = conv_len * width * 2
    ssm = heads * 128 * 128 * 4
    page = conv + 2 * ssm + 2 * ssm // 32
    ms = SimpleNamespace(
        shapes=((conv_len, width), (heads, 128, 128)), dtypes=(torch.bfloat16, torch.float32), page_size_bytes=page
    )
    fa = SimpleNamespace(num_kv_heads=1, head_size=256, head_size_v=256, block_size=ssm // 256, page_size_bytes=page)
    for blocks in (1, 3):
        # Nonzero absolute storage offset catches slice-offset mistakes.
        backing = torch.full((blocks * page + 512,), 91, dtype=torch.int8)
        raw = backing[256:-256]
        k, v, ks, vs = c8_contiguous_views(raw, fa, 512, ms)
        c, s = mamba_contiguous_views(raw, ms)
        assert all(t.is_contiguous() for t in (k, v, ks, vs, c, s))
        assert s.data_ptr() == k.data_ptr()
        assert s.numel() * 4 == k.numel()
        assert v.data_ptr() == k.data_ptr() + k.numel()
        assert vs.data_ptr() + vs.numel() == raw.data_ptr() + raw.numel()
        ks.fill_(17)
        vs.fill_(23)
        v.view(torch.uint8).fill_(29)
        c.zero_()
        s.zero_()
        assert (ks == 17).all() and (vs == 23).all() and (v.view(torch.uint8) == 29).all()
        # Each SSM row overlaps exactly its own logical K block, no neighbours.
        k.view(torch.uint8).fill_(37)
        s[-1].zero_()
        logical_k = k.view(torch.uint8).view(blocks, -1)
        assert (logical_k[-1] == 0).all()
        if blocks > 1:
            assert (logical_k[:-1] == 37).all()
        assert (backing[:256] == 91).all() and (backing[-256:] == 91).all()
    with pytest.raises(ValueError, match="K block bytes == SSM bytes"):
        c8_contiguous_views(raw, SimpleNamespace(**{**vars(fa), "block_size": 512}), 512, ms)


@pytest.mark.parametrize("heads,width", [(8, 1536), (16, 3072)])
@pytest.mark.parametrize("prefix", [False, True])
def test_configuration_restores_geometry_after_backend_selection(heads, width, prefix):
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _: ((6, width), (heads, 128, 128)),
        get_mamba_state_dtype_from_config=lambda _: (torch.bfloat16, torch.float32),
    )
    cache = SimpleNamespace(
        block_size=512,
        mamba_page_size_padded=None,
        mamba_block_size=133120,
        enable_prefix_caching=prefix,
        mamba_cache_mode="align",
        cache_dtype="mxfp8",
    )
    config = SimpleNamespace(
        cache_config=cache,
        parallel_config=SimpleNamespace(),
        use_v2_model_runner=False,
        model_config=SimpleNamespace(
            architecture="Qwen3_5ForConditionalGeneration",
            is_hybrid=True,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 256,
        ),
    )
    with patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls", return_value=(model_cls, None)):
        configure_contiguous_c8_hybrid_layout(config)
        expected = (cache.block_size, cache.mamba_page_size_padded, cache.mamba_block_size)
        # The post-load hook must restore both fields, not just block_size.
        cache.block_size = 1536
        cache.mamba_page_size_padded = 811008
        configure_contiguous_c8_hybrid_layout(config)
    ssm = heads * 128 * 128 * 4
    assert expected == (ssm // 256, 6 * width * 2 + 2 * ssm + ssm // 16, ssm // 256)
    assert (cache.block_size, cache.mamba_page_size_padded, cache.mamba_block_size) == expected
    assert use_contiguous_c8_hybrid_layout(config)
    config.use_v2_model_runner = True
    assert not use_contiguous_c8_hybrid_layout(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_post_load_hook_selects_matching_layout_policy(enabled):
    from vllm.platforms.interface import Platform

    from vllm_ascend.platform import NPUPlatform

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            mamba_page_size_padded=1099776,
            enable_prefix_caching=True,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
        model_config=SimpleNamespace(),
    )
    with (
        patch("vllm_ascend.worker.c8_hybrid_layout.use_contiguous_c8_hybrid_layout", return_value=enabled),
        patch("vllm_ascend.worker.c8_hybrid_layout.configure_contiguous_c8_hybrid_layout") as configure,
        patch.object(Platform, "update_block_size_for_backend") as generic,
    ):
        NPUPlatform.update_block_size_for_backend(config)
    if enabled:
        configure.assert_called_once_with(config)
        generic.assert_not_called()
    else:
        configure.assert_not_called()
        generic.assert_called_once_with(config)
