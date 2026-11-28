# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from dataclasses import replace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackend,
    is_mxfp_cache_spec,
    mxfp_cache_spec,
    mxfp_cache_views_for_spec,
    mxfp_packet_section_sizes,
    mxfp_packet_size_bytes,
    mxfp_paged_cache_views,
)
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace


def _spec(ratio=1, heads=1, v_dim=256, padding=0):
    spec = mxfp_cache_spec(
        FullAttentionSpec(
            block_size=512 * ratio,
            num_kv_heads=heads,
            head_size=256,
            head_size_v=v_dim,
            dtype=torch.float16,
        )
    )
    return replace(spec, page_size_padded=spec.page_size_bytes + padding * ratio)


@pytest.mark.parametrize("ratio", [1, 2, 8])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("v_dim", [128, 256])
@pytest.mark.parametrize("padding", [0, 256])
def test_page_views_partition_payload_and_preserve_padding(ratio, heads, v_dim, padding):
    spec = _spec(ratio, heads, v_dim, padding)
    offset, num_pages, guard = 137, 2, 97
    backing = torch.full((offset + num_pages * spec.page_size_bytes + guard,), 93, dtype=torch.int8)
    raw = backing[offset:-guard]
    k, v, ks, vs = mxfp_cache_views_for_spec(raw, spec, 512)
    pitch = spec.page_size_bytes // ratio
    k_bytes, ks_bytes, v_bytes, vs_bytes = mxfp_packet_section_sizes(heads, 512, 256, v_dim)
    assert k.shape == (num_pages * ratio, heads, 8, 512, 32)
    assert ks.shape == (num_pages * ratio, heads, 32, 4, 16, 2)
    assert vs.shape == (num_pages * ratio, heads, v_dim // 16, 8, 16, 2)
    assert [t.storage_offset() for t in (k, ks, v, vs)] == [
        offset,
        offset + k_bytes,
        offset + k_bytes + ks_bytes,
        offset + k_bytes + ks_bytes + v_bytes,
    ]
    for marker, view in enumerate((k, ks, v, vs), start=1):
        assert view.stride(0) == pitch
        assert view[0].is_contiguous()
        view.view(torch.uint8).fill_(marker)
    pages = raw.view(num_pages * ratio, pitch)
    start = 0
    for marker, size in enumerate((k_bytes, ks_bytes, v_bytes, vs_bytes), start=1):
        assert (pages[:, start : start + size] == marker).all()
        start += size
    assert (pages[:, start:] == 93).all()
    assert (backing[:offset] == 93).all()
    assert (backing[-guard:] == 93).all()


def test_spec_keeps_real_head_dimensions_and_metadata():
    original = FullAttentionSpec(
        block_size=512, num_kv_heads=1, head_size=256, head_size_v=128, dtype=torch.float16, non_causal=True
    )
    spec = mxfp_cache_spec(original)
    assert type(spec) is FullAttentionSpec
    assert (spec.head_size, spec.head_size_v) == (256, 128)
    assert spec.non_causal is True
    assert spec.state_content_bytes == 396
    assert spec.page_size_bytes == 512 * 396
    assert original.state_content_bytes is None


def test_d256_page_budget_includes_both_scales_once():
    assert mxfp_packet_size_bytes(1, 512, 256, 256) == 270336
    assert _spec().page_size_bytes == 270336


@pytest.mark.parametrize("ratio", [1, 2, 8])
def test_cyclic_copy_moves_all_four_payloads_with_nonzero_offset(ratio):
    spec = _spec(ratio, padding=256)
    offset, guard = 128, 256
    backing = torch.full((offset + 3 * spec.page_size_bytes + guard,), 91, dtype=torch.int8)
    raw = backing[offset:-guard]
    views = mxfp_cache_views_for_spec(raw, spec, 512)
    pitch = spec.page_size_bytes // ratio
    for section, view in enumerate(views):
        for block in range(3 * ratio):
            view[block].view(torch.uint8).fill_(section * 17 + block + 1)
    before = [t.view(torch.uint8).clone().unflatten(0, (3, ratio)) for t in views]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2), KVCacheBlockCopy(src_block_id=2, dst_block_id=0)]
    # Duplicated entries catch repeated copies of an aliased region in a cycle.
    copy_kv_cache_blocks_inplace([views, views], 3, copies)
    for view, snapshot in zip(views, before):
        pages = view.view(torch.uint8).unflatten(0, (3, ratio))
        assert torch.equal(pages[0], snapshot[2])
        assert torch.equal(pages[2], snapshot[0])
        assert torch.equal(pages[1], snapshot[1])
    payload = mxfp_packet_size_bytes(1, 512, 256, 256)
    assert (raw.view(3 * ratio, pitch)[:, payload:] == 91).all()
    assert (backing[:offset] == 91).all() and (backing[-guard:] == 91).all()


@pytest.mark.parametrize("case", ["raw_short", "pitch_short", "raw_stride", "bad_v_dim", "bad_k_dim"])
def test_invalid_views_fail_before_as_strided(case):
    raw = torch.zeros(2 * 270336, dtype=torch.int8)
    kwargs = dict(num_kernel_blocks=2, num_kv_heads=1, k_dim=256, v_dim=256, block_size=512)
    if case == "raw_short":
        raw = raw[:-1]
    elif case == "pitch_short":
        kwargs["page_stride_bytes"] = 270335
    elif case == "raw_stride":
        raw = raw[::2]
    elif case == "bad_v_dim":
        kwargs["v_dim"] = 129
    else:
        kwargs["k_dim"] = 255
    with pytest.raises(ValueError):
        mxfp_paged_cache_views(raw, **kwargs)


def test_invalid_scheduler_geometry_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        mxfp_cache_views_for_spec(torch.empty(1, dtype=torch.int8), _spec(), 1024)
    with pytest.raises(ValueError, match="whole pages"):
        mxfp_cache_views_for_spec(torch.empty(1, dtype=torch.int8), _spec(), 512)
    with pytest.raises(ValueError, match="smaller"):
        mxfp_cache_spec(replace(_spec(), page_size_padded=1))


@pytest.mark.parametrize("block_size", [1, 512])
def test_mxfp_backend_customize_spec_supports_real_and_per_token_probe(block_size):
    original = FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=64, head_size_v=128, dtype=torch.bfloat16
    )
    customized = AscendC8MXFPAttentionBackend.customize_spec(original)
    expected_state_bytes = 64 + 128 + (64 + 128) // 32
    assert is_mxfp_cache_spec(customized)
    assert customized.dtype == torch.float8_e4m3fn
    assert customized.state_content_bytes == expected_state_bytes
    assert customized.page_size_bytes == 2 * block_size * expected_state_bytes

    plain_fp8 = replace(original, dtype=torch.float8_e4m3fn)
    assert not is_mxfp_cache_spec(original)
    assert not is_mxfp_cache_spec(plain_fp8)

    swa = SlidingWindowSpec(block_size=512, num_kv_heads=2, head_size=64, dtype=torch.bfloat16, sliding_window=128)
    assert AscendC8MXFPAttentionBackend.customize_spec(swa) is swa
    assert not is_mxfp_cache_spec(swa)


@pytest.mark.parametrize("mamba_bytes", [262144, 532480])
def test_platform_hybrid_alignment_uses_mxfp_customized_page_size(monkeypatch, mamba_bytes):
    from types import SimpleNamespace

    from vllm.config import CacheConfig
    from vllm.model_executor.models import ModelRegistry

    import vllm_ascend.core.kv_cache_dtype_handlers  # noqa: F401
    from vllm_ascend.platform import NPUPlatform

    class _FakeHybridModel:
        @staticmethod
        def get_mamba_state_shape_from_config(_cfg):
            return ((mamba_bytes // 2,),)

        @staticmethod
        def get_mamba_state_dtype_from_config(_cfg):
            return (torch.bfloat16,)

    cache_config = CacheConfig(block_size=512, cache_dtype="mxfp8", mamba_cache_mode="none")
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            use_mla=False,
            is_hybrid=True,
            architecture="Qwen3_5ForCausalLM",
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            hf_text_config=SimpleNamespace(model_type="qwen3_5"),
            max_model_len=32768,
            get_num_kv_heads=lambda _pc: 2,
            get_head_size=lambda: 256,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
        use_flashinfer_replayssm=False,
    )
    monkeypatch.setattr(
        NPUPlatform, "_find_non_ssm_backends", classmethod(lambda cls, _cfg: [AscendC8MXFPAttentionBackend])
    )
    monkeypatch.setattr(ModelRegistry, "resolve_model_cls", lambda *_args, **_kwargs: (_FakeHybridModel, "FakeHybrid"))

    NPUPlatform.update_block_size_for_backend(vllm_config)

    expected_page = mxfp_packet_size_bytes(2, 512, 256, 256)
    assert cache_config.block_size == 512
    assert cache_config.mamba_page_size_padded == expected_page
