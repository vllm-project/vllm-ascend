import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


def test_ascend_mla_page_size_honors_hybrid_cache_padding():
    spec = AscendMLAAttentionSpec(
        block_size=768,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=912384,
    )

    assert spec.real_page_size_bytes == 884736
    assert spec.page_size_bytes == 912384

    merged = AscendMLAAttentionSpec.merge([spec, spec])
    assert merged.real_page_size_bytes == 884736
    assert merged.page_size_padded == 912384
    assert merged.page_size_bytes == 912384
