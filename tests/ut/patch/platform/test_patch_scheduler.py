from types import SimpleNamespace

from vllm_ascend.patch.platform import patch_scheduler


def _dummy_scheduler(block_size: int = 16):
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size), use_eagle=False)


def _dummy_request(num_tokens: int = 128):
    return SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
    )


def test_mamba_block_aligned_split_accepts_common_prefix_limit():
    result = patch_scheduler._mamba_block_aligned_split(
        _dummy_scheduler(),
        _dummy_request(),
        64,
        0,
        0,
        32,
    )

    assert result == 32
