from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler

import vllm_ascend.patch.platform.patch_mamba_block_aligned_split  # noqa: F401


def _scheduler_stub() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=1600),
        use_eagle=True,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=16,
    )


def _request_stub(num_computed_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=2002,
        num_tokens=2002,
        shared_prefix_boundary=0,
    )


def test_fragmented_chunk_past_last_cache_position_is_not_forced_to_align():
    num_scheduled = Scheduler._mamba_block_aligned_split(_scheduler_stub(), _request_stub(0), 364)

    assert num_scheduled == 364


def test_mid_block_chunk_past_last_cache_position_is_not_forced_to_realign():
    num_scheduled = Scheduler._mamba_block_aligned_split(_scheduler_stub(), _request_stub(331), 1000)

    assert num_scheduled == 1000
