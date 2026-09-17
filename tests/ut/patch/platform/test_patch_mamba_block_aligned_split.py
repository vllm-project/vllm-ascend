# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from types import SimpleNamespace

import vllm.v1.core.sched.scheduler as scheduler_module

from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import (
    _mamba_block_aligned_split,
    _original_mamba_block_aligned_split,
)
from vllm_ascend.utils import vllm_version_is


def _scheduler(
    *,
    is_kv_consumer: bool | None,
    is_kv_producer: bool | None = None,
    uses_sparse_index_kpool: bool = False,
):
    if is_kv_consumer is None:
        kv_transfer_config = None
    else:
        role = {"is_kv_consumer": is_kv_consumer}
        if is_kv_producer is not None:
            role["is_kv_producer"] = is_kv_producer
        kv_transfer_config = SimpleNamespace(**role)
    # vLLM main added `mamba_has_prefill_checkpoint_blocks` (gated by
    # MambaSpec.num_prefill_checkpoint_blocks) to the boundary split; v0.28.0
    # does not define it.
    scheduler_kwargs: dict = {}
    if not vllm_version_is("0.28.0"):
        scheduler_kwargs["mamba_has_prefill_checkpoint_blocks"] = False
        scheduler_kwargs["mamba_fine_grained_prefix_cache"] = False
    indexer_config = {"index_topk": 2048, "index_kpool": 4} if uses_sparse_index_kpool else {}
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(),
                hf_text_config=SimpleNamespace(**indexer_config),
            ),
        ),
        cache_config=SimpleNamespace(block_size=384),
        block_size=128,
        use_eagle=True,
        use_eagle_block_drop=True,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=384,
        mamba_partial_cache_hit=False,
        **scheduler_kwargs,
    )


def _request(
    *,
    num_computed_tokens: int = 379,
    num_prompt_tokens: int = 380,
    num_tokens: int = 380,
):
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_tokens=num_tokens,
        shared_prefix_boundary=0,
    )


def test_pd_consumer_preserves_complete_speculative_window():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=True),
        _request(),
        num_new_tokens=8,
    )

    assert result == 8


def test_producer_retains_upstream_mamba_boundary_split():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False),
        _request(),
        num_new_tokens=8,
    )

    assert result == 5


def test_non_pd_request_retains_upstream_mamba_boundary_split():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=None),
        _request(),
        num_new_tokens=8,
    )

    assert result == 5


def test_partial_hit_does_not_stop_at_shared_prefix_junction():
    request = _request(
        num_computed_tokens=0,
        num_prompt_tokens=2000,
        num_tokens=2000,
    )
    request.shared_prefix_boundary = 600

    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False),
        request,
        num_new_tokens=1000,
    )

    # Normal alignment ends at 768. The upstream junction stop would cut this
    # chunk at 384 and create a cold-path-absent recurrent-kernel boundary.
    assert result == 768


def test_pd_consumer_preserves_window_after_external_cache_hit():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=True),
        _request(num_computed_tokens=0),
        num_new_tokens=8,
        num_external_computed_tokens=379,
    )

    assert result == 8


def test_kv_both_cold_prefill_retains_mamba_boundary_split():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=True),
        _request(
            num_computed_tokens=0,
            num_prompt_tokens=4800,
            num_tokens=4800,
        ),
        num_new_tokens=4800,
    )

    # 4800 rounds down to 4608; EAGLE keeps one 384-token verifier block.
    assert result == 4224


def test_producer_splits_window_after_external_cache_hit():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False),
        _request(num_computed_tokens=0),
        num_new_tokens=8,
        num_external_computed_tokens=379,
    )

    assert result == 5


def test_producer_decode_fast_path_remains_unsplit():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False),
        _request(num_computed_tokens=380),
        num_new_tokens=8,
    )

    assert result == 8


def test_sparse_index_kpool_prefill_uses_resolved_common_block_size():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False, uses_sparse_index_kpool=True),
        _request(
            num_computed_tokens=128,
            num_prompt_tokens=500,
            num_tokens=500,
        ),
        num_new_tokens=200,
    )

    assert result == 128


def test_sparse_index_kpool_stops_at_last_cacheable_boundary():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False, uses_sparse_index_kpool=True),
        _request(
            num_computed_tokens=220,
            num_prompt_tokens=500,
            num_tokens=500,
        ),
        num_new_tokens=100,
    )

    assert result == 36


def test_sparse_index_kpool_does_not_round_small_chunk_to_zero():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False, uses_sparse_index_kpool=True),
        _request(
            num_computed_tokens=128,
            num_prompt_tokens=500,
            num_tokens=500,
        ),
        num_new_tokens=64,
    )

    assert result == 64


def test_sparse_index_kpool_pd_consumer_still_preserves_verifier_window():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=True, uses_sparse_index_kpool=True),
        _request(),
        num_new_tokens=8,
    )

    assert result == 8


def test_patch_is_registered_with_upstream_signature():
    registered = scheduler_module.Scheduler._mamba_block_aligned_split
    # The EAGLE-backoff suppression for producers and standalone instances is
    # inlined in the split itself (``_skips_eagle_block_drop``), so the
    # method is replaced directly - no outer wrapper is registered.
    assert registered is _mamba_block_aligned_split
    assert inspect.signature(_mamba_block_aligned_split) == inspect.signature(_original_mamba_block_aligned_split)


def test_producer_cold_prefill_suppresses_eagle_backoff():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=False, is_kv_producer=True),
        _request(
            num_computed_tokens=0,
            num_prompt_tokens=4800,
            num_tokens=4800,
        ),
        num_new_tokens=4800,
    )

    # Without the backoff the chunk reaches the final full-page boundary
    # (4608 = 12 x 384) instead of stopping one verifier block short (4224).
    assert result == 4608


def test_standalone_cold_prefill_suppresses_eagle_backoff():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=None),
        _request(
            num_computed_tokens=0,
            num_prompt_tokens=4800,
            num_tokens=4800,
        ),
        num_new_tokens=4800,
    )

    assert result == 4608
