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


def _scheduler(*, is_kv_consumer: bool | None):
    kv_transfer_config = None if is_kv_consumer is None else SimpleNamespace(is_kv_consumer=is_kv_consumer)
    # vLLM main added `mamba_has_prefill_checkpoint_blocks` (gated by
    # MambaSpec.num_prefill_checkpoint_blocks) to the boundary split; v0.27.1
    # does not define it.
    scheduler_kwargs: dict = {}
    if not vllm_version_is("0.27.1"):
        scheduler_kwargs["mamba_has_prefill_checkpoint_blocks"] = False
    return SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_transfer_config),
        cache_config=SimpleNamespace(block_size=384),
        use_eagle=True,
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


def test_pd_consumer_preserves_window_after_external_cache_hit():
    result = _mamba_block_aligned_split(
        _scheduler(is_kv_consumer=True),
        _request(num_computed_tokens=0),
        num_new_tokens=8,
        num_external_computed_tokens=379,
    )

    assert result == 8


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


def test_patch_is_registered_with_upstream_signature():
    assert scheduler_module.Scheduler._mamba_block_aligned_split is _mamba_block_aligned_split
    assert inspect.signature(_mamba_block_aligned_split) == inspect.signature(_original_mamba_block_aligned_split)
