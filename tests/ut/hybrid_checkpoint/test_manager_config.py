#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the opt-in gate of frontier checkpoints."""

from types import SimpleNamespace

import pytest

from vllm_ascend.hybrid_checkpoint.manager import check_frontier_checkpoint_config

BLOCK = 16


def _config(**overrides):
    cache = SimpleNamespace(
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        prefix_cache_retention_interval=0,
        use_replayssm=False,
    )
    parallel = SimpleNamespace(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    scheduler = SimpleNamespace(long_prefill_token_threshold=0)
    config = SimpleNamespace(
        cache_config=cache,
        parallel_config=parallel,
        scheduler_config=scheduler,
        speculative_config=None,
        kv_transfer_config=None,
    )
    for path, value in overrides.items():
        section, _, field = path.partition("__")
        setattr(getattr(config, section), field, value)
    return config


def _check(config, use_v2_model_runner=True, enable_kv_cache_events=False):
    return check_frontier_checkpoint_config(
        config,
        block_size=BLOCK,
        use_v2_model_runner=use_v2_model_runner,
        enable_kv_cache_events=enable_kv_cache_events,
    )


def test_accepts_the_supported_configuration():
    assert _check(_config()) is None


def test_requires_prefix_caching_on_the_aligned_mamba_mode():
    with pytest.raises(ValueError, match="prefix caching"):
        _check(_config(cache_config__enable_prefix_caching=False))
    with pytest.raises(ValueError, match="prefix caching"):
        _check(_config(cache_config__mamba_cache_mode="align_decode"))


def test_requires_model_runner_v2():
    with pytest.raises(ValueError, match="Model Runner V2"):
        _check(_config(), use_v2_model_runner=False)


def test_requires_retention_interval_zero():
    with pytest.raises(ValueError, match="retention_interval"):
        _check(_config(cache_config__prefix_cache_retention_interval=4096))


def test_rejects_a_long_prefill_threshold_below_the_block_size():
    with pytest.raises(ValueError, match="long_prefill_token_threshold"):
        _check(_config(scheduler_config__long_prefill_token_threshold=BLOCK - 1))


def test_a_threshold_at_or_above_the_block_size_is_accepted():
    assert _check(_config(scheduler_config__long_prefill_token_threshold=BLOCK)) is None


def test_rejects_replayssm():
    with pytest.raises(ValueError, match="ReplaySSM"):
        _check(_config(cache_config__use_replayssm=True))


def test_rejects_kv_cache_events():
    # The port publishes no recovery-unit events, so it cannot serve a
    # consumer that expects them.
    with pytest.raises(ValueError, match="KV cache events"):
        _check(_config(), enable_kv_cache_events=True)


def test_rejects_context_parallel():
    with pytest.raises(ValueError, match="context parallel"):
        _check(_config(parallel_config__decode_context_parallel_size=2))
    with pytest.raises(ValueError, match="context parallel"):
        _check(_config(parallel_config__prefill_context_parallel_size=2))
