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

import pytest
import vllm.v1.core.sched.scheduler as scheduler_module

import vllm_ascend.patch.platform.patch_mamba_block_aligned_split as mod
from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import (
    _mamba_block_aligned_split,
    _original_mamba_block_aligned_split,
)


def _scheduler(
    *,
    is_kv_consumer: bool | None,
    is_kv_producer: bool | None = None,
    uses_sparse_index_kpool: bool = False,
):
    kv_transfer_config = None if is_kv_consumer is None else SimpleNamespace(is_kv_consumer=is_kv_consumer)
    if kv_transfer_config is not None and is_kv_producer is not None:
        kv_transfer_config.is_kv_producer = is_kv_producer
    # vLLM main added `mamba_has_prefill_checkpoint_blocks` (gated by
    # MambaSpec.num_prefill_checkpoint_blocks) to the boundary split.
    scheduler_kwargs: dict = {}
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
    # The producer/standalone EAGLE-backoff suppression is applied inline
    # inside _mamba_block_aligned_split; no separate wrapper remains.
    assert registered is _mamba_block_aligned_split
    assert inspect.signature(_mamba_block_aligned_split) == inspect.signature(_original_mamba_block_aligned_split)


# ---------------------------------------------------------------------------
# Inline producer/standalone suppression: the drop knobs are cleared for the
# duration of the upstream call and restored afterwards. The original is
# monkeypatched so the observed knobs are version-independent.
# ---------------------------------------------------------------------------


def _fake_original_recording(observed: dict):
    def _fake_original(self, request, num_new_tokens, nlc=0, nec=0):
        observed["use_eagle"] = self.use_eagle
        observed["use_eagle_block_drop"] = getattr(self, "use_eagle_block_drop", None)
        return 42

    return _fake_original


def test_producer_clears_drop_knobs_around_upstream_call(monkeypatch):
    observed: dict[str, bool | None] = {}
    monkeypatch.setattr(mod, "_original_mamba_block_aligned_split", _fake_original_recording(observed))
    scheduler = _scheduler(is_kv_consumer=False, is_kv_producer=True)
    result = _mamba_block_aligned_split(scheduler, _request(), num_new_tokens=8)
    assert result == 42
    # The original sees the EAGLE drop disabled...
    assert observed == {"use_eagle": False, "use_eagle_block_drop": False}
    # ...and the scheduler's own knobs are restored afterwards.
    assert scheduler.use_eagle is True
    assert scheduler.use_eagle_block_drop is True


def test_standalone_clears_drop_knobs_around_upstream_call(monkeypatch):
    observed: dict[str, bool | None] = {}
    monkeypatch.setattr(mod, "_original_mamba_block_aligned_split", _fake_original_recording(observed))
    scheduler = _scheduler(is_kv_consumer=None)
    result = _mamba_block_aligned_split(scheduler, _request(), num_new_tokens=8)
    assert result == 42
    assert observed == {"use_eagle": False, "use_eagle_block_drop": False}
    assert scheduler.use_eagle is True
    assert scheduler.use_eagle_block_drop is True


def test_consumer_keeps_drop_knobs_around_upstream_call(monkeypatch):
    observed: dict[str, bool | None] = {}
    monkeypatch.setattr(mod, "_original_mamba_block_aligned_split", _fake_original_recording(observed))
    # Decode consumer with a computed prefix: the verifier window is preserved
    # by the early return, the original is never reached.
    scheduler = _scheduler(is_kv_consumer=True)
    result = _mamba_block_aligned_split(scheduler, _request(), num_new_tokens=8)
    assert result == 8
    assert observed == {}

    # Cold kv_both prefill reaches the original with the knobs untouched.
    scheduler = _scheduler(is_kv_consumer=True, is_kv_producer=True)
    result = _mamba_block_aligned_split(
        scheduler,
        _request(num_computed_tokens=0),
        num_new_tokens=8,
    )
    assert result == 42
    assert observed == {"use_eagle": True, "use_eagle_block_drop": True}


def test_producer_restores_drop_knobs_on_exception(monkeypatch):
    def _boom(self, request, num_new_tokens, nlc=0, nec=0):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_original_mamba_block_aligned_split", _boom)
    scheduler = _scheduler(is_kv_consumer=False, is_kv_producer=True)
    with pytest.raises(RuntimeError, match="boom"):
        _mamba_block_aligned_split(scheduler, _request(), num_new_tokens=8)
    assert scheduler.use_eagle is True
    assert scheduler.use_eagle_block_drop is True


def test_producer_handles_missing_drop_attributes(monkeypatch):
    monkeypatch.setattr(
        mod,
        "_original_mamba_block_aligned_split",
        lambda self, request, num_new_tokens, nlc=0, nec=0: 42,
    )
    scheduler = _scheduler(is_kv_consumer=False, is_kv_producer=True)
    del scheduler.use_eagle
    del scheduler.use_eagle_block_drop
    result = _mamba_block_aligned_split(scheduler, _request(), num_new_tokens=8)
    assert result == 42
    assert not hasattr(scheduler, "use_eagle")
    assert not hasattr(scheduler, "use_eagle_block_drop")
