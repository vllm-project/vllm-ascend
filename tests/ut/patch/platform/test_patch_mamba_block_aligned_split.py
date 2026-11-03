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
from unittest.mock import Mock

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
    scheduler_kwargs["mamba_shared_prefix_checkpoint"] = False
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


@pytest.mark.parametrize(
    ("is_kv_consumer", "is_kv_producer", "computed", "prompt", "new_tokens", "local", "external", "sparse"),
    [
        (True, False, 379, 380, 8, 0, 0, False),
        (None, None, 0, 380, 8, 379, 0, False),
        (True, False, 0, 380, 8, 0, 379, False),
        (True, True, 0, 4800, 4800, 0, 0, False),
        (False, True, 128, 500, 200, 16, 32, True),
    ],
    ids=["consumer_verifier", "standalone_local_hit", "consumer_external_hit", "kv_both_cold", "producer_sparse"],
)
def test_delegates_boundary_selection_and_arguments_to_native(
    monkeypatch, is_kv_consumer, is_kv_producer, computed, prompt, new_tokens, local, external, sparse
):
    original = Mock(wraps=_original_mamba_block_aligned_split)
    monkeypatch.setattr(mod, "_original_mamba_block_aligned_split", original)
    scheduler = _scheduler(
        is_kv_consumer=is_kv_consumer,
        is_kv_producer=is_kv_producer,
        uses_sparse_index_kpool=sparse,
    )
    request = _request(num_computed_tokens=computed, num_prompt_tokens=prompt, num_tokens=prompt)

    result = _mamba_block_aligned_split(
        scheduler,
        request,
        num_new_tokens=new_tokens,
        num_new_local_computed_tokens=local,
        num_external_computed_tokens=external,
    )

    original.assert_called_once_with(scheduler, request, new_tokens, local, external)
    native_scheduler = _scheduler(
        is_kv_consumer=is_kv_consumer,
        is_kv_producer=is_kv_producer,
        uses_sparse_index_kpool=sparse,
    )
    if is_kv_consumer is None or (is_kv_producer and not is_kv_consumer):
        native_scheduler.use_eagle = False
        native_scheduler.use_eagle_block_drop = False
    expected = _original_mamba_block_aligned_split(native_scheduler, request, new_tokens, local, external)
    assert result == expected


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
    # A consumer with a computed prefix delegates verifier padding to vLLM.
    scheduler = _scheduler(is_kv_consumer=True)
    result = _mamba_block_aligned_split(scheduler, _request(), num_new_tokens=8)
    assert result == 42
    assert observed == {"use_eagle": True, "use_eagle_block_drop": True}
    assert scheduler.use_eagle is True
    assert scheduler.use_eagle_block_drop is True

    # Cold kv_both prefill reaches the original with the knobs untouched.
    scheduler = _scheduler(is_kv_consumer=True, is_kv_producer=True)
    result = _mamba_block_aligned_split(
        scheduler,
        _request(num_computed_tokens=0),
        num_new_tokens=8,
    )
    assert result == 42
    assert observed == {"use_eagle": True, "use_eagle_block_drop": True}
    assert scheduler.use_eagle is True
    assert scheduler.use_eagle_block_drop is True


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
