# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.

from types import SimpleNamespace

import pytest

import vllm_ascend.patch.platform.patch_layerwise_offload_scheduler as layerwise_patch


def _request(*, computed_tokens=0, prompt_tokens=1024):
    return SimpleNamespace(
        num_computed_tokens=computed_tokens,
        num_prompt_tokens=prompt_tokens,
    )


@pytest.fixture
def fake_async_scheduler(monkeypatch):
    class FakeAsyncScheduler:
        def _mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens=0,
            num_external_computed_tokens=0,
        ):
            del request
            self.original_calls.append(
                (num_new_local_computed_tokens, num_external_computed_tokens)
            )
            return num_new_tokens - 1

    monkeypatch.setattr(layerwise_patch, "AsyncScheduler", FakeAsyncScheduler)
    return FakeAsyncScheduler


def _enabled_scheduler(fake_async_scheduler):
    layerwise_patch.apply_layerwise_offload_scheduler_patch()
    scheduler = fake_async_scheduler()
    scheduler.original_calls = []
    scheduler.store_granularity = 128
    setattr(scheduler, layerwise_patch._LAYERWISE_SCHEDULER_MARKER, True)
    return scheduler


def test_patch_delegates_unmarked_schedulers_and_is_idempotent(fake_async_scheduler):
    layerwise_patch.apply_layerwise_offload_scheduler_patch()
    first_hook = fake_async_scheduler._mamba_block_aligned_split
    scheduler = fake_async_scheduler()
    scheduler.original_calls = []

    assert scheduler._mamba_block_aligned_split(_request(), 257) == 256
    assert scheduler.original_calls == [(0, 0)]

    layerwise_patch.apply_layerwise_offload_scheduler_patch()
    assert fake_async_scheduler._mamba_block_aligned_split is first_hook


def test_patch_aligns_prefill_chunk_and_computed_deltas(fake_async_scheduler):
    scheduler = _enabled_scheduler(fake_async_scheduler)

    result = scheduler._mamba_block_aligned_split(
        _request(computed_tokens=128),
        200,
        num_external_computed_tokens=128,
    )

    assert result == 128
    assert scheduler.original_calls == []


@pytest.mark.parametrize(
    ("computed_tokens", "proposed_tokens", "expected"),
    [(900, 124, 124), (1024, 4, 4), (0, 0, 0)],
)
def test_patch_preserves_final_prefill_decode_and_empty_chunks(
    fake_async_scheduler, computed_tokens, proposed_tokens, expected
):
    scheduler = _enabled_scheduler(fake_async_scheduler)
    assert scheduler._mamba_block_aligned_split(
        _request(computed_tokens=computed_tokens), proposed_tokens
    ) == expected


def test_alignment_rejects_invalid_granularity():
    with pytest.raises(ValueError, match="granularity"):
        layerwise_patch.align_prefill_chunk_for_store(_request(), 0, 128, 0)


def test_patch_fails_when_upstream_hook_is_missing(monkeypatch):
    class SchedulerWithoutHook:
        pass

    monkeypatch.setattr(layerwise_patch, "AsyncScheduler", SchedulerWithoutHook)
    with pytest.raises(RuntimeError, match="pre-allocation hook"):
        layerwise_patch.apply_layerwise_offload_scheduler_patch()
