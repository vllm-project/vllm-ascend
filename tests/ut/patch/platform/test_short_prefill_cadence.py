# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_mamba_block_aligned_split as patch


def scheduler(requests, *, interval=2):
    return SimpleNamespace(
        requests={
            index: SimpleNamespace(num_prompt_tokens=prompt, max_tokens=output)
            for index, (prompt, output) in enumerate(requests)
        },
        local_prefill_interval=1,
        hash_block_size=64,
        cache_config=SimpleNamespace(block_size=2048),
        _ascend_short_prefill_interval=interval,
        _ascend_gdn_checkpoint_policy=SimpleNamespace(max_tokens=8192),
    )


@pytest.mark.parametrize(
    ("requests", "expected"),
    [
        ([(2048, 1), (2048, 8)], 3),
        ([(2048, 8), (2032, 8)], 2),
        ([(2048, 8), (1984, 8)], 1),
        ([(7000, 2), (7000, 8)], 2),
        ([(7000, 2), (7063, 8)], 2),
        ([(7000, 2), (7064, 8)], 1),
        ([(7000, 2), (8000, 3)], 1),
        ([(8190, 2)], 2),
        ([(7000, 2), (131072, 1024)], 1),
        ([(8000, 1024)], 1),
        ([], 1),
    ],
)
def test_short_cadence_uses_existing_scheduler_only_when_bounded(monkeypatch, requests, expected):
    instance = scheduler(requests)
    monkeypatch.setattr(patch, "_original_schedule", lambda state, throttle: (state.local_prefill_interval, throttle))
    assert patch._schedule(instance, throttle_prefills=True) == (expected, True)
    assert instance.local_prefill_interval == 1


def test_unconfigured_and_explicit_cadence_are_left_to_core(monkeypatch):
    instance = scheduler([(7000, 2)], interval=1)
    instance.local_prefill_interval = 3
    monkeypatch.setattr(patch, "_original_schedule", lambda state, _throttle: state.local_prefill_interval)
    assert patch._schedule(instance) == 3


def test_core_failure_restores_interval(monkeypatch):
    instance = scheduler([(7000, 2)])

    def failing(state, _throttle):
        assert state.local_prefill_interval == 2
        raise RuntimeError("from core")

    monkeypatch.setattr(patch, "_original_schedule", failing)
    with pytest.raises(RuntimeError, match="from core"):
        patch._schedule(instance)
    assert instance.local_prefill_interval == 1


def test_cadence_option_is_registered_and_requires_native_context_limit():
    from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig

    options = dict(sparse_kv_offload_config=SparseKVOffloadConfig())
    assert AscendConfig(**options).gdn_short_prefill_schedule_interval == 1
    configured = AscendConfig(
        **options,
        gdn_short_prefill_schedule_interval=2,
        gdn_native_prefill_checkpoints=True,
        gdn_internal_prefill_checkpoints=True,
        gdn_native_checkpoint_max_tokens=8192,
    )
    assert configured.gdn_short_prefill_schedule_interval == 2
    with pytest.raises(ValueError, match="requires native checkpoints"):
        AscendConfig(**options, gdn_short_prefill_schedule_interval=2)
