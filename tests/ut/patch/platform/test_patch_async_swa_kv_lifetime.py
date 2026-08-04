# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

import vllm_ascend.patch.platform.patch_async_swa_kv_lifetime as patch


def test_schedule_output_tracks_in_flight_tokens(monkeypatch):
    request = SimpleNamespace(num_in_flight_tokens=0)
    scheduler = SimpleNamespace(requests={"request": request})
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request": 3})

    monkeypatch.setattr(patch, "_original_update_after_schedule", lambda *_args: None)
    monkeypatch.setattr(patch, "_original_update_from_output", lambda *_args: "output")

    patch._patched_update_after_schedule(scheduler, scheduler_output)
    assert request.num_in_flight_tokens == 3

    assert patch._patched_update_from_output(scheduler, scheduler_output, SimpleNamespace()) == "output"
    assert request.num_in_flight_tokens == 0


def test_allocate_prunes_on_processed_token_basis(monkeypatch):
    pruned_at = []
    swa_manager = SimpleNamespace(
        kv_cache_spec=SlidingWindowSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=512,
        )
    )
    mamba_manager = MambaManager.__new__(MambaManager)
    mamba_manager.kv_cache_spec = MambaSpec(
        block_size=1,
        shapes=((1,),),
        dtypes=(torch.float32,),
        num_speculative_blocks=1,
    )
    mamba_manager.num_speculative_blocks = 1
    mamba_manager.mamba_cache_mode = "none"
    request = SimpleNamespace(
        request_id="request",
        num_in_flight_tokens=1,
    )

    monkeypatch.setattr(
        patch,
        "_original_remove_skipped_blocks",
        lambda manager, _request_id, num_tokens: pruned_at.append((type(manager.kv_cache_spec), num_tokens)),
    )

    def original_allocate_slots(_self, current_request):
        patch._patched_remove_skipped_blocks(swa_manager, current_request.request_id, 159)
        mamba_manager.remove_skipped_blocks(current_request.request_id, 159)

    monkeypatch.setattr(patch, "_original_allocate_slots", original_allocate_slots)

    patch._patched_allocate_slots(SimpleNamespace(), request)
    assert pruned_at == [(SlidingWindowSpec, 158), (MambaSpec, 158)]

    patch._patched_remove_skipped_blocks(swa_manager, request.request_id, 159)
    assert pruned_at == [
        (SlidingWindowSpec, 158),
        (MambaSpec, 158),
        (SlidingWindowSpec, 159),
    ]


def test_connector_prunes_on_processed_token_basis(monkeypatch):
    pruned_at = []
    manager = SimpleNamespace(
        kv_cache_spec=SlidingWindowSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=512,
        )
    )
    request = SimpleNamespace(
        request_id="request",
        num_in_flight_tokens=1,
    )

    monkeypatch.setattr(
        patch,
        "_original_remove_skipped_blocks",
        lambda _self, _request_id, num_tokens: pruned_at.append(num_tokens),
    )

    def original_connector_finished(_self, current_request):
        patch._patched_remove_skipped_blocks(manager, current_request.request_id, 159)
        return False, None

    monkeypatch.setattr(patch, "_original_connector_finished", original_connector_finished)

    assert patch._patched_connector_finished(SimpleNamespace(), request) == (
        False,
        None,
    )
    assert pruned_at == [158]


def test_swa_admission_accounts_for_concurrent_batches(monkeypatch):
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=512,
    )
    manager = SimpleNamespace(
        kv_cache_spec=spec,
        _max_admission_blocks_per_request=None,
    )
    vllm_config = SimpleNamespace(
        max_concurrent_batches=2,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        model_config=SimpleNamespace(max_model_len=2048),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    def original_scheduler_init(scheduler, _vllm_config):
        scheduler.max_model_len = 2048
        scheduler.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=(manager,)))

    monkeypatch.setattr(patch, "_original_scheduler_init", original_scheduler_init)

    scheduler = SimpleNamespace()
    patch._patched_scheduler_init(scheduler, vllm_config)

    assert manager._max_admission_blocks_per_request == 97
    assert spec.max_memory_usage_bytes(vllm_config) == 97 * spec.page_size_bytes
