# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_request, create_vllm_config
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform import patch_mamba_block_aligned_split as split_patch


def _make_scheduler(
    monkeypatch,
    tmp_path,
    scheduler_cls,
    *,
    has_mamba,
    cache_mode,
    budget,
    role="kv_both",
    speculative_method=None,
    num_speculative_tokens=2,
    eagle_block_drop=True,
):
    config = create_vllm_config(
        max_num_batched_tokens=128,
        block_size=16,
        speculative_method=speculative_method,
        num_speculative_tokens=num_speculative_tokens,
    )
    if config.speculative_config is not None:
        # Use the real MTP configuration and derived lookahead properties.
        config.speculative_config.draft_model_config.hf_config.num_nextn_predict_layers = num_speculative_tokens
        config.speculative_config.disable_eagle_block_drop = not eagle_block_drop
    # BalanceScheduler initializes real runtime reporting; isolate its writes
    # without requiring the source checkout to be writable.
    config.additional_config.update(
        runtime_config_path=str(tmp_path / "runtime_config.json"),
        runtime_report_dir=str(tmp_path / "report"),
        runtime_config_hot_reload=False,
    )
    config.model_config.max_model_len = 128
    config.scheduler_config.max_model_len = 128
    config.scheduler_config.max_num_batched_tokens = budget
    config.scheduler_config.async_scheduling = scheduler_cls is AsyncScheduler
    config.cache_config.mamba_cache_mode = cache_mode
    config.cache_config.enable_prefix_caching = not has_mamba or cache_mode == "align"
    config.cache_config.num_gpu_blocks = 128
    config.kv_transfer_config = (
        None if role is None else KVTransferConfig(kv_connector="MooncakeConnector", kv_role=role)
    )
    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.requires_kv_delivery = False
    connector.supports_divergent_local_hybrid_hits = False
    connector.get_num_new_matched_tokens.return_value = (0, False)
    connector.build_connector_meta.return_value = None
    connector.take_events.return_value = ()
    monkeypatch.setattr(scheduler_module.KVConnectorFactory, "create_connector", lambda **kwargs: connector)

    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    groups = [
        KVCacheGroupSpec(
            ["attention"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
        )
    ]
    if has_mamba:
        groups.append(
            KVCacheGroupSpec(
                ["state"],
                MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode=cache_mode),
            )
        )
    kv_cache_config = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=groups)
    scheduler = scheduler_cls(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        structured_output_manager=StructuredOutputManager(config),
    )
    assert scheduler_cls._mamba_block_aligned_split is split_patch._mamba_block_aligned_split
    return scheduler


def _prefill_intervals(scheduler, request):
    scheduler.add_request(request)
    intervals: list[tuple[int, int]] = []
    while request.num_computed_tokens < request.num_prompt_tokens:
        start = request.num_computed_tokens
        output = scheduler.schedule()
        count = output.num_scheduled_tokens[request.request_id]
        end = request.num_computed_tokens
        assert 0 < count <= scheduler.max_num_scheduled_tokens
        assert end == start + count <= request.num_prompt_tokens
        assert not intervals or intervals[-1][1] == start
        intervals.append((start, end))
        if end < request.num_prompt_tokens:
            # Non-final prefills sample no token. Retire this CPU-simulated
            # forward before entering the real scheduler's running loop.
            feedback = create_model_runner_output([request])
            feedback.sampled_token_ids = [[]]
            scheduler.update_from_output(output, feedback)
    return intervals


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
@pytest.mark.parametrize("has_mamba,cache_mode", [(False, "none"), (False, "align"), (True, "none")])
def test_attention_and_nonalign_schedulers_do_not_enter_mamba_split(
    monkeypatch, tmp_path, scheduler_cls, has_mamba, cache_mode
):
    scheduler = _make_scheduler(
        monkeypatch, tmp_path, scheduler_cls, has_mamba=has_mamba, cache_mode=cache_mode, budget=4
    )
    split = Mock(side_effect=AssertionError("This scheduler must not call Mamba alignment"))
    monkeypatch.setattr(scheduler, "_mamba_block_aligned_split", split)
    request = create_request(1, num_tokens=11, block_size=16)

    assert _prefill_intervals(scheduler, request) == [(0, 4), (4, 8), (8, 11)]
    split.assert_not_called()


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
def test_generic_hybrid_scheduler_keeps_reusable_prefill_boundary(monkeypatch, tmp_path, scheduler_cls, role):
    scheduler = _make_scheduler(
        monkeypatch, tmp_path, scheduler_cls, has_mamba=True, cache_mode="align", budget=32, role=role
    )
    split = Mock(wraps=scheduler._mamba_block_aligned_split)
    monkeypatch.setattr(scheduler, "_mamba_block_aligned_split", split)
    request = create_request(2, num_tokens=49, block_size=16)

    assert _prefill_intervals(scheduler, request) == [(0, 32), (32, 48), (48, 49)]
    assert split.call_count == 3


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_generic_hybrid_scheduler_keeps_chunks_without_an_alignment_stop(monkeypatch, tmp_path, scheduler_cls):
    scheduler = _make_scheduler(monkeypatch, tmp_path, scheduler_cls, has_mamba=True, cache_mode="align", budget=4)
    split = Mock(wraps=scheduler._mamba_block_aligned_split)
    monkeypatch.setattr(scheduler, "_mamba_block_aligned_split", split)
    request = create_request(3, num_tokens=49, block_size=16)

    intervals = _prefill_intervals(scheduler, request)

    assert [end - start for start, end in intervals] == [4] * 12 + [1]
    assert intervals[1] == (4, 8)
    assert split.call_count == len(intervals)


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
@pytest.mark.parametrize("role", ["kv_consumer", "kv_both"])
@pytest.mark.parametrize("lookahead", [1, 2, 4])
@pytest.mark.parametrize("eagle_block_drop", [False, True])
@pytest.mark.parametrize("budget", [16, 32])
def test_mtp_consumer_prefill_keeps_legacy_forward_progress(
    monkeypatch, tmp_path, scheduler_cls, role, lookahead, eagle_block_drop, budget
):
    scheduler = _make_scheduler(
        monkeypatch,
        tmp_path,
        scheduler_cls,
        has_mamba=True,
        cache_mode="align",
        budget=budget,
        role=role,
        speculative_method="mtp",
        num_speculative_tokens=lookahead,
        eagle_block_drop=eagle_block_drop,
    )
    assert scheduler.vllm_config.speculative_config.method == "mtp"
    assert scheduler.num_prefill_lookahead == lookahead
    assert scheduler.use_eagle and scheduler.use_eagle_block_drop == eagle_block_drop
    assert scheduler.num_hash_lookahead_tokens == 0
    assert scheduler.need_mamba_block_aligned_split
    assert not scheduler.mamba_has_prefill_checkpoint_blocks
    request = create_request(401, num_tokens=49, block_size=16)

    intervals = _prefill_intervals(scheduler, request)

    assert intervals[-1][1] == 49
    if budget == 32:
        assert intervals == [(0, 32), (32, 49)]
    # Every unfinished prefill leaves the real MTP drafter's prompt runway.
    assert all(49 - end >= lookahead for _, end in intervals[:-1])
