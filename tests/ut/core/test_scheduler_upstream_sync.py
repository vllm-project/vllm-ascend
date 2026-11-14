# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.metrics.stats import KV_FETCH_IN_PROGRESS
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.core.test_recompute_scheduler import _make_mm_recompute_scheduler
from tests.ut.core.test_scheduler_pause_drain import DYNTRA_SCHEDULERS, SCHEDULERS, make_scheduler, step
from tests.ut.core.test_score_encoder_cache_scheduler import _request as make_mm_request
from tests.ut.kv_offload.utils import create_model_runner_output, create_request, create_vllm_config
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkAsyncScheduler, ProfilingChunkScheduler
from vllm_ascend.core.scheduler_utils import should_skip_external_kv
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

LOAD_PATHS = tuple(
    pytest.param(cls, prefetch, id=f"{cls.__name__}-{'prefetch' if prefetch else 'schedule'}")
    for cls, prefetch in [*((cls, False) for cls in SCHEDULERS), *((cls, True) for cls in DYNTRA_SCHEDULERS)]
)


class HybridConnector(KVConnectorBase_V1, SupportsHMA):
    """Spec for mocks that honor the multi-group connector lifecycle."""


@pytest.fixture(autouse=True)
def reset_ascend_config():
    clear_ascend_config()
    yield
    clear_ascend_config()


@pytest.fixture(params=SCHEDULERS, ids=lambda cls: cls.__name__)
def scheduler(request):
    return make_scheduler(request.param)


def make_group_scheduler(scheduler_cls, specs, num_blocks, num_spec_tokens, *, mamba_mode="none"):
    config = create_vllm_config(
        block_size=specs[-1].block_size,
        speculative_method="ngram",
        num_speculative_tokens=num_spec_tokens,
    )
    config.kv_transfer_config = None
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.num_gpu_blocks = num_blocks
    config.cache_config.mamba_cache_mode = mamba_mode
    if any(isinstance(spec, MambaSpec) for spec in specs):
        config.cache_config.enable_prefix_caching = mamba_mode == "align"
        config.model_config.max_model_len = 128
        config.scheduler_config.max_model_len = 128
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer-{i}"], spec) for i, spec in enumerate(specs)],
    )
    scheduler = scheduler_cls(
        vllm_config=config,
        kv_cache_config=kv_config,
        block_size=config.cache_config.block_size,
        hash_block_size=16,
        structured_output_manager=StructuredOutputManager(config),
        log_stats=True,
    )
    if isinstance(scheduler, BalanceScheduler):
        scheduler._balance_enabled = True
        scheduler.balance_queue = [torch.tensor([0], dtype=torch.int)]
    connector = MagicMock(spec=HybridConnector)
    connector.get_num_new_matched_tokens.side_effect = lambda req, local: (req.num_tokens - local, True)
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.has_pending_block_frees.return_value = False
    connector.get_external_cache_hit_sources.return_value = None
    connector.get_kv_connector_stats.return_value = None
    connector.take_events.return_value = ()
    connector.request_finished.return_value = (False, None)
    connector.request_finished_all_groups = MagicMock(return_value=(False, None))
    scheduler.connector = connector
    scheduler._kv_fetch_stages = {}
    return scheduler


def start_loads(scheduler, prefetch):
    if prefetch:
        scheduler._lb_kv_prefetch_enabled = True
        scheduler.prepare_dyntra_lb_step()
    else:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens == 0


@pytest.mark.parametrize("scheduler_cls,prefetch", LOAD_PATHS)
@pytest.mark.parametrize(
    "swa,block_size,num_spec_tokens,num_blocks",
    [(False, 16, 3, 8), (True, 16, 3, 17), (True, 64, 31, 40)],
    ids=["single-group", "multiple-groups", "mixed-block-sizes"],
)
def test_async_load_keeps_promotion_margin(scheduler_cls, prefetch, swa, block_size, num_spec_tokens, num_blocks):
    # Based on upstream test_async_load_reserves_blocks_for_promotion_margin.
    full = FullAttentionSpec(block_size=block_size, num_kv_heads=4, head_size=16, dtype=torch.float16)
    specs = [full]
    if swa:
        specs.append(
            SlidingWindowSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=16,
                dtype=torch.float16,
                sliding_window=4096,
                page_size_padded=full.page_size_bytes,
            )
        )
    scheduler = make_group_scheduler(scheduler_cls, specs, num_blocks, num_spec_tokens)
    first, second = [create_request(i, num_tokens=block_size * n, max_tokens=1) for i, n in ((1, 4), (2, 3))]
    for req in (first, second):
        scheduler.add_request(req)
    assert scheduler.num_spec_tokens == num_spec_tokens > 0
    assert scheduler._spec_decode_step_blocks > 0

    start_loads(scheduler, prefetch)

    assert first.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert first in scheduler.kv_holding_waiting
    assert second.status == RequestStatus.WAITING
    assert not any(scheduler.kv_cache_manager.get_block_ids(second.request_id))
    assert scheduler._kv_fetch_stages[first] == KV_FETCH_IN_PROGRESS
    # Stop new admissions and prove the reserved space lets the holder advance.
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    scheduler.update_from_output(
        scheduler.schedule(), create_model_runner_output([], finished_recving={first.request_id})
    )
    output = scheduler.schedule()
    assert first.status == RequestStatus.RUNNING
    assert output.num_scheduled_tokens[first.request_id] > 0
    assert second.request_id not in output.num_scheduled_tokens


@pytest.mark.parametrize("scheduler_cls,prefetch", LOAD_PATHS)
@pytest.mark.parametrize("mode,mamba_block_size", [("none", 128), ("align", 16)])
def test_async_load_does_not_double_reserve_mamba(scheduler_cls, prefetch, mode, mamba_block_size):
    specs = [
        FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=16, dtype=torch.float16),
        MambaSpec(
            block_size=mamba_block_size,
            shapes=((16,), (16,)),
            dtypes=(torch.float16,),
            mamba_cache_mode=mode,
            num_speculative_blocks=3,
        ),
    ]
    scheduler = make_group_scheduler(scheduler_cls, specs, 18, 3, mamba_mode=mode)
    requests = [create_request(i, num_tokens=n, max_tokens=1) for i, n in ((1, 64), (2, 48))]
    for req in requests:
        scheduler.add_request(req)

    start_loads(scheduler, prefetch)

    assert scheduler._spec_decode_step_blocks == 1
    assert all(req.status == RequestStatus.WAITING_FOR_REMOTE_KVS for req in requests)
    scheduler.update_from_output(
        scheduler.schedule(), create_model_runner_output([], finished_recving={req.request_id for req in requests})
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[requests[0].request_id] > 0
    # Full-sequence admission may serialize the promotions; both must progress.
    scheduler.finish_requests(requests[0].request_id, RequestStatus.FINISHED_ABORTED)
    if requests[1].request_id not in output.num_scheduled_tokens:
        assert scheduler.schedule().num_scheduled_tokens[requests[1].request_id] > 0


@pytest.mark.parametrize("scheduler_cls,prefetch", LOAD_PATHS)
def test_async_load_waits_for_inflight_execution_fence(scheduler_cls, prefetch):
    scheduler = make_scheduler(scheduler_cls)
    request = create_request(1, num_tokens=48)
    scheduler.add_request(request)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    scheduler.defer_async_loads = True
    scheduler.sched_step_seq = 2
    scheduler.processed_step_seq = 1

    start_loads(scheduler, prefetch)

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.deferred_loads[request.request_id][:2] == (2, 32)
    scheduler.connector.update_state_after_alloc.assert_not_called()
    scheduler._drain_deferred_loads()
    scheduler.connector.update_state_after_alloc.assert_not_called()
    scheduler.processed_step_seq = 2
    scheduler._drain_deferred_loads()
    scheduler.connector.update_state_after_alloc.assert_called_once()
    assert not scheduler.deferred_loads


@pytest.mark.parametrize("scheduler_cls,prefetch", LOAD_PATHS)
def test_skip_external_kv_uses_native_request_policy(scheduler_cls, prefetch):
    scheduler = make_scheduler(scheduler_cls)
    request = create_request(1, num_tokens=48)
    # Rebuild the actual Request after changing its sampling policy.
    params = request.sampling_params.clone()
    params.skip_reading_prefix_cache = True
    request = Request(
        request_id="skip", prompt_token_ids=list(request.prompt_token_ids), sampling_params=params, pooling_params=None
    )
    scheduler.add_request(request)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    assert should_skip_external_kv(request)

    if prefetch:
        start_loads(scheduler, True)
        assert request.status == RequestStatus.WAITING
        assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))
        scheduler.connector.update_state_after_alloc.assert_not_called()
    else:
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {request.request_id: 48}
        assert not output.has_sync_kv_loads
    assert request not in scheduler.kv_holding_waiting


def test_external_kv_flag_compatibility_preserves_native_value():
    assert should_skip_external_kv(SimpleNamespace(skip_reading_prefix_cache=True))
    assert not should_skip_external_kv(SimpleNamespace(skip_reading_external_kv=False, skip_reading_prefix_cache=True))
    with pytest.raises(AttributeError):
        should_skip_external_kv(SimpleNamespace())


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("external_tokens", [4, 8])
def test_dyntra_prefetch_reconciles_partial_local_hit(prefetch, external_tokens):
    scheduler = make_scheduler(DYNTRA_SCHEDULERS[0])
    request = create_request(1, num_tokens=16)
    scheduler.add_request(request)
    empty = scheduler.kv_cache_manager.empty_kv_cache_blocks
    scheduler._get_local_prefix_cache_hit = MagicMock(return_value=(empty, 5, 0, False))
    scheduler.kv_cache_manager.truncate_computed_blocks = MagicMock(return_value=empty)
    scheduler.connector.get_num_new_matched_tokens.return_value = (external_tokens, True)

    if prefetch:
        start_loads(scheduler, True)
    else:
        scheduler.schedule()

    scheduler.connector.get_num_new_matched_tokens.assert_called_once_with(request, 0)
    if external_tokens > 5:
        scheduler.kv_cache_manager.truncate_computed_blocks.assert_called_once_with(empty, 0)
        assert request.num_computed_tokens == 8
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    else:
        scheduler.kv_cache_manager.truncate_computed_blocks.assert_not_called()
        assert request.status != RequestStatus.WAITING_FOR_REMOTE_KVS


def test_sync_load_flag_is_per_step(scheduler):
    scheduler.connector.get_num_new_matched_tokens.return_value = (16, False)
    request = create_request(1, num_tokens=32)
    scheduler.add_request(request)
    assert step(scheduler).has_sync_kv_loads
    assert not step(scheduler).has_sync_kv_loads


def test_streaming_sessions_hold_admission_capacity(scheduler):
    scheduler.num_waiting_for_streaming_input = scheduler.max_num_active_reqs
    request = create_request(1, num_tokens=16)
    scheduler.add_request(request)
    assert not scheduler.schedule().num_scheduled_tokens
    assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))


def test_unscheduled_running_lora_still_counts(scheduler):
    first = create_request(1, num_tokens=16)
    first.lora_request = SimpleNamespace(lora_int_id=1)
    scheduler.lora_config = SimpleNamespace(max_loras=1)
    scheduler.add_request(first)
    step(scheduler)
    first.next_decode_eligible_step = scheduler.current_step + 10
    second = create_request(2, num_tokens=16)
    second.lora_request = SimpleNamespace(lora_int_id=2)
    scheduler.add_request(second)

    assert not scheduler.schedule().num_scheduled_tokens
    assert second in scheduler.waiting
    assert not any(scheduler.kv_cache_manager.get_block_ids(second.request_id))


@pytest.mark.parametrize("scheduler_cls", [ProfilingChunkScheduler, ProfilingChunkAsyncScheduler])
def test_profiling_cap_precedes_mamba_and_prefill_lookahead(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    request = create_request(1, num_tokens=128)
    scheduler.max_num_scheduled_tokens = 32
    scheduler.add_request(request)
    output = scheduler.schedule()
    runner_output = create_model_runner_output([request])
    runner_output.sampled_token_ids = [[]]
    scheduler.update_from_output(output, runner_output)
    scheduler.profiling_chunk_manager = SimpleNamespace(
        is_ready=True, predictor=SimpleNamespace(target_latency=1.0), predict_chunk_size=MagicMock(return_value=23)
    )
    scheduler.need_mamba_block_aligned_split = True
    align = MagicMock(side_effect=lambda req, n, *args: n // 16 * 16)
    scheduler._mamba_block_aligned_split = align
    with patch.object(scheduler, "_reserve_prefill_lookahead", wraps=scheduler._reserve_prefill_lookahead) as reserve:
        output = scheduler.schedule()
    assert align.call_args.args[1] == 23
    assert reserve.call_args.args[2] == 16
    assert output.num_scheduled_tokens == {request.request_id: 16}


@pytest.mark.parametrize("v2", [False, True])
def test_resumed_request_uses_native_model_runner_contract(scheduler, v2):
    request = create_request(1, num_tokens=16)
    scheduler.add_request(request)
    step(scheduler)
    scheduler.running.remove(request)
    request.status = RequestStatus.PREEMPTED
    scheduler.kv_holding_waiting.prepend_request(request)
    scheduler.use_v2_model_runner = v2

    output = scheduler.schedule()

    assert request.request_id in output.num_scheduled_tokens
    if v2:
        assert [req.req_id for req in output.scheduled_new_reqs] == [request.request_id]
        assert request.request_id not in output.scheduled_cached_reqs.req_ids
    else:
        assert not output.scheduled_new_reqs
        assert request.request_id in output.scheduled_cached_reqs.req_ids
        assert scheduler.prev_step_scheduled_req_ids == {request.request_id}


@pytest.mark.parametrize("scheduler_cls", SCHEDULERS, ids=lambda cls: cls.__name__)
def test_encoder_blocked_waiter_does_not_block_text_request(scheduler_cls):
    scheduler = _make_mm_recompute_scheduler(scheduler_cls)
    blocked = make_mm_request("mm", [("image", [True, True])])
    following = create_request(2, num_tokens=16)
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.add_request(blocked)
    scheduler.add_request(following)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {following.request_id: 16}
    assert blocked in scheduler.waiting
    assert not any(scheduler.kv_cache_manager.get_block_ids(blocked.request_id))


@pytest.mark.parametrize("scheduler_cls", SCHEDULERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("resumed", [False, True])
def test_encoder_readiness_blocks_running_and_resumed_requests(scheduler_cls, resumed):
    scheduler = _make_mm_recompute_scheduler(scheduler_cls)
    request = make_mm_request("mm", [("image", [True, True])])
    request.max_tokens = 3
    scheduler.add_request(request)
    step(scheduler)
    if resumed:
        scheduler.running.remove(request)
        request.status = RequestStatus.PREEMPTED
        scheduler.kv_holding_waiting.prepend_request(request)
    scheduler.ec_connector = MagicMock()
    scheduler.ec_connector.ensure_cache_available.return_value = False

    output = scheduler.schedule()

    assert not output.num_scheduled_tokens
    scheduler.ec_connector.ensure_cache_available.assert_called_once()


@pytest.mark.parametrize("scheduler_cls,prefetch", LOAD_PATHS)
@pytest.mark.parametrize("window_loaded", [False, True])
def test_async_prefix_replay_respects_loaded_window(scheduler_cls, prefetch, window_loaded):
    scheduler = make_scheduler(scheduler_cls)
    scheduler.prefix_replay_tokens = 16
    scheduler.prefix_replay_group_ids = (0,)
    request = create_request(1, num_tokens=48)
    scheduler.add_request(request)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,) if window_loaded else ()

    start_loads(scheduler, prefetch)
    scheduler.update_from_output(
        scheduler.schedule(), create_model_runner_output([], finished_recving={request.request_id})
    )
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == (16 if window_loaded else 32)
    assert request.replay_start == (0 if window_loaded else 16)


@pytest.mark.parametrize("clip_alignment", [False, True])
def test_spec_padding_respects_model_limit_and_alignment(scheduler, clip_alignment):
    scheduler.num_spec_tokens = 3
    scheduler.num_lookahead_tokens = 3
    scheduler.dynamic_sd_lookup = None
    # One token remains after a synchronous external cache hit.
    request = create_request(1, num_tokens=16)
    scheduler.connector.get_num_new_matched_tokens.return_value = (15, False)
    scheduler.add_request(request)
    if clip_alignment:
        scheduler.need_mamba_block_aligned_split = True
        scheduler._mamba_block_aligned_split = MagicMock(return_value=2)
    else:
        # Four padded rows would fit, but there is no room for the sampled token.
        scheduler.max_model_len = 19

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 1}
    assert not output.scheduled_spec_decode_tokens


@pytest.mark.parametrize("max_tokens,expected_skip", [(1, True), (3, False)])
def test_drafts_skipped_only_when_next_step_cannot_use_them(scheduler, max_tokens, expected_skip):
    scheduler.can_skip_drafts = True
    request = create_request(1, num_tokens=16, max_tokens=max_tokens)
    scheduler.add_request(request)
    assert scheduler.schedule().skip_drafts is expected_skip


def test_prompt_scoring_preserves_installed_native_cache_policy(scheduler):
    original = create_request(1, num_tokens=48)
    params = original.sampling_params.clone()
    params.prompt_logprobs = 1
    request = Request(
        request_id="score",
        prompt_token_ids=list(original.prompt_token_ids),
        sampling_params=params,
        pooling_params=None,
    )
    scheduler.add_request(request)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    native_skip = (
        request.skip_reading_external_kv
        if hasattr(request, "skip_reading_external_kv")
        else request.skip_reading_prefix_cache
    )
    assert should_skip_external_kv(request) == native_skip

    output = scheduler.schedule()

    if native_skip:
        assert output.num_scheduled_tokens == {request.request_id: 48}
    else:
        assert not output.num_scheduled_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_preempting_unscheduled_earlier_victim_keeps_following_requests(scheduler):
    victim, retried, following = [create_request(i, num_tokens=16) for i in range(1, 4)]
    for request in (victim, retried, following):
        scheduler.add_request(request)
    step(scheduler)
    scheduler.policy = SchedulingPolicy.PRIORITY
    victim.priority = 10
    victim.next_decode_eligible_step = scheduler.current_step + 10
    scheduler.connector.update_state_before_preempt = MagicMock(return_value=True)
    allocate = scheduler.kv_cache_manager.allocate_slots
    failed = False

    def fail_once(request, *args, **kwargs):
        nonlocal failed
        if request is retried and not failed:
            failed = True
            return None
        return allocate(request, *args, **kwargs)

    with patch.object(scheduler.kv_cache_manager, "allocate_slots", side_effect=fail_once):
        output = scheduler.schedule()

    assert failed
    assert victim.status == RequestStatus.PREEMPTED
    assert set(output.num_scheduled_tokens) == {retried.request_id, following.request_id}


@pytest.mark.parametrize("scheduler_cls", SCHEDULERS, ids=lambda cls: cls.__name__)
def test_v2_encoder_payload_is_released_after_consumption(scheduler_cls):
    scheduler = _make_mm_recompute_scheduler(scheduler_cls)
    scheduler.use_v2_model_runner = True
    request = make_mm_request("mm", [("image", [True, True])])
    request.max_tokens = 3
    request.mm_features[0].data = MultiModalKwargsItem({})
    scheduler.add_request(request)
    first = step(scheduler)
    assert first.scheduled_encoder_inputs == {request.request_id: [0]}
    assert scheduler._pending_encoder_input_ids[request.request_id] == {0}

    output = scheduler.schedule()

    assert output.encoder_input_updates == {request.request_id: {0: None}}
    assert not scheduler._pending_encoder_input_ids[request.request_id]


@pytest.mark.parametrize("scheduler_cls,prefetch", LOAD_PATHS)
def test_aux_connector_caps_async_hit_and_receives_metadata(scheduler_cls, prefetch):
    scheduler = make_scheduler(scheduler_cls)
    request = create_request(1, num_tokens=48)
    scheduler.add_request(request)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    aux = MagicMock()
    aux.max_cache_hit.return_value = 16
    aux.cap_external_hit.return_value = 16
    metadata = object()
    aux.build_connector_meta.return_value = metadata
    scheduler.aux_output_connector = aux

    start_loads(scheduler, prefetch)

    assert request.num_computed_tokens == 16
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    aux.max_cache_hit.assert_called_once_with(request)
    aux.cap_external_hit.assert_called_once_with(request, 0, 32)
    aux.request_loading.assert_called_once_with(request, 16)
    assert scheduler.schedule().aux_output_connector_metadata is metadata
