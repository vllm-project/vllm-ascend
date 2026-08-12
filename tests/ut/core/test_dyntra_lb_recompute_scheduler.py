# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.v1.request import RequestStatus

import vllm_ascend.core.recompute_scheduler as recompute_scheduler_module
from tests.ut.core.test_dyntra_lb_scheduler import (
    create_dyntra_lb_scheduler,
    make_dyntra_test_config,
)
from tests.ut.kv_offload.utils import (
    create_model_runner_output,
    create_request,
)
from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBPolicyMixin
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    DyntraLBRecomputeScheduler,
)


def _create_dyntra_lb_recompute_scheduler():
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    vllm_config.additional_config = {
        "scheduler_config": {
            "dyntra_lb_config": {
                "enabled": True,
            }
        }
    }
    return create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBRecomputeScheduler,
    )


def test_dyntra_lb_recompute_schedulers_use_policy_mixin():
    assert issubclass(DyntraLBRecomputeScheduler, DyntraLBPolicyMixin)
    assert issubclass(AsyncDyntraLBRecomputeScheduler, DyntraLBPolicyMixin)
    assert (
        DyntraLBRecomputeScheduler._apply_load_balance_modifications
        is DyntraLBPolicyMixin._apply_load_balance_modifications
    )
    assert AsyncDyntraLBRecomputeScheduler._can_admit_waiting_request is DyntraLBPolicyMixin._can_admit_waiting_request


def test_dyntra_lb_recompute_invokes_policy_hooks():
    scheduler = _create_dyntra_lb_recompute_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler._apply_load_balance_modifications = MagicMock()
    scheduler._can_admit_waiting_request = MagicMock(return_value=False)

    scheduler_output = scheduler.schedule()

    scheduler._apply_load_balance_modifications.assert_called_once_with()
    scheduler._can_admit_waiting_request.assert_called_once_with(request)
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.running
    assert request.request_id not in scheduler_output.num_scheduled_tokens


def test_dyntra_lb_recompute_emits_scheduler_diagnostics(monkeypatch):
    scheduler = _create_dyntra_lb_recompute_scheduler()
    summaries = []
    monkeypatch.setattr(
        recompute_scheduler_module,
        "diagnostics_enabled",
        lambda _config: True,
    )
    monkeypatch.setattr(
        recompute_scheduler_module,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    scheduler.schedule()

    assert len(summaries) == 1


def test_dyntra_lb_recompute_prefetch_waits_for_offload_store():
    scheduler = _create_dyntra_lb_recompute_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    request.status = RequestStatus.PREEMPTED

    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (None, False)
    scheduler._lb_kv_prefetch_enabled = True

    candidates = scheduler.prepare_dyntra_lb_step()

    assert candidates == []
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting
    assert request not in scheduler.skipped_waiting
    scheduler.connector.update_state_after_alloc.assert_not_called()


def test_dyntra_lb_recompute_prefetch_restores_ready_cpu_kv():
    scheduler = _create_dyntra_lb_recompute_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    request.status = RequestStatus.PREEMPTED

    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (8, True)
    scheduler._lb_kv_prefetch_enabled = True

    candidates = scheduler.prepare_dyntra_lb_step()

    assert candidates == []
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 8
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.waiting
    assert request in scheduler._inflight_prefills
    scheduler.connector.update_state_after_alloc.assert_called_once()
    update_request, allocated_blocks, external_tokens = scheduler.connector.update_state_after_alloc.call_args.args
    assert update_request is request
    assert allocated_blocks.get_block_ids()
    assert external_tokens == 8


def test_dyntra_lb_pause_with_retained_kv_skips_prefetch():
    scheduler = _create_dyntra_lb_recompute_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output([request]),
    )
    retained_block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)

    scheduler.running.remove(request)
    scheduler._lb_pause_request(request, 0.0)
    scheduler.connector = MagicMock()
    scheduler._lb_kv_prefetch_enabled = True

    candidates = scheduler.prepare_dyntra_lb_step()

    assert candidates == [request]
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens > 0
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id) == (retained_block_ids)
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()
