# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reservation inheritance must compose with either upstream SWA baseline."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.ut.core.test_scheduler_kv_reservation import _add_remote_requests, _make_scheduler
from tests.ut.kv_offload.utils import create_model_runner_output
from vllm_ascend.core.scheduler import AscendAsyncScheduler
from vllm_ascend.core.scheduler_kv_reservation import initialize_spec_reservation, spec_step_blocks


@pytest.mark.parametrize("num_spec_tokens,lookahead,expected", [(0, 0, 0), (31, 0, 3), (31, 31, 5)])
def test_margin_uses_effective_manager_block_sizes_and_excludes_mamba(num_spec_tokens, lookahead, expected):
    # The manager's allocation unit can exceed the spec's block size under
    # context parallelism. Mamba accounts for speculative states separately.
    attention = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float16,),
        num_speculative_blocks=31,
    )
    managers = [
        SimpleNamespace(block_size=64, kv_cache_spec=attention),
        SimpleNamespace(block_size=16, kv_cache_spec=attention),
        SimpleNamespace(block_size=16, kv_cache_spec=mamba),
    ]
    scheduler = SimpleNamespace(
        num_spec_tokens=num_spec_tokens,
        num_lookahead_tokens=lookahead,
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=managers)),
    )
    initialize_spec_reservation(scheduler)
    assert spec_step_blocks(scheduler) == expected


def test_async_scheduler_keeps_upstream_policy_with_balance_disabled():
    scheduler = _make_scheduler(AscendAsyncScheduler, speculative_method="mtp")
    assert not scheduler._balance_enabled
    request, _ = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={request.request_id}))

    output = scheduler.schedule()
    # The upstream async policy pads the first decode for speculative tokens.
    assert output.num_scheduled_tokens[request.request_id] == 1 + scheduler.num_spec_tokens
    assert request.num_output_placeholders > 0


@pytest.mark.parametrize("swa_install_order", ["before_import", "before_init", "after_init"])
@pytest.mark.parametrize("async_import_order", ["early", "late"])
def test_inherited_reservation_and_swa_in_fresh_process(swa_install_order, async_import_order):
    # Process isolation observes the original upstream class before any global
    # patches and also catches late SWA installation on existing instances.
    repo = Path(__file__).resolve().parents[3]
    script = """
import runpy
from types import SimpleNamespace
from unittest.mock import MagicMock
import torch
from vllm.triton_utils import HAS_TRITON, triton
if HAS_TRITON:
    triton.runtime.driver.active = MagicMock()
    triton.runtime.driver.active.get_current_target.return_value = SimpleNamespace(backend="cpu")
from vllm.v1.core.sched.scheduler import Scheduler as OriginalScheduler
if ASYNC_IMPORT_ORDER == "early":
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

def methods():
    return (
        OriginalScheduler.__init__, OriginalScheduler.schedule,
        OriginalScheduler._request_remaining_blocks,
        OriginalScheduler._inflight_prefill_reserved_blocks,
        getattr(OriginalScheduler, "_spec_decode_step_blocks", None),
    )

original = methods()
replay_config = SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True))
if INSTALL_ORDER == "before_import":
    # Importing the platform package also imports Balance and its Ascend base.
    # Load this existing standalone patch first to exercise the opposite order.
    legacy = runpy.run_path(SWA_PATH)
    if not hasattr(OriginalScheduler, "_mark_prefix_replay"):
        # The public installer imports the platform payload patch afterward,
        # which would eagerly import Balance. Install its real scheduler/cache
        # components first; normal platform setup below installs the payload.
        legacy["_patch_scheduler"]()
        legacy["_patch_cache"]()
before_import = methods()
# Install the normal CPU hardware boundaries before loading platform classes.
runpy.run_path(CONFTEST_PATH)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm_ascend.core.scheduler import AscendScheduler, AscendAsyncScheduler
from vllm_ascend.core.scheduler_kv_reservation import SpecDecodeReservationMixin
assert methods() == before_import
assert AsyncScheduler in AscendAsyncScheduler.__mro__

from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import apply_legacy_replay_patch
from tests.ut.core.test_scheduler_kv_reservation import (
    _make_scheduler, _assert_async_load_pressure, _assert_first_async_admission,
)
from tests.ut.kv_offload.utils import create_request
from tests.ut.patch.platform.test_patch_swa_bounded_replay_legacy import (
    replay_scheduler,
    test_actual_scheduler_replays_across_chunks_and_preserves_shared_blocks,
)
from vllm.v1.request import RequestStatus
assert methods() == before_import
assert methods()[2:] == original[2:]
classes = (AscendScheduler, AscendAsyncScheduler, BalanceScheduler)
for cls in classes:
    assert cls.__mro__.count(SpecDecodeReservationMixin) == 1
assert AscendAsyncScheduler._update_after_schedule is AsyncScheduler._update_after_schedule
assert AscendAsyncScheduler._update_request_with_output is AsyncScheduler._update_request_with_output

if INSTALL_ORDER == "before_init":
    apply_legacy_replay_patch(replay_config)
before_init = methods()
schedulers = [
    (_make_scheduler(cls, num_blocks=blocks, bounded_replay=replay), blocks)
    for cls in classes
    for replay in (False, True)
    for blocks in (23, 24, 38)
]
assert methods() == before_init

# A raw upstream instance must retain its original helper semantics. Installing
# or constructing an Ascend subclass cannot fix another backend by mutation.
raw = _make_scheduler(OriginalScheduler)
assert not hasattr(raw, "_ascend_spec_step_blocks")
assert raw._inflight_prefill_reserved_blocks() == 0
if original[-1] is not None:
    assert raw._spec_decode_step_blocks() == 1
assert methods() == before_init

balance_schedulers = [
    (_make_scheduler(cls, balance_enabled=enabled, num_speculative_tokens=0), enabled)
    for cls in (AscendScheduler, AscendAsyncScheduler)
    for enabled in (False, True)
]
# Use a real replay window as well as the zero-window admission cases below.
# In the after_init case, both construction and import precede SWA installation.
replay_fixture = replay_scheduler.__wrapped__(SimpleNamespace(param=(False, AscendAsyncScheduler)))
replay_instance = next(replay_fixture)

if INSTALL_ORDER == "after_init":
    apply_legacy_replay_patch(replay_config)
installed = methods()
apply_legacy_replay_patch(replay_config)
assert methods() == installed
assert methods()[2:] == original[2:]

for scheduler, blocks in schedulers:
    # Plain SlidingWindowSpec has no replay window, so a connector is valid
    # with either flag value. Both SWA wrapper branches must retain reservation.
    if blocks == 38:
        _assert_async_load_pressure(scheduler)
    else:
        _assert_first_async_admission(scheduler, blocks == 24)
    assert methods() == installed

# A full remote DP rank must prevent local admission only while Balance is
# enabled. The default async subclass must retain the same gate as sync.
for scheduler, balance_enabled in balance_schedulers:
    scheduler.balance_queue = [torch.tensor([scheduler.max_num_running_reqs]), torch.tensor([0])]
    request = create_request(request_id=1, num_tokens=16, block_size=16)
    scheduler.connector.matched_tokens[request.request_id] = 0
    scheduler.connector.load_async = False
    scheduler.add_request(request)
    output = scheduler.schedule()
    if balance_enabled:
        assert request.status == RequestStatus.WAITING, (type(scheduler).__name__, request.status)
        assert not output.num_scheduled_tokens
        assert not scheduler.running
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
            assert not manager.req_to_blocks.get(request.request_id)
        scheduler.balance_queue = [torch.tensor([0]), torch.tensor([0])]
        output = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert output.num_scheduled_tokens == {request.request_id: 16}

# Exercise the actual 32-token replay of a 96-token cached prefix, shared block
# ownership, and preempt/resume payload after every supported install order.
try:
    test_actual_scheduler_replays_across_chunks_and_preserves_shared_blocks(replay_instance, 96)
finally:
    replay_fixture.close()
"""
    script = (
        script.replace("INSTALL_ORDER", repr(swa_install_order))
        .replace("ASYNC_IMPORT_ORDER", repr(async_import_order))
        .replace("SWA_PATH", repr(str(repo / "vllm_ascend/patch/platform/patch_swa_bounded_replay_legacy.py")))
        .replace("CONFTEST_PATH", repr(str(repo / "tests/ut/conftest.py")))
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
