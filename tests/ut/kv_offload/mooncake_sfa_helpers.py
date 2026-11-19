# SPDX-License-Identifier: Apache-2.0
"""Reuse generic request fixtures with the specialized scheduler/receiver."""

from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.scheduler import MooncakeSFAConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import MooncakeSFAPullRecvingThread

from .mooncake_v2.test_pull_scheduler import make_pull_scheduler as make_base_scheduler
from .mooncake_v2.test_pull_worker import make_thread as make_base_thread


def make_thread(**overrides):
    thread = make_base_thread(thread_cls=MooncakeSFAPullRecvingThread, **overrides)
    if thread.staging_reader is None:
        # Address/lifecycle tests use a synchronous reader without Host copies.
        def read(session, local, remote, lengths, request, **kwargs):
            return thread.engine.batch_transfer_sync_read(session, local, remote, lengths)

        thread.staging_reader = SimpleNamespace(batch_transfer_sync_read=read, drain_pending_landing=lambda: None)
    return thread


def make_pull_scheduler():
    scheduler = make_base_scheduler(scheduler_cls=MooncakeSFAConnectorScheduler)
    scheduler._copy_sfa_slot_allocator = None
    scheduler._copy_sfa_deferred_releases = set()
    return scheduler
