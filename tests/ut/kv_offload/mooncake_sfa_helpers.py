# SPDX-License-Identifier: Apache-2.0
"""Reuse generic request fixtures with the specialized scheduler/receiver."""

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.scheduler import MooncakeSFAConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import MooncakeSFAPullRecvingThread

from .mooncake_v2.test_pull_scheduler import make_pull_scheduler as make_base_scheduler
from .mooncake_v2.test_pull_worker import make_thread as make_base_thread


def make_thread(**overrides):
    return make_base_thread(thread_cls=MooncakeSFAPullRecvingThread, **overrides)


def make_pull_scheduler():
    scheduler = make_base_scheduler(scheduler_cls=MooncakeSFAConnectorScheduler)
    scheduler._copy_sfa_slot_allocator = None
    scheduler._copy_sfa_deferred_releases = set()
    return scheduler
