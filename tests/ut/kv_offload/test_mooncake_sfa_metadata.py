# SPDX-License-Identifier: Apache-2.0
"""SFA metadata conversion preserves the public builder's state transitions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import scheduler as module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import (
    MooncakeSFAConnectorMetadata,
    MooncakeSFAReqMeta,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaTopkSlotAllocator

from .mooncake_v2.helpers import make_blocks
from .mooncake_v2.test_pull_scheduler import heartbeat_request
from .mooncake_v2.test_pull_scheduler import make_pull_scheduler as make_base_scheduler
from .mooncake_v2.test_pull_worker import make_req_meta


def make_pull_scheduler():
    scheduler = make_base_scheduler(scheduler_cls=module.MooncakeSFAConnectorScheduler)
    scheduler._copy_sfa_slot_allocator = None
    scheduler._copy_sfa_deferred_releases = set()
    return scheduler


def test_conversion_retains_public_fields_and_snapshots_slot_zero():
    public = MooncakeConnectorMetadata()
    public.requests = {"request": make_req_meta()}
    public.reqs_in_batch = {"request"}
    public.requests_to_send = {"producer": 1.0}
    public.future_field = object()
    bindings = {"request": 0}
    converted = MooncakeSFAConnectorMetadata.from_mooncake(public, bindings)
    bindings.clear()
    assert converted.copy_sfa_slots == {"request": 0}
    assert isinstance(converted.requests["request"], MooncakeSFAReqMeta)
    assert converted.requests["request"].copy_sfa_slot == 0
    assert converted.requests["request"].local_block_ids is public.requests["request"].local_block_ids
    assert converted.reqs_in_batch == {"request"}
    assert converted.requests_to_send == {"producer": 1.0}
    assert converted.future_field is public.future_field
    assert not isinstance(public.requests["request"], MooncakeSFAReqMeta)


def test_builder_consumes_public_queues_once_and_returns_extended_requests():
    scheduler = make_pull_scheduler()
    scheduler._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(1)
    request = heartbeat_request(request_id="request", remote_num_prompt_tokens=32)
    scheduler.update_state_after_alloc(request, make_blocks(), 32)
    scheduler._reqs_in_batch = {"request"}
    metadata = scheduler.build_connector_meta(MagicMock())
    assert metadata.copy_sfa_slots == {"request": 0}
    assert metadata.requests["request"].copy_sfa_slot == 0
    assert metadata.reqs_in_batch == {"request"}
    assert scheduler._reqs_need_recv == {}
    assert scheduler._reqs_in_batch == set()
    assert scheduler.build_connector_meta(MagicMock()).requests == {}


def test_missing_reservation_fails_before_the_public_builder_consumes_requests():
    scheduler = make_pull_scheduler()
    scheduler._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(1)
    request = heartbeat_request(request_id="request", remote_num_prompt_tokens=32)
    scheduler.update_state_after_alloc(request, make_blocks(), 32)
    scheduler._copy_sfa_slot_allocator.release("request")
    pending = dict(scheduler._reqs_need_recv)
    with pytest.raises(ValueError, match="no reserved slots"):
        scheduler.build_connector_meta(MagicMock())
    assert scheduler._reqs_need_recv == pending


def test_sfa_policy_is_ready_before_parent_constructor_starts_background_work(monkeypatch):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
    )
    monkeypatch.setattr(
        module,
        "init_ascend_config",
        lambda _: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(use_fused_copy_sfa=True)),
    )
    constructed = []

    def initialize_parent(self, *_args):
        assert self._copy_sfa_slot_allocator.can_bind("request")
        assert self._copy_sfa_deferred_releases == set()
        constructed.append(self)

    monkeypatch.setattr(MooncakePullConnectorScheduler, "__init__", initialize_parent)
    scheduler = module.MooncakeSFAConnectorScheduler(config, "engine", object())
    assert constructed == [scheduler]
