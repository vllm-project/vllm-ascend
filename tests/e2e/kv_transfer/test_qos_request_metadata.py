# SPDX-License-Identifier: Apache-2.0
"""Installed-stack metadata integration; no model/NPU transfer is exercised here."""

from types import SimpleNamespace

import msgspec
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import KvQosPolicy


def test_openai_request_to_connector_metadata():
    entry = CompletionRequest(
        model="code-fixture", prompt=[1, 2, 3], max_tokens=1, kv_transfer_params={"kv_priority": 2}
    )
    req = Request("r", [1, 2, 3], entry.to_sampling_params(1), None)
    scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
    scheduler.qos_policy = KvQosPolicy.from_config({"priority_to_qos": {"0": 0, "2": 7}})
    scheduler.kv_role = "kv_both"
    scheduler.consumer_is_to_put = False
    scheduler._unfinished_requests = {"r": (req, [[0]])}
    scheduler._unfinished_request_ids = {"r"}
    scheduler._request_trackers = {}
    scheduler._preempted_req_ids = set()
    scheduler._loading_req_ids = set()
    scheduler._delayed_free_req_ids = set()
    # Block allocation is a fixture; priority propagation is the actual method.
    scheduler._process_new_request = lambda *args: ReqMeta(
        req_id="r", token_len_chunk=3, block_ids_by_group=[[0]], block_hashes=[b"a"], can_save=True, load_spec=None
    )
    scheduler.touch_sending_mamba_blocks = lambda _: None
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id="r")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )
    metadata = scheduler.build_connector_meta(output)
    requests = msgspec.msgpack.decode(msgspec.msgpack.encode(metadata.requests))
    assert requests[0]["kv_priority"] == 2
    assert scheduler.qos_policy.select(requests[0]["kv_priority"]) == 7
