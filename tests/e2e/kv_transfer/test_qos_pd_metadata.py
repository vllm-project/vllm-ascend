# SPDX-License-Identifier: Apache-2.0
"""Installed real Request/scheduler metadata test. No model or NPU I/O."""

import msgspec
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import KvQosPolicy


def test_real_request_p_handoff_d_metadata():
    policy = KvQosPolicy.from_config({"priority_to_qos": {"0": 0, "2": 7}})
    entry = CompletionRequest(
        model="metadata-fixture",
        prompt=[1] * 256,
        max_tokens=1,
        kv_transfer_params={"kv_priority": 2, "do_remote_decode": True},
    )
    p_request = Request("P", [1] * 256, entry.to_sampling_params(1), None)
    p_request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    p_request.append_output_token_ids(3)
    p = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    p.qos_policy = policy
    p.block_size, p.engine_id, p.side_channel_host, p.side_channel_port = 128, "P-engine", "localhost", 9000
    p.pcp_size = p.dcp_size = p.tp_size = 1
    p.multi_nodes_meta_mapping, p._reqs_need_send = {}, {}
    # KV allocator geometry is a fixture; request/connector methods are real.
    p._get_transfer_block_ids = lambda ids, n: ids
    p._get_swa_transfer_block_ids = lambda ids: ids
    delay, handoff = p.request_finished(p_request, ([0, 1],))
    assert delay and handoff["kv_priority"] == 2 and handoff["remote_request_id"] == "P"
    d_entry = CompletionRequest(model="metadata-fixture", prompt=[1] * 256, max_tokens=8, kv_transfer_params=handoff)
    d_request = Request("D", [1] * 256, d_entry.to_sampling_params(8), None)
    d = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    d.qos_policy = policy
    d._reqs_need_recv = {"D": (d_request, ([2, 3],), ([2, 3],), 256)}
    d._reqs_need_send, d._reqs_in_batch = {}, set()
    metadata = d.build_connector_meta(None)
    requests = msgspec.msgpack.decode(msgspec.msgpack.encode(metadata.requests))
    assert requests["D"]["kv_priority"] == 2
    assert policy.select(requests["D"]["kv_priority"]) == 7
