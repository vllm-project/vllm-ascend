# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import types

from tests.ut.kv_connector.utils import (create_request, create_scheduler,
                                         create_vllm_config)
from vllm_ascend.distributed.llmdatadist_c_mgr_connector import (
    LLMDataDistCMgrConnectorMetadata, LLMDataDistCMgrConnectorWorker, LLMRole)


def test_basic_inferface():
    """Unit test for basic LLMDataDistCMgrConnector interface functionality."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers LLMDataDistCMgrConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, LLMDataDistCMgrConnectorMetadata)

    assert len(kv_connector_metadata.requests) == 1
    assert request_id in kv_connector_metadata.requests
    req_meta = kv_connector_metadata.requests[request_id]

    for block_id, block in zip(
            req_meta.local_block_ids, scheduler.kv_cache_manager.coordinator.
            single_type_managers[0].req_to_blocks[request_id]):
        assert block_id == block.block_id


def test_read_agent_metadata():
    rank_table = {
        "version":
        "1.2",
        "server_count":
        "2",
        "prefill_device_list": [{
            "server_id": "192.168.1.1",
            "device_id": "0",
            "device_ip": "10.30.0.1",
            "cluster_id": "0",
        }, {
            "server_id": "192.168.1.1",
            "device_id": "1",
            "device_ip": "10.30.0.2",
            "cluster_id": "1",
        }, {
            "server_id": "192.168.1.2",
            "device_id": "0",
            "device_ip": "10.30.0.3",
            "cluster_id": "2",
        }, {
            "server_id": "192.168.1.2",
            "device_id": "1",
            "device_ip": "10.30.0.4",
            "cluster_id": "3",
        }]
    }

    def get_device_ip(worker_local_ip, worker_tp_rank, worker_visible_devices):
        old_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        worker = types.SimpleNamespace()
        worker.local_ip = worker_local_ip
        worker.tp_rank = worker_tp_rank
        worker.llm_datadist_role = LLMRole.PROMPT
        worker.pcp_rank = 0
        worker.tp_size = worker_tp_rank + 1
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = worker_visible_devices
        agent_metadata = LLMDataDistCMgrConnectorWorker.read_agent_metadata(
            worker, rank_table)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = old_visible_devices
        return agent_metadata.device_ip

    assert get_device_ip("192.168.1.1", 0, "0") == "10.30.0.1"
    assert get_device_ip("192.168.1.1", 0, "1") == "10.30.0.2"
    assert get_device_ip("192.168.1.2", 0, "0") == "10.30.0.3"
    assert get_device_ip("192.168.1.2", 0, "1") == "10.30.0.4"
    assert get_device_ip("192.168.1.1", 0, "0,1") == "10.30.0.1"
    assert get_device_ip("192.168.1.1", 1, "0,1") == "10.30.0.2"
    assert get_device_ip("192.168.1.1", 0, "") == "10.30.0.1"
    assert get_device_ip("192.168.1.1", 1, "") == "10.30.0.2"
