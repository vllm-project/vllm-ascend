#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Any
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeAgentMetadata,
    MooncakeConnector,
    MooncakeConnectorScheduler,
)


def _make_mooncake_metadata(engine_id: str, logical_rank: int, handshake_port: int) -> MooncakeAgentMetadata:
    return MooncakeAgentMetadata(
        engine_id=engine_id,
        te_rpc_port=10000 + logical_rank,
        kv_group2layeridx={},
        block_size=16,
        kv_caches_base_addr=[],
        block_size_scale=[],
        num_blocks=0,
        block_lens=[],
        block_strides=[],
        local_ip=f"10.0.0.{logical_rank + 1}",
        handshake_port=handshake_port,
        logical_rank=logical_rank,
    )


def test_multi_connector_preserves_logical_rank_handshake_metadata_for_mooncake() -> None:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.multi_nodes_meta_mapping = {}

    mooncake_connector = MooncakeConnector.__new__(MooncakeConnector)
    mooncake_connector.connector_scheduler = scheduler

    non_mooncake_connector = MagicMock()
    multi_connector = AscendMultiConnector.__new__(AscendMultiConnector)
    multi_connector._connectors = [mooncake_connector, non_mooncake_connector]

    metadata: dict[int | tuple[int, int], Any] = {
        0: _make_mooncake_metadata("prefill-0", logical_rank=0, handshake_port=45100),
        4: _make_mooncake_metadata("prefill-4", logical_rank=4, handshake_port=45104),
    }

    multi_connector.set_xfer_handshake_metadata_pp_aware(metadata)

    assert scheduler.multi_nodes_meta_mapping == {
        "0": {
            "host": "10.0.0.1",
            "engine_id": "prefill-0",
            "handshake_port": 45100,
        },
        "4": {
            "host": "10.0.0.5",
            "engine_id": "prefill-4",
            "handshake_port": 45104,
        },
    }
    non_mooncake_connector.set_xfer_handshake_metadata.assert_not_called()
    non_mooncake_connector.set_xfer_handshake_metadata_pp_aware.assert_not_called()
