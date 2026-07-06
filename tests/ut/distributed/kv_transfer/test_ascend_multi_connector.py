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
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeAgentMetadata,
    MooncakeConnector,
)


def _make_connector_with_children(*children: Any) -> AscendMultiConnector:
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = list(children)
    return connector


def _make_mooncake_child() -> MooncakeConnector:
    child = MooncakeConnector.__new__(MooncakeConnector)
    child.set_xfer_handshake_metadata = MagicMock()  # type: ignore[method-assign]
    child.set_xfer_handshake_metadata_pp_aware = MagicMock()  # type: ignore[method-assign]
    return child


def _make_mooncake_metadata(engine_id: str) -> MooncakeAgentMetadata:
    return MooncakeAgentMetadata(
        engine_id=engine_id,
        te_rpc_port=10000,
        kv_group2layeridx={},
        block_size=16,
        kv_caches_base_addr=[],
        block_size_scale=[],
        num_blocks=0,
        block_lens=[],
        block_strides=[],
        local_ip="10.0.0.1",
        handshake_port=45100,
    )


def test_mooncake_tuple_handshake_metadata_only_routes_to_mooncake() -> None:
    mooncake_child = _make_mooncake_child()
    non_mooncake_child = MagicMock()
    connector = _make_connector_with_children(mooncake_child, non_mooncake_child)
    metadata: dict[int | tuple[int, ...], Any] = {
        (0, 0, 0): _make_mooncake_metadata("engine-0"),
        (0, 1, 0): _make_mooncake_metadata("engine-1"),
    }

    connector.set_xfer_handshake_metadata_pp_aware(metadata)

    mooncake_child.set_xfer_handshake_metadata.assert_not_called()  # type: ignore[attr-defined]
    mooncake_child.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(metadata)  # type: ignore[attr-defined]
    non_mooncake_child.set_xfer_handshake_metadata.assert_not_called()
    non_mooncake_child.set_xfer_handshake_metadata_pp_aware.assert_not_called()


def test_pp_tp_tuple_handshake_metadata_keeps_parent_behavior() -> None:
    child = MagicMock()
    connector = _make_connector_with_children(child)
    metadata: dict[int | tuple[int, ...], Any] = {(0, 0): MagicMock()}

    with patch.object(MultiConnector, "set_xfer_handshake_metadata_pp_aware", autospec=True) as mock_parent:
        connector.set_xfer_handshake_metadata_pp_aware(metadata)

    mock_parent.assert_called_once_with(connector, metadata)
    child.set_xfer_handshake_metadata.assert_not_called()


def test_mooncake_handshake_metadata_without_mooncake_raises() -> None:
    child = MagicMock()
    connector = _make_connector_with_children(child)
    metadata: dict[int | tuple[int, ...], Any] = {(0, 0, 0): _make_mooncake_metadata("engine-0")}

    with pytest.raises(ValueError, match="requires a MooncakeConnector"):
        connector.set_xfer_handshake_metadata_pp_aware(metadata)

    child.set_xfer_handshake_metadata.assert_not_called()


def test_mixed_handshake_metadata_keys_raise() -> None:
    connector = _make_connector_with_children(MagicMock())
    metadata: dict[int | tuple[int, ...], Any] = {
        0: MagicMock(),
        (0, 1): MagicMock(),
    }

    with pytest.raises(ValueError, match="Mixed int logical-rank"):
        connector.set_xfer_handshake_metadata_pp_aware(metadata)
