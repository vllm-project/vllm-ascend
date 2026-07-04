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


def _make_connector_with_children(*children: MagicMock) -> AscendMultiConnector:
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = list(children)
    return connector


def test_logical_rank_handshake_metadata_uses_child_set_xfer() -> None:
    child_a = MagicMock()
    child_b = MagicMock()
    connector = _make_connector_with_children(child_a, child_b)
    metadata: dict[int | tuple[int, int], Any] = {
        0: MagicMock(),
        2: MagicMock(),
    }

    connector.set_xfer_handshake_metadata_pp_aware(metadata)

    child_a.set_xfer_handshake_metadata.assert_called_once_with(metadata)
    child_b.set_xfer_handshake_metadata.assert_called_once_with(metadata)
    child_a.set_xfer_handshake_metadata_pp_aware.assert_not_called()
    child_b.set_xfer_handshake_metadata_pp_aware.assert_not_called()


def test_pp_tp_handshake_metadata_keeps_parent_behavior() -> None:
    child = MagicMock()
    connector = _make_connector_with_children(child)
    metadata: dict[int | tuple[int, int], Any] = {(0, 0): MagicMock()}

    with patch.object(MultiConnector, "set_xfer_handshake_metadata_pp_aware", autospec=True) as mock_parent:
        connector.set_xfer_handshake_metadata_pp_aware(metadata)

    mock_parent.assert_called_once_with(connector, metadata)
    child.set_xfer_handshake_metadata.assert_not_called()


def test_mixed_handshake_metadata_keys_raise() -> None:
    connector = _make_connector_with_children(MagicMock())
    metadata: dict[int | tuple[int, int], Any] = {
        0: MagicMock(),
        (0, 1): MagicMock(),
    }

    with pytest.raises(ValueError, match="Mixed int logical-rank"):
        connector.set_xfer_handshake_metadata_pp_aware(metadata)
