# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector


@pytest.mark.parametrize("chosen_connector", [None, 1])
def test_update_state_after_alloc_forwards_real_blocks(chosen_connector: int | None) -> None:
    connector: Any = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [MagicMock(), MagicMock()]
    connector._requests_to_connector = {}

    request = SimpleNamespace(request_id="request-1")
    blocks = MagicMock()
    num_external_tokens = 17
    if chosen_connector is not None:
        connector._requests_to_connector[request.request_id] = chosen_connector

    connector.update_state_after_alloc(request, blocks, num_external_tokens)

    for index, sub_connector in enumerate(connector._connectors):
        expected_external_tokens = num_external_tokens if index == chosen_connector else 0
        sub_connector.update_state_after_alloc.assert_called_once_with(request, blocks, expected_external_tokens)
    blocks.new_empty.assert_not_called()


def test_update_state_after_alloc_keeps_mooncake_layerwise_transfer() -> None:
    class FakeMooncakeLayerwiseConnector:
        def __init__(self) -> None:
            self.update_state_after_alloc = MagicMock()

    connector: Any = AscendMultiConnector.__new__(AscendMultiConnector)
    chosen = MagicMock()
    layerwise = FakeMooncakeLayerwiseConnector()
    connector._connectors = [chosen, layerwise]
    connector._requests_to_connector = {"request-1": 0}
    request = SimpleNamespace(request_id="request-1")
    blocks = MagicMock()

    with patch(
        "vllm_ascend.distributed.kv_transfer.ascend_multi_connector.MooncakeLayerwiseConnector",
        FakeMooncakeLayerwiseConnector,
    ):
        connector.update_state_after_alloc(request, blocks, 17)

    chosen.update_state_after_alloc.assert_called_once_with(request, blocks, 17)
    layerwise.update_state_after_alloc.assert_called_once_with(request, blocks, 17)
