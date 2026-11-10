# SPDX-License-Identifier: Apache-2.0
"""Constructor selection keeps native Store and CANN range protocols separate."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.cann_multi_connector import AscendMultiConnector as CANNMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.cann_mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector as CANNLayerwiseConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cann.ascend_store_connector import (
    AscendStoreConnector as CANNStoreConnector,
)


@pytest.mark.parametrize("key", ["use_layerwise_range", "hybrid_cache_c128_chunk", "layerwise_host_kv_offload"])
def test_cann_store_options_select_the_paired_protocol_once(key):
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={key: True}))
    with patch.object(CANNStoreConnector, "__init__", return_value=None) as initialize:
        connector = AscendStoreConnector(config, "worker", None)
    assert type(connector) is CANNStoreConnector
    initialize.assert_called_once_with(config, "worker", None)


@pytest.mark.parametrize("options", [{}, {"use_layerwise": True}, {"load_async": True}, {"use_layerwise_range": False}])
def test_regular_store_stays_on_native_032(options):
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=options))
    with patch.object(AscendStoreConnector, "__init__", return_value=None):
        connector = AscendStoreConnector(config, "worker", None)
    assert type(connector) is AscendStoreConnector


@pytest.mark.parametrize(
    "connector_type,cann_type",
    [
        (AscendMultiConnector, CANNMultiConnector),
        (MooncakeLayerwiseConnector, CANNLayerwiseConnector),
    ],
)
def test_workspace_option_selects_matching_coordinator_and_transport(connector_type, cann_type):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"layerwise_host_kv_offload": True})
    )
    with patch.object(cann_type, "__init__", return_value=None) as initialize:
        connector = connector_type(config, "worker", None)
    assert type(connector) is cann_type
    initialize.assert_called_once_with(config, "worker", None)
