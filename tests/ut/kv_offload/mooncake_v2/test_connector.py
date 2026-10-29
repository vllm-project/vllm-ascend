# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakeBaseConnector,
    MooncakeConnector,
    MooncakePullConnector,
    MooncakePushConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata


def make_facade() -> MooncakeBaseConnector:
    connector = MooncakeBaseConnector.__new__(MooncakeBaseConnector)
    connector.connector_scheduler = MagicMock()
    connector.connector_worker = MagicMock()
    connector._connector_metadata = MooncakeConnectorMetadata()
    return connector


def test_mooncake_connector_aliases_pull_connector() -> None:
    assert MooncakeConnector is MooncakePullConnector


def test_required_layout_is_hnd_for_regular_attention_and_none_for_mla() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) == "HND"  # type: ignore[arg-type]

    config.model_config.use_mla = True
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) is None  # type: ignore[arg-type]

    config.model_config = None
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) is None  # type: ignore[arg-type]


def test_facade_delegates_scheduler_methods() -> None:
    connector = make_facade()
    request = MagicMock()
    blocks = MagicMock()
    output = MagicMock()
    connector.connector_scheduler.get_num_new_matched_tokens.return_value = (8, True)
    connector.connector_scheduler.request_finished.return_value = (True, {"remote": True})

    assert connector.get_num_new_matched_tokens(request, 4) == (8, True)
    connector.update_state_after_alloc(request, blocks, 8)
    connector.update_connector_output(output)
    assert connector.request_finished(request, [1, 2]) == (True, {"remote": True})

    connector.connector_scheduler.update_state_after_alloc.assert_called_once_with(request, blocks, 8)
    connector.connector_scheduler.request_finished.assert_called_once_with(request, ([1, 2],))


def test_facade_delegates_worker_methods_and_metadata() -> None:
    connector = make_facade()
    connector.connector_worker.get_finished.return_value = ({"sent"}, {"received"})
    connector.connector_worker.get_block_ids_with_load_errors.return_value = {10}
    connector.connector_worker.xfer_handshake_metadata = MagicMock()
    connector._connector_metadata.requests["request"] = MagicMock()

    connector.register_kv_caches({})
    connector.start_load_kv(MagicMock())

    assert connector.get_finished(set()) == ({"sent"}, {"received"})
    assert connector.get_block_ids_with_load_errors() == {10}
    assert connector.get_handshake_metadata() is connector.connector_worker.xfer_handshake_metadata
    connector.connector_worker.start_load_kv.assert_called_once_with(connector._connector_metadata)


def test_pull_connector_selects_implementation_by_role() -> None:
    config = MagicMock()
    config.kv_transfer_config.engine_id = "engine"
    config.kv_transfer_config.kv_role = "kv_producer"
    kv_cache_config = MagicMock()

    def initialize_base(self, *_args, **_kwargs) -> None:
        self.engine_id = "engine"
        self.connector_scheduler = None
        self.connector_worker = None

    with (
        patch.object(MooncakeBaseConnector, "__init__", initialize_base),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector.MooncakePullConnectorScheduler",
            return_value=MagicMock(),
        ) as scheduler_cls,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector.MooncakePullConnectorWorker",
            return_value=MagicMock(),
        ) as worker_cls,
    ):
        scheduler_connector = MooncakePullConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        worker_connector = MooncakePullConnector(config, KVConnectorRole.WORKER, kv_cache_config)

    assert scheduler_connector.connector_scheduler is scheduler_cls.return_value
    assert worker_connector.connector_worker is worker_cls.return_value


def test_push_connector_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        MooncakePushConnector(MagicMock(), KVConnectorRole.SCHEDULER, MagicMock())
