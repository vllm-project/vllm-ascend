"""Regression tests for Ascend MultiConnector completion aggregation."""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
    AscendMultiKVConnectorMetadata,
)


class _FinishedConnector:
    def __init__(self, results):
        self._results = iter(results)

    def get_finished(self, finished_req_ids):
        return next(self._results, (None, None))

    def shutdown(self):
        return None


class _RequestFinishedConnector:
    def __init__(self, async_save):
        self.async_save = async_save

    def request_finished(self, request, blocks):
        return self.async_save, None

    def request_finished_all_groups(self, request, block_ids):
        return self.async_save, None


def _make_finished_connector(*connectors, async_save_sources=None, extra_async_saves=None):
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = list(connectors)
    connector._extra_async_saves = extra_async_saves or {}
    connector._finished_sending_connectors = {}
    connector._finished_recving_emitted = OrderedDict()
    connector._async_save_sources = async_save_sources or {}
    connector._eligible_finished_req_ids = set()
    return connector


def test_duplicate_async_save_from_one_connector_waits_for_other_connector():
    request_id = "req-duplicate"
    connector = _make_finished_connector(
        _FinishedConnector([({request_id}, None), ({request_id}, None), (None, None)]),
        _FinishedConnector([(None, None), (None, None), ({request_id}, None)]),
        async_save_sources={request_id: (0, 1)},
        extra_async_saves={request_id: 1},
    )

    assert connector.get_finished({request_id}) == (None, None)
    assert connector.get_finished(set()) == (None, None)
    assert connector.get_finished(set()) == ({request_id}, None)


def test_async_save_is_emitted_once_after_all_expected_connectors_finish():
    request_id = "req-terminal"
    connector = _make_finished_connector(
        _FinishedConnector([({request_id}, None), ({request_id}, None)]),
        _FinishedConnector([({request_id}, None), ({request_id}, None)]),
        async_save_sources={request_id: (0, 1)},
        extra_async_saves={request_id: 1},
    )

    assert connector.get_finished({request_id}) == ({request_id}, None)
    assert connector.get_finished(set()) == (None, None)


def test_single_async_save_duplicate_is_ignored():
    request_id = "req-single"
    connector = _make_finished_connector(
        _FinishedConnector([({request_id}, None), ({request_id}, None)]),
        async_save_sources={request_id: (0,)},
    )

    assert connector.get_finished({request_id}) == ({request_id}, None)
    assert connector.get_finished(set()) == (None, None)


def test_unexpected_connector_does_not_satisfy_async_save():
    request_id = "req-wrong-source"
    connector = _make_finished_connector(
        _FinishedConnector([(None, None), ({request_id}, None)]),
        _FinishedConnector([({request_id}, None), (None, None)]),
        async_save_sources={request_id: (0,)},
    )

    assert connector.get_finished({request_id}) == (None, None)
    assert connector.get_finished(set()) == ({request_id}, None)


def test_completion_before_announcement_is_emitted_after_announcement():
    request_id = "req-late-announcement"
    connector = _make_finished_connector(
        _FinishedConnector([({request_id}, None), (None, None)]),
        async_save_sources={request_id: (0,)},
    )

    assert connector.get_finished(set()) == (None, None)
    assert connector.get_finished({request_id}) == ({request_id}, None)


def test_sync_finished_request_does_not_create_eligibility():
    request_id = "req-sync"
    connector = _make_finished_connector(_FinishedConnector([({request_id}, None)]))

    assert connector.get_finished({request_id}) == (None, None)
    assert connector._eligible_finished_req_ids == set()


def test_unknown_async_save_is_not_forwarded():
    request_id = "req-unknown"
    connector = _make_finished_connector(_FinishedConnector([({request_id}, None)]))

    assert connector.get_finished(set()) == (None, None)


def test_receive_completion_is_deduplicated():
    request_id = "req-receive"
    connector = _make_finished_connector(
        _FinishedConnector([(None, {request_id}), (None, {request_id})])
    )

    assert connector.get_finished(set()) == (None, {request_id})
    assert connector.get_finished(set()) == (None, None)


def test_metadata_carries_async_save_source_indexes():
    metadata = AscendMultiKVConnectorMetadata(metadata=(), async_save_sources={"req": (1, 2)})
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = []
    connector._extra_async_saves = {}
    connector._async_save_sources = {}
    connector.bind_connector_metadata(metadata)

    assert connector._async_save_sources == {"req": (1, 2)}


def test_request_finished_records_async_save_source_indexes():
    request = SimpleNamespace(request_id="req-sources")
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [
        _RequestFinishedConnector(True),
        _RequestFinishedConnector(False),
        _RequestFinishedConnector(True),
    ]
    connector._all_support_hma = True
    connector._extra_async_saves = {}
    connector._async_save_sources = {}
    connector._requests_to_connector = {}

    assert connector.request_finished_all_groups(request, ([1],)) == (True, None)
    assert connector._async_save_sources == {"req-sources": (0, 2)}
    assert connector._extra_async_saves == {"req-sources": 1}


def test_shutdown_clears_completion_tracking():
    request_id = "req-shutdown"
    connector = _make_finished_connector(
        _FinishedConnector([({request_id}, None)]),
        async_save_sources={request_id: (0,)},
        extra_async_saves={request_id: 1},
    )
    connector.get_finished({request_id})

    connector.shutdown()

    assert connector._finished_sending_connectors == {}
    assert connector._extra_async_saves == {}
    assert connector._async_save_sources == {}
    assert connector._finished_recving_emitted == {}
    assert connector._eligible_finished_req_ids == set()


def test_update_state_after_alloc_preserves_full_block_observer():
    selected = SimpleNamespace(
        requires_full_blocks_on_update_after_alloc=False,
        update_state_after_alloc=MagicMock(),
    )
    observer = SimpleNamespace(
        requires_full_blocks_on_update_after_alloc=True,
        update_state_after_alloc=MagicMock(),
    )
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [selected, observer]
    connector._requests_to_connector = {"req": 0}
    blocks = SimpleNamespace(new_empty=lambda: "empty")
    request = SimpleNamespace(request_id="req")

    connector.update_state_after_alloc(request, blocks, 16)

    selected.update_state_after_alloc.assert_called_once_with(request, blocks, 16)
    observer.update_state_after_alloc.assert_called_once_with(request, blocks, 16)
