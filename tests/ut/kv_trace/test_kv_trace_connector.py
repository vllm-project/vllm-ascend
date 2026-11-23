# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector observations retain native completion/failure fields without polling."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

from .test_kv_trace_runtime import message
from .test_kv_trace_runtime import runtime_fixture as runtime_fixture


def test_public_request_plan_does_not_export_token_payloads(runtime):
    request = NS(
        local_block_ids=[1], remote_block_ids=[2], num_external_tokens=16, remote_request_id="remote", token_ids=[100]
    )
    output = message()
    output.kv_connector_metadata = NS(requests={"r": request}, requests_to_send={"r": object()})
    with runtime.execute_scope(output):
        pass
    event = next(event for event in runtime.events if event.get("boundary") == "connector.input")
    assert event["requests"]["r"] == {
        "local_block_ids": [1],
        "remote_block_ids": [2],
        "num_external_tokens": 16,
        "remote_request_id": "remote",
    }
    assert event["send_request_ids"] == ["r"]


def test_unknown_metadata_is_only_identified(runtime):
    runtime.connector_subject.record_plan(object(), runtime.report)
    assert runtime.events[-1]["metadata_type"] == "builtins.object"


def test_async_results_are_not_polled_and_failed_receive_is_preserved(runtime):
    connector = NS(finished_sending=None, finished_recving={"r"}, failed_recving={"r"}, invalid_block_ids={2})
    native = NS(kv_connector_output=connector)
    getter = Mock(return_value=native)
    output = runtime._guard_output(NS(get_output=getter))
    getter.assert_not_called()
    assert output.get_output() is native
    getter.assert_called_once_with()
    event = runtime.events[-1]
    assert event["finished_recving"] == event["failed_recving"] == ["r"]
    assert event["invalid_block_ids"] == [2]
