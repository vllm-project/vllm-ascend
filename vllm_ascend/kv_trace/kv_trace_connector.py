# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record public connector plans and native results without polling queues."""

from collections.abc import Mapping
from typing import Any


def record_plan(metadata: Any, report: Any) -> None:
    if metadata is None:
        return
    fields: dict[str, Any] = {"metadata_type": f"{type(metadata).__module__}.{type(metadata).__qualname__}"}
    requests = getattr(metadata, "requests", None)
    if isinstance(requests, Mapping):
        fields["requests"] = {}
        for request_id, request in requests.items():
            plan = {}
            for name in (
                "local_block_ids",
                "remote_block_ids",
                "num_computed_tokens",
                "num_external_tokens",
                "remote_engine_id",
                "remote_request_id",
            ):
                value = request.get(name) if isinstance(request, Mapping) else getattr(request, name, None)
                if value is not None:
                    plan[name] = value
            fields["requests"][request_id] = plan
    sending = getattr(metadata, "requests_to_send", None)
    if isinstance(sending, Mapping):
        fields["send_request_ids"] = list(sending)
    report("kv_trace.boundary", boundary="connector.input", **fields)


def record_output(output: Any, report: Any) -> None:
    connector = getattr(output, "kv_connector_output", None)
    if connector is None:
        return
    fields = {
        name: None if (value := getattr(connector, name, None)) is None else sorted(value)
        for name in ("finished_sending", "finished_recving", "failed_recving", "invalid_block_ids")
    }
    report("kv_trace.boundary", boundary="connector.output", **fields)
