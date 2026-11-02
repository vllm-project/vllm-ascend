# SPDX-License-Identifier: Apache-2.0
"""Actual upstream cancellation methods keep DMA destinations owned."""

import ast
import enum
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def actual_classes(modules):
    upstream = Path(__file__).resolve().parents[6] / "vllm/vllm/v1"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    status = next(
        node
        for node in ast.parse((upstream / "request.py").read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "RequestStatus"
    )
    namespace = {"enum": enum}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, status], type_ignores=[])), "request.py", "exec"),
        namespace,
    )
    request_status = namespace["RequestStatus"]
    path = upstream / "core/sched/scheduler.py"
    scheduler = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    wanted = {
        "finish_requests",
        "_free_request",
        "_free_blocks",
        "_free_request_blocks",
        "_request_blocks_can_be_freed",
        "_update_from_kv_xfer_finished",
        "_connector_finished",
    }
    methods = [node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {
        "RequestStatus": request_status,
        "SupportsHMA": modules.base.SupportsHMA,
        "logger": logging.getLogger("pd-scheduler-test"),
        "KV_FETCH_COMPLETED_WAITING": "complete",
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return request_status, type("ActualSchedulerLifecycle", (), namespace)


@pytest.mark.parametrize("received_before_cancel", [False, True])
def test_cancel_waits_for_actual_receive_drain(modules, received_before_cancel):
    status, scheduler_class = actual_classes(modules)

    class Request:
        request_id = "D"
        client_index = 0
        num_computed_tokens = 257
        num_prompt_tokens = 257
        num_in_flight_tokens = 0
        kv_transfer_params = None

        def is_finished(self):
            return status.is_finished(self.status)

    request = Request()
    request.status = status.WAITING_FOR_REMOTE_KVS
    connector = object.__new__(modules.connector.HiSparsePDConnector)
    connector.producer = False
    connector._composed = True
    connector.loads = {}
    connector.pending_exports = set()
    freed = []
    scheduler = scheduler_class()
    scheduler.__dict__.update(
        requests={"D": request},
        connector=connector,
        running=[],
        waiting=NS(remove_requests=lambda _: None),
        kv_holding_waiting=NS(remove_requests=lambda _: None),
        deferred_waiting=set(),
        finished_recving_kv_req_ids=set(),
        failed_recving_kv_req_ids=set(),
        aux_output_connector=None,
        ec_connector=None,
        _inflight_prefills={request},
        _set_kv_fetch_stage=lambda *_: None,
        encoder_cache_manager=NS(free=lambda _: None),
        finished_req_ids=set(),
        finished_req_ids_dict=None,
        defer_block_free=False,
        vllm_config=NS(kv_transfer_config=NS(is_kv_producer=False)),
        kv_cache_manager=NS(
            remove_skipped_blocks=lambda **_: None,
            get_block_ids_for_computed_tokens=lambda **_: ([2, 6, 3],),
            free=lambda req: freed.append(req.request_id),
        ),
    )

    def result(done):
        return NS(finished_sending=set(), finished_recving={"D"} if done else set(), failed_recving=set())

    if received_before_cancel:
        scheduler._update_from_kv_xfer_finished(result(True))
        assert not freed and scheduler.finished_recving_kv_req_ids == {"D"}
    assert scheduler.finish_requests("D", status.FINISHED_ABORTED) == [request]
    assert request.is_finished()
    if not received_before_cancel:
        # The D connector itself returns False; the scheduler's existing
        # WAITING_FOR_REMOTE_KVS fence still owns all destination pages.
        assert not freed and scheduler.requests == {"D": request}
        scheduler._update_from_kv_xfer_finished(result(False))
        assert not freed and "D" in scheduler.requests
        scheduler._update_from_kv_xfer_finished(result(True))
    assert freed == ["D"] and not scheduler.requests
