#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""UT: ActionExecutor resolve/order."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.observability.runtime_guard.action.actions import (
    ActionExecutor,
    order_incident_actions,
)
from vllm_ascend.observability.runtime_guard.io import has_nonempty_sampled_row
from vllm_ascend.observability.runtime_guard.state import MANUAL_TRIGGER_TYPE, Incident


def test_order_incident_actions_report_before_dump():
    assert order_incident_actions(["dump_kv", "report"]) == ["report", "dump_kv"]
    assert order_incident_actions(["report"]) == ["report"]
    assert order_incident_actions(["dump_kv"]) == ["dump_kv"]
    # Dedupe; report before dump_kv.
    assert order_incident_actions(["dump_kv", "report", "report"]) == [
        "report",
        "dump_kv",
    ]


def test_resolve_actions_defaults_and_override():
    rc = MagicMock()
    rc.actions_default_on_trigger.return_value = ["report"]
    rc.detector_section.return_value = {"on_trigger": ["report", "dump_kv"], "dump_kv": {"scope": "request"}}
    rc.action_queue_max_size.return_value = 8
    ex = ActionExecutor(
        runner=SimpleNamespace(),
        runtime_config=rc,
        report_writer=MagicMock(),
        quota=MagicMock(),
        action_queue=MagicMock(),
    )
    names, det = ex.resolve_actions("token_repeat")
    assert names == ["report", "dump_kv"]
    assert det["dump_kv"]["scope"] == "request"

    names2, _ = ex.resolve_actions("token_repeat", override=["report"])
    assert names2 == ["report"]


def test_resolve_actions_falls_back_when_on_trigger_missing():
    rc = MagicMock()
    rc.actions_default_on_trigger.return_value = None
    rc.detector_section.return_value = {}
    rc.action_queue_max_size.return_value = 8
    ex = ActionExecutor(
        runner=SimpleNamespace(),
        runtime_config=rc,
        report_writer=MagicMock(),
        quota=MagicMock(),
        action_queue=MagicMock(),
    )
    names, _ = ex.resolve_actions("token_repeat")
    assert names == ["report"]


def test_has_nonempty_sampled_row_gate():
    assert has_nonempty_sampled_row(["a"], [[1, 2]]) is True
    assert has_nonempty_sampled_row(["a"], [[-1, -1]]) is False
    assert has_nonempty_sampled_row(["a"], [[]]) is False
    assert has_nonempty_sampled_row([], [[1]]) is False
    assert has_nonempty_sampled_row(["a"], None) is False


def _bare_executor(**kwargs) -> ActionExecutor:
    rc = MagicMock()
    rc.actions_default_on_trigger.return_value = ["report"]
    rc.detector_section.return_value = {"on_trigger": ["report", "dump_kv"]}
    rc.action_queue_max_size.return_value = 8
    defaults = dict(
        runner=SimpleNamespace(),
        runtime_config=rc,
        report_writer=MagicMock(),
        quota=MagicMock(),
        action_queue=MagicMock(),
    )
    defaults.update(kwargs)
    return ActionExecutor(**defaults)


def test_handle_returns_early_when_detection_gated():
    ex = _bare_executor()
    ex.can_run_detection = MagicMock(return_value=False)
    ex.resolve_actions = MagicMock()
    ex.handle(
        Incident(incident_type="token_repeat", req_id="r1"),
        detail={},
    )
    ex.resolve_actions.assert_not_called()
    ex.action_queue.submit.assert_not_called()


def test_handle_write_report_false_strips_report_action():
    ex = _bare_executor()
    ex.can_run_detection = MagicMock(return_value=True)
    prepared: list[str] = []

    def _get_action(name: str):
        act = MagicMock()
        act.name = name
        act.sync_only = False
        act.heavy = False

        def _prepare(_ctx):
            prepared.append(name)
            return object()

        act.prepare = MagicMock(side_effect=_prepare)
        return act

    with patch(
        "vllm_ascend.observability.runtime_guard.action.actions.get_action",
        side_effect=_get_action,
    ):
        ex.handle(
            Incident(incident_type="token_repeat", req_id="r1"),
            detail={},
            write_report=False,
        )
    assert "report" not in prepared
    assert prepared == ["dump_kv"]
    assert ex.action_queue.submit.call_count == 1


def test_handle_manual_trigger_injects_dump_kv_all_requests():
    ex = _bare_executor()
    ex.can_run_detection = MagicMock(return_value=True)
    rc = ex._runtime_config
    rc.detector_section.return_value = {"on_trigger": ["report"]}
    captured: list[dict] = []

    def _get_action(name: str):
        act = MagicMock()
        act.name = name
        act.sync_only = False
        act.heavy = name == "dump_kv"

        def _prepare(ctx):
            captured.append(dict(ctx.action_overrides))
            return None if name == "dump_kv" else object()

        act.prepare = MagicMock(side_effect=_prepare)
        act.run = MagicMock()
        return act

    with patch(
        "vllm_ascend.observability.runtime_guard.action.actions.get_action",
        side_effect=_get_action,
    ):
        ex.handle(
            Incident(incident_type=MANUAL_TRIGGER_TYPE, req_id="__manual_trigger__"),
            detail={"source": "ut"},
        )
    assert captured
    overrides = captured[0]
    assert "dump_kv" in overrides["_actions"]
    assert overrides["dump_kv"]["scope"] == "all_requests"
