# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.worker.v2.spec_decode.diagnostics import enable_budget_debug, enable_draft_graph_debug


@pytest.mark.parametrize("enable", [enable_budget_debug, enable_draft_graph_debug])
def test_info_mode_does_not_wrap_or_touch_manager(enable):
    logger = Mock()
    logger.isEnabledFor.return_value = False
    manager = SimpleNamespace()
    enable(manager, logger)
    assert vars(manager) == {}
    logger.debug.assert_not_called()


def test_budget_debug_preserves_arguments_result_and_state():
    logger = Mock()
    logger.isEnabledFor.return_value = True
    state = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 4)
    original = Mock(return_value=6)
    manager = SimpleNamespace(get_num_tokens=original, _batch_budget=state)
    enable_budget_debug(manager, logger)
    assert manager.get_num_tokens("tokens", drafts="drafts") == 6
    original.assert_called_once_with("tokens", drafts="drafts")
    assert manager._batch_budget is state
    assert state == ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 4)
    assert logger.debug.call_args.args[1:7] == (2, 6, 4, 3, 3, 2)


def test_graph_debug_preserves_descriptor_identity():
    logger = Mock()
    logger.isEnabledFor.return_value = True
    desc = SimpleNamespace(cg_mode="FULL", num_tokens=48, num_reqs=16, uniform_token_count=3)
    original = Mock(return_value=desc)
    manager = SimpleNamespace(dispatch=original, _capture_descs={"FULL": [desc]})
    enable_draft_graph_debug(manager, logger)
    assert manager.dispatch(num_tokens=48, num_reqs=16, uniform_token_count=3) is desc
    original.assert_called_once_with(num_tokens=48, num_reqs=16, uniform_token_count=3)
    assert logger.debug.call_args.args[1:] == ("FULL", 48, 16, 3)


def test_debug_does_not_swallow_upstream_errors():
    logger = Mock()
    logger.isEnabledFor.return_value = True
    manager = SimpleNamespace(get_num_tokens=Mock(side_effect=ValueError("upstream")))
    enable_budget_debug(manager, logger)
    with pytest.raises(ValueError, match="upstream"):
        manager.get_num_tokens({}, {})
