# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.compilation import acl_graph
from vllm_ascend.compilation.acl_graph import (
    clear_graph_params_for_recapture,
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    set_graph_params,
    update_graph_params_workspaces,
)
from vllm_ascend.device_allocator.sleep_mem_optimized import AclGraphSleepWakeupManager


@pytest.fixture(autouse=True)
def reset_graph_params():
    names = ("_graph_params", "_draft_graph_params", "_draft_graph_prefill_params")
    previous = {name: getattr(acl_graph, name) for name in names}
    for name in names:
        setattr(acl_graph, name, None)
    yield
    for name, value in previous.items():
        setattr(acl_graph, name, value)


def _forward_context(*, has_lora: bool, mode: CUDAGraphMode = CUDAGraphMode.FULL):
    return SimpleNamespace(
        cudagraph_runtime_mode=mode,
        batch_descriptor=BatchDescriptor(num_tokens=4, has_lora=has_lora),
    )


def test_graph_params_isolate_base_and_lora_with_the_same_num_tokens() -> None:
    set_graph_params([4])

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=False),
    ):
        base_params = get_graph_params()
        base_params.events[4].append("base")

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=True),
    ):
        lora_params = get_graph_params()
        lora_params.events[4].append("lora")

    assert base_params is not lora_params
    assert base_params.events[4] == ["base"]
    assert lora_params.events[4] == ["lora"]


def test_graph_params_use_base_storage_outside_full_graph() -> None:
    set_graph_params([4])

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=True, mode=CUDAGraphMode.PIECEWISE),
    ):
        piecewise_params = get_graph_params()

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        side_effect=AssertionError,
    ):
        no_context_params = get_graph_params()

    assert piecewise_params is no_context_params


def test_workspace_update_targets_the_active_route() -> None:
    set_graph_params([4])
    base_workspace = torch.empty(1)
    lora_workspace = torch.empty(2)

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=False),
    ):
        update_graph_params_workspaces(4, base_workspace)
        base_params = get_graph_params()

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=True),
    ):
        update_graph_params_workspaces(4, lora_workspace)
        lora_params = get_graph_params()

    assert base_params.workspaces[4] is base_workspace
    assert lora_params.workspaces[4] is lora_workspace


def test_draft_graph_params_are_isolated_by_route() -> None:
    set_draft_graph_params([4])
    set_draft_graph_prefill_params([4])

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=False),
    ):
        base_draft = get_draft_graph_params()
        base_prefill = get_draft_graph_prefill_params()

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(has_lora=True),
    ):
        lora_draft = get_draft_graph_params()
        lora_prefill = get_draft_graph_prefill_params()

    assert base_draft is not lora_draft
    assert base_prefill is not lora_prefill


def test_sleep_clears_both_base_and_lora_graph_params() -> None:
    set_graph_params([4])
    params = list(acl_graph.iter_graph_params())
    assert len(params) == 2
    for graph_params in params:
        graph_params.workspaces[4] = torch.empty(1)
        graph_params.events[4].append(object())

    AclGraphSleepWakeupManager.clear_all_attention_workspaces()
    AclGraphSleepWakeupManager.reset_all_graph_params()

    for graph_params in params:
        assert graph_params.workspaces[4] is None
        assert graph_params.events[4] == []


def test_recapture_clears_all_graph_params_routes() -> None:
    set_graph_params([4])
    set_draft_graph_params([4])
    set_draft_graph_prefill_params([4])
    params = list(acl_graph.iter_graph_params())

    for graph_params in params:
        graph_params.events[4].append(object())
        graph_params.workspaces[4] = torch.empty(1)
        graph_params.handles[4].append(object())
        graph_params.attn_params[4].append(object())

    clear_graph_params_for_recapture()

    for graph_params in params:
        assert graph_params.events[4] == []
        assert graph_params.workspaces[4] is None
        assert graph_params.handles[4] == []
        assert graph_params.attn_params[4] == []
