# SPDX-License-Identifier: Apache-2.0
"""Offload graph admission and role staging through existing worker hooks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager

from vllm_ascend.distributed.expert_transfer import expert_offload_utils as expert_offload
from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import DSparkExpertPool
from vllm_ascend.distributed.expert_transfer.expert_offload_manager import ExpertOffloadManager


def make_manager():
    manager = ExpertOffloadManager()
    manager.dspark_expert_pool = DSparkExpertPool(
        torch.device("cpu"), 12, 2, 2, num_speculative_tokens=5, num_device_experts=84
    )
    model = torch.nn.Module()
    model._expert_offload_manager = manager
    return manager, model


def make_state(rows, drafts, prefilling):
    return SimpleNamespace(
        num_scheduled_tokens=np.array(rows),
        num_draft_tokens_np=drafts,
        is_prefilling_np=prefilling,
        has_prefill=any(prefilling),
    )


def make_graph(load_hook_class, model, *, width=6):
    graph_class = load_hook_class(
        "vllm_ascend/worker/v2/aclgraph_utils.py",
        "ModelAclGraphManager",
        {"dispatch"},
        ModelCudaGraphManager,
        {
            "expert_offload": expert_offload,
            "BatchExecutionDescriptor": BatchExecutionDescriptor,
            "CUDAGraphMode": CUDAGraphMode,
        },
    )
    graph = object.__new__(graph_class)
    graph.model_runner = SimpleNamespace(model=model)
    captured = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=2 * width, num_reqs=2, uniform_token_count=width
    )
    graph._graphs_captured = True
    graph._lora_dispatch_map = {}
    graph._decode_descs = {(width, 0, 1): [captured] * 3}
    graph._candidates = {(n, 0): [captured] for n in range(1, 2 * width + 1)}
    return graph


@pytest.mark.parametrize(
    "rows,drafts,prefilling,width,want_full",
    [
        ([6], [5], [False], 6, True),
        ([6, 6], [5, 5], [False, False], 6, True),
        ([6], [5], [True], 6, False),
        ([6, 6], [5, 5], [False, True], 6, False),
        ([6], [2], [False], 6, False),
        ([3], [2], [False], 3, False),
        ([1, 6], [0, 5], [False, False], None, False),
    ],
)
def test_graph_manager_rejects_incompatible_pool_layout_before_replay(
    load_hook_class, rows, drafts, prefilling, width, want_full
):
    manager, model = make_manager()
    manager.prepare_batch(make_state(rows, drafts, prefilling))
    graph = make_graph(load_hook_class, model)
    actual = graph.dispatch(len(rows), sum(rows), width, 0)
    assert actual.cg_mode == (CUDAGraphMode.FULL if want_full else CUDAGraphMode.NONE)
    assert actual.num_tokens == (12 if want_full else sum(rows))
    if not want_full:
        assert actual.num_reqs == len(rows)


@pytest.mark.parametrize("owner", ["absent", "native_offload", "dummy_pool"])
def test_pool_admission_preserves_unrelated_and_dummy_graphs(load_hook_class, owner):
    manager, model = make_manager()
    if owner == "absent":
        del model._expert_offload_manager
    elif owner == "native_offload":
        manager.dspark_expert_pool = None
        manager.prepare_batch(make_state([6], [5], [True]))
    else:
        manager.prepare_batch(make_state([6], [5], [True]))
        manager.prepare_batch(None)
    graph = make_graph(load_hook_class, model)
    assert graph.dispatch(1, 6, 6, 0).cg_mode == CUDAGraphMode.FULL


def test_pool_admission_preserves_single_token_native_graphs(load_hook_class):
    manager, model = make_manager()
    manager.prepare_batch(make_state([1], None, [False]))
    graph = make_graph(load_hook_class, model, width=1)
    assert graph.dispatch(1, 1, 1, 0).cg_mode == CUDAGraphMode.FULL


def test_existing_batch_hook_resets_stale_pool_for_real_and_dummy_steps():
    manager, model = make_manager()
    pool = manager.dspark_expert_pool
    runner = SimpleNamespace(model=model)
    for state, want_prefill in (
        (make_state([6], [5], [False]), False),
        (None, True),
        (make_state([6], [5], [True]), True),
        (make_state([6], [5], [False]), False),
    ):
        pool.active = True
        expert_offload.prepare_batch(runner, state)
        assert not pool.active
        assert manager.has_prefill is want_prefill


@pytest.mark.parametrize("structured", [False, True])
def test_model_state_stages_current_real_roles_and_preserves_capture_roles(load_hook_class, structured):
    manager, model = make_manager()
    pool = manager.dspark_expert_pool

    class Parent:
        def preprocess_state(self, input_batch, block_tables, kv_cache_config, num_computed_tokens):
            self.preprocessed_rows = input_batch.num_tokens

    state_class = load_hook_class(
        "vllm_ascend/worker/v2/model_states/default.py",
        "AscendModelState",
        {"preprocess_state"},
        Parent,
        {"expert_offload": expert_offload},
    )
    state = state_class()
    state.model = model
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=12,
        num_tokens_after_padding=12,
        query_start_loc_np=np.array([0, 6, 12]),
        num_draft_tokens_per_req=[5, 5],
        is_prefilling_np=[False, False],
        has_prefill=False,
        has_structured_output_reqs=structured,
    )
    address = pool.roles.data_ptr()
    expert_offload.prepare_capture(model, batch)
    assert pool.active
    for real_rows in (6, 12, 6):
        batch.num_reqs = real_rows // 6
        batch.num_tokens = real_rows
        batch.num_draft_tokens_per_req = [5] * batch.num_reqs
        batch.is_prefilling_np = [False] * batch.num_reqs
        manager.prepare_batch(make_state([6] * batch.num_reqs, batch.num_draft_tokens_per_req, batch.is_prefilling_np))
        state.preprocess_state(batch, (), None, torch.empty(0))
        assert pool.active
        assert pool.roles.tolist() == [1, 1, 2, 2, 2, 2] * batch.num_reqs + [0] * (12 - real_rows)
        assert pool.roles.data_ptr() == address
        assert state.preprocessed_rows == real_rows
    batch.has_prefill = True
    batch.is_prefilling_np = [True]
    manager.prepare_batch(make_state([6], [5], [True]))
    state.preprocess_state(batch, (), None, torch.empty(0))
    assert not pool.active
    manager.prepare_batch(None)
    expert_offload.prepare_capture(model, batch)
    assert pool.active
