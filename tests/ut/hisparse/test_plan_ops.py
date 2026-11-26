# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Ascend HiSparse SparseKvPlan adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from vllm_ascend.device.device_config import AscendDeviceType
from vllm_ascend.hisparse import plan_ops


def _runtime(*, top_k=2048, max_union_rows=8192, region_stride=8192, block_size=128, eligible=True):
    return SimpleNamespace(
        _ascend_sparse_kv_plan_eligible=eligible,
        index_group=SimpleNamespace(
            shared_topk=SimpleNamespace(
                device_topk_rows=SimpleNamespace(shape=(None, top_k)),
            ),
        ),
        max_union_rows=max_union_rows,
        max_swap_rows=8192,
        region_stride=region_stride,
        hot=SimpleNamespace(block_size=block_size),
    )


def test_sparse_kv_plan_support_valid_mtp3_geometry(monkeypatch):
    monkeypatch.setattr(plan_ops, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(plan_ops, "_sparse_kv_plan_dependencies_available", lambda: True)
    assert plan_ops.SparseKvPlanWorkspace.supports(_runtime())


def test_sparse_kv_plan_support_rejects_incompatible_runtime(monkeypatch):
    monkeypatch.setattr(plan_ops, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(plan_ops, "_sparse_kv_plan_dependencies_available", lambda: True)

    assert not plan_ops.SparseKvPlanWorkspace.supports(_runtime(top_k=1024))
    assert not plan_ops.SparseKvPlanWorkspace.supports(_runtime(region_stride=6144))
    assert not plan_ops.SparseKvPlanWorkspace.supports(_runtime(block_size=24))
    assert not plan_ops.SparseKvPlanWorkspace.supports(_runtime(eligible=False))


def test_sparse_kv_plan_swap_in_runs_on_copy_stream_and_stages_followers(monkeypatch):
    topk = torch.empty((4, 2048), dtype=torch.int32)
    request_ids = torch.arange(2, dtype=torch.int64)
    lengths = torch.zeros(4, dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    active_rows = torch.tensor([4], dtype=torch.int32)
    physical = torch.empty((8, 2048), dtype=torch.int32)
    metadata = SimpleNamespace(
        plan_request_ids=request_ids,
        plan_stable_prefix_lens=lengths,
        plan_visible_seq_lens=lengths,
        plan_token_to_req=token_to_req,
        plan_active_rows=active_rows,
        plan_max_token=100,
    )

    plan = MagicMock()
    leader_event = SimpleNamespace(record=Mock())
    follower_event = SimpleNamespace(record=Mock())
    follower = SimpleNamespace(_swap_rows=Mock(), _layer_ready_event=follower_event)
    copy_stream = MagicMock()
    runtime = SimpleNamespace(
        cache_ops=SimpleNamespace(plan_workspace=plan),
        _sparse_kv_plan_inputs=metadata,
        is_group_leader=True,
        index_group=SimpleNamespace(
            copy_stream=copy_stream,
            logical_topk_ready=None,
            followers=[follower],
            shared_topk=SimpleNamespace(physical_topk_indices=torch.empty((8, 2048), dtype=torch.int32)),
        ),
        _step_rows=Mock(return_value=slice(0, 4)),
        _swap_rows=Mock(),
        _layer_ready_event=leader_event,
    )
    compute_stream = Mock()
    monkeypatch.setattr("vllm.utils.torch_utils.current_stream", lambda: compute_stream)
    monkeypatch.setattr("vllm.forward_context.in_piecewise_cudagraph", lambda: False)

    result = plan_ops._sparse_plan_swap_in(
        runtime,
        Mock(),
        resident=SimpleNamespace(decode_batch=True),
        req_id_per_token=torch.arange(4, dtype=torch.int32),
        logical_topk_indices=topk,
    )

    assert result.data_ptr() == physical.data_ptr()
    assert result.shape == (4, 2048)
    copy_stream.wait_stream.assert_called_once_with(compute_stream)
    assert copy_stream.__enter__.called
    plan.execute.assert_called_once_with(
        topk,
        request_ids,
        lengths,
        lengths,
        token_to_req,
        active_rows,
        max_token=100,
    )
    runtime._swap_rows.assert_called_once_with(slice(0, 4))
    follower._swap_rows.assert_called_once_with(slice(0, 4))
    leader_event.record.assert_called_once_with(copy_stream)
    follower_event.record.assert_called_once_with(copy_stream)
    assert runtime._sparse_kv_plan_inputs is None


def test_sparse_kv_plan_op_returns_none_without_package(monkeypatch):
    monkeypatch.setattr(plan_ops, "_has_sparse_kv_plan_package", lambda _package: False)

    assert plan_ops.sparse_kv_plan_op() is None


def test_sparse_kv_plan_op_returns_none_when_module_is_missing(monkeypatch):
    package = "cann_ops_transformer_sparsekvplan"
    error = ModuleNotFoundError(
        f"No module named '{package}.ops'",
        name=f"{package}.ops",
    )
    import_module = MagicMock(side_effect=error)
    monkeypatch.setattr(plan_ops, "_SPARSE_KV_PLAN_PACKAGES", (package,))
    monkeypatch.setattr(plan_ops, "_has_sparse_kv_plan_package", lambda _package: True)
    monkeypatch.setattr(plan_ops, "import_module", import_module)

    assert plan_ops.sparse_kv_plan_op() is None
    import_module.assert_called_once_with(f"{package}.ops")


def test_sparse_kv_plan_op_requires_callable_api(monkeypatch):
    package = "cann_ops_transformer_sparsekvplan"
    module = SimpleNamespace(sparse_kv_plan=None)
    monkeypatch.setattr(plan_ops, "_SPARSE_KV_PLAN_PACKAGES", (package,))
    monkeypatch.setattr(plan_ops, "_has_sparse_kv_plan_package", lambda _package: True)
    monkeypatch.setattr(plan_ops, "import_module", MagicMock(return_value=module))

    assert plan_ops.sparse_kv_plan_op() is None


def test_sparse_kv_plan_op_returns_callable_api(monkeypatch):
    package = "cann_ops_transformer_sparsekvplan"
    operation = MagicMock()
    module = SimpleNamespace(sparse_kv_plan=operation)
    import_module = MagicMock(return_value=module)
    monkeypatch.setattr(plan_ops, "_SPARSE_KV_PLAN_PACKAGES", (package,))
    monkeypatch.setattr(plan_ops, "_has_sparse_kv_plan_package", lambda _package: True)
    monkeypatch.setattr(plan_ops, "import_module", import_module)

    assert plan_ops.sparse_kv_plan_op() is operation
    import_module.assert_called_once_with(f"{package}.ops")


def test_sparse_kv_plan_op_continues_to_next_installed_package(monkeypatch):
    first = "cann_ops_transformer_qliorder"
    second = "cann_ops_transformer_sparsekvplan"
    operation = MagicMock()
    first_error = ModuleNotFoundError(
        f"No module named '{first}.ops'",
        name=f"{first}.ops",
    )
    import_module = MagicMock(side_effect=[first_error, SimpleNamespace(sparse_kv_plan=operation)])
    monkeypatch.setattr(plan_ops, "_SPARSE_KV_PLAN_PACKAGES", (first, second))
    monkeypatch.setattr(plan_ops, "_has_sparse_kv_plan_package", lambda _package: True)
    monkeypatch.setattr(plan_ops, "import_module", import_module)

    assert plan_ops.sparse_kv_plan_op() is operation
    assert import_module.call_args_list == [
        ((f"{first}.ops",),),
        ((f"{second}.ops",),),
    ]


def test_sparse_kv_plan_op_propagates_internal_import_failure(monkeypatch):
    package = "cann_ops_transformer_sparsekvplan"
    error = ModuleNotFoundError("No module named 'internal_dependency'", name="internal_dependency")
    monkeypatch.setattr(plan_ops, "_SPARSE_KV_PLAN_PACKAGES", (package,))
    monkeypatch.setattr(plan_ops, "_has_sparse_kv_plan_package", lambda _package: True)
    monkeypatch.setattr(plan_ops, "import_module", MagicMock(side_effect=error))

    with pytest.raises(ModuleNotFoundError, match="internal_dependency"):
        plan_ops.sparse_kv_plan_op()


def test_sparse_kv_plan_execute_uses_torch_extension_api():
    workspace = object.__new__(plan_ops.SparseKvPlanWorkspace)
    workspace.op = MagicMock()
    workspace.group_width = 2
    workspace.max_rows = 4
    workspace.capacity = 4096
    workspace.block_size = 128
    for name in (
        "last_req_ids",
        "slot_to_token",
        "lru_slots",
        "current_slots",
        "miss_count",
        "miss_tokens",
        "miss_slots",
        "plan_workspace",
        "row_modes",
        "resident_rows",
        "resident_block_table",
    ):
        setattr(workspace, name, MagicMock())

    topk = torch.empty((4, 2048), dtype=torch.int32)
    request_ids = torch.arange(2, dtype=torch.int64)
    lengths = torch.zeros(4, dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    active_rows = torch.tensor([4], dtype=torch.int32)

    workspace.execute(
        topk,
        request_ids,
        lengths,
        lengths,
        token_to_req,
        active_rows,
        max_token=100,
    )

    kwargs = workspace.op.call_args.kwargs
    assert kwargs["request_ids"] is request_ids
    assert kwargs["topk_indices"] is topk
    assert kwargs["stable_prefix_lens"] is lengths
    assert kwargs["visible_seq_lens"] is lengths
    assert kwargs["token_to_req"] is token_to_req
    assert kwargs["active_rows"].data_ptr() == active_rows.data_ptr()
    assert kwargs["active_rows"].shape == (1,)
    assert kwargs["capacity"] == 4096
    assert kwargs["max_token"] == 100
    assert kwargs["block_size"] == 128
