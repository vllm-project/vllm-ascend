# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks of production FT methods, independent of NPU package imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_definitions(path, names, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def test_scale_down_routing_is_dense_and_preserves_storage():
    ns = load_definitions(
        "vllm_ascend/worker/sentinel/eplb_redistribute.py",
        ["build_orig_to_dense_rank_table", "densify_routing_table_physical_ids"],
        torch=torch,
    )
    table = ns["build_orig_to_dense_rank_table"](4, {1})
    assert table.tolist() == [0, -1, 1, 2]
    routes = torch.tensor([[0, 1], [4, 7]], dtype=torch.int32)
    ptr = routes.data_ptr()
    ns["densify_routing_table_physical_ids"](routes, table, 2)
    assert routes.tolist() == [[0, 1], [2, 5]]
    assert routes.data_ptr() == ptr
    for invalid in (-1, 2):
        with pytest.raises(RuntimeError):
            ns["densify_routing_table_physical_ids"](torch.tensor([invalid]), table, 2)


def test_scale_down_preconditions_reject_unsupported_runners_and_backends():
    config = SimpleNamespace(enable_fused_mc2=False, enable_mc2_hierarchy_comm=False)
    ns = load_definitions(
        "vllm_ascend/worker/sentinel/npu_worker_sentinel.py",
        ["WorkerSentinel"],
        GPUWorkerSentinel=object,
        get_ascend_config=lambda: config,
        torch_npu=SimpleNamespace(npu_moe_distribute_dispatch_v2=Mock()),
    )
    sentinel = ns["WorkerSentinel"].__new__(ns["WorkerSentinel"])
    sentinel.worker = SimpleNamespace(
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(eplb_state=object()),
        parallel_config=SimpleNamespace(eplb_config=SimpleNamespace(num_redundant_experts=1)),
    )
    sentinel._resolve_drafter = Mock()
    with pytest.raises(ValueError, match="v2 model runner"):
        sentinel._validate_scale_down_preconditions()
    sentinel.worker.use_v2_model_runner = True
    sentinel.worker.parallel_config.eplb_config.num_redundant_experts = 0
    with pytest.raises(ValueError, match="redundant"):
        sentinel._validate_scale_down_preconditions()
    sentinel.worker.parallel_config.eplb_config.num_redundant_experts = 1
    sentinel._validate_scale_down_preconditions()
    config.enable_fused_mc2 = True
    with pytest.raises(ValueError, match="fused_mc2"):
        sentinel._validate_scale_down_preconditions()
