# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check KV writer dependency loading without importing the NPU model stack."""

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_scatter():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_attn_kv_plan.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_get_kv_compress_epilog"
    ]
    plan = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DsaAttnKvPlan")
    functions.append(
        next(node for node in plan.body if isinstance(node, ast.FunctionDef) and node.name == "dsa_kv_compress_scatter")
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    namespace = SimpleNamespace()
    native_scatter, bf16_scatter = Mock(), Mock()
    torch_api = SimpleNamespace(
        uint8=torch.uint8,
        ops=SimpleNamespace(
            cann_ops_transformer=namespace,
            _C_ascend=SimpleNamespace(npu_scatter_nd_update_sk=native_scatter),
        ),
    )
    load = Mock()
    scope = {
        "torch": torch_api,
        "torch_npu": SimpleNamespace(npu_scatter_nd_update_=bf16_scatter),
        "lru_cache": lru_cache,
        "import_module": load,
    }
    exec(compile(module, str(source), "exec"), scope)
    return scope["dsa_kv_compress_scatter"], load, namespace, native_scatter, bf16_scatter


def test_fp8_scatter_loads_epilog_without_query_metadata():
    scatter, load, namespace, _, _ = _load_scatter()
    plan = SimpleNamespace(uses_sparse_flash_mla=False, uses_kv_compress_epilog=True)
    cache = torch.zeros(2, 32, 1, 608, dtype=torch.uint8)
    updates = torch.ones(3, 1, 512, dtype=torch.bfloat16)
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)
    events = []
    epilog = Mock(side_effect=lambda **kwargs: events.append("write"))

    def register(name):
        events.append("import")
        namespace.kv_compress_epilog = epilog

    load.side_effect = register
    # An empty-query PCP rank still writes the restored global KV rows.
    # Start with no registered epilog and do not build any query metadata.
    scatter(plan, cache, updates, slots)
    scatter(plan, cache, updates, slots)

    assert events == ["import", "write", "write"]
    load.assert_called_once_with("cann_ops_transformer")
    assert epilog.call_count == 2
    passed = epilog.call_args.kwargs
    assert passed["cache"].data_ptr() == cache.data_ptr()
    assert passed["cache"].dtype == torch.uint8
    assert passed["x"].shape == (3, 512)
    assert passed["x"].data_ptr() == updates.data_ptr()
    assert passed["slot_mapping"] is slots


@pytest.mark.parametrize("failure", [ImportError("missing package"), RuntimeError("package initialization failed")])
def test_fp8_scatter_preserves_import_errors(failure):
    scatter, load, _, _, _ = _load_scatter()
    plan = SimpleNamespace(uses_sparse_flash_mla=False, uses_kv_compress_epilog=True)
    load.side_effect = failure

    with pytest.raises(type(failure)) as raised:
        scatter(plan, torch.zeros(1), torch.zeros(1, 512), torch.tensor([0]))

    assert raised.value is failure


def test_fp8_scatter_reports_missing_epilog_after_import():
    scatter, load, _, _, _ = _load_scatter()
    plan = SimpleNamespace(uses_sparse_flash_mla=False, uses_kv_compress_epilog=True)

    with pytest.raises(AttributeError, match="kv_compress_epilog"):
        scatter(plan, torch.zeros(1), torch.zeros(1, 512), torch.tensor([0]))

    load.assert_called_once_with("cann_ops_transformer")


@pytest.mark.parametrize(
    "uses_sparse_flash_mla,uses_kv_compress_epilog", [(True, False), (False, False), (False, True)]
)
def test_scatter_loads_no_package_for_none_updates(uses_sparse_flash_mla, uses_kv_compress_epilog):
    scatter, load, _, native, bf16 = _load_scatter()
    plan = SimpleNamespace(
        uses_sparse_flash_mla=uses_sparse_flash_mla,
        uses_kv_compress_epilog=uses_kv_compress_epilog,
    )

    scatter(plan, torch.zeros(1), None, torch.tensor([0]))

    load.assert_not_called()
    native.assert_not_called()
    bf16.assert_not_called()


@pytest.mark.parametrize("uses_sparse_flash_mla", [False, True])
def test_other_scatter_plans_do_not_load_cann(uses_sparse_flash_mla):
    scatter, load, _, native, bf16 = _load_scatter()
    plan = SimpleNamespace(uses_sparse_flash_mla=uses_sparse_flash_mla, uses_kv_compress_epilog=False)
    cache = torch.zeros(2, 32, 1, 512, dtype=torch.bfloat16)
    updates = torch.ones(1, 1, 512, dtype=torch.bfloat16)
    slots = torch.tensor([0], dtype=torch.int32)

    scatter(plan, cache, updates, slots)

    load.assert_not_called()
    (bf16 if uses_sparse_flash_mla else native).assert_called_once()
    (native if uses_sparse_flash_mla else bf16).assert_not_called()
