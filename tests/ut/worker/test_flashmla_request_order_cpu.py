# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU request-order checks; the framework parent is emulated, not NPU execution."""

import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
REQUEST_FIELDS = (
    "num_scheduled_tokens",
    "idx_mapping_np",
    "prefill_len_np",
    "num_computed_prefill_tokens_np",
    "is_prefilling_np",
)


@pytest.fixture
def runner():
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "gather_batch_req_state"
    )

    class Parent:
        def gather_batch_req_state(self, scheduler_output, dummy_run):
            return scheduler_output.state, scheduler_output.uniform

    owner = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"np": np, "Parent": Parent}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, owner], type_ignores=[])), str(path), "exec"), scope
    )
    instance = scope["Runner"]()
    instance.vllm_config = SimpleNamespace(kv_transfer_config=None)
    instance.flashmla_executor = object()
    instance.decode_query_len = 1
    instance._pd_uniform_decode_token_count = Mock(return_value=1)
    return instance


def make_batch(*, new_fields, optional_values):
    fields = ["req_ids", *REQUEST_FIELDS, "num_tokens", "has_prefill"]
    values = dict(
        req_ids=["fresh_prompt", "decode_b", "prompt_tail", "decode_a"],
        num_scheduled_tokens=np.array([1, 3, 4, 2]),
        idx_mapping_np=np.array([10, 11, 12, 13]),
        prefill_len_np=np.array([1, 20, 40, 30]),
        num_computed_prefill_tokens_np=np.array([0, 20, 39, 30]),
        is_prefilling_np=np.array([True, False, True, False]),
        num_tokens=10,
        has_prefill=True,
    )
    if new_fields:
        fields.extend(["num_draft_tokens_np", "prefill_runs_as_decode_np", "decode_graph_eligible"])
        values.update(
            num_draft_tokens_np=np.array([0, 2, 3, 1]) if optional_values else None,
            prefill_runs_as_decode_np=np.array([False, False, True, False]) if optional_values else None,
            decode_graph_eligible=False,
        )
    batch_type = namedtuple("Batch", fields)
    return batch_type(**values)


@pytest.mark.parametrize("new_fields,optional_values", [(False, False), (True, False), (True, True)])
def test_flashmla_order_keeps_all_request_fields_aligned(runner, new_fields, optional_values):
    state = make_batch(new_fields=new_fields, optional_values=optional_values)
    scheduler = SimpleNamespace(state=state, uniform=None)
    result, uniform = runner.gather_batch_req_state(scheduler, False)
    order = [1, 3, 0, 2]
    assert result.req_ids == ["decode_b", "decode_a", "fresh_prompt", "prompt_tail"]
    for field in REQUEST_FIELDS:
        np.testing.assert_array_equal(getattr(result, field), getattr(state, field)[order])
    assert result.num_tokens == state.num_tokens
    assert result.has_prefill and uniform is None
    # A fresh one-token prompt remains prefill and follows genuine decodes.
    assert result.is_prefilling_np[2]
    if new_fields:
        assert not result.decode_graph_eligible
        for field in ("num_draft_tokens_np", "prefill_runs_as_decode_np"):
            if optional_values:
                np.testing.assert_array_equal(getattr(result, field), getattr(state, field)[order])
            else:
                assert getattr(result, field) is None


def test_without_flashmla_keeps_parent_request_order(runner):
    runner.flashmla_executor = None
    state = make_batch(new_fields=True, optional_values=True)
    result, uniform = runner.gather_batch_req_state(SimpleNamespace(state=state, uniform=4), False)
    assert result is state and uniform == 4


def test_flashmla_dummy_batch_keeps_parent_result(runner):
    result = runner.gather_batch_req_state(SimpleNamespace(state=None, uniform=4), True)
    assert result == (None, 4)


def test_pd_recompute_retains_main_uniform_decode_helper(runner):
    runner.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_consumer=True, is_kv_producer=False)
    state = make_batch(new_fields=True, optional_values=True)
    state.num_scheduled_tokens[2] = 1
    result, uniform = runner.gather_batch_req_state(SimpleNamespace(state=state, uniform=None), False)
    assert result.req_ids == ["decode_b", "prompt_tail", "decode_a", "fresh_prompt"]
    runner._pd_uniform_decode_token_count.assert_called_once()
    assert uniform == 1 and result.has_prefill
