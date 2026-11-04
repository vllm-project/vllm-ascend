# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the PP speculation draft-token handoff helper."""

from __future__ import annotations
import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v1/model_runner.py"


class _PPGroup(SimpleNamespace):
    world_size = 2
    is_last_rank = False


def _load_runner_cls():
    tree = ast.parse(RUNNER_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    cls.body = [
        node for node in cls.body if getattr(node, "name", None) in {"_fill_pp_spec_token_ids", "_is_pd_prefill_worker"}
    ]

    class Parent:
        pass

    pp_group = _PPGroup()
    namespace = {"Parent": Parent, "get_pp_group": lambda: pp_group, "Callable": object}
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(
        compile(module, str(RUNNER_SOURCE), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["NPUModelRunner"], pp_group


NPUModelRunner, PP_GROUP = _load_runner_cls()


def _make_runner(draft, *, spec=True, producer=False, consumer=False, world_size=2):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(method="mtp") if spec else None
    runner.is_kv_producer = producer
    runner.is_kv_consumer = consumer
    runner.take_calls = 0

    def take_draft_token_ids():
        runner.take_calls += 1
        return draft

    runner.take_draft_token_ids = take_draft_token_ids
    PP_GROUP.world_size = world_size
    return runner


def test_maps_drafts_by_output_request_order():
    runner = _make_runner(SimpleNamespace(req_ids=["c", "a"], draft_token_ids=[[3, 4], [1, 2]]))
    output = SimpleNamespace()

    runner._fill_pp_spec_token_ids(output, ["a", "b", "c"])

    assert output.spec_token_ids == [[1, 2], [], [3, 4]]


def test_no_drafts_leaves_output_untouched():
    runner = _make_runner(None)
    output = SimpleNamespace()

    runner._fill_pp_spec_token_ids(output, ["a"])

    assert not hasattr(output, "spec_token_ids")
    assert runner.take_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 1},
        {"spec": False},
        {"producer": True},
    ],
)
def test_gated_off_paths_do_not_consume_drafts(kwargs):
    runner = _make_runner(SimpleNamespace(req_ids=["a"], draft_token_ids=[[1]]), **kwargs)
    output = SimpleNamespace()

    runner._fill_pp_spec_token_ids(output, ["a"])

    assert runner.take_calls == 0
    assert not hasattr(output, "spec_token_ids")


def test_empty_drafts_produce_empty_lists():
    runner = _make_runner(SimpleNamespace(req_ids=[], draft_token_ids=[]))
    output = SimpleNamespace()

    runner._fill_pp_spec_token_ids(output, ["a", "b"])

    assert output.spec_token_ids == [[], []]
