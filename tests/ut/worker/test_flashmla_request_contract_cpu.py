# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source-contract checks against the installed vLLM lane, without NPU work.

Execute upstream request gathering and descriptor matching with the candidate
override. No handwritten parent substitutes for vLLM's graph eligibility rules.
"""

import ast
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _node(source, name, owner=None):
    body = ast.parse(source).body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def _compile(nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), __file__, "exec"), scope)


@pytest.fixture
def contract():
    spec = find_spec("vllm")
    if spec is None or spec.origin is None:
        pytest.skip("requires the source of the tested vLLM lane")
    assert spec is not None and spec.origin is not None
    source_root = Path(spec.origin).parent
    upstream = (source_root / "v1/worker/gpu/model_runner.py").read_text(encoding="utf-8")
    utils = (source_root / "v1/worker/utils.py").read_text(encoding="utf-8")
    graphs = (source_root / "v1/worker/gpu/cudagraph_utils.py").read_text(encoding="utf-8")
    candidate = (ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text(encoding="utf-8")
    scope: dict[str, Any] = {"np": np, "NamedTuple": NamedTuple}
    _compile(
        [
            _node(upstream, "BatchReqState"),
            _node(upstream, "sort_batch_req_ids"),
            _node(utils, "is_uniform_query_len"),
            _node(utils, "get_uniform_decode_token_count"),
            _node(graphs, "_is_compatible"),
        ],
        scope,
    )
    main_contract = "decode_graph_eligible" in scope["BatchReqState"]._fields
    for name, source, owner, base in (
        ("Parent", upstream, "GPUModelRunner", ""),
        ("Candidate", candidate, "NPUModelRunner", "(Parent)"),
    ):
        cls = ast.parse(f"class {name}{base}: pass").body[0]
        assert isinstance(cls, ast.ClassDef)
        cls.body = [_node(source, "gather_batch_req_state", owner)]
        _compile([cls], scope)
    scope["vllm_model_runner"] = SimpleNamespace(get_uniform_decode_token_count=scope["get_uniform_decode_token_count"])
    override = _node(candidate, "NPUModelRunner")
    version_split = next(
        n
        for n in override.body
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "_pd_uniform_decode_token_count" for child in n.body
        )
    )
    pd_helper = (version_split.orelse if main_contract else version_split.body)[0]
    _compile([pd_helper], scope)
    scope["Candidate"]._pd_uniform_decode_token_count = scope["_pd_uniform_decode_token_count"]

    def gather(rows, *, external=True, pd=False, dummy=False):
        # row = request id, computed prompt tokens, scheduled tokens, draft count.
        req_ids = [row[0] for row in rows]
        runner = scope["Candidate"]()
        runner.req_states = SimpleNamespace(
            req_id_to_index={name: index for index, name in enumerate(req_ids)},
            prefill_len=SimpleNamespace(np=np.full(len(rows), 100)),
            num_computed_prefill_tokens=np.array([row[1] for row in rows]),
        )
        runner.decode_query_len = max(row[2] for row in rows)
        runner.pcp_manager = None
        runner.adaptive_verification = None
        transfer = SimpleNamespace(is_kv_consumer=True, is_kv_producer=False) if pd else None
        runner.vllm_config = SimpleNamespace(kv_transfer_config=transfer)
        runner.flashmla_executor = object() if external else None
        scheduler = SimpleNamespace(
            num_scheduled_tokens={row[0]: row[2] for row in rows},
            total_num_scheduled_tokens=sum(row[2] for row in rows),
            scheduled_spec_decode_tokens={row[0]: [0] * row[3] for row in rows if row[3]},
        )
        original = scope["Parent"].gather_batch_req_state(runner, scheduler, dummy)
        actual = runner.gather_batch_req_state(scheduler, dummy)
        return original, actual

    return SimpleNamespace(gather=gather, main=main_contract, compatible=scope["_is_compatible"])


@pytest.mark.parametrize(
    "rows",
    [
        [("tail", 99, 1, 0)],
        [("tail", 99, 1, 0), ("decode", 100, 1, 0)],
        [("tail", 99, 4, 3), ("decode", 100, 4, 3)],
        [("fresh", 0, 1, 0)],
        [("middle", 4, 1, 0)],
        [("chunk", 4, 2, 0)],
    ],
)
def test_fia_prefill_cannot_match_a_captured_decode_graph(contract, rows):
    (original, original_uniform), (state, uniform) = contract.gather(rows)
    assert state.has_prefill and uniform is None
    if contract.main:
        assert state.decode_graph_eligible is False
        if rows[0][0] in ("tail", "middle"):
            assert original.decode_graph_eligible and original_uniform is not None
    order = np.argsort(original.is_prefilling_np, kind="stable")
    assert state.req_ids == [original.req_ids[index] for index in order]
    for field in original._fields:
        expected, actual = getattr(original, field), getattr(state, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected[order])
    common = dict(num_reqs=len(rows), num_tokens=state.num_tokens, num_active_loras=0, num_ubatches=1)
    uniform_desc = SimpleNamespace(**common, uniform_token_count=max(row[2] for row in rows), max_query_len=None)
    assert not contract.compatible(uniform_desc, uniform_token_count=uniform, max_query_len=None, **common)
    varlen_desc = SimpleNamespace(**common, uniform_token_count=None, max_query_len=4)
    assert not contract.compatible(varlen_desc, uniform_token_count=uniform, max_query_len=None, **common)


@pytest.mark.parametrize("widths,expected_uniform", [([1, 1], 1), ([1, 4], None)])
def test_real_decode_keeps_upstream_graph_shape(contract, widths, expected_uniform):
    rows = [(str(index), 100, width, 0) for index, width in enumerate(widths)]
    _, (state, uniform) = contract.gather(rows)
    assert not state.has_prefill and uniform == expected_uniform
    if contract.main:
        assert state.decode_graph_eligible


@pytest.mark.parametrize("external", [False, True])
def test_dummy_request_state_keeps_upstream_result(contract, external):
    original, actual = contract.gather([("dummy", 0, 1, 0)], external=external, dummy=True)
    assert actual == original


def test_disabled_backend_keeps_upstream_prompt_tail_policy(contract):
    (original, original_uniform), (state, uniform) = contract.gather([("tail", 99, 1, 0)], external=False)
    assert state.has_prefill and uniform == original_uniform
    if contract.main:
        assert state.decode_graph_eligible == original.decode_graph_eligible


@pytest.mark.parametrize("mixed", [False, True])
def test_existing_pd_reclassification_stays_consistent(contract, mixed):
    rows = [("tail", 99, 1, 0)]
    if mixed:
        rows.append(("fresh", 0, 1, 0))
    _, (state, uniform) = contract.gather(rows, pd=True)
    assert state.has_prefill is mixed
    assert uniform == (None if mixed else 1)
    if contract.main:
        assert state.decode_graph_eligible is not mixed


def test_pd_reclassification_preserves_ineligibility_for_uniform_graphs(contract):
    (original, _), (state, uniform) = contract.gather([("tail", 98, 2, 0)], pd=True)
    assert not state.has_prefill
    if contract.main:
        assert not original.decode_graph_eligible
        assert not state.decode_graph_eligible
        assert uniform is None
        common = dict(num_reqs=1, num_tokens=2, num_active_loras=0, num_ubatches=1)
        descriptor = SimpleNamespace(**common, uniform_token_count=2, max_query_len=None)
        assert not contract.compatible(descriptor, uniform_token_count=uniform, max_query_len=2, **common)
    else:
        # The legacy lane has no independent graph-eligibility field.
        assert uniform == 2
