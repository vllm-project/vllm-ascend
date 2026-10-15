# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route propagation checks using production statements, without NPU work."""

import ast
from collections import namedtuple
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNNER = "vllm_ascend/worker/v2/model_runner.py"


def node(path, name, owner=None):
    body = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), __file__, "exec"), scope
    )


@pytest.fixture
def phase():
    scope = {"np": np}
    execute([node("vllm_ascend/attention/flashmla.py", "flashmla_prefill_flags")], scope)
    return scope["flashmla_prefill_flags"]


def state(computed, width, *, modern=False, mask=None):
    fields = "req_ids num_scheduled_tokens num_computed_prefill_tokens_np is_prefilling_np has_prefill"
    values = [["request"], np.array([width]), np.array([computed]), np.array([True]), True]
    if modern:
        fields += " prefill_runs_as_decode_np"
        values.append(None if mask is None else np.array([mask]))
    return namedtuple("State", fields)(*values)


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize(
    "computed,width,drafts,prefill",
    [(99, 1, 0, False), (99, 4, 3, False), (4, 1, 0, False), (0, 1, 0, True), (0, 4, 3, True), (4, 2, 0, True)],
)
def test_phase_promotes_only_one_new_token_over_context(phase, modern, computed, width, drafts, prefill):
    batch = state(computed, width, modern=modern, mask=not prefill)
    semantic = batch.is_prefilling_np.copy()
    result = phase(batch, {"request": [0] * drafts}, allow_short_extends=True)
    assert result.tolist() == [prefill]
    np.testing.assert_array_equal(batch.is_prefilling_np, semantic)
    assert batch.has_prefill
    assert not np.shares_memory(result, batch.is_prefilling_np)


@pytest.mark.parametrize("mask", [None, False])
def test_main_does_not_replace_an_absent_or_false_upstream_mask(phase, mask):
    batch = state(99, 1, modern=True, mask=mask)
    assert phase(batch, {}, allow_short_extends=True).tolist() == [True]


def test_pcp_does_not_promote_prompt_extensions(phase):
    assert phase(state(99, 1, modern=True, mask=True), {}, allow_short_extends=False).tolist() == [True]


@pytest.mark.parametrize("native_dcp", [False, True])
def test_mixed_native_dcp_groups_keep_a_shared_prefill_order(phase, native_dcp):
    builders = [
        SimpleNamespace(dcp_enabled=True, flashmla_state=object()),
        SimpleNamespace(dcp_enabled=native_dcp, flashmla_state=None),
    ]
    groups = [[SimpleNamespace(get_metadata_builder=lambda _, b=b: b) for b in builders]]
    batch = state(99, 1, modern=True, mask=True)
    assert phase(batch, {}, allow_short_extends=True, attn_groups=groups).tolist() == [native_dcp]
    assert batch.is_prefilling_np.tolist() == [True]


def test_input_phase_is_saved_before_adaptive_compaction(phase):
    method = node(RUNNER, "_prepare_inputs_impl", "NPUModelRunner")
    phase_assignment = next(
        n
        for n in method.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "flashmla_is_prefilling_np" for t in n.targets)
    )
    compaction = next(
        n
        for n in method.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "adaptive_verification_active"
    )
    assert method.body.index(phase_assignment) < method.body.index(compaction)
    batch = state(99, 4)
    drafts = np.array([3])

    def compact(counts, scheduled, logits):
        counts.fill(0)
        return np.array([1]), logits

    scope = {
        "flashmla_prefill_flags": phase,
        "self": SimpleNamespace(flashmla_executor=object(), pcp_manager=None),
        "batch_req_state": batch,
        "scheduler_output": SimpleNamespace(scheduled_spec_decode_tokens={"request": [0, 0, 0]}),
        "adaptive_verification_active": True,
        "adaptive_verification_manager": SimpleNamespace(compact_batch=compact),
        "num_draft_tokens_per_req": drafts,
        "num_scheduled_tokens_np": batch.num_scheduled_tokens,
        "cu_num_logits_np": np.array([0, 4]),
    }
    execute([phase_assignment, compaction], scope)
    assert drafts.tolist() == [0]
    assert scope["num_scheduled_tokens_np"].tolist() == [1]
    assert scope["flashmla_is_prefilling_np"].tolist() == [False]
    # Re-deriving from the mutated draft counts would wrongly choose prefill.
    assert phase(batch, {"request": []}, allow_short_extends=True).tolist() == [True]
    input_assignment = next(
        n
        for n in method.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "AscendInputBatch"
    )
    selected = copy(input_assignment)
    selected.value = copy(input_assignment.value)
    assert isinstance(selected.value, ast.Call)
    selected.value.keywords = [
        k for k in selected.value.keywords if k.arg in ("is_prefilling_np", "has_prefill", "flashmla_is_prefilling_np")
    ]
    scope["AscendInputBatch"] = SimpleNamespace
    execute([selected], scope)
    assert scope["input_batch"].has_prefill
    assert scope["input_batch"].is_prefilling_np.tolist() == [True]
    assert scope["input_batch"].flashmla_is_prefilling_np.tolist() == [False]


@pytest.mark.parametrize("capture", [False, True])
def test_factory_replaces_phase_only_for_the_external_builder(capture):
    method = node("vllm_ascend/worker/v2/attn_utils.py", "build_attn_metadata")
    loop = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and [t.id for t in n.target.elts if isinstance(t, ast.Name)] == ["variant_names", "variant_common"]
    )
    semantic = torch.tensor([True, False])
    execution = torch.tensor([False, False])
    common = SimpleNamespace(is_prefilling=semantic, positions=torch.arange(2))
    metadata: dict[str, Any] = {}
    scope = {
        "copy": copy,
        "flashmla_is_prefilling": execution,
        "for_cudagraph_capture": capture,
        "attn_metadata_extra_kwargs": {},
        "GDNAttentionMetadataBuilder": type("GDN", (), {}),
        "num_actual_reqs": 2,
        "is_dsa_builder": False,
        "executor": None,
        "attn_metadata": metadata,
    }
    for name, external in (("flash", True), ("state", False)):
        builder = SimpleNamespace(
            flashmla_state=object() if external else None,
            build=lambda **kwargs: kwargs["common_attn_metadata"],
            build_for_cudagraph_capture=lambda common, **kwargs: common,
        )
        scope.update(attn_metadata_builder=builder, metadata_variants=[([name], common)])
        execute([loop], scope)
    assert metadata["flash"] is not common
    assert metadata["flash"].is_prefilling is execution
    assert metadata["flash"].positions is common.positions
    assert metadata["state"] is common and common.is_prefilling is semantic


@pytest.mark.parametrize("kind", ["default", "mamba_hybrid"])
def test_model_state_passes_independent_phase_without_changing_semantics(kind):
    owner = "AscendModelState" if kind == "default" else "AscendMambaHybridModelState"
    method = node(f"vllm_ascend/worker/v2/model_states/{kind}.py", "prepare_attn", owner)
    phase_assignment = next(
        n
        for n in method.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "flashmla_phase" for t in n.targets)
    )
    build_call = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and any(k.arg == "flashmla_is_prefilling" for k in n.keywords)
    )
    selected = copy(build_call)
    selected.func = ast.Name(id="record_build", ctx=ast.Load())
    selected.keywords = [k for k in selected.keywords if k.arg == "flashmla_is_prefilling"]
    batch = SimpleNamespace(is_prefilling_np=np.array([True]), flashmla_is_prefilling_np=np.array([False]))
    record = Mock()
    execute(
        [phase_assignment, ast.Expr(value=selected)], {"torch": torch, "input_batch": batch, "record_build": record}
    )
    assert record.call_args.kwargs["flashmla_is_prefilling"].tolist() == [False]
    assert batch.is_prefilling_np.tolist() == [True]
