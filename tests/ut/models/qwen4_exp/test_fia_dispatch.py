# SPDX-License-Identifier: Apache-2.0
"""Mask semantics and scheduler-shape coverage without requiring a native NPU."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp"


def _load_source(filename, namespace):
    source = MODEL_DIR / filename
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(module, str(source), "exec"), namespace)


@pytest.fixture
def implementation():
    namespace = {"torch": torch, "dataclass": dataclass, "__name__": __name__}
    _load_source("gbsa.py", namespace)
    _load_source("fia.py", namespace)
    _load_source("attention_dispatch.py", namespace)
    return namespace


def test_mask_discards_invalid_and_duplicate_indices(implementation):
    indices = torch.tensor([[[0, 2, 5, -1], [1, 3, 7, -1], [0, 4, -1, -1], [2, 2, 5, -1]]])
    mask = implementation["build_selected_token_mask"](indices, 6, 8)
    expected = torch.tensor(
        [[1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0]],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask[0, 0], expected)


@pytest.mark.parametrize("visible", range(12))
def test_compact_groups_and_implicit_tail_respect_each_query_position(implementation, visible):
    # Deliberately poison explicit tail slots: GBSA derives its tail from visibility.
    compact = torch.tensor([[0, 1, 8, -1, 999, 999, 999]], dtype=torch.int32)
    selected = implementation["expand_qsa_selected_tokens"](compact, torch.tensor([visible]))
    allowed = implementation["build_selected_token_mask"](selected[None], torch.tensor([[visible]]), 16)
    expected = torch.zeros(16, dtype=torch.bool)
    expected[: min(visible // 4 * 4, 8)] = True
    expected[visible // 4 * 4 : visible] = True
    torch.testing.assert_close(allowed[0, 0, 0], expected)


def test_fia_passes_gqa_heads_causal_mask_and_zeroes_empty_rows(implementation):
    calls = []

    def fia(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(kwargs["query"]), None

    implementation["torch_npu"] = SimpleNamespace(npu_fused_infer_attention_score_v2=fia)
    query = torch.zeros(4, 4, 8, dtype=torch.bfloat16)
    key = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    compact = torch.tensor([[0, -1, -1, -1]] * 4)
    result = implementation["qsa_fia_attention"](
        query,
        key,
        key,
        compact,
        torch.tensor([0, 2, 3, 4]),
        4,
        torch.empty_like(query),
    )
    assert calls[0]["num_query_heads"] == 4
    assert calls[0]["num_key_value_heads"] == 1
    assert calls[0]["block_table"] is None
    assert calls[0]["atten_mask"][0, 0, 1].tolist() == [False, False, True, True]
    assert torch.all(result[0] == 0)
    assert torch.all(result[1:] == 1)


@pytest.mark.parametrize("groups", [31, 128, 511, 512])
@pytest.mark.parametrize("tail", range(4))
@pytest.mark.parametrize("holes", [False, True])
def test_fia_mask_matches_latest_gbsa_explicit_tail_policy(implementation, groups, tail, holes):
    from tests.ut.models.qwen4_exp.gbsa_reference import reference_prepare

    visible = 4096 + tail
    indices = torch.full((1, groups + 3), -1, dtype=torch.int32)
    indices[0, :groups] = torch.arange(groups - 1, -1, -1)
    if holes:
        indices[0, 2:groups:7] = -1
    lengths = torch.tensor([visible], dtype=torch.int32)
    packed, counts, *_ = reference_prepare(
        indices,
        torch.tensor([visible - 1]),
        lengths,
        torch.tensor([0]),
        torch.arange(33, dtype=torch.int32)[None],
        1,
    )
    expected = torch.zeros(visible, dtype=torch.bool)
    for group in packed[0, 0, : counts[0, 0]].tolist():
        expected[group * 4 : min(group * 4 + 4, visible)] = True
    selected = implementation["expand_qsa_selected_tokens"](indices, lengths)
    actual = implementation["build_selected_token_mask"](selected[None], lengths[None], visible)
    torch.testing.assert_close(actual[0, 0, 0], expected)


def test_fia_descriptor_page_mapping_is_zero_copy(implementation):
    storage = torch.arange(8 * 128 * 2 * 4).reshape(8, 128, 2, 4)
    cache = storage.as_strided((3, 128, 2, 4), (2 * 128 * 8, 8, 4, 1))
    key, value, table, width = implementation["prepare_qsa_fia_paged_kv"](
        cache,
        cache,
        torch.tensor([[2, 0, -1]], dtype=torch.int32),
        257,
    )
    assert key.data_ptr() == cache.data_ptr()
    assert value.data_ptr() == cache.data_ptr()
    assert table.tolist() == [[4, 0, 0]]
    assert width == 384
    torch.testing.assert_close(key[4], cache[2].flatten(1))


@pytest.mark.parametrize(
    "query_length,kv_length,prefill",
    [
        (1, 1, True),
        (1, 32768, True),
        (4096, 4096, True),
        (4096, 32768, True),
        (1, 128, False),
        (4, 16383, False),
        (8, 32768, False),
    ],
)
def test_route_uses_explicit_phase_not_query_or_kv_length(implementation, query_length, kv_length, prefill):
    plan = implementation["build_qsa_request_plan"](
        torch.tensor([0, query_length]), torch.tensor([kv_length]), query_length, torch.tensor([prefill])
    )
    assert plan[0].is_prefill is prefill
    assert plan[0].use_paged_kv is (query_length != kv_length)
    if not prefill:
        assert not plan[0].use_fia


@pytest.mark.parametrize(
    "query_length,kv_length,expected_fia",
    [
        (1, 24575, True),
        (1, 24576, True),
        (1, 24577, False),
        (4, 24578, False),
        (4096, 24576, True),
        (4096, 28671, False),
        (4096, 28672, False),
        (4096, 32768, False),
        (131072, 131072, False),
    ],
)
def test_total_kv_policy_including_large_initial_prefill(implementation, query_length, kv_length, expected_fia):
    plan = implementation["build_qsa_request_plan"](
        torch.tensor([0, query_length]), torch.tensor([kv_length]), query_length, torch.tensor([True])
    )
    request = plan[0]
    assert request.use_fia is expected_fia
    assert request.fia_query_end == (query_length if expected_fia else 0)
    assert request.gbsa_query_start == (query_length if expected_fia else 0)


def run_stub_batch(implementation, phases, query_lengths, kv_lengths):
    starts = [0]
    for length in query_lengths:
        starts.append(starts[-1] + length)
    total = starts[-1]
    q = torch.arange(total, dtype=torch.float32).reshape(total, 1, 1)
    out = torch.zeros_like(q)
    calls = []

    def prepare(k, v, table, kv):
        return k, v, table, kv

    def fia(q, k, v, idx, visible, kv, out, **kwargs):
        calls.append(("fia", q.flatten().tolist()))
        out.fill_(3)

    def gbsa(q, k, v, idx, table, requests, positions, lengths, out):
        calls.append(("gbsa", q.flatten().tolist()))
        out.fill_(7)
        return out

    implementation.update(prepare_qsa_fia_paged_kv=prepare, qsa_fia_attention=fia, qsa_gbsa_attention=gbsa)
    plan = implementation["build_qsa_request_plan"](
        torch.tensor(starts), torch.tensor(kv_lengths), total, torch.tensor(phases)
    )
    requests = torch.repeat_interleave(torch.arange(len(phases)), torch.tensor(query_lengths))
    implementation["dispatch_qsa_attention"](
        q,
        q,
        q,
        q,
        q,
        torch.zeros(total, 7),
        torch.zeros(len(phases), 256),
        requests,
        torch.arange(total),
        torch.tensor(kv_lengths),
        out,
        plan,
    )
    expected = []
    for phase, query_length, kv_length in zip(phases, query_lengths, kv_lengths):
        fia_rows = 0
        if phase:
            fia_rows = query_length if kv_length <= 24576 else 0
        expected.extend([3] * fia_rows)
        expected.extend([7] * (query_length - fia_rows))
    torch.testing.assert_close(out.flatten(), torch.tensor(expected, dtype=torch.float32))
    return calls


def test_pure_prefill_routes_by_logical_kv_length(implementation):
    calls = run_stub_batch(implementation, [True, True, True], [2, 4, 1], [4096, 24578, 24577])
    assert calls == [("fia", [0, 1]), ("gbsa", [2, 3, 4, 5, 6])]


def test_pure_decode_is_one_gbsa_call_including_mtp(implementation):
    calls = run_stub_batch(implementation, [False, False, False], [1, 4, 8], [1, 8192, 32768])
    assert calls == [("gbsa", list(range(13)))]


def test_mixed_batch_runs_prefill_first_then_contiguous_decode_spans(implementation):
    calls = run_stub_batch(
        implementation,
        [False, True, True, False, False, True, False],
        [1, 2, 1, 1, 4, 1, 1],
        [4096, 24576, 24577, 16384, 12, 1, 8192],
    )
    assert calls == [
        ("fia", [1, 2]),
        ("fia", [9]),
        ("gbsa", [0]),
        ("gbsa", [3, 4, 5, 6, 7, 8]),
        ("gbsa", [10]),
    ]


def test_decode_sentinel_bypasses_request_planner(implementation):
    calls = []
    implementation["qsa_gbsa_attention"] = lambda *args: calls.append(args) or args[-1]
    q = torch.zeros(4, 3, 8)
    implementation["dispatch_qsa_attention"](
        q, q, q, q, q, torch.zeros(4, 7), torch.zeros(1, 1), torch.zeros(4), torch.arange(4), torch.tensor([4]), q, None
    )
    assert len(calls) == 1 and calls[0][0] is q


def test_padding_requests_are_ignored(implementation):
    plan = implementation["build_qsa_request_plan"](
        torch.tensor([0, 1, 3, 8]), torch.tensor([4096, 32768, 0]), 3, torch.tensor([False, True, False])
    )
    assert [(r.query_start, r.query_end, r.is_prefill) for r in plan] == [(0, 1, False), (1, 3, True)]


def test_phase_metadata_required_to_cover_requests(implementation):
    with pytest.raises(ValueError, match="every request"):
        implementation["build_qsa_request_plan"](
            torch.tensor([0, 1, 2]), torch.tensor([10, 20]), 2, torch.tensor([False])
        )


def load_builder():
    source = MODEL_DIR / "qsa.py"
    builder = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendQSAMetadataBuilder"
    )

    class Base:
        def build(self, *args, **kwargs):
            return SimpleNamespace(block_table="kept")

    plans = []

    def plan(*args):
        plans.append(args)
        return ("prefill-plan",)

    namespace = dict(
        upstream_qsa=SimpleNamespace(Qwen4ExpQSAMetadataBuilder=Base),
        AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH="uniform"),
        CommonAttentionMetadata=object,
        build_qsa_request_plan=plan,
    )
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["AscendQSAMetadataBuilder"](), plans


def test_metadata_builder_keeps_decode_capture_on_gbsa():
    builder, plans = load_builder()
    assert builder._cudagraph_support == "uniform"
    common = SimpleNamespace(is_prefilling=torch.tensor([False]), seq_lens_cpu_upper_bound=None)
    assert builder.build(0, common).qsa_request_plan is None
    # Capture dummy inputs can look like prefills; the decode capture contract
    # must not freeze FIA or host KV lengths into the graph.
    common.is_prefilling = torch.tensor([True])
    assert builder.build_for_cudagraph_capture(common).qsa_request_plan is None
    assert plans == []


def test_metadata_builder_uses_phase_flags_and_host_lengths_for_prefill():
    builder, plans = load_builder()
    common = SimpleNamespace(
        is_prefilling=torch.tensor([True]),
        seq_lens_cpu_upper_bound=torch.tensor([32768]),
        query_start_loc_cpu=torch.tensor([0, 4096]),
        num_actual_tokens=4096,
    )
    meta = builder.build(0, common)
    assert meta.block_table == "kept" and meta.qsa_request_plan == ("prefill-plan",)
    assert plans[0][-1] is common.is_prefilling


def test_metadata_builder_rejects_unknown_phase_instead_of_guessing():
    builder, _ = load_builder()
    with pytest.raises(ValueError, match="is_prefilling"):
        builder.build(0, SimpleNamespace(is_prefilling=None))


@pytest.mark.parametrize(
    "architecture,has_prefill,dummy",
    [
        ("Qwen4ExpForConditionalGeneration", True, False),
        ("Qwen4ExpForConditionalGeneration", False, False),
        ("OtherModel", True, False),
        ("Qwen4ExpForConditionalGeneration", True, True),
    ],
)
def test_runner_keeps_qwen_prompt_tails_out_of_decode_graph(architecture, has_prefill, dummy):
    source = MODEL_DIR.parents[1] / "worker/v2/model_runner.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(
        n
        for n in module.body
        if isinstance(n, ast.ClassDef)
        and any(isinstance(m, ast.FunctionDef) and m.name == "gather_batch_req_state" for m in n.body)
    )
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "gather_batch_req_state")
    method.args.args[1].annotation = None
    from collections import namedtuple

    State = namedtuple("State", "has_prefill decode_graph_eligible prefill_runs_as_decode_np")
    state = State(has_prefill, True, "tail")

    class Base:
        def gather_batch_req_state(self, *args):
            return state, 1

    namespace = dict(Base=Base)
    child = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[child], type_ignores=[])), str(source), "exec"), namespace)
    runner = namespace["Runner"]()
    runner._uses_flashmla = False
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))
    actual, uniform = runner.gather_batch_req_state(None, dummy)
    changed = architecture == "Qwen4ExpForConditionalGeneration" and has_prefill and not dummy
    assert actual.decode_graph_eligible is (not changed)
    assert actual.prefill_runs_as_decode_np == (None if changed else "tail")
    assert uniform == (None if changed else 1)


def test_fia_helper_tiles_mask_memory_for_eligible_prefill(implementation):
    calls = []
    implementation["QSA_FIA_MASK_BUDGET_BYTES"] = 64
    implementation["torch_npu"] = SimpleNamespace(
        npu_fused_infer_attention_score_v2=lambda **kwargs: (
            calls.append(kwargs) or torch.ones_like(kwargs["query"]),
            None,
        )
    )
    query = torch.zeros(5, 2, 4, dtype=torch.bfloat16)
    key = torch.zeros(32, 1, 4, dtype=query.dtype)
    implementation["qsa_fia_attention"](
        query,
        key,
        key,
        torch.tensor([[0, -1, -1, -1]] * 5),
        torch.tensor([28, 29, 30, 31, 32]),
        32,
        torch.empty_like(query),
    )
    assert [c["query"].shape[1] for c in calls] == [2, 2, 1]
