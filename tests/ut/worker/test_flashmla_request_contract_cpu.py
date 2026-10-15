# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source-contract checks against the paired vLLM release, without NPU work.

Execute upstream request gathering and descriptor matching with the candidate
override. No handwritten parent substitutes for vLLM's graph eligibility rules.
"""

import ast
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import Mock

import numpy as np
import pytest
import torch

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
    backend = (ROOT / "vllm_ascend/attention/flashmla.py").read_text(encoding="utf-8")
    scope: dict[str, Any] = {"np": np, "NamedTuple": NamedTuple}
    _compile(
        [
            _node(upstream, "BatchReqState"),
            _node(upstream, "sort_batch_req_ids"),
            _node(utils, "is_uniform_query_len"),
            _node(utils, "get_uniform_decode_token_count"),
            _node(graphs, "_is_compatible"),
            _node(backend, "prepare_flashmla_batch_req_state"),
            _node(backend, "split_flashmla_requests"),
        ],
        scope,
    )
    assert "decode_graph_eligible" not in scope["BatchReqState"]._fields, "requires the release request-state ABI"
    scope["vllm_model_runner"] = SimpleNamespace(get_uniform_decode_token_count=scope["get_uniform_decode_token_count"])
    for name, source, owner, base in (
        ("Parent", upstream, "GPUModelRunner", ""),
        ("Candidate", candidate, "NPUModelRunner", "(Parent)"),
    ):
        cls = ast.parse(f"class {name}{base}: pass").body[0]
        assert isinstance(cls, ast.ClassDef)
        cls.body = [_node(source, "gather_batch_req_state", owner)]
        _compile([cls], scope)

    def gather(rows, *, external=True, pd=False, dummy=False, with_inputs=False):
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
        runner._uses_flashmla = external
        # Selection is independent of the metadata executor's lifecycle.
        runner.flashmla_executor = None if external else object()
        scheduler = SimpleNamespace(
            num_scheduled_tokens={row[0]: row[2] for row in rows},
            total_num_scheduled_tokens=sum(row[2] for row in rows),
            scheduled_spec_decode_tokens={row[0]: [0] * row[3] for row in rows if row[3]},
        )
        original = scope["Parent"].gather_batch_req_state(runner, scheduler, dummy)
        actual = runner.gather_batch_req_state(scheduler, dummy)
        if with_inputs:
            return runner, scheduler, actual
        return original, actual

    return SimpleNamespace(
        gather=gather,
        compatible=scope["_is_compatible"],
        candidate=candidate,
        upstream=upstream,
        scope=scope,
    )


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
    (original, _), (state, uniform) = contract.gather(rows)
    assert state.has_prefill and uniform is None
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


@pytest.mark.parametrize("external", [False, True])
def test_dummy_request_state_keeps_upstream_result(contract, external):
    original, actual = contract.gather([("dummy", 0, 1, 0)], external=external, dummy=True)
    assert actual == original


@pytest.mark.parametrize(
    "rows",
    [
        [("tail", 99, 1, 0)],
        [("tail", 99, 1, 0), ("decode", 100, 1, 0)],
        [("prefill", 10, 4, 3), ("decode", 100, 4, 0), ("tail", 99, 1, 0)],
    ],
)
def test_disabled_backend_keeps_upstream_result(contract, rows):
    (original, original_uniform), (state, uniform) = contract.gather(rows, external=False)
    assert uniform == original_uniform
    for field in original._fields:
        expected, actual = getattr(original, field), getattr(state, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


@pytest.mark.parametrize("mixed", [False, True])
def test_existing_pd_reclassification_stays_consistent(contract, mixed):
    rows = [("tail", 99, 1, 0)]
    if mixed:
        rows.append(("fresh", 0, 1, 0))
    _, (state, uniform) = contract.gather(rows, pd=True)
    assert state.has_prefill is mixed
    assert uniform == (None if mixed else 1)


def test_pd_reclassification_keeps_release_uniform_graph_contract(contract):
    _, (state, uniform) = contract.gather([("tail", 98, 2, 0)], pd=True)
    assert not state.has_prefill
    assert uniform == 2


def test_reordered_mixed_batch_keeps_derived_inputs_aligned(contract):
    """Run production input/attention preparation with CPU device primitives.

    The scheduled prefill has drafts and sorts before decode upstream. Fixed
    request identities below catch a partial reorder reaching positions,
    sampling or KV consumers; these mocks do not validate device kernels.
    """
    rows = [("prefill", 10, 4, 3), ("decode", 100, 4, 0), ("prompt-tail", 99, 1, 0)]
    runner, scheduler, (state, uniform) = contract.gather(rows, with_inputs=True)
    assert state.req_ids == ["decode", "prefill", "prompt-tail"]
    assert state.has_prefill and uniform is None
    scope = dict(contract.scope)

    def copy_to_cpu(values, *, device=None, out=None):
        values = torch.as_tensor(values)
        return values.clone() if out is None else out.copy_(values)

    def positions_and_lengths(mapping, starts, computed, positions, lengths):
        for row, index in enumerate(mapping):
            start, end = int(starts[row]), int(starts[row + 1])
            positions[start:end] = torch.arange(end - start) + computed[index]
            lengths[row] = computed[index] + end - start

    def sampling_indices(ids, mapping, sampled, starts, lengths, prompts, drafts, cu_logits, total, bonus):
        return torch.cat(
            [torch.arange(starts[row + 1] - cu_logits[row + 1] + cu_logits[row], starts[row + 1]) for row in range(3)]
        )

    def expand_mapping(mapping, total, cu_logits, query_len):
        counts = cu_logits[1:] - cu_logits[:-1]
        return torch.repeat_interleave(mapping, counts), torch.cat([torch.arange(count) for count in counts])

    prefill_inputs = Mock()
    scope.update(
        torch=torch,
        CUDAGraphMode=SimpleNamespace(FULL="full"),
        async_copy_to_gpu=copy_to_cpu,
        build_attn_state=Mock(return_value="mixed"),
        expand_idx_mapping=expand_mapping,
        prepare_prefill_inputs=prefill_inputs,
        prepare_pos_seq_lens=positions_and_lengths,
        combine_sampled_and_draft_tokens=sampling_indices,
        AscendInputBatch=SimpleNamespace,
        update_cos_sin=Mock(),
        vllm_model_runner=SimpleNamespace(
            pcp=SimpleNamespace(maybe_partition_pcp_batch=lambda manager, batch, batch_desc: batch)
        ),
    )
    _compile(
        [
            _node(contract.candidate, "prepare_inputs", "NPUModelRunner"),
            _node(contract.upstream, "prepare_attn", "GPUModelRunner"),
        ],
        scope,
    )
    num_tokens = state.num_tokens
    runner.device = torch.device("cpu")
    runner.max_num_reqs = len(rows)
    runner.use_fia = True
    runner.kv_cache_config = object()
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.eplb = SimpleNamespace(set_batch_phase=Mock())
    runner._check_finegrained_tp_graph_step = Mock()
    runner.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        query_start_loc=torch.zeros(len(rows) + 2, dtype=torch.int32),
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        seq_lens=torch.zeros(len(rows), dtype=torch.int32),
        seq_lens_np=np.array([104, 14, 100], dtype=np.int32),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool),
    )
    runner._update_seq_lens_cpu = Mock()
    computed = np.array([row[1] for row in rows], dtype=np.int32)
    runner.req_states.num_computed_tokens_np = computed
    runner.req_states.num_computed_tokens = SimpleNamespace(gpu=torch.from_numpy(computed))
    runner.req_states.prefill_len.gpu = torch.full((len(rows),), 100, dtype=torch.int32)
    runner.req_states.all_token_ids = SimpleNamespace(gpu=object())
    runner.req_states.next_prefill_tokens = object()
    runner.req_states.last_sampled_tokens = object()
    runner.req_states.draft_tokens = object()
    scheduler.has_structured_output_requests = False
    descriptor = SimpleNamespace(cg_mode="none", num_tokens=num_tokens, num_reqs=None)
    batch = scope["prepare_inputs"](runner, scheduler, state, descriptor)
    runner._update_seq_lens_cpu.assert_called_once_with(scheduler, ["decode", "prefill", "prompt-tail"])
    np.testing.assert_array_equal(batch.idx_mapping_np, [1, 0, 2])
    np.testing.assert_array_equal(batch.query_start_loc_np, [0, 4, 8, 9])
    np.testing.assert_array_equal(batch.cu_num_logits_np, [0, 1, 5, 6])
    np.testing.assert_array_equal(batch.expanded_idx_mapping, [1, 0, 0, 0, 0, 2])
    np.testing.assert_array_equal(batch.positions, [100, 101, 102, 103, 10, 11, 12, 13, 99])
    np.testing.assert_array_equal(batch.seq_lens, [104, 14, 100])
    np.testing.assert_array_equal(batch.logits_indices, [3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(prefill_inputs.call_args.args[2], [1, 0, 2])

    def slots(mapping, starts, positions, *, num_tokens_padded):
        owners = torch.repeat_interleave(mapping, starts[1:] - starts[:-1])
        return (owners * 1000 + positions[:num_tokens_padded]).unsqueeze(0)

    runner.block_tables = SimpleNamespace(
        gather_block_tables=lambda mapping, **kwargs: (mapping[:, None] + 1000,),
        compute_slot_mappings=slots,
    )
    block_tables, slot_mapping = scope["prepare_attn"](runner, batch)
    np.testing.assert_array_equal(block_tables[0].flatten(), [1001, 1000, 1002])
    np.testing.assert_array_equal(slot_mapping[0], [1100, 1101, 1102, 1103, 10, 11, 12, 13, 2099])
    common = SimpleNamespace(
        is_prefilling=torch.from_numpy(batch.is_prefilling_np),
        query_start_loc_cpu=torch.from_numpy(batch.query_start_loc_np),
        num_reqs=batch.num_reqs,
        num_actual_tokens=batch.num_tokens,
    )
    assert scope["split_flashmla_requests"](common) == (1, 2, 4, 5)
