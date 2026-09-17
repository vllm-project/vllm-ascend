# SPDX-License-Identifier: Apache-2.0
"""PD prompt-tail graph dispatch keeps real prompt input preparation intact."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_runner_methods(base):
    path = Path(__file__).parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"gather_batch_req_state", "_get_attention_prefill_mask", "_pad_query_start_loc_for_fia"}
    ]
    namespace = {
        "GPUModelRunner": base,
        "np": np,
        "is_pd_decode_node": lambda config: config.consumer,
        "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(cls), str(path), "exec"), namespace)
    return namespace["NPUModelRunner"]


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize(
    "case,expected_uniform",
    [
        ("received_tail", 4),
        ("steady_decode", 4),
        ("missing_prefix", None),
        ("cold_prompt", None),
        ("resumed_output", None),
        ("short_verifier", None),
        ("missing_draft_rows", None),
        ("producer", None),
    ],
)
def test_pd_tail_dispatch_matches_attention_phase(dcp, concurrency, case, expected_uniform):
    prompt_len = 131072 if case != "cold_prompt" else 1
    computed = prompt_len - 1
    prefill_len = prompt_len
    query_len = 4
    if case == "steady_decode":
        computed = prompt_len + 10
    elif case == "missing_prefix":
        computed -= 3
    elif case == "resumed_output":
        prefill_len += 4
    elif case == "short_verifier":
        query_len = 2
    req_ids = [f"r{i}" for i in range(concurrency)]
    state = SimpleNamespace(
        req_ids=req_ids,
        idx_mapping_np=np.arange(concurrency),
        num_scheduled_tokens=np.full(concurrency, query_len),
        prefill_len_np=np.full(concurrency, prefill_len),
        is_prefilling_np=np.full(concurrency, computed < prefill_len),
        has_prefill=computed < prefill_len,
    )
    original_mask = state.is_prefilling_np.copy()

    class Base:
        def gather_batch_req_state(self, output, dummy_run):
            return state, 4 if case == "steady_decode" else None

    runner = load_runner_methods(Base)()
    runner.vllm_config = SimpleNamespace(consumer=case != "producer")
    runner.use_dcp = dcp > 1
    runner.decode_query_len = 4
    runner.req_states = SimpleNamespace(
        num_computed_tokens_np=np.full(concurrency, computed),
        prompt_len=SimpleNamespace(np=np.full(concurrency, prompt_len)),
    )
    output = SimpleNamespace(
        scheduled_spec_decode_tokens={r: [-1] * (2 if case == "missing_draft_rows" else 3) for r in req_ids}
    )
    returned_state, uniform = runner.gather_batch_req_state(output, False)
    assert uniform == expected_uniform
    assert returned_state is state
    np.testing.assert_array_equal(state.is_prefilling_np, original_mask)
    assert state.has_prefill == (computed < prefill_len)
    if case == "received_tail":
        assert state.has_prefill  # prepare_prefill_inputs still reads the real tail.
        assert not runner._get_attention_prefill_mask(state).any()
        runner.compilation_config = SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY")
        # The TP8 graph bucket pads one real request to two q=4 rows.
        padded_reqs = max(concurrency, 2)
        query_start = np.zeros(padded_reqs + 2, dtype=np.int32)
        query_start[: concurrency + 1] = np.arange(concurrency + 1) * 4
        query_start, count = runner._pad_query_start_loc_for_fia(
            padded_reqs * 4, padded_reqs, concurrency, query_start, "FULL", padded_reqs
        )
        assert count == padded_reqs
        np.testing.assert_array_equal(query_start[: count + 1], np.arange(count + 1) * 4)


def test_dummy_graph_capture_keeps_upstream_descriptor():
    class Base:
        def gather_batch_req_state(self, output, dummy_run):
            return None, 4

    runner = load_runner_methods(Base)()
    assert runner.gather_batch_req_state(None, True) == (None, 4)
