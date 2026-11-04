# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production adapter methods against the observed upstream contract."""

import __future__

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def spec():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDSparkSpeculator")
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ["propose", "build_draft_attn_metadatas", "_draft_metadata_request_count"]
    ]

    class Parent:
        def _build_uniform_attn_metadata(self, **kwargs):
            n = kwargs["num_reqs"]
            p = kwargs["batch_desc"].num_reqs
            # Same CPU endpoint copy that failed in installed vLLM, rather than an
            # expectation about which adapter attribute should be set.
            endpoint = torch.empty(p + 1, dtype=torch.int32)
            endpoint[: n + 1] = torch.from_numpy(np.arange(n + 1, dtype=np.int32) * kwargs["num_query_per_req"])
            endpoint[n:] = endpoint[n]
            self.observed.append(dict(real=n, padded=p, endpoints=endpoint.tolist()))
            return {"draft": object()}

        def propose(self, *args, **kwargs):
            input_batch = args[0]
            sync = args[11]
            dummy = args[12]
            actual = min(input_batch.num_reqs, sync.num_reqs) if dummy and sync is not None else input_batch.num_reqs
            assert actual == self.expected_actual
            result = self.build_draft_attn_metadatas(self.graph_padded, input_batch.seq_lens_cpu_upper_bound)
            if self.fail_parent:
                raise RuntimeError("parent failure after metadata")
            return result

    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Adapter",
                bases=[ast.Name(id="Parent", ctx=ast.Load())],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(
        Parent=Parent,
        torch=torch,
        contextmanager=contextmanager,
        AscendAttentionState=SimpleNamespace(ChunkedPrefill="chunked_prefill"),
        BatchExecutionDescriptor=SimpleNamespace,
        CUDAGraphMode=SimpleNamespace(FULL="full"),
        build_attn_metadata_wrapper=nullcontext,
        build_attn_metadata_factory=lambda *a, **kw: nullcontext(),
    )
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    obj = namespace["Adapter"]()
    obj.replicated_pcp = False
    obj.enable_adaptive_verification = False
    obj.use_dcp = False
    obj.attn_architecture = "GQA"
    obj.input_buffers = SimpleNamespace(positions=object())
    obj.max_num_tokens = 512
    obj.attn_vllm_config = SimpleNamespace(parallel_config=object())
    obj._group_causal = False
    obj._prepare_draft_dcp_metadata_inputs = lambda real, padded, step: (None, torch.zeros(padded, dtype=torch.bool))
    obj.observed = []
    obj.fail_parent = False
    return obj


def invoke(spec, *, real=4, sync=3, dummy=True, skip=False):
    batch = SimpleNamespace(
        num_reqs=real,
        is_prefilling_np=np.zeros(real, dtype=np.bool_),
        seq_lens_cpu_upper_bound=torch.ones(real, dtype=torch.int32),
    )
    state = None if sync is None else SimpleNamespace(num_reqs=sync)
    result = spec.propose(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=None,
        aux_hidden_states=None,
        num_sampled=None,
        num_rejected=None,
        last_sampled=None,
        next_prefill_tokens=None,
        temperature=None,
        seeds=None,
        dp_sync_state=state,
        dummy_run=dummy,
        skip_attn_for_dummy_run=skip,
    )
    assert batch.num_reqs == real
    assert getattr(spec, "_draft_metadata_num_reqs", None) is None
    return result


@pytest.mark.parametrize("width", [3, 7, 8])
def test_idle_dp_dummy_reduces_four_target_requests_to_three_draft_slots(spec, width):
    spec.num_query_per_req = width
    spec.expected_actual = 3
    spec.graph_padded = 3
    assert invoke(spec)
    assert spec.observed == [dict(real=3, padded=3, endpoints=[0, width, 2 * width, 3 * width])]


@pytest.mark.parametrize("width", [3, 7, 8])
@pytest.mark.parametrize(
    "real,sync,dummy,padded", [(1, 3, False, 3), (4, 6, True, 6), (4, None, True, 4), (4, 3, False, 4)]
)
def test_real_requests_and_extra_graph_padding_remain_intact(spec, width, real, sync, dummy, padded):
    spec.num_query_per_req = width
    spec.expected_actual = min(real, sync) if dummy and sync is not None else real
    spec.graph_padded = padded
    invoke(spec, real=real, sync=sync, dummy=dummy)
    row = spec.observed[0]
    assert row["real"] == spec.expected_actual
    assert row["endpoints"] == [min(i, row["real"]) * width for i in range(padded + 1)]


@pytest.mark.parametrize("kind", ["profiling", "replicated", "adaptive"])
def test_paths_that_discard_target_dp_sync_keep_original_batch(spec, kind):
    spec.num_query_per_req = 3
    spec.expected_actual = 4
    spec.graph_padded = 4
    spec.replicated_pcp = kind == "replicated"
    spec.enable_adaptive_verification = kind == "adaptive"
    invoke(spec, skip=kind == "profiling")
    assert spec.observed[0]["real"] == 4


def test_count_is_restored_when_parent_raises(spec):
    spec.num_query_per_req = 3
    spec.expected_actual = 3
    spec.graph_padded = 3
    spec.fail_parent = True
    with pytest.raises(RuntimeError, match="parent failure after metadata"):
        invoke(spec)
    assert getattr(spec, "_draft_metadata_num_reqs", None) is None


def test_real_requests_cannot_be_silently_dropped_to_fit_graph(spec):
    spec.num_query_per_req = 3
    spec.expected_actual = 4
    spec.graph_padded = 3
    with pytest.raises(AssertionError):
        invoke(spec, dummy=False)
    assert getattr(spec, "_draft_metadata_num_reqs", None) is None


def test_nested_count_scopes_restore_outer_value(spec):
    with spec._draft_metadata_request_count(2):
        with spec._draft_metadata_request_count(3):
            assert spec._draft_metadata_num_reqs == 3
        assert spec._draft_metadata_num_reqs == 2
    assert spec._draft_metadata_num_reqs is None
