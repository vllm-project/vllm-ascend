# SPDX-License-Identifier: Apache-2.0
"""Actual query gather/finish routing and async dependency contracts."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "tokens,decode,dsa,prepared",
    [
        (1, True, False, False),
        (6, True, False, True),
        (12, True, False, True),
        (13, True, False, False),
        (6, False, False, False),
        (6, True, True, False),
    ],
)
def test_query_gather_waits_before_final_unpack_and_preserves_fallback(tokens, decode, dsa, prepared):
    source = ROOT / "vllm_ascend/attention/context_parallel/sfa_cp.py"
    tree = ast.parse(source.read_text())
    context_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DCPGatherContext")
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFADCPImpl")
    methods = [
        n
        for n in impl.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_start_dcp_query_gather", "_finish_dcp_gather", "_record_query_gather_context")
    ]
    for method in methods:
        method.decorator_list = []
    events = []
    group = object()
    handle = SimpleNamespace(wait=lambda: events.append("wait"))

    def gather(value, selected_group):
        assert selected_group is group
        events.append("gather")
        return torch.cat([value] * 8, dim=0), handle

    def prep(qn, qr):
        events.append("prepare")
        return torch.cat((qn, qr), dim=-1).permute(1, 0, 2).contiguous()

    def unpack(value):
        assert events[-1] == "wait"
        events.append("unpack")
        return tuple(part.contiguous() for part in value.permute(1, 0, 2).split((512, 64), dim=-1))

    namespace = {
        "torch": torch,
        "M": object,
        "AscendSFADCPMetadata": SimpleNamespace,
        "NamedTuple": NamedTuple,
        "all_gather_async": gather,
        "can_prepare_query": lambda qn, qr: 1 < qn.shape[0] <= 12,
        "prepare_query_head_major": prep,
        "unpack_query": unpack,
    }
    exec(compile(ast.Module(body=[context_class, *methods], type_ignores=[]), str(source), "exec"), namespace)
    context_type = namespace["DCPGatherContext"]

    def legacy(value, dim, split_sizes):
        send = value if dim == 0 else value.permute(1, 0, 2).contiguous()
        gathered, work = gather(send, group)
        # Four-argument construction must keep older KV/query callers working.
        return context_type(gathered, work, None if dim == 0 else (1, 0, 2), split_sizes)

    state = SimpleNamespace(
        enable_dsa_cp=dsa,
        dcp_size=8,
        dcp_group=group,
        _start_dcp_gather=legacy,
        _parallel_query_gather_dim=lambda: 0 if dsa else 1,
        _has_prefill=lambda metadata: metadata.num_prefills > 0,
    )
    state._start_dcp_query_gather = lambda *args, **kwargs: namespace["_start_dcp_query_gather"](state, *args, **kwargs)
    qn = torch.arange(tokens * 8 * 512).to(torch.bfloat16).view(8, tokens, 512).transpose(0, 1)
    qr = torch.ones(tokens, 8, 64, dtype=torch.bfloat16)
    metadata = SimpleNamespace(num_prefills=0 if decode else 1, dcp_context=SimpleNamespace(gather_context=None))
    namespace["_record_query_gather_context"](state, qn, qr, metadata)
    context = metadata.dcp_context.gather_context
    if not decode:
        assert context is None and not events
        return
    assert context.query_prepared is prepared
    assert "wait" not in events
    result = namespace["_finish_dcp_gather"](context)
    dim = 0 if dsa else 1
    assert torch.equal(result[0], torch.cat([qn] * 8, dim=dim))
    assert torch.equal(result[1], torch.cat([qr] * 8, dim=dim))
    assert events.count("gather") == events.count("wait") == 1
    assert ("unpack" in events) is prepared
    if prepared:
        assert all(part.is_contiguous() for part in result)


@pytest.mark.parametrize("tokens,expected", [(0, False), (1, False), (2, True), (6, True), (12, True), (13, False)])
def test_query_shape_gate_keeps_single_token_and_large_batch_paths(tokens, expected):
    source = ROOT / "vllm_ascend/ops/triton/sfa_dcp_query.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "can_prepare_query")
    namespace = {
        "torch": torch,
        "triton": object(),
        "_LOCAL_HEADS": 8,
        "_NOPE_DIM": 512,
        "_ROPE_DIM": 64,
        "_MAX_DECODE_TOKENS": 12,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    device = SimpleNamespace(type="npu")
    qn = SimpleNamespace(
        device=device, dtype=torch.bfloat16, ndim=3, shape=(tokens, 8, 512), stride=lambda: (512, tokens * 512, 1)
    )
    qr = SimpleNamespace(
        device=device, dtype=torch.bfloat16, ndim=3, shape=(tokens, 8, 64), stride=lambda: (512, 64, 1)
    )
    assert namespace["can_prepare_query"](qn, qr) is expected
