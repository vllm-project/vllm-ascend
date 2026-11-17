# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise SFA routing without initializing the device or loading Triton."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_methods(path, owner, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=[*future, *methods], type_ignores=[]), path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("pcp,dcp", [(1, 2), (2, 2), (2, 4)])
@pytest.mark.parametrize("tokens,rope_dim", [(0, 0), (1, 0), (3, 4)])
def test_query_gather_keeps_physical_pcp_shards_and_waits(supported, pcp, dcp, tokens, rope_dim):
    class Context(NamedTuple):
        gathered: torch.Tensor
        handle: object
        restore_perm: tuple
        split_sizes: tuple

    events = []
    group, tp_group = object(), object()

    def gather(tensor, selected):
        assert selected is (tp_group if pcp > 1 else group)
        events.append("gather")
        return torch.cat([tensor] * (dcp // pcp)), SimpleNamespace(wait=lambda: events.append("wait"))

    prep = Mock(side_effect=lambda q, r: torch.cat((q, r), dim=-1).transpose(0, 1).contiguous())

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            assert supported, f"Unsupported query gather imported {name}"
            return SimpleNamespace(prep_query_head_major=prep)
        return builtins.__import__(name, *args, **kwargs)

    namespace = load_methods(
        "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "AscendSFADCPImpl",
        ("_start_dcp_query_gather", "_finish_dcp_packed_gather", "_finish_dcp_gather"),
        {
            "torch": torch,
            "DCPGatherContext": Context,
            "supports_triton": lambda: supported,
            "get_pcp_group": lambda: SimpleNamespace(world_size=pcp),
            "get_tp_group": lambda: tp_group,
            "all_gather_async": gather,
            "__builtins__": {**vars(builtins), "__import__": guarded_import},
        },
    )
    state = SimpleNamespace(dcp_group=group, dcp_size=dcp)
    q = torch.arange(tokens * 2 * 8, dtype=torch.float32).reshape(tokens, 2, 8)
    rope = torch.ones(tokens, 2, rope_dim)
    context = namespace["_start_dcp_query_gather"](state, q, rope if rope_dim else None)
    assert events == ([] if pcp == dcp else ["gather"])
    cls = SimpleNamespace(_finish_dcp_packed_gather=namespace["_finish_dcp_packed_gather"])
    actual = namespace["_finish_dcp_gather"](cls, context)
    torch.testing.assert_close(actual[0], torch.cat([q] * (dcp // pcp), dim=1))
    torch.testing.assert_close(actual[1], torch.cat([rope] * (dcp // pcp), dim=1))
    assert events == ([] if pcp == dcp else ["gather", "wait"])
    assert prep.call_count == int(supported)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("world,rank,interleave", [(2, 0, 1), (3, 2, 3), (8, 1, 128)])
def test_native_sparse_remap_preserves_exact_integers_and_order(dtype, world, rank, interleave):
    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            raise AssertionError(f"Native sparse remap imported {name}")
        return builtins.__import__(name, *args, **kwargs)

    namespace = load_methods(
        "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "AscendSFADCPImpl",
        ("_remap_sparse_indices",),
        {
            "torch": torch,
            "supports_triton": lambda: False,
            "__builtins__": {**vars(builtins), "__import__": guarded_import},
        },
    )
    values = [-1, 0, 1, 2, 3, 127, 128, 129, 2**24 + 1, 2**30 + 3]
    if dtype == torch.int64:
        values += [2**40 + 1, 2**40 + 128]
    indices = torch.tensor([values, list(reversed(values))], dtype=dtype)
    state = SimpleNamespace(dcp_size=world, dcp_rank=rank, _dcp_interleave_size=interleave, _dcp_index_topk=len(values))
    expected = []
    for row in indices.tolist():
        local = [
            (value // (world * interleave)) * interleave + value % interleave
            for value in row
            if value >= 0 and (value // interleave) % world == rank
        ]
        expected.append(local + [-1] * (len(row) - len(local)))
    result = namespace["_remap_sparse_indices"](state, indices)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype), rtol=0, atol=0)
    empty = indices[:, :0]
    assert namespace["_remap_sparse_indices"](state, empty) is empty


def test_indexer_warmup_without_triton_does_not_resolve_kernel_names():
    namespace = load_methods(
        "vllm_ascend/attention/indexer.py",
        "AscendSFAIndexerBackend",
        ("register_k_cache_warmup",),
        {"supports_triton": lambda: False},
    )
    namespace["register_k_cache_warmup"](object())
