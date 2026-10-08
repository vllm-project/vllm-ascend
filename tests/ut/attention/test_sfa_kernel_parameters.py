# SPDX-License-Identifier: Apache-2.0
"""Keep runtime workload values out of the DCP kernels' specialization keys."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton"
# Ordinary runtime arguments can retain Triton's unit/divisibility metadata.
# Exact workload values must not become constexpr specialization keys.
ALIGNMENT_METADATA = {
    "_pack_token_peer": {"output_s0", "output_s1", "lse_s0", "lse_s1"},
    "_fused_merge_kernel": {"out_s0", "out_s1", "out_s2", "head_dim"},
    "_prepare_query_head_major": {"ns0", "ns1", "ns2", "rs0", "rs1", "rs2"},
}
CASES: list[tuple[str, str, set[str], set[str]]] = [
    ("sfa_dcp_exchange.py", "_pack_t1_peer", set(), set()),
    (
        "sfa_dcp_exchange.py",
        "_pack_token_peer",
        {"LOCAL_HEADS", "HEAD_WORDS", "STATS_TILE"},
        {"output_s0", "output_s1", "lse_s0", "lse_s1", "tokens"},
    ),
    (
        "sfa_dcp_merge.py",
        "_fused_merge_kernel",
        {"RANKS", "NATIVE_LAYOUT", "BLOCK_D", "LSE_RANK_STRIDE"},
        {
            "out_s0",
            "out_s1",
            "out_s2",
            "lse_s1",
            "head_count",
            "head_dim",
            "total_tiles",
            "num_programs",
        },
    ),
    (
        "sfa_dcp_query.py",
        "_prepare_query_head_major",
        {"NOPE_DIM", "ROPE_DIM"},
        {"ns0", "ns1", "ns2", "rs0", "rs1", "rs2", "tokens", "programs", "heads"},
    ),
    ("sfa_dcp_query.py", "_unpack_query_fragments", {"NOPE_DIM", "ROPE_DIM"}, {"tokens", "programs", "heads"}),
    ("sfa_indexer_store.py", "_store_indexer_key_scale", set(), {"capacity"}),
]


@pytest.mark.parametrize("filename,kernel,configuration,runtime", CASES)
def test_kernel_parameter_classification(filename, kernel, configuration, runtime):
    tree = ast.parse((ROOT / filename).read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == kernel)
    constants = {
        a.arg
        for a in function.args.args
        if isinstance(a.annotation, ast.Attribute) and a.annotation.attr == "constexpr"
    }
    assert constants == configuration
    assert all(name.isupper() for name in constants)
    if runtime:
        jit = next(d for d in function.decorator_list if isinstance(d, ast.Call))
        excluded = ast.literal_eval(next(k.value for k in jit.keywords if k.arg == "do_not_specialize"))
        alignment = ALIGNMENT_METADATA.get(kernel, set())
        assert runtime - alignment <= set(excluded)
        assert not alignment & set(excluded), "do not discard useful runtime layout metadata"
    used = {
        n.id
        for statement in function.body
        for n in ast.walk(statement)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assert {a.arg for a in function.args.args} <= used, "unused arguments must not create specialization keys"
