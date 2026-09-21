# SPDX-License-Identifier: Apache-2.0
"""Exercise exchange routing and invalid-rank semantics without an NPU."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "tokens,ranks,scatter,supported,pcp",
    [
        (1, 8, 1, True, False),
        (6, 8, 1, True, False),
        (12, 8, 1, True, False),
        (13, 8, 1, True, False),
        (48, 8, 1, True, False),
        (192, 8, 1, True, False),
        (193, 8, 1, False, False),
        (6, 2, 1, True, False),
        (8, 8, 0, True, False),
        (6, 8, 1, True, True),
        (6, 1, 1, True, True),
        (6, 1, 1, True, False),
    ],
)
@pytest.mark.parametrize("budget", [None, 32, 48, 192, 512])
def test_registered_op_keeps_upstream_fallback_and_output_dtype(tokens, ranks, scatter, supported, pcp, budget):
    calls = []
    group = object()
    output = torch.zeros(tokens, 64, 512, dtype=torch.bfloat16)
    lse = torch.zeros(tokens, 64, 1)
    shape = (tokens, 64 // ranks, 512) if scatter else (tokens // ranks, 64, 512)

    def exchange(out, stats, selected_group):
        assert selected_group is group and out is output and stats is lse
        calls.append("exchange")
        return torch.ones(shape, dtype=torch.float32)

    def transfer(recv, send, *, group):
        calls.append("upstream_collective")

    namespace = {
        "torch": torch,
        "dist": SimpleNamespace(all_to_all_single=transfer),
        "can_exchange": lambda *_: supported,
        "exchange": exchange,
        "pack_sfa_dcp_output_lse": lambda *_, **kwargs: torch.empty(8, 16),
        "fused_sfa_dcp_lse_combine": lambda *_, **kwargs: torch.ones(shape, dtype=output.dtype),
    }
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_combine", namespace)

    def pcp_gather(value, dim):
        assert dim == 0
        calls.append("pcp_gather")
        return value

    pcp_group = SimpleNamespace(all_gather=pcp_gather) if pcp else None
    result = run(output, lse, ranks, scatter, group, pcp_group, decode_token_budget=budget)
    assert result.dtype == output.dtype and result.shape == shape
    if ranks == 8 and scatter == 1 and supported and not pcp and (budget is None or tokens <= budget):
        assert calls == ["exchange"]
    else:
        assert calls == (["upstream_collective"] if ranks > 1 else []) + (["pcp_gather"] if pcp else [])


@pytest.mark.parametrize(
    "tokens,expected", [(0, False), (1, True), (6, True), (12, True), (13, True), (48, True), (192, True), (193, False)]
)
def test_exchange_gate(tokens, expected):
    source = ROOT / "vllm_ascend/ops/triton/sfa_dcp_exchange.py"
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("_MAX_DECODE_TOKENS", "_HEADS", "_HEAD_DIM")
    }
    namespace = {"torch": torch, "triton": object(), **constants}
    run = load_function("vllm_ascend/ops/triton/sfa_dcp_exchange.py", "can_exchange", namespace)
    device = SimpleNamespace(type="npu")
    output = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=3, shape=(tokens, 64, 512))
    lse = SimpleNamespace(device=device, dtype=torch.float32, shape=(tokens, 64, 1))
    assert run(output, lse) is expected


@pytest.mark.parametrize("token_dim", [1, 2])
def test_merge_masks_invalid_rank_outputs_before_weighting(token_dim):
    spec = importlib.util.spec_from_file_location("merge_under_test", ROOT / "vllm_ascend/ops/triton/sfa_dcp_merge.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    parts = torch.full((8, 2, 3, 4), float("nan"))
    lse = torch.full((8, 2, 3), float("-inf"))
    lse[0] = float("nan")
    lse[1] = float("inf")
    parts[2] = 2.0
    lse[2] = 0.0
    parts[3] = 4.0
    lse[3] = 0.0
    # One position has no valid ranks; even NaN payloads must yield zero.
    lse[:, 0, 0] = float("-inf")
    expected = torch.full((2, 3, 4), 3.0)
    expected[0, 0] = 0.0
    expected = expected.movedim(token_dim - 1, 0).contiguous()
    assert torch.equal(merge.fused_merge(parts, lse, token_dim), expected)


def test_custom_op_fake_retains_shape_dtype_and_does_not_collect():
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_fake", {"torch": torch})
    output = torch.empty(6, 64, 512, dtype=torch.bfloat16)
    result = run(output, torch.empty(6, 64, 1), 8, 1, "unused")
    assert result.shape == (6, 8, 512) and result.dtype == output.dtype


@pytest.mark.parametrize("supported", [False, True])
def test_missing_scatter_group_raises_without_communication(supported):
    def forbidden(*args, **kwargs):
        raise AssertionError("communication attempted without a group")

    namespace = {
        "torch": torch,
        "can_exchange": lambda *_: supported,
        "exchange": forbidden,
        "dist": SimpleNamespace(all_to_all_single=forbidden),
        "pack_sfa_dcp_output_lse": lambda *args, **kwargs: torch.empty(8, 16),
    }
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_combine", namespace)
    with pytest.raises(ValueError, match="explicit All2All group"):
        run(torch.empty(6, 64, 512, dtype=torch.bfloat16), torch.empty(6, 64, 1), 8, 1, None)


@pytest.mark.parametrize("defer", [False, True])
def test_new_main_merge_modes_never_take_output_only_fast_path(defer):
    """Sparse-offload callers need packed contributions or the merged LSE."""
    packed = torch.empty(8, 8, 6, 516)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("output-only fast path loses upstream merge state")

    def combine(recv, head_dim, *, scatter_dim, return_lse):
        assert recv.shape == packed.shape and head_dim == 512 and scatter_dim == 1
        assert return_lse
        calls.append("merge_lse")
        return torch.empty(6, 8, 513)

    namespace = {
        "torch": torch,
        "can_exchange": lambda *_: True,
        "exchange": forbidden,
        "dist": SimpleNamespace(all_to_all_single=lambda *args, **kwargs: calls.append("collective")),
        "pack_sfa_dcp_output_lse": lambda *args, **kwargs: packed,
        "fused_sfa_dcp_lse_combine": combine,
    }
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_combine", namespace)
    result = run(
        torch.empty(6, 64, 512),
        torch.empty(6, 64, 1),
        8,
        1,
        object(),
        return_lse=not defer,
        defer_combine=defer,
        decode_token_budget=192,
    )
    assert result.shape == (packed.shape if defer else (6, 8, 513))
    assert calls == (["collective"] if defer else ["collective", "merge_lse"])


def test_registered_operator_forwards_budget_and_fake_keeps_shape():
    group = SimpleNamespace(world_size=8, device_group=object())
    calls = []

    def combine(*args, **kwargs):
        calls.append(kwargs["decode_token_budget"])
        return args[0]

    namespace = {"torch": torch, "_groups": {"dcp": lambda: group}, "sfa_dcp_a2a_fused_combine": combine}
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused", namespace)
    fake = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_fake", {"torch": torch})
    output = torch.empty(48, 64, 512, dtype=torch.bfloat16)
    lse = torch.empty(48, 64, 1)
    for budget in (32, 192):
        run(output, lse, 8, 1, "dcp", decode_token_budget=budget)
        result = fake(output, lse, 8, 1, "dcp", decode_token_budget=budget)
        assert result.shape == (48, 8, 512) and result.dtype == output.dtype
    assert calls == [32, 192]


def test_attention_passes_current_scheduler_budget_without_global_config():
    source = ROOT / "vllm_ascend/attention/context_parallel/sfa_cp.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AscendSFADCPImpl"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_merge_dcp_outputs")
    budgets = []

    def invoke(*args, **kwargs):
        budgets.append(kwargs["decode_token_budget"])
        return args[0]

    namespace = {"torch": SimpleNamespace(ops=SimpleNamespace(vllm=SimpleNamespace(sfa_dcp_a2a_fused=invoke)))}
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), namespace)
    scheduler = SimpleNamespace(max_num_batched_tokens=48)
    state = SimpleNamespace(
        dcp_size=8,
        dcp_group=SimpleNamespace(unique_name="dcp"),
        vllm_config=SimpleNamespace(scheduler_config=scheduler),
    )
    for budget in (48, 128):
        scheduler.max_num_batched_tokens = budget
        namespace["_merge_dcp_outputs"](state, object(), object())
    assert budgets == [48, 128]
