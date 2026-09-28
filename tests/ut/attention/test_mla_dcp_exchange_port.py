# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch guards; distributed payload/parity checks live in the benchmark."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/mla_dcp_exchange.py"
tree = ast.parse(SOURCE.read_text())
guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "can_exchange")
scope = {"torch": torch, "triton": object(), "_LOCAL_HEADS": 6, "_HEAD_DIM": 512, "_MAX_DECODE_TOKENS": 12}
exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"), scope)


def tensor(shape, dtype, device="npu"):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=SimpleNamespace(type=device))


@pytest.mark.parametrize("ranks", [2, 8, 16])
@pytest.mark.parametrize("tokens", [1, 8, 12])
def test_supported_shapes(ranks, tokens):
    output = tensor((tokens, ranks * 6, 512), torch.bfloat16)
    stats = tensor((tokens, ranks * 6, 1), torch.float32)
    assert scope["can_exchange"](output, stats, ranks)


@pytest.mark.parametrize(
    "tokens,ranks,heads,dtype,device",
    [
        (0, 16, 96, torch.bfloat16, "npu"),
        (13, 16, 96, torch.bfloat16, "npu"),
        (1, 1, 6, torch.bfloat16, "npu"),
        (1, 8, 64, torch.bfloat16, "npu"),
        (1, 16, 96, torch.float32, "npu"),
        (1, 16, 96, torch.bfloat16, "cpu"),
    ],
)
def test_fallback_shapes(tokens, ranks, heads, dtype, device):
    assert not scope["can_exchange"](
        tensor((tokens, heads, 512), dtype, device), tensor((tokens, heads, 1), torch.float32, device), ranks
    )


@pytest.mark.parametrize(
    "ranks,tokens,enabled",
    [(2, 1, False), (8, 1, False), (16, 1, True), (2, 8, False), (8, 12, False), (16, 6, False), (16, 13, False)],
)
def test_mla_routes_only_measured_wins(ranks, tokens, enabled):
    source = SOURCE.parents[2] / "attention/context_parallel/mla_cp.py"
    route = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.If) and "can_exchange(" in ast.unparse(n.test)
    )
    state = {
        "self": SimpleNamespace(
            dcp_size=ranks, dcp_device_group=None, kv_lora_rank=512, _merge_dcp_attention_output=lambda *args: "native"
        ),
        "attn_output": tensor((tokens, ranks * 6, 512), torch.bfloat16),
        "softmax_lse": tensor((tokens, ranks * 6, 1), torch.float32),
        "can_exchange": scope["can_exchange"],
        "exchange": lambda *args: "packed",
    }
    exec(compile(ast.Module(body=[route], type_ignores=[]), str(source), "exec"), state)
    assert state["attn_output"] == ("packed" if enabled else "native")


@pytest.mark.parametrize("lse_value", [float("-inf"), -10.0, 10.0])
def test_native_transport_empty_identity(lse_value):
    source = SOURCE.parents[2] / "attention/context_parallel/common_cp.py"
    fn = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_process_attn_out_lse"
    )
    state = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), state)
    output = torch.zeros(1, 6, 512)
    lse = torch.full((1, 6, 1), lse_value)
    result = state["_process_attn_out_lse"](output, lse, dcp_size=1)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[..., :512], output, atol=0, rtol=0)
    expected = torch.finfo(torch.float32).min if lse_value == float("-inf") else lse_value
    assert (result[..., 512:] == expected).all()
