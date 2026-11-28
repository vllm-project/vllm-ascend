# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/ops/triton/fla/fused_gdn_prepare.py"
# Load Python dispatch without importing NPU-only registrations. Hardware tests
# exercise the real registered operator and Triton kernel separately.
ns = {
    "torch": torch,
    "_BLOCK_SIZE": 1024,
    "_PREFILL_MIN_TOKENS": 1024,
    "_PREFILL_TILE_ROWS": 4,
    "_PREFILL_MAX_WIDTH": 8192,
}
for node in ast.parse(SOURCE.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "supports_gdn_prepare",
        "_prepare_fake",
        "_prepare",
    ):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
supports = ns["supports_gdn_prepare"]


def inputs(tokens=1024, heads=32, width=8192, padding=0, ba_dtype=torch.float32):
    qkvz = torch.randn(tokens + 1, width + heads * 128 + padding, dtype=torch.bfloat16)[1:, : width + heads * 128]
    ba = torch.randn(tokens + 1, heads * 2 + padding, dtype=ba_dtype)[1:, : heads * 2]
    return qkvz, ba


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("padding", [0, 17])
def test_tp_local_strides_and_fake_metadata(tp, padding):
    heads, width = 32 // tp, 8192 // tp
    x, ba = inputs(heads=heads, width=width, padding=padding)
    assert supports(x, ba, width, heads, 128)
    qkv, b, a, core = ns["_prepare_fake"](x, ba, width, heads, 128)
    assert qkv.shape == (1024, width)
    assert core.shape == (1024, heads, 128)
    assert a.shape == b.shape == (1024, heads)
    assert qkv.dtype == core.dtype == x.dtype
    assert a.dtype == b.dtype == ba.dtype
    assert all(t.is_contiguous() for t in (qkv, b, a, core))


@pytest.mark.parametrize(
    "case", ["zero_width", "zero_heads", "bad_shape", "integer", "column_stride", "replicated_ba", "rank", "tokens"]
)
def test_unsupported_inputs_fall_back(case):
    x, ba = inputs()
    width, heads = 8192, 32
    if case == "zero_width":
        width = 0
    elif case == "zero_heads":
        heads = 0
    elif case == "bad_shape":
        x = x[:, :-1]
    elif case == "integer":
        x = x.int()
    elif case == "column_stride":
        x = x[:, ::2]
    elif case == "replicated_ba":
        ba = ba.repeat(1, 2)
    elif case == "rank":
        x = x.unsqueeze(0)
    else:
        ba = ba[:1023]
    assert not supports(x, ba, width, heads, 128)
    with pytest.raises(ValueError, match="Unsupported"):
        ns["_prepare"](x, ba, width, heads, 128)


@pytest.fixture
def launch(monkeypatch):
    calls = []

    class Kernel:
        def __init__(self, prefill=False):
            self.prefill = prefill

        def __getitem__(self, grid):
            def run(qkv, b, a, core, x, ba, *args, **tiles):
                if self.prefill:
                    tokens, xs, bs, width, heads, core_width = args
                    assert tokens == x.shape[0]
                else:
                    xs, bs, width, heads, core_width = args
                calls.append((grid, x.shape[0], xs, bs, tiles))
                qkv.copy_(x[:, :width])
                b.copy_(ba[:, :heads])
                a.copy_(ba[:, heads:])
                core.zero_()

            return run

    monkeypatch.setitem(ns, "fused_gdn_prepare_kernel", Kernel())
    monkeypatch.setitem(ns, "fused_gdn_prepare_prefill_kernel", Kernel(prefill=True))
    monkeypatch.setitem(
        ns,
        "triton",
        SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b, next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
    )
    return calls


@pytest.mark.parametrize("tokens", [0, 1, 17, 1023, 1024, 1025])
def test_launcher_grid_strides_and_outputs(launch, tokens):
    x, ba = inputs(tokens=tokens, padding=13)
    qkv, b, a, core = ns["_prepare"](x, ba, 8192, 32, 128)
    assert len(launch) == int(tokens > 0)
    if tokens == 0:
        assert qkv.shape == (0, 8192) and core.shape == (0, 32, 128)
        return
    grid, count, xs, bs, tiles = launch[0]
    assert grid == (((tokens + 3) // 4,) if tokens >= 1024 else (tokens, 8))
    assert (count, xs, bs) == (tokens, x.stride(0), ba.stride(0))
    assert tiles == (
        {"BLOCK_ROWS": 4, "BLOCK_WIDTH": 8192, "BLOCK_HEADS": 32} if tokens >= 1024 else {"BLOCK_SIZE": 1024}
    )
    assert torch.equal(qkv, x[:, :8192])
    assert torch.equal(b, ba[:, :32]) and torch.equal(a, ba[:, 32:])
    assert torch.count_nonzero(core) == 0


@pytest.mark.parametrize("tokens,replicated", [(1024, False), (1025, False), (4, False), (1024, True)])
def test_actual_gdn_forward_dispatch(monkeypatch, launch, tokens, replicated):
    source = (ROOT / "vllm_ascend/attention/gdn.py").read_text()
    cls = next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "AscendGatedDeltaNetAttention"
    )
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    node.decorator_list = []
    x, ba = inputs(tokens=tokens)
    if replicated:
        ba = ba.repeat(1, 2)
    original_x, original_ba = x.clone(), ba.clone()

    def fused(x, ba, width, heads, dim):
        qkv, b, a, core = ns["_prepare"](x, ba, width, heads, dim)
        return qkv, x[:, width:].reshape(tokens, heads, dim), b, a, core

    def core(qkv, b, a, out, prefix, flag):
        assert torch.equal(qkv, original_x[:, :8192])
        assert torch.equal(b, original_ba[:, :32])
        assert torch.equal(a, original_ba[:, 32:64])
        assert torch.count_nonzero(out) == 0
        out.fill_(2)

    monkeypatch.setattr(torch.ops.vllm, "qwen_gdn_attention_core", core, raising=False)
    scope = {
        "torch": torch,
        "rearrange": rearrange,
        "supports_gdn_prepare": supports,
        "fused_gdn_prepare": fused,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "gdn.py", "exec"), scope)
    layer = SimpleNamespace(
        out_proj=lambda x: (x, None),
        in_proj_qkvz=lambda h: (x, None),
        in_proj_ba=lambda h: (ba, None),
        gqa_interleaved_layout=False,
        key_dim=2048,
        value_dim=4096,
        tp_size=1,
        head_v_dim=128,
        num_v_heads=32,
        prefix="test",
        norm=lambda out, z: out * z,
        _split_ba_for_tp=lambda ba: (ba[:, :32], ba[:, 32:64]),
    )
    output = scope["forward"](layer, torch.zeros(tokens, 1, dtype=torch.bfloat16))
    assert torch.equal(output, 2 * original_x[:, 8192:])
    assert len(launch) == int(not replicated)
    assert torch.equal(x, original_x) and torch.equal(ba, original_ba)


@pytest.mark.parametrize("width", [1023, 1024, 1025, 16385])
def test_column_tail_and_wide_projection(launch, width):
    x, ba = inputs(tokens=2, width=width)
    qkv, b, a, core = ns["_prepare"](x, ba, width, 32, 128)
    assert launch[0][0] == (2, (max(width, 4096) + 1023) // 1024)
    assert torch.equal(qkv, x[:, :width])
    assert torch.count_nonzero(core) == 0


@pytest.mark.parametrize(
    "dtype,width,prefill", [(torch.float32, 8192, False), (torch.bfloat16, 8193, False), (torch.float16, 8192, True)]
)
def test_large_batch_dispatch_limits(launch, dtype, width, prefill):
    x, ba = inputs(tokens=1024, width=width)
    x = x.to(dtype)
    qkv, b, a, core = ns["_prepare"](x, ba, width, 32, 128)
    assert len(launch) == 1
    assert (len(launch[0][0]) == 1) == prefill
    assert torch.equal(qkv, x[:, :width])
    assert torch.count_nonzero(core) == 0
