# SPDX-License-Identifier: Apache-2.0
"""Execute production DCP kernels with Triton's CPU interpreter, without NPU.

The interpreter checks indexing and math, not Ascend compilation or timing.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def kernels():
    triton = pytest.importorskip("triton")
    from triton import language as tl  # type: ignore[import-untyped]
    from triton.runtime.interpreter import InterpretedFunction  # type: ignore[import-untyped]
    from triton.runtime.jit import JITFunction  # type: ignore[import-untyped]

    module = ModuleType("dcp_mask_interpreter")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        torch=torch,
        tl=tl,
        triton=SimpleNamespace(jit=InterpretedFunction, next_power_of_2=triton.next_power_of_2, cdiv=triton.cdiv),
        DCP_HEAD_SCATTER_SINGLE_ROW=0,
        DCP_HEAD_SCATTER_MULTI_ROW=1,
        get_current_hardware_profile=lambda: SimpleNamespace(dcp_head_scatter_mode=0),
        # CPU Triton has no Ascend get_element intrinsic. Gather the selected scalar rank weight.
        get_element=lambda values, indices: tl.reshape(tl.gather(values, tl.full((1,), indices[0], tl.int32), 0), ()),
    )
    for relative in ("dcp_a2a_batched.py", "dcp_a2a_rank_tiled.py", "dcp_a2a.py"):
        path = ROOT / "vllm_ascend/ops/triton/dcp" / relative
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
        if relative == "dcp_a2a_rank_tiled.py":
            tree.body = [
                n for n in tree.body if getattr(n, "name", None) == "_fused_dcp_lse_combine_rank_tiled_static_kernel"
            ]
        tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    # Ascend's interpreter does not rewrite standard-library JIT helpers
    # when Triton was imported earlier without TRITON_INTERPRET=1.
    with pytest.MonkeyPatch.context() as patch:
        # Ordinary dispatch cases below exercise the scalar and batched combine paths.
        patch.setattr(module, "_rank_tiled_mode", lambda: "off")
        for name, helper in list(vars(tl).items()):
            if isinstance(helper, JITFunction):
                patch.setattr(tl, name, InterpretedFunction(helper.fn))
        yield module
    del sys.modules[module.__name__]


def launch(kernel, grid, *args, **kwargs):
    kwargs.pop("multibuffer", None)
    kernel[grid](*args, **kwargs)


@pytest.mark.parametrize(
    "dtype,scatter_dim,tokens,heads,dim,cores,mode",
    [
        (torch.float32, 0, 6, 4, 8, 16, 0),
        (torch.float16, 1, 7, 4, 8, 16, 0),
        (torch.bfloat16, 1, 3, 4, 8, 16, 0),
        (torch.bfloat16, 1, 9, 4, 8, 1, 0),
        (torch.bfloat16, 1, 9, 4, 512, 1, 0),
        (torch.bfloat16, 1, 9, 4, 512, 2, 1),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_masked_pack_and_local_combine(
    kernels, monkeypatch, dtype, scatter_dim, tokens, heads, dim, cores, mode, masked
):
    """Cover both kernel routes, partial tiles, strided LSE, and finite garbage."""
    torch.manual_seed(115)
    monkeypatch.setattr(kernels, "get_current_hardware_profile", lambda: SimpleNamespace(dcp_head_scatter_mode=mode))
    world = 2
    outputs = torch.randn(world, tokens, heads, dim, dtype=dtype)
    # A3 returns head-major LSE; keep that noncontiguous TN1 layout.
    lses = torch.randn(world, heads, tokens).transpose(1, 2).unsqueeze(-1)
    masks = torch.zeros(world, tokens, dtype=torch.bool)
    masks[0, ::2] = True
    masks[1, ::3] = True
    if not masked:
        masks.zero_()
    # Invalid rows may contain finite stale values or NaN, not just +inf LSE.
    outputs[0, 0] = torch.nan
    lses[0, 0] = torch.inf
    outputs[1, 1] = torch.nan
    lses[1, 1] = -torch.inf
    local_t, local_h = (tokens // world, heads) if scatter_dim == 0 else (tokens, heads // world)
    extra = kernels._lse_pack_dim(dtype)
    sends = []
    for rank in range(world):
        send = torch.empty(
            world,
            (tokens if scatter_dim == 0 else heads) // world,
            heads if scatter_dim == 0 else tokens,
            dim + extra,
            dtype=dtype,
        )
        kernels._launch_pack_dcp_output_lse(
            outputs[rank],
            lses[rank],
            None,
            send,
            scatter_dim,
            cores,
            launch,
            masks[rank] if masked else None,
        )
        sends.append(send)
    for dst in range(world):
        recv = torch.stack([send[dst] for send in sends])
        local = torch.randn(local_t, local_h, dim, dtype=dtype)
        local_lse = torch.randn(local_h, local_t).T.unsqueeze(-1)
        local_mask = torch.zeros(local_t, dtype=torch.bool)
        if masked:
            local_mask[::2] = True
        local[0] = torch.nan
        local_lse[0] = torch.inf
        actual = torch.empty_like(local)
        kernels._launch_fused_dcp_lse_combine(
            recv,
            actual,
            dim,
            scatter_dim,
            False,
            local,
            local_lse,
            cores,
            launch,
            local_mask if masked else None,
        )
        token_slice = slice(dst * local_t, (dst + 1) * local_t) if scatter_dim == 0 else slice(None)
        head_slice = slice(dst * local_h, (dst + 1) * local_h) if scatter_dim == 1 else slice(None)
        history = outputs[:, token_slice, head_slice].double()
        stats = lses[:, token_slice, head_slice].double()
        history_mask = masks[:, token_slice, None, None]
        values = torch.cat((history, local[None].double()))
        stats = torch.cat((stats, local_lse[None].double()))
        invalid = torch.cat(
            (history_mask.expand(-1, -1, local_h, -1), local_mask[None, :, None, None].expand(1, -1, local_h, -1))
        )
        valid = ~invalid & torch.isfinite(stats)
        weights = torch.nan_to_num(stats.masked_fill(~valid, -torch.inf).softmax(0), nan=0.0)
        expected = (values.masked_fill(~valid, 0) * weights).sum(0).to(dtype)
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.02 if dtype == torch.bfloat16 else 0.002,
            rtol=0.02,
            msg=lambda diff, dst=dst: f"destination rank {dst}: {diff}",
        )
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize("bad", ["dtype", "shape", "stride"])
def test_mask_validation(kernels, bad):
    output = torch.empty(6, 4, 8)
    mask = {
        "dtype": torch.zeros(6),
        "shape": torch.zeros(5, dtype=torch.bool),
        "stride": torch.zeros(12, dtype=torch.bool)[::2],
    }[bad]
    with pytest.raises(ValueError, match="output mask"):
        kernels._validate_output_mask(mask, output)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("tokens", [3, 129])
def test_warmup_registers_masked_launches_only_for_a3(kernels, tokens, masked):
    path = ROOT / "vllm_ascend/ops/triton/dcp/dcp_warmup.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_record_dcp_layout", "_output_and_lse")
    ]

    def tensor(dtype, *, shape, strides=None):
        if strides is None:
            return torch.empty(shape, dtype=dtype, device="meta")
        return torch.empty_strided(shape, strides, dtype=dtype, device="meta")

    scope = dict(kernels.__dict__, _Tensor=tensor, _use_fast_exchange=lambda *args: False)
    exec(compile(tree, str(path), "exec"), scope)
    launches = []
    scope["_record_dcp_layout"](
        torch.bfloat16,
        tokens,
        8,
        512,
        2,
        1,
        False,
        True,
        "head_major_lse",
        24,
        lambda kernel, grid, *args, **kwargs: launches.append((kernel.fn.__name__, kwargs)),
        masked_output=masked,
    )
    masked_pack = [kw for name, kw in launches if "pack" in name and kw["HAS_OUTPUT_MASK"]]
    masked_merge = [kw for name, kw in launches if "combine" in name and kw["HAS_LOCAL_OUTPUT_MASK"]]
    assert len(masked_pack) == len(masked_merge) == int(masked)


def test_masked_static_combine_ignores_invalid_payloads(kernels):
    ranks, tokens, dim = 8, 4, 512
    recv = torch.zeros(ranks, 1, tokens, dim + 4, dtype=torch.bfloat16)
    recv[..., :dim] = torch.nan
    # Rank zero contributes 2 on the first two tokens (wire code 128 means LSE zero).
    recv[0, 0, :2, :dim] = 2
    recv[0, 0, :2, dim] = 128
    local = torch.full((tokens, 1, dim), 4.0, dtype=recv.dtype)
    local[1] = torch.nan
    mask = torch.tensor([False, True, False, True])
    lse = torch.zeros(tokens, 1)
    output = torch.empty_like(local)
    kernels._fused_dcp_lse_combine_rank_tiled_static_kernel[(1,)](
        recv,
        output,
        local,
        lse,
        *local.stride(),
        *lse.stride(),
        NUM_HEADS=1,
        NUM_TOKENS=tokens,
        DCP_SIZE=ranks,
        HEAD_DIM=dim,
        BLOCK_R=ranks,
        BLOCK_D=dim,
        local_output_mask_ptr=mask,
        HAS_LOCAL_OUTPUT_MASK=True,
    )
    expected = torch.tensor([3, 2, 4, 0], dtype=local.dtype)
    torch.testing.assert_close(output, expected[:, None, None].expand_as(output))
