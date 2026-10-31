import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Strict-assertion companion to test_dequant_swiglu_quant.py.
#
# The base file asserts y with atol=1 against a golden that rounds swiglu to
# bf16 before quantizing, while the kernel quantizes the fp32 swiglu
# directly - so a large slice of real defects (wrong per-row activation
# scale, wrong per-group weight scale, swapped act/gate halves) hides inside
# that tolerance. The optimization work on this operator is gated here
# instead:
#   - golden mirrors the kernel's exact numeric chain in fp32
#     ((ws*x)*act -> silu with alpha/beta/clamp -> row max -> *1/127 ->
#     divide -> RINT -> int8), with no bf16 pre-round;
#   - y: atol=1 / rtol=0 - one int8 step of headroom for the ~1-2 ulp fp32
#     difference between the c220 vector Exp and the CPU exp at RINT
#     boundaries; anything beyond one step is a defect;
#   - scale: rtol=1e-5 - row-mixing or a wrong max moves it by orders of
#     magnitude;
#   - all-zero rows pinned exactly (scale==0, y==0 - matches both the kernel
#     and CANN's npu_dynamic_quant padding semantics);
#   - shape matrix crosses the ubFactorDimx tails (odd/prime row counts per
#     2H), the 36-core boundary (rows 73/74), per-group row-count tails of
#     the MoE group path (ragged sizes incl. empty groups and the cut-group
#     variant), activate_left both ways, activation_scale=None, and the
#     1D/2D weight_scale layouts.

seed = 7
torch.manual_seed(seed)

# Fp32 constant identical to the kernel's DYNAMIC_QUANT_FACTOR (1.0f/127.0f).
INV_127 = 1.0 / 127.0


def dequant_swiglu_quant_golden(
    x,
    weight_scale,
    activation_scale,
    *,
    swiglu_mode=1,
    clamp_limit=0.0,
    glu_alpha=1.0,
    glu_bias=0.0,
    activate_left=True,
    group_sizes=None,
):
    """CPU fp32 reference following the kernel's op order exactly."""
    x = x.cpu()
    ws = weight_scale.cpu()
    if group_sizes is not None:
        # weight_scale is [group_num, 2H]; expand to per-row [T, 2H].
        sizes = torch.tensor(group_sizes, dtype=torch.long)
        row_of_group = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
        ws_rows = ws[row_of_group]
    else:
        ws_rows = ws.reshape(1, -1)
    if activation_scale is None:
        act = torch.ones(x.shape[0], 1)
    else:
        act = activation_scale.cpu().reshape(-1, 1)

    # Kernel: Cast int32->fp32; Mul(ws_broadcast, x); Mul(act_broadcast, .)
    f = x.to(torch.float32) * ws_rows * act
    half = f.shape[-1] // 2
    if activate_left:
        act_half, gate_half = f[..., :half], f[..., half:]
    else:
        act_half, gate_half = f[..., half:], f[..., :half]
    if swiglu_mode == 1:
        if clamp_limit > 0.0:
            # Kernel order: Mins/Maxs on the gate half (both sides) and the
            # act half (max only), then Adds(glu_bias) on the gate half.
            act_half = act_half.clamp(max=clamp_limit)
            gate_half = gate_half.clamp(min=-clamp_limit, max=clamp_limit)
        gate_half = gate_half + glu_bias
        # Kernel: Muls(-alpha); Exp; Adds(1.0); Div(act, act, denom)
        denom = 1.0 + torch.exp(-glu_alpha * act_half)
        sig = act_half / denom
    else:
        denom = 1.0 + torch.exp(-act_half)
        sig = act_half / denom
    swiglu = sig * gate_half

    # Kernel: Abs; row max tree; Muls(1/127); Div; RINT; int8.
    row_max = swiglu.abs().amax(dim=-1, keepdim=True)
    scale = row_max * INV_127
    y = torch.round(swiglu / scale).to(torch.int8)
    return y, scale.squeeze(-1)


def make_inputs(rows, x_last_dim, *, amp=1000, ws_scale=0.001, act_range=4.0):
    x = torch.randint(-amp, amp, (rows, x_last_dim), dtype=torch.int32)
    weight_scale = torch.randn(x_last_dim, dtype=torch.float32) * ws_scale
    activation_scale = torch.rand(rows, 1, dtype=torch.float32) * act_range + 0.5
    return x, weight_scale, activation_scale


def run_kernel(x, weight_scale, activation_scale, *, swiglu_mode=1, clamp_limit=0.0,
               glu_alpha=1.0, glu_bias=0.0, activate_left=True, group_index=None,
               quant_mode=1):
    y, s = torch.ops._C_ascend.npu_dequant_swiglu_quant(
        x=x.npu(), weight_scale=weight_scale.npu(),
        activation_scale=None if activation_scale is None else activation_scale.npu(),
        bias=None, quant_scale=None, quant_offset=None,
        group_index=None if group_index is None else group_index.npu(),
        activate_left=activate_left, quant_mode=quant_mode,
        swiglu_mode=swiglu_mode, clamp_limit=clamp_limit,
        glu_alpha=glu_alpha, glu_bias=glu_bias)
    torch.npu.synchronize()
    return y.cpu(), s.cpu()


def check(y, s, y_g, s_g):
    torch.testing.assert_close(y, y_g, atol=1, rtol=0)
    torch.testing.assert_close(s, s_g, rtol=1e-5, atol=0)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "rows, x_last_dim",
    [
        # ubFactorDimx per 2H (swiglu_mode=1): 512->24, 1024->9, 2048->5,
        # 4096->2. Rows chosen to hit tile tails (rows % p != 0), tiny
        # single-tile shapes, and the 36-core boundary.
        (1, 512), (25, 512),          # p=24: tail 1
        (2, 1024), (11, 1024),        # p=9: tail 2
        (3, 2048), (7, 2048),         # p=5: tails 3 and 2
        (2, 4096), (3, 4096), (5, 4096),   # p=2: tails 1
        (37, 4096), (73, 4096),       # 36/37 rows-per-core boundary
        (293, 4096), (4608, 2048),    # multi-tile, base-test large shape
    ],
)
def test_strict_mode1_default(rows, x_last_dim):
    x, ws, act = make_inputs(rows, x_last_dim)
    y, s = run_kernel(x, ws, act)
    y_g, s_g = dequant_swiglu_quant_golden(x, ws, act)
    check(y, s, y_g, s_g)


@pytest.mark.parametrize(
    "clamp_limit, glu_alpha, glu_bias, activate_left, use_act_scale",
    [
        (7.0, 1.0, 0.0, True, True),      # clamp active (V3.2 style)
        (0.0, 1.702, 0.5, True, True),    # non-trivial alpha/beta
        (0.0, 1.0, 0.0, False, True),     # halves swapped
        (0.0, 1.0, 0.0, True, False),     # activationScaleIsEmpty path
    ],
)
def test_strict_mode1_variants(clamp_limit, glu_alpha, glu_bias, activate_left,
                               use_act_scale):
    rows, x_last_dim = 37, 4096
    x, ws, act = make_inputs(rows, x_last_dim)
    if not use_act_scale:
        act = None
    y, s = run_kernel(x, ws, act, clamp_limit=clamp_limit, glu_alpha=glu_alpha,
                      glu_bias=glu_bias, activate_left=activate_left)
    y_g, s_g = dequant_swiglu_quant_golden(
        x, ws, act, clamp_limit=clamp_limit, glu_alpha=glu_alpha,
        glu_bias=glu_bias, activate_left=activate_left)
    check(y, s, y_g, s_g)


def test_strict_mode1_weight_scale_2d():
    # No-group path also accepts [1, 2H] weight scales.
    rows, x_last_dim = 19, 2048
    x, ws, act = make_inputs(rows, x_last_dim)
    ws2d = ws.reshape(1, -1)
    y, s = run_kernel(x, ws2d, act)
    y_g, s_g = dequant_swiglu_quant_golden(x, ws2d, act)
    check(y, s, y_g, s_g)


@pytest.mark.parametrize(
    "rows, x_last_dim",
    [(2, 4096), (64, 4096), (129, 2048)],
)
def test_strict_mode0_plain_swiglu(rows, x_last_dim):
    # w8a8 MC2 fallback path: swiglu_mode=0 ignores clamp/alpha/beta.
    x, ws, act = make_inputs(rows, x_last_dim)
    y, s = run_kernel(x, ws, act, swiglu_mode=0, clamp_limit=7.0,
                      glu_alpha=1.702, glu_bias=0.5)
    y_g, s_g = dequant_swiglu_quant_golden(x, ws, act, swiglu_mode=0)
    check(y, s, y_g, s_g)


@pytest.mark.parametrize(
    "sizes, x_last_dim, swiglu_mode",
    [
        ([1, 0, 2], 4096, 0),                       # empty group in the middle
        ([0, 1, 2, 3, 5, 8, 13, 21], 4096, 0),      # ragged sizes
        ([2] * 64, 4096, 0),                        # 64 groups x 2 rows
        ([2] * 256, 4096, 0),                       # 256 groups -> cut-group
        ([2] * 128, 4096, 0),                       # >32 groups, cut-group
        ([2, 2, 2, 2], 4096, 1),                    # group + swiglu_mode=1
        ([5] * 40, 2048, 0),                        # >16 rows/group: no cut
    ],
)
def test_strict_group_path(sizes, x_last_dim, swiglu_mode):
    rows = sum(sizes)
    x, ws1d, act = make_inputs(rows, x_last_dim)
    # The group path requires 2D weight scales [group_num, 2H].
    ws = torch.randn(len(sizes), x_last_dim, dtype=torch.float32) * 0.001
    group_index = torch.tensor(sizes, dtype=torch.int64)
    y, s = run_kernel(x, ws, act, swiglu_mode=swiglu_mode, group_index=group_index)
    y_g, s_g = dequant_swiglu_quant_golden(
        x, ws, act, swiglu_mode=swiglu_mode, group_sizes=sizes)
    check(y, s, y_g, s_g)


def test_strict_zero_rows():
    # All-zero rows (padding): scale must be exactly 0 and y exactly 0,
    # matching CANN's npu_dynamic_quant behavior on the same input.
    rows, x_last_dim = 4, 512
    x, ws, act = make_inputs(rows, x_last_dim)
    x[1] = 0
    x[3] = 0
    y, s = run_kernel(x, ws, act)
    y_g, s_g = dequant_swiglu_quant_golden(x, ws, act)
    # Golden yields NaN for 0/0 on the zero rows; pin the kernel contract.
    assert torch.equal(s[1], torch.tensor(0.0))
    assert torch.equal(s[3], torch.tensor(0.0))
    assert torch.equal(y[1], torch.zeros(x_last_dim // 2, dtype=torch.int8))
    assert torch.equal(y[3], torch.zeros(x_last_dim // 2, dtype=torch.int8))
    # Non-zero rows still compare against the golden.
    for r in (0, 2):
        torch.testing.assert_close(y[r], y_g[r], atol=1, rtol=0)
        torch.testing.assert_close(s[r], s_g[r], rtol=1e-5, atol=0)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_strict_extreme_int32_range():
    # Full-range int32 x with tiny scales: silu saturates but fp32 math and
    # the row max stay finite; exercises the Cast int32->fp32 path.
    rows, x_last_dim = 8, 1024
    x = torch.randint(-(2**28), 2**28, (rows, x_last_dim), dtype=torch.int32)
    ws = torch.randn(x_last_dim, dtype=torch.float32) * 1e-8
    act = torch.rand(rows, 1) * 4 + 0.5
    y, s = run_kernel(x, ws, act)
    y_g, s_g = dequant_swiglu_quant_golden(x, ws, act)
    check(y, s, y_g, s_g)


def test_strict_all_negative_rows():
    # Rows whose swiglu is entirely negative: the row max is |min| and the
    # int8 output must saturate at -127, not wrap.
    rows, x_last_dim = 4, 512
    x = -torch.randint(1, 1000, (rows, x_last_dim), dtype=torch.int32)
    ws = torch.ones(x_last_dim, dtype=torch.float32) * 0.001
    act = torch.ones(rows, 1)
    y, s = run_kernel(x, ws, act)
    y_g, s_g = dequant_swiglu_quant_golden(x, ws, act)
    check(y, s, y_g, s_g)
    assert y.min() >= -127 and y.max() <= 127


def _reject(fn):
    with pytest.raises(Exception):
        fn()


def test_reject_odd_last_dim():
    x, ws, act = make_inputs(4, 512)
    x_odd = x[:, :511].contiguous()
    _reject(lambda: run_kernel(x_odd, ws[:511], act))


def test_reject_last_dim_not_64_aligned():
    # 160 % 64 != 0: the DskTiling path requires x last dim divisible by 64.
    x = torch.randint(-100, 100, (4, 160), dtype=torch.int32)
    ws = torch.randn(160, dtype=torch.float32) * 0.001
    act = torch.rand(4, 1) * 4 + 0.5
    _reject(lambda: run_kernel(x, ws, act))


def test_reject_weight_scale_wrong_size():
    x, ws, act = make_inputs(4, 512)
    _reject(lambda: run_kernel(x, ws[:-64], act))


def test_reject_weight_scale_1d_with_groups():
    x, ws, act = make_inputs(8, 512)
    group_index = torch.tensor([2] * 4, dtype=torch.int64)
    _reject(lambda: run_kernel(x, ws, act, group_index=group_index))


def test_reject_group_index_wrong_dtype():
    x, _, act = make_inputs(8, 512)
    ws = torch.randn(4, 512, dtype=torch.float32) * 0.001
    group_index = torch.tensor([2] * 4, dtype=torch.int32)
    _reject(lambda: run_kernel(x, ws, act, group_index=group_index))


def test_reject_invalid_quant_mode():
    x, ws, act = make_inputs(4, 512)
    _reject(lambda: run_kernel(x, ws, act, quant_mode=5))


def test_reject_negative_clamp_limit():
    x, ws, act = make_inputs(4, 512)
    _reject(lambda: run_kernel(x, ws, act, clamp_limit=-1.0))


def test_reject_invalid_swiglu_mode():
    x, ws, act = make_inputs(4, 512)
    _reject(lambda: run_kernel(x, ws, act, swiglu_mode=3))


def test_reject_quant_offset_in_dynamic_group_mode():
    # Dynamic quantization with a group index only supports quant_offset=None.
    x = torch.randint(-100, 100, (8, 512), dtype=torch.int32)
    ws = torch.randn(4, 512, dtype=torch.float32) * 0.001
    act = torch.rand(8, 1) * 4 + 0.5
    group_index = torch.tensor([2] * 4, dtype=torch.int64)
    quant_offset = torch.ones(4, 256, dtype=torch.float32)
    with pytest.raises(Exception):
        torch.ops._C_ascend.npu_dequant_swiglu_quant(
            x=x.npu(), weight_scale=ws.npu(), activation_scale=act.npu(),
            bias=None, quant_scale=None, quant_offset=quant_offset.npu(),
            group_index=group_index.npu(), activate_left=True, quant_mode=1,
            swiglu_mode=0, clamp_limit=0.0, glu_alpha=1.0, glu_bias=0.0)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
