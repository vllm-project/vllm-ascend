import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mul_add import muls_add_triton as muls_add_triton_from_mul_add
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# These kernels require Ascend A2/A3 (910B/C). Skip on other hardware, aligned
# with the upstream guard added in #18431.
_REQUIRES_A2_A3 = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)

DEVICE = "npu"
# Tolerances per dtype, aligned with neighboring Triton op tests
# (see test_rms_norm.py). muls_add is an elementwise op, so tolerances stay
# tight: the only source of numerical drift is dtype rounding of `x * scale`.
TOLERANCES = {
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (2e-2, 5e-2),
    torch.float32: (1e-4, 1e-4),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def _ref(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """PyTorch baseline for muls_add: output = x * scale + y (elementwise)."""
    return x * scale + y


# ---------------------------------------------------------------------------
# Original correctness cases (kept unchanged so existing CI coverage is
# preserved) plus a few more shapes / dtypes.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@pytest.mark.parametrize(
    ("shape", "dtype", "scale"),
    [
        ((1, 2048), torch.float16, 1.25),
        ((4000, 2048), torch.float16, 0.75),
        ((4, 2048), torch.bfloat16, 1.0),
        ((8, 4096), torch.float32, 0.5),
    ],
)
@torch.inference_mode()
def test_muls_add_triton_correctness(shape, dtype, scale):
    """compare the correctness of muls_add_triton with the PyTorch baseline implementation."""
    device = "npu"

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    y = torch.randn(*shape, dtype=dtype, device=device)

    out_triton = muls_add_triton(x, y, scale)
    out_ref = _ref(x, y, scale)

    rtol, atol = TOLERANCES[dtype]

    assert out_triton.shape == out_ref.shape
    assert out_triton.dtype == out_ref.dtype
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Boundary: different rank shapes (1D, 3D). The kernel only relies on
# x.numel() and x.shape[-1], so multi-rank tensors must work too.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        pytest.param((2048,), torch.float16, id="1d-fp16"),
        pytest.param((2, 4, 1024), torch.bfloat16, id="3d-bf16"),
        pytest.param((1, 1, 4096), torch.float32, id="3d-single-row-fp32"),
    ],
)
@torch.inference_mode()
def test_muls_add_triton_various_ranks(shape, dtype):
    torch.manual_seed(1)
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    y = torch.randn(*shape, dtype=dtype, device=DEVICE)
    scale = 0.875

    out_triton = muls_add_triton(x, y, scale)
    out_ref = _ref(x, y, scale)

    rtol, atol = TOLERANCES[dtype]
    assert out_triton.shape == out_ref.shape
    torch.testing.assert_close(
        out_triton.float().cpu(),
        out_ref.float().cpu(),
        rtol=rtol,
        atol=atol,
    )


# ---------------------------------------------------------------------------
# Boundary: scale edge values (0, negative, large). scale=0 must degenerate
# to `0 + y == y` exactly; negative / large exercise the multiply path.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@pytest.mark.parametrize(
    ("scale", "dtype"),
    [
        pytest.param(0.0, torch.float32, id="scale-zero"),
        pytest.param(-1.5, torch.float16, id="scale-negative"),
        pytest.param(1e4, torch.bfloat16, id="scale-large"),
    ],
)
@torch.inference_mode()
def test_muls_add_triton_scale_boundary(scale, dtype):
    torch.manual_seed(2)
    shape = (2, 2048)
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    y = torch.randn(*shape, dtype=dtype, device=DEVICE)

    out_triton = muls_add_triton(x, y, scale)
    out_ref = _ref(x, y, scale)

    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(
        out_triton.float().cpu(),
        out_ref.float().cpu(),
        rtol=rtol,
        atol=atol,
    )


# ---------------------------------------------------------------------------
# Boundary: tiny / non-power-of-two sizes exercise the `offsets < n_elements`
# tail mask path of the kernel.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@pytest.mark.parametrize(
    ("n", "dtype"),
    [
        pytest.param(1, torch.float16, id="size-1"),
        pytest.param(7, torch.float16, id="size-7-non-pow2"),
        pytest.param(1025, torch.bfloat16, id="size-1025-tail"),
    ],
)
@torch.inference_mode()
def test_muls_add_triton_tail_mask(n, dtype):
    torch.manual_seed(3)
    x = torch.randn(n, dtype=dtype, device=DEVICE)
    y = torch.randn(n, dtype=dtype, device=DEVICE)
    scale = 1.5

    out_triton = muls_add_triton(x, y, scale)
    out_ref = _ref(x, y, scale)

    rtol, atol = TOLERANCES[dtype]
    assert out_triton.shape == (n,)
    torch.testing.assert_close(
        out_triton.float().cpu(),
        out_ref.float().cpu(),
        rtol=rtol,
        atol=atol,
    )


# ---------------------------------------------------------------------------
# Boundary: non-contiguous (sliced) input. The kernel reads inputs via linear
# pointer offsets and ignores strides, so a non-contiguous tensor would
# silently compute on the wrong elements. The entry point now guards this
# with an explicit `is_contiguous()` assertion; verify it surfaces the misuse
# as an error rather than a wrong result.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@torch.inference_mode()
def test_muls_add_triton_non_contiguous_raises():
    x_base = torch.randn(4, 2048, dtype=torch.float32, device=DEVICE)
    y_base = torch.randn(4, 2048, dtype=torch.float32, device=DEVICE)
    x = x_base[:, ::2]  # non-contiguous
    y = y_base[:, ::2]

    with pytest.raises(AssertionError, match="contiguous"):
        muls_add_triton(x, y, 1.0)


# ---------------------------------------------------------------------------
# Negative: mismatched shapes must hit the `assert x.shape == y.shape` guard.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@torch.inference_mode()
def test_muls_add_triton_shape_mismatch_raises():
    x = torch.randn(2, 2048, dtype=torch.float16, device=DEVICE)
    y = torch.randn(3, 2048, dtype=torch.float16, device=DEVICE)

    with pytest.raises(AssertionError, match="same shape"):
        muls_add_triton(x, y, 1.0)


# ---------------------------------------------------------------------------
# Parity: the two twin implementations (mul_add.py and muls_add.py) are
# intentionally identical today; keep them in lockstep so a future divergence
# is surfaced by CI rather than silently changing DeepSeek v4/v4.1 outputs.
# ---------------------------------------------------------------------------
@_REQUIRES_A2_A3
@torch.inference_mode()
def test_mul_add_and_muls_add_implementations_match():
    torch.manual_seed(4)
    shape = (2, 2048)
    dtype = torch.bfloat16
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    y = torch.randn(*shape, dtype=dtype, device=DEVICE)
    scale = 1.25

    out_mul_add = muls_add_triton_from_mul_add(x, y, scale)
    out_muls_add = muls_add_triton(x, y, scale)

    torch.testing.assert_close(
        out_mul_add.float().cpu(),
        out_muls_add.float().cpu(),
        rtol=0.0,
        atol=0.0,
    )
