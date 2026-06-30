# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402


def _has_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="solve_tri requires an NPU device")


def _make_strict_lower_triangular(shape, layout, *, dtype=torch.float16):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype) * 0.05
    bt = shape[-1]
    if layout == "bhtd":
        batch, heads, seq_len, _ = shape
        for b in range(batch):
            for h in range(heads):
                for start in range(0, seq_len, bt):
                    block = torch.tril(x[b, h, start : start + bt, :], diagonal=-1)
                    x[b, h, start : start + bt, :] = block
    elif layout == "bsnd":
        batch, seq_len, heads, _ = shape
        for b in range(batch):
            for h in range(heads):
                for start in range(0, seq_len, bt):
                    block = torch.tril(x[b, start : start + bt, h, :], diagonal=-1)
                    x[b, start : start + bt, h, :] = block
    else:
        raise AssertionError(f"unexpected layout {layout}")
    return x


def _iter_blocks(x, layout):
    bt = x.shape[-1]
    if layout == "bhtd":
        batch, heads, seq_len, _ = x.shape
        for b in range(batch):
            for h in range(heads):
                for start in range(0, seq_len, bt):
                    yield x[b, h, start : start + bt, :]
    elif layout == "bsnd":
        batch, seq_len, heads, _ = x.shape
        for b in range(batch):
            for h in range(heads):
                for start in range(0, seq_len, bt):
                    yield x[b, start : start + bt, h, :]
    else:
        raise AssertionError(f"unexpected layout {layout}")


def _solve_tri_golden(x, layout):
    out = torch.empty_like(x, dtype=torch.float32)
    bt = x.shape[-1]
    eye = torch.eye(bt, dtype=torch.float32)
    if layout == "bhtd":
        batch, heads, seq_len, _ = x.shape
        for b in range(batch):
            for h in range(heads):
                for start in range(0, seq_len, bt):
                    block = x[b, h, start : start + bt, :].float()
                    out[b, h, start : start + bt, :] = torch.linalg.inv(eye + block)
    elif layout == "bsnd":
        batch, seq_len, heads, _ = x.shape
        for b in range(batch):
            for h in range(heads):
                for start in range(0, seq_len, bt):
                    block = x[b, start : start + bt, h, :].float()
                    out[b, start : start + bt, h, :] = torch.linalg.inv(eye + block)
    return out.to(x.dtype)


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("bhtd", (1, 2, 32, 16)),
        ("bsnd", (1, 32, 2, 16)),
    ],
)
def test_npu_solve_tri_matches_cpu_golden(layout, shape):
    x = _make_strict_lower_triangular(shape, layout)
    expected = _solve_tri_golden(x, layout)

    actual = torch.ops._C_ascend.npu_solve_tri(x.npu(), layout=layout).cpu()

    np.testing.assert_allclose(
        actual.float().numpy(),
        expected.float().numpy(),
        rtol=1e-2,
        atol=1e-2,
    )

    bt = shape[-1]
    eye = torch.eye(bt, dtype=torch.float32)
    max_err = 0.0
    for a_block, y_block in zip(_iter_blocks(x, layout), _iter_blocks(actual, layout)):
        err = ((eye + a_block.float()) @ y_block.float() - eye).abs().max().item()
        max_err = max(max_err, err)
    assert max_err < 2e-2


def test_npu_solve_tri_meta_shape():
    x = torch.empty((1, 2, 32, 16), dtype=torch.float16, device="meta")
    out = torch.ops._C_ascend.npu_solve_tri(x, layout="bhtd")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device.type == "meta"
