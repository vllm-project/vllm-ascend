# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401,E402


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
    ("layout", "shape", "dtype"),
    [
        ("bhtd", (1, 2, 32, 16), torch.float16),
        ("bsnd", (1, 32, 2, 16), torch.float16),
        ("bhtd", (1, 16, 64, 16), torch.float16),
        ("bhtd", (1, 16, 128, 64), torch.float16),
        ("bhtd", (1, 16, 128, 128), torch.bfloat16),
        ("bsnd", (1, 128, 16, 64), torch.bfloat16),
        ("bsnd", (1, 128, 16, 128), torch.bfloat16),
        ("bhtd", (2, 8, 256, 64), torch.float16),
        ("bhtd", (1, 4, 64, 32), torch.bfloat16),
    ],
)
def test_npu_solve_tri_matches_cpu_golden(layout, shape, dtype):
    x = _make_strict_lower_triangular(shape, layout, dtype=dtype)
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


def _all_npu_devices():
    return list(range(torch.npu.device_count()))


@pytest.mark.parametrize("device_id", _all_npu_devices())
@pytest.mark.parametrize(
    ("layout", "shape", "dtype"),
    [
        ("bhtd", (1, 16, 128, 64), torch.float16),
        ("bhtd", (1, 16, 128, 128), torch.bfloat16),
        ("bsnd", (1, 128, 16, 128), torch.bfloat16),
    ],
)
def test_npu_solve_tri_all_devices(device_id, layout, shape, dtype):
    torch.npu.set_device(device_id)
    x = _make_strict_lower_triangular(shape, layout, dtype=dtype)
    expected = _solve_tri_golden(x, layout)

    actual = torch.ops._C_ascend.npu_solve_tri(x.to(f"npu:{device_id}"), layout=layout).cpu()
    torch.npu.synchronize(device_id)

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


@pytest.mark.parametrize("device_id", _all_npu_devices())
def test_npu_solve_tri_stress(device_id):
    torch.npu.set_device(device_id)
    shape = (1, 16, 256, 128)
    for i in range(50):
        x = _make_strict_lower_triangular(shape, "bhtd", dtype=torch.bfloat16)
        torch.ops._C_ascend.npu_solve_tri(x.to(f"npu:{device_id}"), layout="bhtd")
    torch.npu.synchronize(device_id)
