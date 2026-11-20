# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import sfa_v1


def _project(projected, weight, rope_width):
    heads, nope_width, _ = weight.shape
    layer = SimpleNamespace(
        q_proj=lambda x: (x, None),
        local_num_heads=heads,
        qk_head_dim=nope_width + rope_width,
        qk_nope_head_dim=nope_width,
        qk_rope_head_dim=rope_width,
        W_UK_T=weight,
    )
    return sfa_v1.AscendSFAImpl._q_proj_and_k_up_proj(layer, projected)


def _transpose_bmm(x, weight, *, perm_x1, perm_x2, perm_y):
    return torch.bmm(x.permute(perm_x1), weight.permute(perm_x2)).permute(perm_y)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op_available", [True, False])
def test_q_projection_layout_and_dispatch(monkeypatch, dtype, op_available):
    fused = Mock(side_effect=_transpose_bmm)
    backend = SimpleNamespace(npu_transpose_batchmatmul=fused) if op_available else SimpleNamespace()
    monkeypatch.setattr(sfa_v1, "torch_npu", backend)

    projected = torch.arange(2 * 3 * 6, dtype=dtype).reshape(2, 18)
    weight = torch.arange(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5)
    nope, rope = projected.view(2, 3, 6).split([4, 2], dim=-1)
    expected = torch.bmm(nope.transpose(0, 1), weight).transpose(0, 1)
    actual, actual_rope = _project(projected, weight, rope_width=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_rope, rope, rtol=0, atol=0, check_stride=True)
    assert actual_rope.data_ptr() == rope.data_ptr()

    if op_available and dtype != torch.float32:
        fused.assert_called_once()
        x, w = fused.call_args.args
        assert x.data_ptr() == nope.data_ptr() and x.stride() == nope.stride()
        assert w is weight
        assert fused.call_args.kwargs == {"perm_x1": (1, 0, 2), "perm_x2": (0, 1, 2), "perm_y": (1, 0, 2)}
    else:
        fused.assert_not_called()


def test_mismatched_weight_dtype_uses_original_bmm(monkeypatch):
    fused = Mock()
    monkeypatch.setattr(sfa_v1, "torch_npu", SimpleNamespace(npu_transpose_batchmatmul=fused))
    projected = torch.ones(2, 18, dtype=torch.float16)
    weight = torch.ones(3, 4, 5, dtype=torch.bfloat16)
    # The original bmm rejects mixed dtypes; the fast path must not cast either operand.
    with pytest.raises(RuntimeError):
        _project(projected, weight, rope_width=2)
    fused.assert_not_called()


@pytest.mark.parametrize("axis", ["tokens", "heads", "nope", "latent"])
@pytest.mark.parametrize("size", [65535, 65536])
def test_operand_dimension_limit(monkeypatch, axis, size):
    dims = {"tokens": 2, "heads": 3, "nope": 4, "latent": 5}
    dims[axis] = size
    _check_shape_dispatch(monkeypatch, **dims, expect_fused=size == 65535)


@pytest.mark.parametrize("tokens,nope,latent", [(0, 4, 5), (2, 0, 5), (2, 4, 0)])
def test_empty_operand_uses_original_bmm(monkeypatch, tokens, nope, latent):
    _check_shape_dispatch(monkeypatch, tokens=tokens, heads=3, nope=nope, latent=latent, expect_fused=False)


def _check_shape_dispatch(monkeypatch, tokens, heads, nope, latent, expect_fused):
    fused = Mock(side_effect=_transpose_bmm)
    monkeypatch.setattr(sfa_v1, "torch_npu", SimpleNamespace(npu_transpose_batchmatmul=fused))
    # Meta tensors exercise the real view/split/bmm shape logic without large allocations.
    projected = torch.empty(tokens, heads * (nope + 2), dtype=torch.float16, device="meta")
    weight = torch.empty(heads, nope, latent, dtype=torch.float16, device="meta")
    result, rope = _project(projected, weight, rope_width=2)
    assert result.shape == (tokens, heads, latent)
    assert rope.shape == (tokens, heads, 2)
    assert fused.call_count == int(expect_fused)
