# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.quantization.methods.kv_cache.turboquant import latent as latent_module
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import CENTROIDS, TurboQuantLatent


def _quantize_reference(latent, centroids):
    norm = (latent.square().sum(-1) + 1e-16).sqrt()
    unit = latent / norm[:, None]
    codes = (unit[:, :, None] >= ((centroids[:-1] + centroids[1:]) / 2)).sum(-1)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, norm.half()


@pytest.mark.parametrize("legacy_hadamard", [False, True])
def test_rotation_preserves_attention_with_sinks(legacy_hadamard):
    torch.manual_seed(0)
    store = TurboQuantLatent(legacy_hadamard=legacy_hadamard)
    q, kv = torch.randn(3, 512), torch.randn(9, 512)
    sinks = torch.randn(3, 1)

    def attention(q, kv):
        scores = torch.cat((q @ kv.T / 512**0.5, sinks), dim=-1)
        return scores.softmax(-1)[:, :-1] @ kv

    expected = attention(q, kv)
    actual = store.inverse(attention(store.forward(q), store.forward(kv)))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("legacy_hadamard", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 17])
def test_packed_layout_norm_correction_and_zero_rows(rows, legacy_hadamard):
    torch.manual_seed(3)
    latent = torch.randn(rows, 1, 512)
    if rows:
        latent[0].zero_()
    store = TurboQuantLatent(legacy_hadamard=legacy_hadamard)
    with patch.object(latent_module, "_get_turbo_quant_op", return_value=_quantize_reference):
        packed = store.compress(latent)
    assert packed.shape == (rows, 1, 258)
    assert packed.dtype == torch.uint8
    assert packed.is_contiguous()
    codes = torch.stack((packed[:, 0, :256] & 15, packed[:, 0, :256] >> 4), dim=-1).flatten(1)
    scale = packed[:, 0, 256:].contiguous().view(torch.float16).float()
    reconstructed = torch.tensor(CENTROIDS)[codes.long()] * scale
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(reconstructed.norm(dim=-1), latent.flatten(1).norm(dim=-1), atol=2e-4, rtol=1e-3)
    if rows > 1:
        assert (store.inverse(reconstructed) - latent[:, 0]).norm() / latent.norm() < 0.15


def test_transform_storage_is_reused():
    store = TurboQuantLatent()
    x = torch.zeros(1, 512)
    store.forward(x)
    pointers = (store.rotation.data_ptr(), store.centroids.data_ptr(), store.norm_lut.data_ptr())
    store.forward(x)
    assert pointers == (store.rotation.data_ptr(), store.centroids.data_ptr(), store.norm_lut.data_ptr())


@pytest.mark.parametrize("legacy_hadamard", [False, True])
def test_transform_sharing_and_inverse_layout(legacy_hadamard):
    first, second = TurboQuantLatent(legacy_hadamard=legacy_hadamard), TurboQuantLatent(legacy_hadamard=legacy_hadamard)
    for store in (first, second):
        store._initialize(torch.device("cpu"))
    for field in ("rotation", "centroids", "norm_lut"):
        a, b = getattr(first, field), getattr(second, field)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert (a.data_ptr() == b.data_ptr()) == legacy_hadamard
    if legacy_hadamard:
        assert first.inverse_rotation is second.inverse_rotation
        assert first.inverse_rotation.is_contiguous()
        torch.testing.assert_close(first.inverse_rotation, first.rotation.T, rtol=0, atol=0)
    else:
        assert first.inverse_rotation is None


@pytest.mark.parametrize("legacy_hadamard", [False, True], ids=["sfa_baseline", "dsv4_historical"])
def test_rotation_and_packing_compatibility(legacy_hadamard):
    store = TurboQuantLatent(legacy_hadamard=legacy_hadamard)
    store._initialize(torch.device("cpu"))
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < 512:
        h = np.block([[h, h], [h, -h]])
    rng = np.random.default_rng(0)
    if legacy_hadamard:
        rng.standard_normal(400_000)
    signs = rng.choice([-1.0, 1.0], 512).astype(np.float32)
    rotation = torch.tensor(signs[:, None] * h / 512**0.5)
    torch.testing.assert_close(store.rotation, rotation, rtol=0, atol=0)
    value = torch.randn(17, 1, 512)
    raw = (value.float().reshape(-1, 512) @ rotation).contiguous()
    packed, norm = _quantize_reference(raw, store.centroids)
    scale = (norm.float() / store.norm_lut[packed.long()].sum(-1).sqrt()).half()
    expected = torch.cat((packed, scale.view(torch.uint8).reshape(-1, 2)), dim=-1).unsqueeze(1)
    with patch.object(latent_module, "_get_turbo_quant_op", return_value=_quantize_reference):
        torch.testing.assert_close(store.compress(value), expected, rtol=0, atol=0)
    inverse_rotation = rotation.T.contiguous() if legacy_hadamard else rotation.T
    torch.testing.assert_close(
        store.inverse(value), (value.flatten(0, 1) @ inverse_rotation).reshape_as(value), rtol=0, atol=0
    )
