# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

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


def test_rotation_preserves_attention_with_sinks():
    torch.manual_seed(0)
    store = TurboQuantLatent()
    q, kv = torch.randn(3, 512), torch.randn(9, 512)
    sinks = torch.randn(3, 1)

    def attention(q, kv):
        scores = torch.cat((q @ kv.T / 512**0.5, sinks), dim=-1)
        return scores.softmax(-1)[:, :-1] @ kv

    expected = attention(q, kv)
    actual = store.inverse(attention(store.forward(q), store.forward(kv)))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("rows", [0, 1, 17])
def test_packed_layout_norm_correction_and_zero_rows(rows):
    torch.manual_seed(3)
    latent = torch.randn(rows, 1, 512)
    if rows:
        latent[0].zero_()
    store = TurboQuantLatent()
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
