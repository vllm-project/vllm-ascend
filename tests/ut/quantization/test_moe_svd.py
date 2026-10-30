import pytest
import torch

from vllm_ascend.quantization.moe_svd import (
    dequantize_modelslim,
    pack_int4,
    quantize_factor,
    rotate_factors,
    svd_factors,
    unpack_int4,
)


def test_signed_nibbles_and_channel_order():
    values = torch.arange(-8, 8, dtype=torch.int8).view(8, 2)
    packed = pack_int4(values)
    assert packed.view(torch.uint8).flatten().tolist() == [168, 185, 236, 253, 32, 49, 100, 117]
    torch.testing.assert_close(unpack_int4(packed), values)
    scales = torch.arange(1, 9).float()[:, None]
    torch.testing.assert_close(dequantize_modelslim(packed, scales, torch.zeros_like(scales)), values * scales)
    with pytest.raises(ValueError, match="offset"):
        dequantize_modelslim(packed, scales, torch.ones_like(scales))


@pytest.mark.parametrize("shape", [(24, 64), (64, 24)])
def test_truncation_matches_svd_optimal_error(shape):
    torch.manual_seed(17)
    matrix = torch.randn(shape)
    left, right, energy = svd_factors(matrix, 12)
    singular = torch.linalg.svdvals(matrix)
    error = (matrix - left @ right).square().sum()
    torch.testing.assert_close(error, singular[12:].square().sum(), atol=1e-3, rtol=1e-5)
    assert abs(energy - float(singular[:12].square().sum() / singular.square().sum())) < 1e-5
    inputs = torch.randn(7, shape[1])
    torch.testing.assert_close((inputs @ right.T) @ left.T, inputs @ (left @ right).T, atol=1e-4, rtol=1e-4)


def test_zero_matrix_and_quantization_bound():
    left, right, energy = svd_factors(torch.zeros(16, 32), 8)
    assert not (left @ right).count_nonzero()
    assert energy == 0
    for matrix in (left, right, torch.randn(32, 64)):
        factor = quantize_factor(matrix)
        assert torch.all((factor.dequantize() - matrix).abs() <= factor.scale / 2 + 1e-6)
        assert factor.weight.numel() * 2 == matrix.numel()


@pytest.mark.parametrize("rank", [0, 3, 18])
def test_invalid_rank(rank):
    with pytest.raises(ValueError, match="Rank"):
        svd_factors(torch.ones(16, 32), rank)


@pytest.mark.parametrize("rank", [32, 48, 1024])
def test_latent_rotation_preserves_product_and_is_reproducible(rank):
    torch.manual_seed(77)
    left, right = torch.randn(12, rank), torch.randn(rank, 20)
    rotated_left, rotated_right = rotate_factors(left, right)
    torch.testing.assert_close(rotated_left @ rotated_right, left @ right, atol=1e-3, rtol=1e-4)
    again_left, again_right = rotate_factors(left, right)
    torch.testing.assert_close(rotated_left, again_left, atol=0, rtol=0)
    torch.testing.assert_close(rotated_right, again_right, atol=0, rtol=0)
