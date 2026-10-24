"""Portable packed INT4 factors for offline MoE SVD conversion.

Matrices use PyTorch linear layout [out, in]. Adjacent output channels are
packed low-nibble first, matching ModelSlim 1.0.0 and Ascend's N-axis packing.
No original dense expert matrix is stored in the factor representation.
"""

from dataclasses import dataclass

import torch

FORMAT_VERSION = 1
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FACTOR_ALIGNMENT = 32


def validate_factor_dimensions(rank: int, hidden_size: int, intermediate_size: int) -> None:
    if type(rank) is not int or rank <= 0 or rank % FACTOR_ALIGNMENT:
        raise ValueError("Low-rank W4A8 requires a positive integer rank aligned to 32")
    if any(type(size) is not int or size <= 0 or size % FACTOR_ALIGNMENT for size in (hidden_size, intermediate_size)):
        raise ValueError("Low-rank W4A8 requires positive hidden and intermediate sizes aligned to 32")
    if rank > min(hidden_size, intermediate_size):
        raise ValueError("Low-rank W4A8 rank must not exceed either expert dimension")


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError("Expected packed int8 [out/2, in] matrix")
    low = packed & 15
    high = (packed >> 4) & 15
    signed = torch.stack((low, high), dim=1).flatten(0, 1)
    return torch.where(signed >= 8, signed - 16, signed)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.int8 or values.ndim != 2 or values.shape[0] % 2:
        raise ValueError("Expected int8 [even out, in] matrix")
    if torch.any((values < -8) | (values > 7)):
        raise ValueError("INT4 values must be in [-8, 7]")
    return ((values[0::2] & 15) | ((values[1::2] & 15) << 4)).contiguous()


def dequantize_modelslim(packed: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    if scale.shape != (packed.shape[0] * 2, 1) or offset.shape != scale.shape:
        raise ValueError("Only ModelSlim per-channel packed INT4 is supported")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid ModelSlim scale")
    # The source checkpoint is symmetric. Reject unknown affine conventions
    # instead of guessing the offset sign or silently ignoring it.
    if torch.count_nonzero(offset):
        raise ValueError("Nonzero ModelSlim weight offsets are unsupported")
    return unpack_int4(packed).float() * scale.float()


@dataclass(frozen=True)
class Int4Factor:
    weight: torch.Tensor
    scale: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return unpack_int4(self.weight).float() * self.scale


def quantize_factor(weight: torch.Tensor) -> Int4Factor:
    if weight.ndim != 2 or weight.shape[0] % 2 or not torch.isfinite(weight).all():
        raise ValueError("Factor must be a finite matrix with even output width")
    weight = weight.float()
    scale = weight.abs().amax(dim=1, keepdim=True) / 7
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    values = (weight / scale).round().clamp(-7, 7).to(torch.int8)
    return Int4Factor(pack_int4(values), scale.contiguous())


def rotate_factors(left: torch.Tensor, right: torch.Tensor, seed: int = 0):
    """Balance the shared latent axis using a signed block Hadamard rotation.

    L D H and H D R have the same product as L R. The rotation is absorbed
    offline into the factors, so it adds no runtime tensors or operations.
    """
    rank = left.shape[1]
    if right.shape[0] != rank:
        raise ValueError("Factor ranks do not match")
    block_size = rank & -rank
    generator = torch.Generator(device=left.device).manual_seed(seed)
    signs = torch.randint(0, 2, (rank,), generator=generator, device=left.device).float() * 2 - 1

    def transform(matrix):
        shape = matrix.shape
        result = (matrix * signs).contiguous()
        stride = 1
        while stride < block_size:
            pairs = result.reshape(*shape[:-1], -1, 2, stride)
            first, second = pairs[..., 0, :], pairs[..., 1, :]
            result = torch.stack((first + second, first - second), dim=-2).reshape(shape)
            stride *= 2
        return result / block_size**0.5

    return transform(left).contiguous(), transform(right.T).T.contiguous()


def svd_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Truncated SVD via the smaller Gram matrix, without dense reconstruction.

    Uses FP32 eigh and matrix products. This is intended for offline conversion;
    the caller controls CPU thread count and device. Zero singular directions
    produce zero columns/rows rather than divisions by zero.
    """
    if weight.ndim != 2 or not 0 < rank <= min(weight.shape) or rank % 2:
        raise ValueError("Rank must be positive, even, and no larger than min(shape)")
    weight = weight.float()
    if not torch.isfinite(weight).all():
        raise ValueError("Non-finite source weights")
    wide = weight.shape[0] <= weight.shape[1]
    matrix = weight if wide else weight.T
    eigenvalues, vectors = torch.linalg.eigh(matrix @ matrix.T)
    eigenvalues = eigenvalues[-rank:].flip(0).clamp_min(0)
    vectors = vectors[:, -rank:].flip(1)
    root_singular = eigenvalues.sqrt().sqrt()
    inverse = torch.where(root_singular > 0, root_singular.clamp_min(1e-30).reciprocal(), 0)
    left = vectors * root_singular
    right = (vectors.T @ matrix) * inverse[:, None]
    energy = float(eigenvalues.sum() / matrix.square().sum().clamp_min(1e-30))
    if not wide:
        left, right = right.T, left.T
    return left.contiguous(), right.contiguous(), energy
