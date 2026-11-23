# SPDX-License-Identifier: Apache-2.0
"""NOVA-KV reference codec: group-VQ keys and packed INT2 values."""

from dataclasses import dataclass, fields

import torch

GROUP_SIZE = 4
CODEBOOK_SIZE = 256
FORMAT_VERSION = 1
SCALE_FLOOR = 1e-8


def pack_int2(symbols: torch.Tensor) -> torch.Tensor:
    if symbols.shape[-1] % GROUP_SIZE:
        raise ValueError("INT2 packing requires a dimension divisible by four")
    parts = symbols.to(torch.int32).reshape(*symbols.shape[:-1], symbols.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    return (parts[..., 0] + parts[..., 1] * 4 + parts[..., 2] * 16 + parts[..., 3] * 64).to(torch.uint8)


def unpack_int2(payload: torch.Tensor) -> torch.Tensor:
    value = payload.to(torch.int32)
    return torch.stack([value % 4, (value // 4) % 4, (value // 16) % 4, value // 64], dim=-1).flatten(-2)


def head_transform(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("thd,hde->the", x.float(), matrix.float())


@dataclass
class NovaCodec:
    key_mean: torch.Tensor
    key_transform: torch.Tensor
    key_inverse: torch.Tensor
    value_transform: torch.Tensor
    codebooks: torch.Tensor

    def __post_init__(self):
        if self.key_mean.ndim != 2:
            raise ValueError("Invalid NOVA-KV mean geometry")
        heads, dim = self.key_mean.shape
        if heads < 1 or dim < GROUP_SIZE or dim % GROUP_SIZE:
            raise ValueError("Invalid NOVA-KV head geometry")
        for matrix in (self.key_transform, self.key_inverse, self.value_transform):
            if matrix.shape != (heads, dim, dim):
                raise ValueError("Invalid NOVA-KV transform geometry")
        if self.codebooks.shape != (heads, dim // GROUP_SIZE, CODEBOOK_SIZE, GROUP_SIZE):
            raise ValueError("Invalid NOVA-KV codebook geometry")
        if any(not tensor.is_floating_point() for tensor in (getattr(self, field.name) for field in fields(self))):
            raise ValueError("NOVA-KV artifacts require floating point tensors")
        device = self.key_mean.device
        self._group_index = torch.arange(dim // GROUP_SIZE, device=device, dtype=torch.long)
        self._head_index = torch.arange(heads, device=device, dtype=torch.long)
        self._value_lut = torch.tensor(
            [[(value >> shift) & 3 for shift in (0, 2, 4, 6)] for value in range(256)],
            device=device,
            dtype=torch.float32,
        )
        # These persistent copies must exist before vLLM profiles memory.
        self._books_float = self.codebooks.float()
        self._book_norms = self._books_float.square().sum(-1).unsqueeze(0)

    @classmethod
    def load(cls, path: str, device: torch.device, rank: int, tp_size: int, heads: int):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported NOVA-KV artifact version")
        total_heads = data["key_mean"].shape[0]
        if tp_size < 1 or not 0 <= rank < tp_size or heads < 1:
            raise ValueError("Invalid NOVA-KV tensor parallel geometry")
        if total_heads >= tp_size and total_heads == tp_size * heads:
            start = rank * heads
        elif total_heads > 0 and tp_size % total_heads == 0 and heads == 1:
            start = rank // (tp_size // total_heads)
        else:
            raise ValueError("NOVA-KV artifact does not match tensor parallel geometry")
        tensors = {}
        for field in fields(cls):
            tensor = data[field.name]
            if tensor.shape[0] != total_heads or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid NOVA-KV artifact tensor: {field.name}")
            dtype = tensor.dtype if field.name == "codebooks" else torch.float32
            tensors[field.name] = tensor[start : start + heads].to(device=device, dtype=dtype)
        codec = cls(**tensors)
        dim = codec.key_mean.shape[-1]
        if codec.codebooks.shape != (heads, dim // GROUP_SIZE, CODEBOOK_SIZE, GROUP_SIZE):
            raise ValueError("Invalid NOVA-KV codebook geometry")
        return codec

    def encode(self, key: torch.Tensor, value: torch.Tensor, chunk_size: int = 256):
        if chunk_size < 1:
            raise ValueError("NOVA-KV codec chunk_size must be positive")
        if key.ndim != 3 or value.shape != key.shape or key.shape[1:] != self.key_mean.shape:
            raise ValueError("NOVA-KV input does not match artifact geometry")
        if not key.shape[0]:
            shape = (*key.shape[:2], key.shape[-1] // GROUP_SIZE)
            return (
                torch.empty(shape, device=key.device, dtype=torch.uint8),
                torch.empty(shape, device=key.device, dtype=torch.uint8),
                torch.empty((*key.shape[:2], 3), device=key.device, dtype=torch.bfloat16),
            )
        transformed = head_transform(key.float() - self.key_mean, self.key_transform)
        key_scale = transformed.square().mean(-1, keepdim=True).sqrt().clamp_min(SCALE_FLOOR)
        normalized = (transformed / key_scale).reshape(key.shape[0], key.shape[1], -1, GROUP_SIZE)
        indices = []
        books, norms = self._books_float, self._book_norms
        # Bound the nearest-neighbour workspace independently of prompt length.
        for rows in normalized.split(chunk_size):
            distances = rows.square().sum(-1, keepdim=True) + norms
            distances = distances - 2 * torch.einsum("thgd,hgcd->thgc", rows, books)
            indices.append(distances.argmin(-1).to(torch.uint8))
        rotated_value = head_transform(value, self.value_transform)
        low = rotated_value.amin(-1, keepdim=True)
        value_scale = ((rotated_value.amax(-1, keepdim=True) - low) / 3).clamp_min(SCALE_FLOOR)
        symbols = ((rotated_value - low) / value_scale).round().clamp(0, 3)
        return torch.cat(indices), pack_int2(symbols), torch.cat([key_scale, value_scale, low], -1).to(torch.bfloat16)

    def decode(
        self,
        key_indices: torch.Tensor,
        value_payload: torch.Tensor,
        scales: torch.Tensor,
        output_dtype: torch.dtype | None = None,
    ):
        tokens, heads, groups = key_indices.shape
        # Direct table indexing avoids materializing an expanded [T,H,G,256,4]
        # view and the 5-D gather index on every decode step.
        flat_index = (
            self._head_index[None, :, None] * groups + self._group_index[None, None, :]
        ) * CODEBOOK_SIZE + key_indices.long()
        transformed_key = (
            self.codebooks.reshape(-1, GROUP_SIZE)
            .index_select(0, flat_index.reshape(-1))
            .reshape(tokens, heads, groups * GROUP_SIZE)
        )
        key = head_transform(transformed_key.float() * scales[..., :1].float(), self.key_inverse) + self.key_mean
        # A small LUT replaces four integer div/mod operations per packed byte.
        rotated_value = self._value_lut[value_payload.long()].reshape(tokens, heads, groups * GROUP_SIZE)
        rotated_value = rotated_value * scales[..., 1:2].float() + scales[..., 2:3].float()
        value = head_transform(rotated_value, self.value_transform.transpose(-1, -2))
        if output_dtype is not None:
            key = key.to(output_dtype)
            value = value.to(output_dtype)
        return key, value


def packet_bytes(head_dim: int) -> int:
    if head_dim < GROUP_SIZE or head_dim % GROUP_SIZE:
        raise ValueError("NOVA-KV head dimension must be divisible by four")
    return head_dim // 2 + 6


def cache_views(raw: torch.Tensor, spec, kernel_block_size: int):
    if isinstance(raw, (tuple, list)):
        raise ValueError("NOVA-KV requires a combined raw cache allocation")
    if raw.dtype not in (torch.int8, torch.uint8) or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("NOVA-KV requires a contiguous one-dimensional byte allocation")
    if raw.numel() % spec.page_size_bytes or spec.page_size_bytes % 2 or raw.storage_offset() % 2:
        raise ValueError("NOVA-KV cache pages must be complete and BF16 aligned")
    pages = raw.numel() // spec.page_size_bytes
    shape = (pages, spec.block_size, spec.num_kv_heads, packet_bytes(spec.head_size))
    if spec.block_size != kernel_block_size:
        raise ValueError("NOVA-KV reference backend does not split kernel pages")
    return torch.as_strided(
        raw.view(torch.uint8),
        shape,
        (spec.page_size_bytes, spec.num_kv_heads * shape[-1], shape[-1], 1),
    )
