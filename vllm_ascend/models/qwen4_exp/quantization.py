# SPDX-License-Identifier: Apache-2.0
"""ModelSlim FA3 metadata, calibrated V parameters and QSA C8 storage."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

C8_SCALE_BYTES = 4
C8_QUANT_MAX = 127
C8_SCALE_MIN = torch.finfo(torch.float32).eps
C8_QSA_QUANT_TYPE = "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL"


@dataclass(frozen=True, kw_only=True)
class QSAC8Spec(FullAttentionSpec):
    """Keep logical head dimensions while accounting for FP32 scales."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype != torch.int8 or self.head_size != self.head_size_v or self.head_size % C8_SCALE_BYTES:
            raise ValueError("QSA C8 requires INT8 storage and equal, FP32-aligned logical K/V widths")
        object.__setattr__(self, "num_head_slots", 2)
        object.__setattr__(self, "state_content_bytes", self.num_kv_heads * self.storage_head_size)

    @property
    def storage_head_size(self) -> int:
        return self.head_size + C8_SCALE_BYTES


def uses_qsa_c8_quantization(quant_config: Any, prefix: str, *, indexer: bool = False) -> bool:
    """Resolve an exact FA owner, independently of linear and MTP schemes.

    A global ``fa_quant_type`` or substring match would incorrectly quantize
    unmarked draft layers. Dynamic-V checkpoints cannot use native INT8 GBSA.
    """
    description = getattr(quant_config, "quant_description", {})
    key = f"{prefix}.quant_type"
    if key not in description:
        return False
    mode = description[key]
    expected = "QK_INT8_DYNAMIC" if indexer else C8_QSA_QUANT_TYPE
    if mode != expected:
        raise ValueError(f"Unsupported QSA FA3 mode {mode!r} for {prefix}; expected {expected}")
    return True


class QSAStaticVParameters(torch.nn.Module):
    """Load ModelSlim's flattened (KV head, channel) scales with QKV TP rules."""

    def __init__(self, total_heads: int, dim: int, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        if min(total_heads, dim, tp_size) <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid QSA V-scale tensor-parallel geometry")
        if max(total_heads, tp_size) % min(total_heads, tp_size):
            raise ValueError("QSA KV heads and TP size must divide each other")
        self.total_heads, self.dim = total_heads, dim
        self.local_heads = max(1, total_heads // tp_size)
        replicas = max(1, tp_size // total_heads)
        self.head_start = (tp_rank // replicas) * self.local_heads
        # A missing scale fails post-load validation instead of silently using 1.
        self.scale = torch.nn.Parameter(
            torch.full((self.local_heads, dim), float("nan"), dtype=torch.float32),
            requires_grad=False,
        )
        self.offset = torch.nn.Parameter(torch.zeros((self.local_heads, dim), dtype=torch.int8), requires_grad=False)
        self.scale.weight_loader = self.load_scale
        self.offset.weight_loader = self.load_offset

    def _shard(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.numel() != self.total_heads * self.dim:
            raise ValueError("QSA V parameter must contain one value per global KV head/channel")
        return weight.reshape(self.total_heads, self.dim).narrow(0, self.head_start, self.local_heads)

    def load_scale(self, param: torch.Tensor, weight: torch.Tensor) -> None:
        if weight.dtype != torch.float32:
            raise ValueError("QSA V scale must be FP32")
        shard = self._shard(weight)
        if not torch.isfinite(shard).all() or not (shard > 0).all():
            raise ValueError("QSA V scale must be finite and positive")
        param.data.copy_(shard)

    def load_offset(self, param: torch.Tensor, weight: torch.Tensor) -> None:
        shard = self._shard(weight)
        if not (shard == 0).all():
            raise ValueError("Native INT8 GBSA requires symmetric V with zero offset")
        param.data.copy_(shard)

    def process_weights_after_loading(self) -> None:
        # Validation happens once at load time, outside graph/forward execution.
        scale, offset = self.scale.detach().cpu(), self.offset.detach().cpu()
        if not torch.isfinite(scale).all() or not (scale > 0).all() or (offset != 0).any():
            raise ValueError("Missing or invalid calibrated QSA V quantization parameters")


def quantize_qsa_values(rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Static V reference with ModelSlim's signed INT8 saturation limits."""
    if rows.ndim != 3 or rows.shape[1:] != scale.shape or scale.dtype != torch.float32:
        raise ValueError("QSA V requires [tokens,heads,dim] and FP32 [heads,dim] scales")
    return (rows.float() / scale).round().clamp(-128, C8_QUANT_MAX).to(torch.int8)


def quantize_qsa_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token/head INT8 reference, also used for small queries."""
    if rows.ndim != 3 or rows.shape[-1] == 0 or not rows.is_floating_point():
        raise ValueError("QSA quantization requires floating [tokens, heads, dim] rows")
    values = rows.float()
    scales = (values.abs().amax(dim=-1) / C8_QUANT_MAX).clamp_min(C8_SCALE_MIN)
    codes = (values / scales.unsqueeze(-1)).round().clamp(-C8_QUANT_MAX, C8_QUANT_MAX).to(torch.int8)
    return codes, scales


def qsa_c8_indexer_cache_views(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the dense INT8 codes and FP16 scales within each indexer page.

    QLI requires dense inner dimensions. Codes and scales share the scheduler
    page lifetime, with spare bytes for FP32 alignment of strided pages.
    """
    if cache.ndim != 4 or cache.dtype != torch.int8 or min(cache.shape) <= 0:
        raise ValueError("QSA C8 cache must be a nonempty 4-D INT8 tensor")
    width = cache.shape[-1] - C8_SCALE_BYTES
    if width <= 0 or width % C8_SCALE_BYTES or cache.stride(-1) != 1:
        raise ValueError("QSA C8 head width must be positive and FP32 aligned")
    if any(stride % C8_SCALE_BYTES for stride in cache.stride()[:-1]) or cache.storage_offset() % C8_SCALE_BYTES:
        raise ValueError("QSA C8 page and record offsets must be FP32 aligned")
    if cache.shape[2] != 1:
        raise ValueError("QSA C8 indexer requires one KV head")
    pages, states = cache.shape[:2]
    codes = cache.as_strided((pages, states, 1, width), (cache.stride(0), width, width, 1))
    scale_storage = cache.view(torch.float16)
    scales = scale_storage.as_strided(
        (pages, states, 1),
        (cache.stride(0) // 2, 1, 1),
        storage_offset=cache.storage_offset() // 2 + states * width // 2,
    )
    return codes, scales
