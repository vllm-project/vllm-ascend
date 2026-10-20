# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 C2 ring compressor, ratio-1 path, and fused RMS normalization."""

import torch
import torch_npu
from torch import nn
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS, uses_a5_packed_cache
from vllm_ascend.utils import maybe_trans_nz


def _compact_to_token_aligned(
    compact: torch.Tensor,
    complete: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Spread compact closed-group rows back onto their closing tokens.

    ``compact`` holds one row per completed group in token order (the
    compressor_v2 output contract, padded at the tail); ``complete[t]``
    marks the closing token of each group. Non-closing rows of ``out`` are
    zeroed and never consumed downstream (their cache slots are -1).
    """
    num_tokens = complete.shape[0]
    if compact.shape[0] == 0 or num_tokens == 0:
        out.zero_()
        return out
    rank = complete.long().cumsum(0) - 1
    rank = rank.clamp_(min=0, max=compact.shape[0] - 1)
    gathered = compact[rank]
    # Masked boolean indexing lowers to dynamic Nonzero and cannot be
    # captured by ACLGraph; broadcast the mask instead.
    out.copy_(
        torch.where(
            complete.unsqueeze(-1),
            gathered,
            torch.zeros_like(gathered),
        ).to(out.dtype)
    )
    return out


class DeepseekV41RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, ratio, vllm_config=None, prefix="compressor"):
        super().__init__()
        self.ratio = ratio
        # The fused ds41 compressor op exists only in the A5 wheel
        # environment; other devices keep the eager Triton pooling path.
        self._fused = uses_a5_packed_cache()
        self._ring_num_cores = None
        self.width = config.head_dim
        dim = config.hidden_size
        self.wkv = nn.Linear(dim, self.width, bias=False, dtype=torch.float32 if ratio == 2 else torch.bfloat16)
        self.norm = DeepseekV41RMSNorm(self.width, config.rms_norm_eps)
        if ratio == 2:
            self.wgate = nn.Linear(dim, self.width, bias=False, dtype=torch.float32)
            # Allocate persistent output before memory profiling, so its footprint
            # is included in the cache budget rather than added after allocation.
            if vllm_config is not None:
                capacity = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
                self.register_buffer(
                    "_ring_pooled",
                    torch.empty(capacity, self.width, dtype=torch.bfloat16, device=self.wkv.weight.device),
                    persistent=False,
                )
            # Standalone unfused-reference tests may supply pages explicitly.
            if vllm_config is not None:
                state_spec = AscendCircularBufferSpec if uses_a5_packed_cache() else CircularBufferSpec
                self.state_cache = DeepseekV41CacheLayer(
                    vllm_config,
                    f"{prefix}.state_cache",
                    state_spec(
                        block_size=STATE_RING_ROWS,
                        num_kv_heads=1,
                        head_size=2 * self.width,
                        dtype=torch.float32,
                        head_size_v=0,
                    ),
                )

    def process_weights_after_loading(self):
        """Adapt the fused compressor operands after weight loading (A5 only)."""
        if not self._fused:
            # The eager Triton path keeps the checkpoint dtypes.
            return
        if self.ratio == 1:
            self.wkv.weight.data = maybe_trans_nz(self.wkv.weight.data)
        elif self.ratio == 2:
            # The fused ds41 compressor op takes BF16 projection weights.
            self.wkv.weight.data = self.wkv.weight.data.to(torch.bfloat16)
            self.wgate.weight.data = self.wgate.weight.data.to(torch.bfloat16)

    def prepare_ring_compressor(self, max_tokens, device):
        """Resolve ring-compressor hardware before capture."""
        if self._fused:
            actual_device = self._ring_pooled.device
            compatible_device = actual_device.type == device.type and (
                device.index is None or actual_device.index == device.index
            )
            if self._ring_pooled.shape[0] < max_tokens or not compatible_device:
                raise ValueError("Ring output capacity/device must be established before memory profiling")
        else:
            from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num

            self._ring_num_cores = _cube_core_num()

    def pool_projected(self, hidden_states, metadata):
        """Fused ds41 compressor path: project + gated-pool inside the op.

        The op (cann_ops_transformer.ops.ds41.compressor) takes the raw
        hidden_states plus the wkv/wgate weights, pools closed groups with a
        per-channel softmax gate over the FP32 ring state, and returns
        compact closed-group rows; the token-aligned spread and norm stay
        outside per the op contract.
        """
        from cann_ops_transformer.ops.ds41 import compressor as compressor_v2

        if not self._fused:
            raise RuntimeError("The fused ds41 compressor path requires the A5 wheel environment")
        if not hasattr(self, "_ring_pooled"):
            raise RuntimeError("Ring compressor must be initialized before graph capture")
        if hidden_states.shape[0] > self._ring_pooled.shape[0]:
            raise ValueError("Compressor batch exceeds its prepared output capacity")
        ring_meta = metadata.c2_ring_metadata
        compact = compressor_v2(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            self.state_cache.kv_cache[0].squeeze(-2),
            state_block_table=ring_meta[4],  # ring page id per request
            cu_seqlens=metadata.query_start_loc.int(),
            seqused=ring_meta[1],  # per-request valid tokens in this chunk
            start_pos=ring_meta[0],  # chunk-first-token absolute position
            cmp_ratio=2,
        )
        num_tokens = hidden_states.shape[0]
        pooled = _compact_to_token_aligned(
            compact,
            metadata.c2_complete_mask[:num_tokens],
            self._ring_pooled[:num_tokens],
        )
        out = self.norm(pooled)
        return out

    def pool_projected_eager(self, kv, scores, metadata):
        """Eager Triton path (non-A5): pool pre-projected KV/scores over the ring state."""
        from vllm_ascend.ops.triton.compressor.compressor_triton import compressor_from_projected

        pooled = compressor_from_projected(
            kv,
            scores,
            self.state_cache.kv_cache[0].squeeze(-2),
            metadata.c2_ring_metadata,
            self._ring_pooled[: kv.shape[0]],
            max_query_len=metadata.max_query_len,
            num_cores=self._ring_num_cores,
        )
        out = self.norm(pooled)
        return out

    def forward(self, x):
        """Project an uncompressed source; ratio-2 uses ``pool_projected``."""
        out = self.norm(self.wkv(x))
        return out
