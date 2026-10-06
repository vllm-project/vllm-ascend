# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2 AscendC ring compression, BF16 projections, and FP32 ring state."""

import torch
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, ratio, vllm_config=None, prefix="compressor"):
        super().__init__()
        self.ratio = ratio
        self.width = config.head_dim
        dim = config.hidden_size
        self.wkv = nn.Linear(dim, self.width, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(self.width, eps=config.rms_norm_eps, dtype=torch.bfloat16)
        if ratio == 2:
            self.wgate = nn.Linear(dim, self.width, bias=False, dtype=torch.bfloat16)
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
                self.state_cache = DeepseekV41CacheLayer(
                    vllm_config,
                    f"{prefix}.state_cache",
                    CircularBufferSpec(
                        block_size=STATE_RING_ROWS,
                        num_kv_heads=1,
                        head_size=2 * self.width,
                        dtype=torch.float32,
                        head_size_v=0,
                    ),
                )

    def prepare_ring_compressor(self):
        """Validate the native backend before graph capture."""
        dim = self.wkv.in_features
        if self.width not in (128, 512) or not 1024 <= dim <= 10240 or dim % 512:
            raise ValueError("AscendC compressor requires D=128/512 and H=1024..10240 aligned to 512")
        if not get_current_hardware_profile().supports(HardwareCapability.DSV41_RING_COMPRESSOR):
            raise ValueError("AscendC V4.1 ring compression is currently supported on A2/A3 only")
        if not hasattr(torch.ops._C_ascend, "compressor_v2"):
            raise RuntimeError("Rebuild vllm-ascend custom ops to enable CompressorV2")

    def compress_native(self, x, metadata):
        """Map native compact groups back to the cache writer's token rows.

        Metadata stays on device, including for padded graph replay and batches
        with no completed groups. RMSNorm and RoPE remain outside the operator.
        """
        tokens = x.shape[0]
        if tokens == 0:
            return self.norm(self._ring_pooled[:0])
        ring = metadata.c2_ring_metadata
        # The native ABI requires the final offset to include graph padding;
        # seqused, not that final offset, determines which tokens update state.
        cu_seqlens = torch.cat((ring[3], ring.new_full((1,), tokens)))
        compact = torch.ops._C_ascend.compressor_v2(
            x.to(torch.bfloat16).contiguous(),
            self.wkv.weight,
            self.wgate.weight,
            self.state_cache.kv_cache[0].squeeze(-2),
            ring[4],
            cu_seqlens,
            ring[1],
            ring[0],
            self.ratio,
        )
        complete = metadata.c2_complete_mask[:tokens]
        rows = (complete.cumsum(0, dtype=torch.int64) - 1).clamp_min(0)
        pooled = self._ring_pooled[:tokens]
        torch.index_select(compact, 0, rows, out=pooled)
        # Unused compact rows are uninitialized, so mask rather than multiply.
        pooled.masked_fill_(~complete.unsqueeze(-1), 0)
        return self.norm(pooled)

    def forward(self, x):
        """Project an uncompressed source; C2 uses ``compress_native``."""
        return self.norm(self.wkv(x))
