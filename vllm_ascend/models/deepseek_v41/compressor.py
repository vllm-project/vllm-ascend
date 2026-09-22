# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2 ring compression with an opt-in fused AscendC backend."""

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
        # Native fusion uses BF16 projections; the reference keeps FP32 weights.
        # Keep the precision change opt-in until model-level accuracy is qualified.
        self.use_ascendc = ratio == 2 and bool(
            vllm_config is not None
            and getattr(vllm_config, "additional_config", {}).get("use_ascendc_compressor", False)
        )
        if self.use_ascendc and (self.width not in (128, 512) or not 1024 <= dim <= 10240 or dim % 512):
            raise ValueError("AscendC compressor requires D=128/512 and H=1024..10240 aligned to 512")
        if self.use_ascendc and not get_current_hardware_profile().supports(HardwareCapability.DSV41_RING_COMPRESSOR):
            raise ValueError("AscendC V4.1 ring compression is currently supported on A2/A3 only")
        projection_dtype = torch.bfloat16 if self.use_ascendc or ratio == 1 else torch.float32
        self.wkv = nn.Linear(dim, self.width, bias=False, dtype=projection_dtype)
        self.norm = RMSNorm(self.width, eps=config.rms_norm_eps, dtype=torch.bfloat16)
        if ratio == 2:
            self.wgate = nn.Linear(dim, self.width, bias=False, dtype=projection_dtype)
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

    def prepare_ring_compressor(self, max_tokens, device):
        """Resolve ring-compressor hardware before capture."""
        if self.use_ascendc:
            if not hasattr(torch.ops._C_ascend, "compressor_v2"):
                raise RuntimeError("Rebuild vllm-ascend custom ops to enable CompressorV2")
            return
        from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num

        self._ring_num_cores = _cube_core_num()

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

    def pool_projected(self, kv, scores, metadata):
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
        return self.norm(pooled)

    def forward(self, x):
        """Project an uncompressed source; C2 selects a ring backend explicitly."""
        return self.norm(self.wkv(x))
