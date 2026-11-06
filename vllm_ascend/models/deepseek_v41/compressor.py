# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 C2 ring compressor, ratio-1 path, and fused RMS normalization."""

import os

import torch
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS, uses_a5_packed_cache


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, ratio, vllm_config=None, prefix="compressor"):
        super().__init__()
        self.ratio = ratio
        self.width = config.head_dim
        dim = config.hidden_size
        # Opt in on every worker before model construction; never toggle after capture.
        self._fuse_c2_proj = ratio == 2 and os.getenv("VLLM_ASCEND_FUSE_V41_C2_PROJ", "0") == "1"
        if self._fuse_c2_proj:
            self.wkv_gate = nn.Linear(dim, 2 * self.width, bias=False, dtype=torch.float32)
        else:
            self.wkv = nn.Linear(dim, self.width, bias=False, dtype=torch.float32 if ratio == 2 else torch.bfloat16)
        self.norm = RMSNorm(self.width, eps=config.rms_norm_eps, dtype=torch.bfloat16)
        if ratio == 2:
            if not self._fuse_c2_proj:
                self.wgate = nn.Linear(dim, self.width, bias=False, dtype=torch.float32)
            projection_weight = self.wkv_gate.weight if self._fuse_c2_proj else self.wkv.weight
            # Allocate persistent output before memory profiling, so its footprint
            # is included in the cache budget rather than added after allocation.
            if vllm_config is not None:
                capacity = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
                self.register_buffer(
                    "_ring_pooled",
                    torch.empty(capacity, self.width, dtype=torch.bfloat16, device=projection_weight.device),
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

    def project_c2(self, hidden_states_fp32):
        """Project C2 using one merged GEMM or the original two GEMMs."""
        # getattr also supports reference fixtures constructed with __new__.
        if not getattr(self, "_fuse_c2_proj", False):
            return self.wkv(hidden_states_fp32), self.wgate(hidden_states_fp32)
        projected = self.wkv_gate(hidden_states_fp32)
        kv, score = projected.split(self.width, dim=-1)
        # The existing Triton pool and ring-update kernels require row stride H.
        return kv.contiguous(), score.contiguous()

    def prepare_ring_compressor(self, max_tokens, device):
        """Resolve ring-compressor hardware before capture."""
        from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num

        self._ring_num_cores = _cube_core_num()

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
        """Project an uncompressed source; ratio-2 uses ``pool_projected``."""
        if self._fuse_c2_proj:
            raise RuntimeError("Fused C2 requires project_c2() followed by pool_projected()")
        return self.norm(self.wkv(x))
