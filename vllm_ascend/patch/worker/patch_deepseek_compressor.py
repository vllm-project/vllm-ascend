import torch
import vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.patch.platform.patch_kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.20.2"):
    from vllm.model_executor.layers import (
        deepseek_compressor,  # type:ignore
        deepseek_v4_attention,  # type:ignore
    )
    from vllm.model_executor.layers.deepseek_compressor import CompressorStateCache  # type:ignore
    from vllm.model_executor.layers.deepseek_v4_attention import DeepseekV4IndexerCache  # type:ignore
else:
    import vllm.models.deepseek_v4.attention as deepseek_v4_attention
    import vllm.models.deepseek_v4.compressor as deepseek_compressor
    from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
    from vllm.models.deepseek_v4.compressor import CompressorStateCache


class AscendCompressorStateCache(CompressorStateCache):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
    ):
        torch.nn.Module.__init__(self)
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        self.compress_ratio = compress_ratio
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        self.block_size = block_size

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        page_size_padded = 16640 if self.state_dim == 2 * 256 and self.compress_ratio == 4 else 131072
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,  # NOTE: FlashMLA requires 576B alignment
            page_size_padded=page_size_padded,
        )

    def forward(self): ...

    def get_attn_backend(self):
        return AscendDSABackend


class AscendDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__(head_dim, dtype, prefix, cache_config, compress_ratio)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendMLAAttentionSpec(  # Only has one vector instead of K + V
            block_size=128,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
            scale_dtype=torch.float16,
        )

    def forward(self): ...

    def get_attn_backend(self):
        return AscendDSABackend


class AscendDeepseekV4SWACache(DeepseekV4SWACache):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__(head_dim, window_size, torch.uint8, prefix, cache_config)
        self.dtype = dtype

        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(cmq): make SWA block size automatically determined and configurable.
        self.block_size = 128

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # TODO(cmq): alignment = 0 if A3 else 128
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="deepseek_v4",
            alignment=None,  # NOTE: FlashMLA requires 576B alignment
        )

    def forward(self): ...

    def get_attn_backend(self):
        return AscendDSABackend


deepseek_compressor.CompressorStateCache = AscendCompressorStateCache
deepseek_v4_attention.DeepseekV4IndexerCache = AscendDeepseekV4IndexerCache
vllm.v1.attention.backends.mla.sparse_swa.DeepseekV4SWACache = AscendDeepseekV4SWACache
