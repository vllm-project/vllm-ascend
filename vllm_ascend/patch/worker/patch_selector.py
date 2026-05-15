from typing import NamedTuple, cast, get_args

import torch
import vllm
from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.attention.selector import _cached_get_attn_backend
from vllm.config.cache import CacheDType
from vllm.logger import init_logger

logger = init_logger(__name__)


class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_compress: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    attn_type: str = AttentionType.DECODER

    def __repr__(self):
        return (f"AttentionSelectorConfig(head_size={self.head_size}, "
                f"dtype={self.dtype}, "
                f"kv_cache_dtype={self.kv_cache_dtype}, "
                f"block_size={self.block_size}, "
                f"use_mla={self.use_mla}, "
                f"has_sink={self.has_sink}, "
                f"use_compress={self.use_compress}, "
                f"use_sparse={self.use_sparse}, "
                f"use_mm_prefix={self.use_mm_prefix}, "
                f"attn_type={self.attn_type})")


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_compress: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    attn_type: str | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}")

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    backend_enum = vllm_config.attention_config.backend

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_compress=use_compress,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        attn_type=attn_type or AttentionType.DECODER,
    )

    return _cached_get_attn_backend(
        backend=backend_enum,
        attn_selector_config=attn_selector_config,
    )


vllm.attention.selector.AttentionSelectorConfig = AttentionSelectorConfig
vllm.attention.selector.get_attn_backend = get_attn_backend
