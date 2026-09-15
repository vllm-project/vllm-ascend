"""Fix MLA rotary cache for models where is_deepseek_mla() is False.

LongCat uses DeepSeek MLA but model_type="longcat" is not in the
is_deepseek_mla() list, so _cos_mla/_sin_mla caches are never initialized.

Migrated from EasyInfer easyinfer/plugins/vllm_ascend/fix_mla_rotary.py.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_rotary_mod: Any = None
_MIN_BUFFER_TOKENS = 8192


def _ensure_mla_caches(rotary_mod: Any, num_tokens: int) -> None:
    """Allocate _cos_mla / _sin_mla if missing, grow if too small."""
    cos_buf = rotary_mod._cos_mla
    sin_buf = rotary_mod._sin_mla
    if cos_buf is not None and sin_buf is not None and cos_buf.size(0) >= num_tokens:
        return

    cos0 = rotary_mod._cos_cache[:1].unsqueeze(1).unsqueeze(2)
    rope_dim = cos0.shape[-1]
    new_size = max(num_tokens, _MIN_BUFFER_TOKENS)
    if cos_buf is not None:
        new_size = max(new_size, cos_buf.size(0) * 2)

    rotary_mod._cos_mla = cos0.new_ones(new_size, 1, 1, rope_dim)
    rotary_mod._sin_mla = cos0.new_zeros(new_size, 1, 1, rope_dim)
    logger.info(
        "[fix_mla_rotary] MLA cos/sin buffers (re)allocated "
        "(rope_dim=%d, tokens=%d)", rope_dim, new_size,
    )


def patch() -> None:
    """Apply MLA rotary fix (call before vllm serve starts)."""
    global _rotary_mod

    import vllm_ascend.ops.rotary_embedding as _mod
    _rotary_mod = _mod

    original = _mod.get_cos_and_sin_mla

    def patched(positions, use_cache=False):
        if use_cache:
            _ensure_mla_caches(_mod, positions.size(0))
        return original(positions, use_cache)

    _mod.get_cos_and_sin_mla = patched
    logger.info("[fix_mla_rotary] Patched rotary_embedding.get_cos_and_sin_mla")

    # Rebind callers that imported the function before the patch
    for caller_name in (
        "vllm_ascend.attention.mla_v1",
        "vllm_ascend.attention.sfa_v1",
    ):
        import sys
        caller = sys.modules.get(caller_name)
        if caller is not None:
            caller.get_cos_and_sin_mla = patched
            logger.info("[fix_mla_rotary] Rebound %s.get_cos_and_sin_mla", caller_name)
