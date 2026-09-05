"""Fix ``extract_layer_index`` for dual-attention models (LongCat-Flash).

Migrated from EasyInfer easyinfer/plugins/vllm_ascend/fix_dual_attention.py.
Stripped of EasyInfer registry dependency; patch() applies fixes directly.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _extract_layer_index_safe(prefix: str, num_attn_module: int = 1) -> int:
    """Like vllm's extract_layer_index but tolerates multi-integer prefixes."""
    int_vals = [int(p) for p in prefix.split(".") if p.lstrip("-").isdigit()]
    if num_attn_module == 1:
        if not int_vals:
            raise ValueError(f"No integer found in layer name: {prefix}")
        return int_vals[0]
    if len(int_vals) <= 2:
        return (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
    raise ValueError(f"layer name {prefix} should contain at most two integers")


_ORIG_INIT: Any = None


def _patch_extract_layer_index_globally() -> None:
    """Swap extract_layer_index to the prefix-tolerant version."""
    import vllm.model_executor.models.utils as _utils

    original = _utils.extract_layer_index
    if original is _extract_layer_index_safe:
        return
    _utils.extract_layer_index = _extract_layer_index_safe

    swapped = []
    for mod in list(sys.modules.values()):
        if mod is None or mod is _utils:
            continue
        mod_dict = getattr(mod, "__dict__", None)
        if not isinstance(mod_dict, dict):
            continue
        if mod_dict.get("extract_layer_index") is original:
            mod.extract_layer_index = _extract_layer_index_safe
            swapped.append(mod.__name__)
    logger.info(
        "[fix_dual_attention] extract_layer_index swapped: %s",
        ", ".join(swapped) if swapped else "(none yet imported)",
    )


def patch() -> None:
    """Apply dual-attention fix (call before vllm serve starts)."""
    import vllm_ascend.patch.worker.patch_deepseek_v2 as _pdv2

    global _ORIG_INIT
    if _ORIG_INIT is not None:
        return

    _ORIG_INIT = _pdv2._deepseek_v2_mla_attention_init
    _patch_extract_layer_index_globally()

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _save = _pdv2.extract_layer_index
        try:
            _pdv2.extract_layer_index = _extract_layer_index_safe
            return _ORIG_INIT(self, *args, **kwargs)
        finally:
            _pdv2.extract_layer_index = _save

    _pdv2._deepseek_v2_mla_attention_init = patched_init

    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
        DeepseekV2MLAAttention.__init__ = patched_init
    except ImportError:
        logger.warning(
            "[fix_dual_attention] could not import DeepseekV2MLAAttention"
        )

    logger.info("[fix_dual_attention] _deepseek_v2_mla_attention_init wrapped")
