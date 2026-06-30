# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Any

import torch


def _safe_bool_call(fn: Any) -> bool:
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if _safe_bool_call(getattr(compiler, "is_compiling", None)):
        return True

    dynamo = getattr(torch, "_dynamo", None)
    return _safe_bool_call(getattr(dynamo, "is_compiling", None))


def _patch_deepseek_v4_attention() -> None:
    try:
        from vllm.model_executor.layers import deepseek_v4_attention
    except ImportError:
        return

    attention_cls = getattr(deepseek_v4_attention, "DeepseekV4MultiHeadLatentAttentionWrapper", None)
    if attention_cls is None:
        return

    original_forward = attention_cls.forward
    if getattr(original_forward, "_vllm_ascend_deepseek_v4_attention_patched", False):
        return

    @wraps(original_forward)
    def patched_forward(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_npu_eager_fallback", False) and _is_compiling():
            out = torch.empty_like(hidden_states)
            torch.ops.vllm.deepseek_v4_attention(
                hidden_states,
                positions,
                out,
                self.layer_name,
            )
            return out
        return original_forward(self, positions, hidden_states, llama_4_scaling)

    patched_forward._vllm_ascend_deepseek_v4_attention_patched = True  # type: ignore[attr-defined]
    attention_cls.forward = patched_forward


_patch_deepseek_v4_attention()
