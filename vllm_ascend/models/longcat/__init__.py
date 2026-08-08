"""LongCat-Flash model patches for vLLM-Ascend.

Call apply() once before vllm serve starts to apply all fixes.
"""
from vllm_ascend.models.longcat.apply import apply

__all__ = ["apply"]
