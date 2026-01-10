from typing import Any, Callable, Optional

import torch
import vllm_ascend.attention.attention_mask as _base_mask


_BASE_BUILDER: Callable[[torch.device], Any] = _base_mask.AttentionMaskBuilder

def _gen_causal_additive_mask_fp16(max_seq_len: int, device: torch.device) -> torch.Tensor:
    tril = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device).tril_()
    upper = ~tril
    m = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
    m.masked_fill_(upper, float("-inf"))
    return m

class _AttentionMaskBuilder310P:
    def __init__(self, device: torch.device):
        self._base = _BASE_BUILDER(device)

        self._fp16_mask_cache: Optional[torch.Tensor] = None
        self._fp16_mask_cached_len: int = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def device(self) -> torch.device:
        return self._base.device
    
    def _get_fp16_mask(self, max_seq_len: int) -> torch.Tensor:
        if self._fp16_mask_cache is None or max_seq_len > self._fp16_mask_cached_len:
            self._fp16_mask_cache = _gen_causal_additive_mask_fp16(max_seq_len, self.device)
            self._fp16_mask_cached_len = max_seq_len
        assert self._fp16_mask_cache is not None
        return self._fp16_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if dtype == torch.float16:
            return self._get_fp16_mask(max_seq_len)
        return self._base.get_attn_mask(max_seq_len, dtype)

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        return self._get_fp16_mask(2048)
    
    def get_attention_mask(self, model_config) -> torch.Tensor:
        if getattr(model_config, "runner_type", None) == "pooling":
            return self._base.get_attn_mask(2048, torch.bool)
        return self.get_splitfuse_attn_mask()


def AttentionMaskBuilder(device: torch.device) -> _AttentionMaskBuilder310P:
    return _AttentionMaskBuilder310P(device)
