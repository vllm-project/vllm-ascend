#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from vllm_ascend.platform import ModelConfig
from vllm_ascend.utils import singleton


def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    mask_value = float("-inf") if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(mask_flag, mask_value)
    return attn_mask


@singleton
class AttentionMaskBuilder:
    # Rounded side of the cached encoder band mask (see
    # ``get_encoder_band_mask``): the square mask is rebuilt only when a batch
    # outgrows it, so a slowly growing batch does not pay the ``n ** 2`` index
    # difference on every step.
    ENCODER_BAND_MASK_ROUND = 1024

    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.chunked_prefill_attn_mask = None
        self.encoder_band_mask = None
        self.encoder_band_mask_key: tuple[int, torch.device] | None = None

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        assert self.attn_mask_cache is not None, "Something is wrong in generate_attn_mask."
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous().to(self.device, non_blocking=True)

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        if self.chunked_prefill_attn_mask is None:
            self.chunked_prefill_attn_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        return self.chunked_prefill_attn_mask

    def get_encoder_band_mask(self, num_tokens: int, sliding_window: int, device: torch.device) -> torch.Tensor:
        """Boolean ``[num_tokens, num_tokens]`` mask, ``True`` where blocked.

        Encoder-only ``sliding_attention`` layers attend to a local band only:
        position ``i`` sees ``j`` when ``abs(i - j) <= sliding_window - 1`` (the
        window boundary is inclusive, so the band is ``2 * sliding_window - 1``
        tokens wide). The mask only depends on ``(sliding_window, device)``, so
        one square is cached and sliced per batch; rebuilding the ``n ** 2``
        index difference every step measured ~2.4 ms at ``n = 6000`` on 910B2.
        """
        key = (sliding_window, device)
        mask = self.encoder_band_mask
        if mask is None or self.encoder_band_mask_key != key or mask.shape[0] < num_tokens:
            size = -(-num_tokens // self.ENCODER_BAND_MASK_ROUND) * self.ENCODER_BAND_MASK_ROUND
            index = torch.arange(size, dtype=torch.int32, device=device)
            mask = (index[:, None] - index[None, :]).abs() >= sliding_window
            self.encoder_band_mask = mask
            self.encoder_band_mask_key = key
        if mask.shape[0] == num_tokens:
            return mask
        return mask[:num_tokens, :num_tokens].contiguous()

    def get_attention_mask(self, causal: bool, model_config: ModelConfig):
        if not causal:
            # FIA applies any provided mask as defaultMask (sparse_mode=0),
            # which would wrongly mask out the upper triangle for
            # bidirectional attention, so non-causal attention must not
            # carry a mask here. The 310P mask builder overrides this
            # because its attention operators require an explicit
            # non-masking mask instead.
            # The one exception is the band a sliding-attention encoder layer
            # needs (``get_encoder_band_mask``): it is applied directly in
            # ``AscendAttentionBackendImpl._forward_encoder_attention`` rather
            # than plumbed
            # through here, because it is per-model window state and not a
            # property of the batch.
            return None

        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()
