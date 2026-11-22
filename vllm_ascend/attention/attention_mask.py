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

from vllm_ascend.attention.tree_spec import (
    align_up,
    dummy_tree_visibility,
    is_tree_verification_capture_shape,
    need_dummy_tree_visibility_for_capture,
    seq_lens_are_tree_query,
    tree_mask_query_capacity,
)
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


def _tree_spec_mask_caps() -> tuple[int, int, int] | None:
    """Max (num_decode, query_len, kv_len) when tree spec is enabled."""
    query_len = tree_mask_query_capacity()
    if query_len is None:
        return None
    max_num_seqs = 1
    max_model_len = query_len
    try:
        from vllm.config import get_current_vllm_config

        vcfg = get_current_vllm_config()
        max_num_seqs = int(vcfg.scheduler_config.max_num_seqs)
        max_model_len = int(vcfg.model_config.max_model_len)
    except (RuntimeError, AssertionError):
        pass
    return (
        max_num_seqs,
        query_len,
        align_up(max_model_len, 128),
    )


@singleton
class AttentionMaskBuilder:
    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.chunked_prefill_attn_mask = None
        # Own the persistent storage used by both the 4D mask and its BSND
        # view. Captured FIA parameters keep weak references to that storage.
        self._tree_attn_mask: torch.Tensor | None = None
        self._tree_spec_caps: tuple[int, int, int] | None = None
        self._tree_mask_cache_key: tuple | None = None

    def configure_tree_mask(self, max_num_decode: int, query_len: int, kv_len: int) -> None:
        """Pin max mask shape from VllmConfig (capture does not always have it)."""
        self._tree_spec_caps = (max_num_decode, query_len, kv_len)

    def as_bsnd_fia_mask(self, batch_size: int, q_len: int) -> torch.Tensor | None:
        """View the prepared [B, 1, Q, KV] storage as an FIA [B, Q, KV] mask."""
        mask = self._tree_attn_mask
        if mask is None or mask.shape[0] < batch_size or mask.shape[2] < q_len:
            return None
        return mask[:batch_size, 0, :q_len]

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

    def get_attention_mask(
        self,
        causal: bool,
        model_config: ModelConfig,
        tree_visibility: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        num_decode: int = 0,
        num_tokens: int = 0,
        for_capture: bool = False,
        common_attn_metadata=None,
    ):
        if not causal:
            # FIA applies any provided mask as defaultMask (sparse_mode=0),
            # which would wrongly mask out the upper triangle for
            # bidirectional attention, so non-causal attention must not
            # carry a mask here. The 310P mask builder overrides this
            # because its attention operators require an explicit
            # non-masking mask instead.
            return None

        if num_decode > 0:
            tree_shaped = num_tokens > 0 and is_tree_verification_capture_shape(num_tokens, num_decode)
            if tree_visibility is None and (
                tree_shaped
                if for_capture
                else (need_dummy_tree_visibility_for_capture() and seq_lens_are_tree_query(seq_lens, num_decode))
            ):
                tree_visibility = dummy_tree_visibility(num_decode, self.device)
            if tree_visibility is not None:
                # Dummy FULL gears that are not k × (1+budget) must keep the
                # 2D splitfuse mask. TND + sparse 3 rejects a 4D tree mask.
                if for_capture and not tree_shaped:
                    return self.get_splitfuse_attn_mask()
                return self.get_tree_attention_mask(
                    tree_visibility,
                    seq_lens,
                    num_decode,
                    for_capture=for_capture,
                    common_attn_metadata=common_attn_metadata,
                )

        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()

    def get_tree_attention_mask(
        self,
        tree_visibility: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decode: int,
        *,
        for_capture: bool,
        common_attn_metadata,
    ) -> torch.Tensor:
        max_nodes = tree_visibility.shape[-1]
        query_len = 1 + max_nodes
        max_caps = self._tree_spec_caps or _tree_spec_mask_caps()
        if max_caps is not None:
            query_len = max(query_len, max_caps[1])
        num_tree_mask_reqs = min(num_decode, tree_visibility.shape[0], seq_lens.shape[0])
        # seq_lens is host metadata (CPU). Caps already cover max_model_len.
        if max_caps is not None:
            alloc_b = max(num_decode, max_caps[0])
            alloc_kv = max_caps[2]
        else:
            alloc_b = num_decode
            alloc_kv = align_up(int(seq_lens[:num_tree_mask_reqs].max()), 128)
        # Allocate to the configured capture caps so updates can fill the same
        # storage before replay, including rows beyond the actual batch.
        mask = self._tree_attn_mask
        if mask is None or alloc_b > mask.shape[0] or query_len > mask.shape[2] or alloc_kv > mask.shape[3]:
            mask = torch.empty(alloc_b, 1, query_len, alloc_kv, dtype=torch.bool, device=self.device)
            self._tree_attn_mask = mask
            self._tree_mask_cache_key = None
        attn_mask = mask[:num_decode, :, :query_len]
        qsl = common_attn_metadata.query_start_loc
        cached_key = self._tree_mask_cache_key
        if (
            cached_key is not None
            and cached_key[0] is tree_visibility
            and cached_key[1] is qsl
            and cached_key[2] == num_decode
            and cached_key[3] == query_len
            and cached_key[4] == for_capture
            and cached_key[5] == seq_lens.data_ptr()
        ):
            return attn_mask
        if num_tree_mask_reqs:
            from vllm_ascend.ops.triton.spec_decode.tree.attention_mask import fill_tree_attention_mask_triton

            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            # The uniform graph gear uses the configured width. For a
            # truncated verification, subtract the input rows actually
            # submitted for this request to find its committed prefix.
            if (
                common_attn_metadata.max_query_len <= query_len
                and int(qsl_cpu[num_tree_mask_reqs]) == num_tree_mask_reqs * query_len
            ):
                root_kv_positions = seq_lens[:num_tree_mask_reqs] - query_len
            else:
                boundaries = qsl_cpu if seq_lens.device.type == "cpu" else qsl
                root_kv_positions = seq_lens[:num_tree_mask_reqs] - boundaries[: num_tree_mask_reqs + 1].diff()
            root_kv_positions = root_kv_positions.to(device=self.device, dtype=torch.int32, non_blocking=True)
            fill_tree_attention_mask_triton(
                attn_mask[:num_tree_mask_reqs], tree_visibility[:num_tree_mask_reqs], root_kv_positions
            )
        if mask.shape[0] > num_tree_mask_reqs:
            mask[num_tree_mask_reqs:, :, :query_len].fill_(True)
        self._tree_mask_cache_key = (
            tree_visibility,
            qsl,
            num_decode,
            query_len,
            for_capture,
            seq_lens.data_ptr(),
        )
        return attn_mask
