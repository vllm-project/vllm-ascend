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
from vllm.distributed import get_pcp_group

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
    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.mla_mask = None
        self.chunked_prefill_attn_mask = None
        self.pcp_mla_mask = None

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

    def get_rswa_attn_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        max_kv_len: int,
        max_query_len: int,
        rswa_window: int,
    ) -> torch.Tensor:
        if not isinstance(rswa_window, int) or isinstance(rswa_window, bool) or rswa_window <= 0:
            raise ValueError(f"rswa_window must be a positive integer, got {rswa_window!r}")
        if not isinstance(max_kv_len, int) or max_kv_len <= 0:
            raise ValueError(f"max_kv_len must be a positive integer, got {max_kv_len!r}")
        if not isinstance(max_query_len, int) or max_query_len <= 0:
            raise ValueError(f"max_query_len must be a positive integer, got {max_query_len!r}")

        prefix_lens = prefix_lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(-1)
        seq_lens = seq_lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(-1)
        query_lens = query_lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(-1)
        num_reqs = prefix_lens.numel()
        if seq_lens.numel() != num_reqs or query_lens.numel() != num_reqs:
            raise ValueError(
                "R-SWA metadata tensors must have the same number of requests: "
                f"prefix={num_reqs}, seq={seq_lens.numel()}, query={query_lens.numel()}"
            )

        # An int8 mask still becomes very large for long-context batches. Fail
        # before allocating a multi-gigabyte temporary instead of surfacing an
        # opaque device OOM from one of the arange/broadcast operations.
        mask_elements = num_reqs * max_query_len * max_kv_len
        if mask_elements > (1 << 29):
            raise ValueError(
                f"R-SWA mask is too large: {num_reqs}*{max_query_len}*{max_kv_len}={mask_elements} elements"
            )

        # Keep validation device-side so V2 prompt lengths do not trigger a
        # synchronous GPU/NPU-to-host copy on the forward path.
        torch._assert(torch.all(prefix_lens >= 0), "R-SWA prefix_lens must be non-negative")
        torch._assert(torch.all(seq_lens >= 0), "R-SWA seq_lens must be non-negative")
        torch._assert(torch.all(query_lens >= 0), "R-SWA query_lens must be non-negative")
        torch._assert(torch.all(prefix_lens <= seq_lens), "R-SWA prefix_lens must be <= seq_lens")
        torch._assert(torch.all(query_lens <= seq_lens), "R-SWA query_lens must be <= seq_lens")
        torch._assert(torch.all(seq_lens <= max_kv_len), "R-SWA seq_lens exceeds max_kv_len")
        torch._assert(torch.all(query_lens <= max_query_len), "R-SWA query_lens exceeds max_query_len")

        if num_reqs == 0:
            return torch.ones(
                (0, 1, max_query_len, max_kv_len),
                dtype=torch.int8,
                device=self.device,
            )

        prefix_lens = prefix_lens.view(-1, 1, 1)
        seq_lens = seq_lens.view(-1, 1, 1)
        query_lens = query_lens.view(-1, 1, 1)

        query_offsets = torch.arange(
            max_query_len,
            device=self.device,
            dtype=torch.int64,
        ).view(1, -1, 1)
        kv_positions = torch.arange(
            max_kv_len,
            device=self.device,
            dtype=torch.int64,
        ).view(1, 1, -1)
        query_positions = seq_lens - query_lens + query_offsets

        valid_queries = query_offsets < query_lens
        valid_keys = kv_positions < seq_lens
        causal = kv_positions <= query_positions
        in_prefix = kv_positions < prefix_lens
        in_window = (query_positions - kv_positions) < rswa_window
        keep = valid_queries & valid_keys & causal & (in_prefix | in_window)

        return (~keep).to(torch.int8).unsqueeze(1).contiguous()

    def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        if self.mla_mask is None or self.mla_mask.dtype != dtype:
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            prefill_mask = torch.triu(torch.ones(512, 512, device=self.device, dtype=dtype), 1)
            self.mla_mask = torch.where(prefill_mask == 1, mask_value, 0).to(dtype)
        return self.mla_mask

    def get_pcp_mla_mask(self, dtype: torch.dtype):
        if self.pcp_mla_mask is None or self.pcp_mla_mask.dtype != dtype:
            self.pcp_mla_mask = torch.triu(torch.ones(512, 512, device=self.device, dtype=dtype), 1)
        return self.pcp_mla_mask

    def get_attention_mask(self, causal: bool, model_config: ModelConfig):
        if not causal:
            # FIA applies any provided mask as defaultMask (sparse_mode=0),
            # which would wrongly mask out the upper triangle for
            # bidirectional attention, so non-causal attention must not
            # carry a mask here. The 310P mask builder overrides this
            # because its attention operators require an explicit
            # non-masking mask instead.
            return None

        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()

    def get_final_mla_mask(self, model_config: ModelConfig):
        if get_pcp_group().world_size > 1:
            return self.get_pcp_mla_mask(model_config.dtype)
        # Prefill stages use 512x512 mask with appropriate dtype
        return self.get_mla_mask(model_config.dtype)
