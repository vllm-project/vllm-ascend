#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""IQuest-Q1 learned-key sink attention on Ascend NPU.

IQuest-Q1 adds a learned *sink key* ``sink_k`` of shape ``(num_kv_heads, head_dim)``
per layer.  It contributes an always-visible, zero-valued KV entry, so it only
rescales the softmax denominator:

    sink_logit = scale * (q . sink_k[head // num_queries_per_kv])
    out        = out_raw * 1 / (1 + exp(sink_logit - lse))

The CANN FIA operand ``learnable_sink`` only accepts a per-query-head *constant*
(bfloat16, shape ``(Q_N,)``), which cannot express the query-dependent
``q . sink_k`` term.  This module therefore takes the alternative route: ask
``npu_fused_infer_attention_score_v2`` for the real softmax LSE
(``return_softmax_lse=True``) and apply the sink correction with plain torch ops.

v0 scope (correctness first): eager only, no ACL graph capture, no ACL graph /
chunked-prefill phase split, no context parallelism.
"""

import torch
import torch_npu

from vllm.model_executor.layers.attention import Attention

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    SWA_INT_MAX,
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    attention_transfer_window,
)


class IQuestQ1AscendAttentionImpl(AscendAttentionBackendImpl):
    """Ascend attention with an iQuest learned sink key owned by the parent layer."""

    # The sink correction needs the real LSE, which the paged-attention path
    # cannot return; vLLM's own LSE-for-decode contract is not used here.
    can_return_lse_for_decode = False
    supports_quant_query_input = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sink_key: torch.Tensor | None = None

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The parent layer owns `sink_k` and its tensor-parallel weight loader.
        # Stash it here because `forward_impl` never sees `layer`.
        sink_key = getattr(layer, "sink_key", None)
        if sink_key is None:
            raise RuntimeError(
                "IQuest-Q1 attention layer is missing `sink_key`; "
                "the model must build it with enable_sink_attention=True."
            )
        self._sink_key = sink_key
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        # Always go through FIA: the paged-attention decode path cannot return
        # the softmax LSE that the sink correction needs.
        with attention_transfer_window():
            return self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        if _EXTRA_CTX.capturing:
            raise NotImplementedError(
                "IQuest-Q1 learned-key sink attention v0 does not support ACL graph "
                "capture. Please run with --enforce-eager."
            )

        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = key[:num_tokens]
            value = value[:num_tokens]
        if block_table is None:
            key = key.contiguous()
            value = value.contiguous()

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            actual_seq_qlen = torch.tensor(
                [1] * len(attn_metadata.seq_lens_list), dtype=torch.int32
            ).cumsum(dim=0)

        sparse_mode = 4 if self.sliding_window is not None else 3
        attn_output, lse = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key.contiguous(),
            value.contiguous(),
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=0,
            atten_mask=attn_metadata.attn_mask,
            sparse_mode=sparse_mode,
            softmax_scale=self.scale,
            block_table=block_table,
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_lengths_kv,
            return_softmax_lse=True,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        self._apply_sink_key(query, attn_output, lse, num_tokens)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _apply_sink_key(
        self,
        query: torch.Tensor,
        attn_output: torch.Tensor,
        lse: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Fold the learned sink into the attention output, in place.

        Equivalent to the reference Triton kernel: the sink is an always-visible
        entry with a zero value, so it only changes the softmax denominator.
        """
        sink_key = self._sink_key
        assert sink_key is not None
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        group = num_heads // num_kv_heads

        # Query heads are laid out kv-major: head h belongs to kv head h // group.
        q = query[:num_tokens].reshape(num_tokens, num_kv_heads, group, head_size)
        sink_logit = (
            q.float() * sink_key.float().unsqueeze(1)  # [KV, 1, D]
        ).sum(-1) * self.scale  # [T, KV, G]
        sink_logit = sink_logit.reshape(num_tokens, num_heads)  # [T, H]

        # TND LSE shape is (T, Q_N, 1); tolerate a (T, Q_N) 2D return as well.
        if lse.dim() == 3:
            lse = lse.reshape(num_tokens, num_heads, 1)
        else:
            lse = lse.reshape(num_tokens, num_heads).unsqueeze(-1)
        lse = lse.float()

        factor = torch.sigmoid(-(sink_logit.unsqueeze(-1) - lse))  # [T, H, 1]
        attn_output[:] = (attn_output.float() * factor).to(attn_output.dtype)


class IQuestQ1AscendAttentionBackend(AscendAttentionBackend):
    """Ascend GQA backend whose impl applies the iQuest learned sink key.

    Inherits ``get_name() == "CUSTOM"`` from the parent; vLLM resolves the
    name through ``AttentionBackendEnum``, so it must stay a registered member.
    """

    @staticmethod
    def get_impl_cls() -> type["IQuestQ1AscendAttentionImpl"]:
        return IQuestQ1AscendAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendAttentionMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        return False


class IQuestQ1AscendAttention(Attention):
    """Attention layer that forwards its learned sink key to the impl."""

    def __init__(self, *args, sink_key: torch.Tensor, **kwargs) -> None:
        kwargs["attn_backend"] = IQuestQ1AscendAttentionBackend
        super().__init__(*args, **kwargs)
        if self.head_size_v != self.head_size:
            raise NotImplementedError(
                "IQuest-Q1 learned sinks require equal query, key and value head sizes"
            )
        if tuple(sink_key.shape) != (self.num_kv_heads, self.head_size):
            raise ValueError(
                "IQuest-Q1 sink key must have shape "
                f"({self.num_kv_heads}, {self.head_size}), got {tuple(sink_key.shape)}"
            )
        # The parent module owns this parameter and its weight loader; keep it
        # out of this module's parameter/submodule registries.
        object.__setattr__(self, "sink_key", sink_key)
