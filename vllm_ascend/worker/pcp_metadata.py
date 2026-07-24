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

import torch

from vllm_ascend.attention.utils import AscendPrefillContextParallelMetadata
from vllm_ascend.worker.pcp_attention_backend import (
    PCPAttentionBackend,
    PCPBackendMetadata,
    PCPMetadataBuildContext,
)


def build_context_parallel_metadata(
    *,
    pcp_use_hybrid_attn: bool,
    num_actual_tokens_pcp_padded: int,
    num_computed_tokens_of_pcp_dcp,
    pcp_unpad_mask: torch.Tensor,
    pcp_padded_tokens_fla: int,
    query_lens_pcp_full_cpu: torch.Tensor,
) -> AscendPrefillContextParallelMetadata:
    return AscendPrefillContextParallelMetadata(
        pcp_use_hybrid_attn=pcp_use_hybrid_attn,
        num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
        num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp,
        pcp_unpad_mask=pcp_unpad_mask,
        pcp_padded_tokens_fla=pcp_padded_tokens_fla,
        query_lens_pcp_full_cpu=query_lens_pcp_full_cpu,
        max_query_len_pcp_full=query_lens_pcp_full_cpu.max().item(),
    )


def apply_dual_chunk_attention_metadata(
    long_seq_metadata: AscendPrefillContextParallelMetadata,
    backend_metadata: PCPBackendMetadata,
) -> None:
    long_seq_metadata.q_head_idx_tensor = backend_metadata.q_head_idx_tensor
    long_seq_metadata.q_tail_idx_tensor = backend_metadata.q_tail_idx_tensor
    long_seq_metadata.q_full_idx = backend_metadata.q_full_idx
    long_seq_metadata.kv_with_q_head_nomask_idx_tensor = backend_metadata.kv_idx_names[
        "kv_with_q_head_nomask_idx_tensor"
    ]
    long_seq_metadata.kv_with_q_head_mask_idx_tensor = backend_metadata.kv_idx_names["kv_with_q_head_mask_idx_tensor"]
    long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = backend_metadata.kv_idx_names[
        "kv_with_q_tail_nomask_idx_tensor"
    ]
    long_seq_metadata.kv_with_q_tail_mask_idx_tensor = backend_metadata.kv_idx_names["kv_with_q_tail_mask_idx_tensor"]
    long_seq_metadata.kv_tail_proj_idx_tensor = backend_metadata.kv_idx_names["kv_tail_proj_idx_tensor"]
    long_seq_metadata.kv_with_q_head_attn_idx_in_tail_tensor = backend_metadata.kv_idx_names[
        "kv_with_q_head_attn_idx_in_tail_tensor"
    ]
    long_seq_metadata.kv_with_q_tail_attn_idx_in_tail_tensor = backend_metadata.kv_idx_names[
        "kv_with_q_tail_attn_idx_in_tail_tensor"
    ]
    long_seq_metadata.attn_mask_seqlens = backend_metadata.extra_long_seq_kwargs["attn_mask_seqlens"]
    long_seq_metadata.head_attn_nomask_seqlens = backend_metadata.extra_long_seq_kwargs["head_attn_nomask_seqlens"]
    long_seq_metadata.tail_attn_nomask_seqlens = backend_metadata.extra_long_seq_kwargs["tail_attn_nomask_seqlens"]
    long_seq_metadata.head_actual_seq_lengths_kv = backend_metadata.extra_long_seq_kwargs["head_actual_seq_lengths_kv"]
    long_seq_metadata.tail_actual_seq_lengths_kv = backend_metadata.extra_long_seq_kwargs["tail_actual_seq_lengths_kv"]
    long_seq_metadata.attn_chunk_seqlens = backend_metadata.attn_chunk_seqlens


def _list_to_tensor(lst: list[int], device, dtype=torch.int32):
    tensor = torch.zeros(len(lst), dtype=dtype, device=device)
    tensor.copy_(torch.tensor(lst, dtype=dtype), non_blocking=True)
    return tensor


def build_dual_chunk_attention_metadata(
    ctx: PCPMetadataBuildContext,
) -> PCPBackendMetadata:
    """Build q/kv index metadata shared by current PCP attention backends."""
    q_head_idx, q_tail_idx = [], []
    kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
    kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
    kv_tail_proj_idx: list[int] = []
    kv_with_q_head_attn_idx_in_tail, kv_with_q_tail_attn_idx_in_tail = [], []
    chunk_seqlens = []
    kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
    head_actual_seq_lengths_kv, tail_actual_seq_lengths_kv = [], []
    q_req_offset = 0
    kv_req_offset = 0
    q_head_chunk_id = ctx.pcp_world_rank
    q_tail_chunk_id = ctx.pcp_world_size * 2 - 1 - ctx.pcp_world_rank

    for i, seq_len_tensor in enumerate(ctx.query_lens):
        if i < ctx.num_decode_reqs:
            continue
        seq_len = int(seq_len_tensor.item())
        chunk_len = seq_len // 2
        chunk_seqlens.append(chunk_len)
        q_head_idx.extend(list(range(q_req_offset, q_req_offset + chunk_len)))
        kv_with_q_head_nomask_idx.extend(list(range(kv_req_offset, kv_req_offset + chunk_len * q_head_chunk_id)))
        kv_with_q_head_mask_idx.extend(
            list(
                range(
                    kv_req_offset + chunk_len * q_head_chunk_id,
                    kv_req_offset + chunk_len * (q_head_chunk_id + 1),
                )
            )
        )
        kv_with_q_head_nomask_seqlens.append(chunk_len * q_head_chunk_id)
        q_tail_idx.extend(list(range(q_req_offset + chunk_len, q_req_offset + chunk_len * 2)))
        kv_with_q_tail_nomask_idx.extend(list(range(kv_req_offset, kv_req_offset + chunk_len * q_tail_chunk_id)))
        kv_with_q_tail_mask_idx.extend(
            list(
                range(
                    kv_req_offset + chunk_len * q_tail_chunk_id,
                    kv_req_offset + chunk_len * (q_tail_chunk_id + 1),
                )
            )
        )
        kv_with_q_tail_nomask_seqlens.append(chunk_len * q_tail_chunk_id)
        tail_proj_offset = len(kv_tail_proj_idx)
        tail_proj_len = chunk_len * (q_tail_chunk_id + 1)
        kv_tail_proj_idx.extend(list(range(kv_req_offset, kv_req_offset + tail_proj_len)))
        kv_with_q_head_attn_idx_in_tail.extend(
            list(range(tail_proj_offset, tail_proj_offset + chunk_len * (q_head_chunk_id + 1)))
        )
        kv_with_q_tail_attn_idx_in_tail.extend(list(range(tail_proj_offset, tail_proj_offset + tail_proj_len)))
        head_actual_seq_lengths_kv.append(len(kv_with_q_head_attn_idx_in_tail))
        tail_actual_seq_lengths_kv.append(len(kv_with_q_tail_attn_idx_in_tail))
        q_req_offset += seq_len
        kv_req_offset += seq_len * ctx.pcp_world_size

    q_head_idx_tensor = _list_to_tensor(q_head_idx, ctx.device)
    q_tail_idx_tensor = _list_to_tensor(q_tail_idx, ctx.device)
    q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
    q_full_idx = q_full_idx.to(torch.float32).argsort().to(torch.int32)

    kv_idx_names = {
        "kv_with_q_head_nomask_idx_tensor": _list_to_tensor(kv_with_q_head_nomask_idx, ctx.device),
        "kv_with_q_head_mask_idx_tensor": _list_to_tensor(kv_with_q_head_mask_idx, ctx.device),
        "kv_with_q_tail_nomask_idx_tensor": _list_to_tensor(kv_with_q_tail_nomask_idx, ctx.device),
        "kv_with_q_tail_mask_idx_tensor": _list_to_tensor(kv_with_q_tail_mask_idx, ctx.device),
        "kv_tail_proj_idx_tensor": _list_to_tensor(kv_tail_proj_idx, ctx.device),
        "kv_with_q_head_attn_idx_in_tail_tensor": _list_to_tensor(kv_with_q_head_attn_idx_in_tail, ctx.device),
        "kv_with_q_tail_attn_idx_in_tail_tensor": _list_to_tensor(kv_with_q_tail_attn_idx_in_tail, ctx.device),
    }

    attn_chunk_seqlens = torch.tensor(chunk_seqlens, dtype=torch.int32)
    attn_mask_seqlens = torch.cumsum(torch.tensor(chunk_seqlens, dtype=torch.int32), dim=0).tolist()
    head_attn_nomask_seqlens = torch.cumsum(
        torch.tensor(kv_with_q_head_nomask_seqlens, dtype=torch.int32), dim=0
    ).tolist()
    tail_attn_nomask_seqlens = torch.cumsum(
        torch.tensor(kv_with_q_tail_nomask_seqlens, dtype=torch.int32), dim=0
    ).tolist()

    return PCPBackendMetadata(
        q_head_idx_tensor=q_head_idx_tensor,
        q_tail_idx_tensor=q_tail_idx_tensor,
        q_full_idx=q_full_idx,
        kv_idx_names=kv_idx_names,
        attn_chunk_seqlens=attn_chunk_seqlens,
        extra_long_seq_kwargs={
            "attn_mask_seqlens": attn_mask_seqlens,
            "head_attn_nomask_seqlens": head_attn_nomask_seqlens,
            "tail_attn_nomask_seqlens": tail_attn_nomask_seqlens,
            "head_actual_seq_lengths_kv": head_actual_seq_lengths_kv,
            "tail_actual_seq_lengths_kv": tail_actual_seq_lengths_kv,
        },
    )


class DualChunkPCPAttentionBackend:
    name = "dual_chunk"

    def build_metadata(self, ctx: PCPMetadataBuildContext) -> PCPBackendMetadata:
        return build_dual_chunk_attention_metadata(ctx)

    def apply_metadata(
        self,
        long_seq_metadata: AscendPrefillContextParallelMetadata,
        metadata: PCPBackendMetadata,
    ) -> None:
        apply_dual_chunk_attention_metadata(long_seq_metadata, metadata)


DUAL_CHUNK_PCP_ATTENTION_BACKEND: PCPAttentionBackend = DualChunkPCPAttentionBackend()
