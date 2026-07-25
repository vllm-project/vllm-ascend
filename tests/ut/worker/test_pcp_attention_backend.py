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

from types import SimpleNamespace

import torch

from vllm_ascend.worker.pcp_attention_backend import (
    PCPMetadataBuildContext,
    select_pcp_attention_backend,
)
from vllm_ascend.worker.pcp_metadata import (
    DUAL_CHUNK_PCP_ATTENTION_BACKEND,
    build_context_parallel_metadata,
)


def _make_context(
    query_lens: list[int],
    *,
    num_decode_reqs: int = 0,
    pcp_world_size: int = 2,
    pcp_world_rank: int = 0,
) -> PCPMetadataBuildContext:
    return PCPMetadataBuildContext(
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        num_decode_reqs=num_decode_reqs,
        pcp_world_size=pcp_world_size,
        pcp_world_rank=pcp_world_rank,
        device="cpu",
        model_config=SimpleNamespace(use_mla=False),
        use_mla=False,
        pcp_use_hybrid_attn=False,
    )


def test_select_pcp_attention_backend_returns_default_dual_chunk_backend():
    assert select_pcp_attention_backend(SimpleNamespace()) is DUAL_CHUNK_PCP_ATTENTION_BACKEND


def test_dual_chunk_backend_builds_expected_indices():
    metadata = DUAL_CHUNK_PCP_ATTENTION_BACKEND.build_metadata(
        _make_context([8, 12], pcp_world_size=2, pcp_world_rank=1)
    )

    assert metadata.q_head_idx_tensor.tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]
    assert metadata.q_tail_idx_tensor.tolist() == [4, 5, 6, 7, 14, 15, 16, 17, 18, 19]
    assert metadata.q_full_idx.tolist() == [0, 1, 2, 3, 10, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19]
    assert metadata.kv_idx_names["kv_with_q_head_nomask_idx_tensor"].tolist() == [0, 1, 2, 3, 16, 17, 18, 19, 20, 21]
    assert metadata.kv_idx_names["kv_with_q_tail_mask_idx_tensor"].tolist() == [8, 9, 10, 11, 28, 29, 30, 31, 32, 33]
    assert metadata.kv_idx_names["kv_tail_proj_idx_tensor"].tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    ]
    assert metadata.extra_long_seq_kwargs["attn_mask_seqlens"] == [4, 10]
    assert metadata.extra_long_seq_kwargs["head_attn_nomask_seqlens"] == [4, 10]
    assert metadata.extra_long_seq_kwargs["tail_attn_nomask_seqlens"] == [8, 20]
    assert metadata.extra_long_seq_kwargs["head_actual_seq_lengths_kv"] == [8, 20]
    assert metadata.extra_long_seq_kwargs["tail_actual_seq_lengths_kv"] == [12, 30]
    assert metadata.attn_chunk_seqlens.tolist() == [4, 6]


def test_dual_chunk_backend_skips_decode_requests_without_offsets():
    metadata = DUAL_CHUNK_PCP_ATTENTION_BACKEND.build_metadata(
        _make_context([1, 8], num_decode_reqs=1, pcp_world_size=2, pcp_world_rank=0)
    )

    assert metadata.q_head_idx_tensor.tolist() == [0, 1, 2, 3]
    assert metadata.q_tail_idx_tensor.tolist() == [4, 5, 6, 7]
    assert metadata.kv_idx_names["kv_with_q_head_mask_idx_tensor"].tolist() == [0, 1, 2, 3]
    assert metadata.kv_idx_names["kv_with_q_tail_mask_idx_tensor"].tolist() == [12, 13, 14, 15]


def test_dual_chunk_backend_applies_metadata_to_long_seq_metadata():
    long_seq_metadata = build_context_parallel_metadata(
        pcp_use_hybrid_attn=False,
        num_actual_tokens_pcp_padded=16,
        num_computed_tokens_of_pcp_dcp=[[[0]], [[0]]],
        pcp_unpad_mask=torch.ones(16, dtype=torch.bool),
        pcp_padded_tokens_fla=0,
        query_lens_pcp_full_cpu=torch.tensor([8], dtype=torch.int32),
    )
    metadata = DUAL_CHUNK_PCP_ATTENTION_BACKEND.build_metadata(_make_context([8]))

    DUAL_CHUNK_PCP_ATTENTION_BACKEND.apply_metadata(long_seq_metadata, metadata)

    assert torch.equal(long_seq_metadata.q_head_idx_tensor, metadata.q_head_idx_tensor)
    assert torch.equal(long_seq_metadata.q_tail_idx_tensor, metadata.q_tail_idx_tensor)
    assert torch.equal(
        long_seq_metadata.kv_with_q_tail_mask_idx_tensor,
        metadata.kv_idx_names["kv_with_q_tail_mask_idx_tensor"],
    )
    assert long_seq_metadata.attn_mask_seqlens == metadata.extra_long_seq_kwargs["attn_mask_seqlens"]
    assert torch.equal(long_seq_metadata.attn_chunk_seqlens, metadata.attn_chunk_seqlens)
