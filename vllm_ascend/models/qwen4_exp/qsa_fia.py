# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Short-sequence QSA attention backed by ``npu_fused_infer_attention_score``.

For short sequences every micro-block selected by the Lightning Indexer covers
the whole context anyway: if the kv length of every request in the step is at
or below the indexer token budget ``K`` (r=4 pooled keys), the sparse core
degenerates into plain full attention.  In that regime we bypass the GBSA
sparse kernel and run the dense FlashAttention operator over the same paged
QSA main K/V cache:

* same K/V pages (block table gathered from QSA metadata per token),
* per-token visibility ``min(logical_position + 1, seq_len)`` reproduces the
  indexer causal rule without materializing an attention mask,
* the indexer still runs and still maintains the side caches (raw ring and
  compressed groups), so a request that later grows past the threshold falls
  back to GBSA seamlessly.

Switches:
* ``VLLM_ASCEND_QSA_SHORTSEQ_FIA=1`` (default) enable the branch,
* ``VLLM_ASCEND_QSA_SHORTSEQ_MAX_LEN=N`` override the threshold; ``0`` means
  auto, i.e. the indexer token budget (2048 for this model).
"""

import os
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.common.qsa_cache import QSAForwardMetadata

SHORTSEQ_FIA_ENV = "VLLM_ASCEND_QSA_SHORTSEQ_FIA"
SHORTSEQ_MAX_LEN_ENV = "VLLM_ASCEND_QSA_SHORTSEQ_MAX_LEN"

# Paged FIAS tiling rejects pages larger than this many tokens
# (768/1024-token pages fail with 561103 on 910B3; 512 and below pass).
FIA_MAX_PAGE = 512


def qsa_shortseq_fia_enabled() -> bool:
    """Read the short-seq dense-attention switch (default: enabled)."""
    return os.environ.get(SHORTSEQ_FIA_ENV, "1") not in ("0", "false", "False")


def qsa_shortseq_max_len(default_limit: int) -> int:
    """Short sequence bound in kv tokens (``0`` → auto: the indexer budget)."""
    try:
        value = int(os.environ.get(SHORTSEQ_MAX_LEN_ENV, "0"))
    except ValueError:
        value = 0
    return value if value > 0 else default_limit


def _assemble_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Normalize the QSA K/V cache to the paged form FIAS expects.

    Accepts ``[blocks, page, nkv, head]`` or ``[blocks, nkv, page, head]``
    (the GBSA path accepts both, so re-use the same reconciliation) and
    returns ``(key, value, block_size)`` with the page axis outermost and
    the head dims fused into one axis.
    """
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA FIA requires 4-D K/V caches")
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA FIA requires matching K/V cache shapes")
    if key_cache.shape[2] == num_kv_heads and key_cache.shape[1] != num_kv_heads:
        # already [blocks, page, nkv, head]
        pass
    elif key_cache.shape[1] == num_kv_heads and key_cache.shape[2] != num_kv_heads:
        # [blocks, nkv, page, head] -> [blocks, page, nkv, head]
        key_cache = key_cache.transpose(1, 2)
        value_cache = value_cache.transpose(1, 2)
    else:
        raise ValueError(
            "QSA FIA cannot locate the kv-head axis in cache "
            f"shape={tuple(key_cache.shape)}, num_kv_heads={num_kv_heads}"
        )
    key = key_cache.contiguous().view(key_cache.shape[0], key_cache.shape[1], -1)
    value = value_cache.contiguous().view(value_cache.shape[0], value_cache.shape[1], -1)
    page = key_cache.shape[1]
    if page > FIA_MAX_PAGE:
        # The paged FIAS tiling only accepts pages up to
        # FIA_MAX_PAGE tokens; the aligned QSA cache can carry larger
        # pages, so re-pack them into sub-pages. The sub-page size must
        # divide the big page exactly (e.g. 768 splits into 3x256), so
        # walk the supported sizes down from FIA_MAX_PAGE. The block
        # table is remapped in lockstep: one big page becomes ``sub``
        # consecutive small pages.
        fia_page = FIA_MAX_PAGE
        while page % fia_page:
            fia_page //= 2
        sub = page // fia_page
        key = (
            key.view(key.shape[0], sub, fia_page, -1)
            .reshape(key.shape[0] * sub, fia_page, -1)
            .contiguous()
        )
        value = (
            value.view(value.shape[0], sub, fia_page, -1)
            .reshape(value.shape[0] * sub, fia_page, -1)
            .contiguous()
        )
        return key, value, fia_page, page
    return key, value, page, page


def qsa_fia_shortseq_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: "QSAForwardMetadata",
    token_to_req: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    """Dense QSA attention for short sequences via FIAS (paged TND).

    ``query``:  ``[num_tokens, num_heads, head_dim]`` bf16.
    ``metadata``: the QSA main-attention metadata (block_table, seq_lens,
        num_actual_tokens).
    ``logical_positions``: per-token logical position from the QSA side-cache
        metadata (the main attention metadata does not carry it).
    ``token_to_req``: side-cache token→request index (same token count).
    ``output``:  ``[num_tokens, num_heads, head_dim]`` (token-major), written
        in place and returned.
    """
    num_tokens = metadata.num_actual_tokens
    token_to_req = token_to_req[:num_tokens]

    key, value, block_size, big_page = _assemble_paged_cache(
        key_cache, value_cache, num_kv_heads
    )

    # Paged access: one block-table row per query token (TND rows are
    # individual tokens, so request pages are gathered per token).
    block_table = metadata.block_table[token_to_req]
    if block_table.dtype not in (torch.int32, torch.int64):
        block_table = block_table.to(torch.int32)
    if big_page != block_size:
        # Remap big-page ids to consecutive FIA_MAX_PAGE sub-page ids:
        # big page b covers sub-pages [b*sub, (b+1)*sub).
        sub = big_page // block_size
        block_table = (
            block_table.unsqueeze(-1) * sub
            + torch.arange(sub, dtype=block_table.dtype, device=block_table.device)
        ).reshape(block_table.shape[0], -1)

    # One query token per sequence row.  In the TND layout the operator reads
    # the per-row kv length from ``actual_seq_lengths_kv`` (verified on 910B3:
    # the value is the row's own visible length, not a cumulative offset), so
    # visibility is the logical position clamped to the kv of the request.
    seq_lens = metadata.seq_lens.to(torch.int64)
    actual_seq_kvlen = torch.minimum(
        logical_positions[:num_tokens].to(torch.int64) + 1,
        seq_lens[token_to_req],
    ).to(torch.int32)
    actual_seq_qlen = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=query.device)

    num_heads = query.shape[1]
    logger.info_once("QSA short-sequence attention is using npu_fused_infer_attention_score (dense).")
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query[:num_tokens].contiguous(),
        key,
        value,
        block_table=block_table,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        sparse_mode=0,
    )
    dims = (num_heads, output.shape[2])
    output[:num_tokens].copy_(attn_output.view(num_tokens, *dims))
    return output
