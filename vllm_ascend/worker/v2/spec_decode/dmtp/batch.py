# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Stable native-MTP inputs and device-only accepted-prefix bookkeeping."""

from dataclasses import dataclass

import torch

from vllm_ascend.worker.v2.spec_decode.dmtp.config import DMTP_PROPOSAL_COUNT
from vllm_ascend.worker.v2.spec_decode.dmtp.native_mtp import NATIVE_BLOCK_SIZE, NativeMTP


@dataclass
class NativeAttentionMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    scheduler_metadata: torch.Tensor


class NativeMTPBatch:
    def __init__(self, native: NativeMTP, max_reqs: int, max_tokens: int):
        self.native = native
        self.max_reqs = max_reqs
        device = native.kv_pool.device
        self.input_ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.positions = torch.ones_like(self.input_ids)
        self.hidden = native.weights.fc.weight.new_zeros(max_tokens, native.weights.fc.out_features)
        self.query_start_loc = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
        self.seq_lens = torch.ones(max_reqs, dtype=torch.int32, device=device)
        self.slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self.last_indices = torch.zeros(max_reqs, dtype=torch.long, device=device)
        self.block_table = torch.empty(max_reqs, native.blocks_per_req, dtype=torch.int32, device=device)
        self.block_offsets = torch.arange(native.blocks_per_req, dtype=torch.int32, device=device)
        self.token_indices = torch.arange(max_tokens, device=device)
        self.request_indices = torch.arange(max_reqs + 1, dtype=torch.int32, device=device)
        self.plans: dict[tuple[int, int, int], torch.Tensor] = {}
        # Padded requests read private, initialized storage and never write KV.
        native.kv_pool[:, max_reqs * native.blocks_per_req :].zero_()

    def prepare(self, batch, hidden, num_sampled, num_rejected, last_sampled, next_prefill_tokens, num_reqs_padded):
        n, t = batch.num_reqs, batch.num_tokens
        offsets = batch.query_start_loc[: n + 1].long()
        ends = offsets[1:] - num_rejected[:n].long()
        slots = batch.idx_mapping[:n].long()
        anchors = torch.where(num_sampled[:n] > 0, last_sampled.view(-1)[slots], next_prefill_tokens[0, slots])
        indices = self.token_indices[:t]
        rows = torch.searchsorted(offsets[1:].contiguous(), indices, right=True)
        # Native row j combines target hidden j with its successor token. The
        # last committed successor is the verifier's bonus / next prompt token.
        successors = batch.input_ids[(indices + 1).clamp(max=t - 1)]
        self.input_ids[:t].copy_(torch.where(indices == ends[rows] - 1, anchors[rows], successors))
        self.hidden[:t].copy_(hidden[:t])
        self.positions[:t].copy_(batch.positions[:t] + 1)
        stride = self.native.blocks_per_req * NATIVE_BLOCK_SIZE
        cache_slots = slots[rows] * stride + batch.positions[:t]
        self.slot_mapping[:t].copy_(torch.where(indices < ends[rows], cache_slots, -1))
        self.last_indices[:n].copy_(ends - 1)
        self.query_start_loc[: n + 1].copy_(offsets)
        # Keep all verification queries, including rejected suffixes, so the
        # right-aligned causal diagonal stays at the original target position.
        # Only the accepted prefix writes KV or supplies T1. The extra cache
        # page bounds reads of discarded queries near max_model_len.
        self.seq_lens[:n].copy_(batch.positions[offsets[:-1]] + offsets[1:] - offsets[:-1])
        self.block_table[:n].copy_(slots[:, None] * self.native.blocks_per_req + self.block_offsets)
        if num_reqs_padded > n:
            padded_tokens = (num_reqs_padded - n) * (DMTP_PROPOSAL_COUNT + 1)
            self.input_ids[t : t + padded_tokens].zero_()
            self.hidden[t : t + padded_tokens].zero_()
            self.positions[t : t + padded_tokens].fill_(1)
            self.slot_mapping[t : t + padded_tokens].fill_(-1)
            self.last_indices[n:num_reqs_padded].fill_(t)
            self.query_start_loc[n + 1 : num_reqs_padded + 1].copy_(
                t + self.request_indices[1 : num_reqs_padded - n + 1] * (DMTP_PROPOSAL_COUNT + 1)
            )
            self.seq_lens[n:num_reqs_padded].fill_(DMTP_PROPOSAL_COUNT + 1)
            self.block_table[n:num_reqs_padded].copy_(self.max_reqs * self.native.blocks_per_req + self.block_offsets)

    def prepare_dummy(self, num_reqs, num_tokens):
        self.input_ids[:num_tokens].zero_()
        self.positions[:num_tokens].fill_(1)
        self.hidden[:num_tokens].zero_()
        self.slot_mapping[:num_tokens].fill_(-1)
        self.last_indices[:num_reqs].zero_()
        self.query_start_loc[: num_reqs + 1].copy_(self.request_indices[: num_reqs + 1] * (num_tokens // num_reqs))
        self.seq_lens[:num_reqs].fill_(num_tokens // num_reqs)
        self.block_table[:num_reqs].copy_(self.max_reqs * self.native.blocks_per_req + self.block_offsets)

    def attention_metadata(self, num_reqs, num_tokens, max_query_len, persistent=False):
        # AICPU tiling launches on an auxiliary stream: perform it before capture
        # / replay, as the shared FA3 backend does, and retain a stable address.
        from flash_attn_npu_3 import get_scheduler_metadata

        attn = self.native.weights.layers[0].self_attn
        plan = get_scheduler_metadata(
            batch_size=num_reqs,
            max_seqlen_q=max_query_len,
            max_seqlen_k=self.native.blocks_per_req * NATIVE_BLOCK_SIZE,
            num_heads_q=attn.config.num_attention_heads,
            num_heads_kv=self.native.num_kv_heads,
            headdim=attn.head_dim,
            cache_seqlens=self.seq_lens[:num_reqs],
            qkv_dtype=self.hidden.dtype,
            cu_seqlens_q=self.query_start_loc[: num_reqs + 1],
            page_size=NATIVE_BLOCK_SIZE,
            causal=True,
            num_splits=0,
        )
        key = num_reqs, num_tokens, max_query_len
        if persistent and key in self.plans:
            self.plans[key].copy_(plan)
        elif persistent:
            self.plans[key] = plan
        return NativeAttentionMetadata(
            self.query_start_loc[: num_reqs + 1],
            self.seq_lens[:num_reqs],
            self.block_table[:num_reqs],
            self.slot_mapping[:num_tokens],
            max_query_len,
            self.plans[key] if persistent else plan,
        )

    def forward(self, num_reqs, num_tokens, metadata, model, t1_hidden, t1):
        hidden = self.native.forward_paged(
            self.input_ids[:num_tokens], self.hidden[:num_tokens], self.positions[:num_tokens], metadata
        )
        t1_hidden[:num_reqs].copy_(hidden[self.last_indices[:num_reqs]])
        t1[:num_reqs].copy_(model.logits_processor.get_top_tokens(model.lm_head, t1_hidden[:num_reqs]))
