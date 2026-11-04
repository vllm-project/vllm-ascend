# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU suffix decoding speculator for Model Runner V2.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
Keep matching, voting and lifecycle semantics aligned with that implementation.
History and corpus addresses are bounded independently of masks for Ascend lowering.

Extends the n-gram speculator with the two parts of SuffixDecoding
(https://arxiv.org/abs/2411.04975) that carry most of its acceptance gain:

* Variable-length matching: drafts continue the occurrence whose preceding
  tokens match the longest suffix of the request's history (up to
  ``suffix_decoding_max_tree_depth``), instead of a fixed n-gram window.
* Cross-request memory: responses of finished requests are appended to a
  device-resident token corpus. A hash index keyed on the last 2 and last 4
  tokens gives candidate positions, and each candidate is extended to the
  longest suffix match, so the index is only an entry point for suffix
  matching.

The propose path never synchronizes with the host. Ingestion reads token
history and lengths directly from ``RequestState`` and processes finished
requests in sorted id order; index updates are order-independent, so every
TP rank holds an identical corpus without any collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_ascend.ops.triton.spec_decode.suffix import (
    _BUCKET_SLOTS,
    _HASH_MUL,
    _INGEST_BLOCK,
    _NUM_LEN_BUCKETS,
    _PROMPT_CONTEXT,
    _STATS_HALF_LIFE,
    _suffix_finalize_kernel,
    _suffix_ingest_kernel,
    _suffix_local_scan_kernel,
    _suffix_observe_kernel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.states import RequestState


class AscendSuffixSpeculator(BaseSpeculator):
    """Device-resident suffix matching and drafting for Ascend MRV2."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: RequestState,
    ):
        if not HAS_TRITON:
            raise RuntimeError("suffix speculative decoding requires Triton.")
        spec = vllm_config.speculative_config
        assert spec is not None
        self.vllm_config = vllm_config
        self.device = device
        self.req_states = req_states
        self.num_speculative_steps: int = spec.num_speculative_tokens
        self.max_depth: int = spec.suffix_decoding_max_tree_depth
        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len

        L = self.max_model_len
        if L >= 1024:
            self.block_l = 256
        elif L >= 256:
            self.block_l = 128
        else:
            self.block_l = max(16, triton.next_power_of_2(max(L, 1)))
        self.n_blocks = triton.cdiv(L, self.block_l)
        self.scratch = torch.zeros((self.max_num_reqs, self.n_blocks), dtype=torch.int64, device=device)

        # Corpus of finished responses separated by -1, plus two hash tables
        # keyed on the last 4 and last 2 tokens before each position. A
        # capacity of 0 disables it.
        self.corpus_len: int = (
            spec.suffix_decoding_corpus_tokens if spec.suffix_decoding_max_cached_requests != 0 else 0
        )
        if self.corpus_len > 0 and self.max_depth < 2:
            # Corpus matches need the 2-token hash key.
            logger.warning(
                "suffix_decoding_max_tree_depth=%d is too short for "
                "cross-request matching (needs >= 2); drafting from each "
                "request's own history only.",
                self.max_depth,
            )
            self.corpus_len = 0
        self.has_corpus = self.corpus_len > 0
        self.num_buckets = triton.next_power_of_2(max(16, self.corpus_len // 4))
        self.corpus = torch.full((max(self.corpus_len, 1),), -1, dtype=torch.int32, device=device)
        self.table = torch.full(
            (2, self.num_buckets if self.has_corpus else 1, _BUCKET_SLOTS),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.corpus_head = torch.zeros(1, dtype=torch.int64, device=device)
        self.doc_seq = torch.zeros(1, dtype=torch.int64, device=device)
        # Host mirror of doc_seq: index entries keep it in 31 bits.
        self.num_docs = 0
        self.ingest_out = torch.zeros(2, dtype=torch.int64, device=device)

        # Batch-ordered draft output, scattered into RequestState.draft_tokens
        # by the model runner (same contract as the model-based speculators).
        self.drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        # Number of real (non-filler) draft tokens per batch row.
        self.num_valid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)

        # Adaptive verification (optional): per-slot confidences from
        # acceptance statistics learned online.
        self.enable_adaptive_verification = spec.enable_adaptive_verification
        self.draft_token_confidence_probs = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self.accept_stats = torch.zeros(
            (2 * _NUM_LEN_BUCKETS, self.num_speculative_steps, 2),
            dtype=torch.int64,
            device=device,
        )
        self.last_bucket = torch.full((self.max_num_reqs,), -1, dtype=torch.int32, device=device)
        self.last_valid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self.num_steps = 0
        # Slots reassigned since the last propose; their stats bookkeeping
        # still belongs to the previous request.
        self.new_slots: list[int] = []

    def add_request(self, req_index: int) -> None:
        if self.enable_adaptive_verification:
            self.new_slots.append(req_index)

    def reset_state(self) -> None:
        """Forget everything learned so far (used after the warmup requests)."""
        self.corpus.fill_(-1)
        self.table.fill_(-1)
        self.corpus_head.zero_()
        self.doc_seq.zero_()
        self.num_docs = 0
        self.accept_stats.zero_()
        self.last_bucket.fill_(-1)
        self.last_valid.zero_()
        self.num_steps = 0
        self.new_slots.clear()

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Add finished responses to the corpus before their slots are freed."""
        if not self.has_corpus:
            return
        req_states = self.req_states
        # Sorted so every TP rank ingests documents in the same order. Starts
        # come from the host copy of prompt_len: the device-visible copy is
        # republished every step and may be reused for a new request before
        # this kernel runs.
        docs = []
        for req_id in sorted(finished_req_ids):
            slot = req_states.req_id_to_index.get(req_id)
            if slot is None:
                continue
            # Keep the prompt's last few tokens (typically the chat template's
            # assistant header) so the first response tokens are indexed by
            # the tokens that precede them.
            start = max(int(req_states.prompt_len.np[slot]) - _PROMPT_CONTEXT, 0)
            docs += [slot, start]
        if not docs:
            return
        num_docs = len(docs) // 2
        if self.num_docs + num_docs >= 1 << 31:
            # Index entries pack doc_seq into 31 bits; start a new index (the
            # corpus tokens stay, only their entries are dropped).
            self.table.fill_(-1)
            self.doc_seq.zero_()
            self.num_docs = 0
        self.num_docs += num_docs
        docs_gpu = async_tensor_h2d(docs, self.device, torch.int32)
        token_ids = self.req_states.all_token_ids.gpu
        _suffix_ingest_kernel[(num_docs, triton.cdiv(self.max_model_len, _INGEST_BLOCK))](
            token_ids,
            token_ids.stride(0),
            docs_gpu,
            num_docs,
            req_states.total_len.gpu,
            self.corpus,
            self.corpus_len,
            self.corpus_head,
            self.doc_seq,
            self.ingest_out[0:1],
            self.ingest_out[1:2],
            self.table,
            HASH_MUL=_HASH_MUL,
            NUM_BUCKETS=self.num_buckets,
            SLOTS=_BUCKET_SLOTS,
            BLOCK=_INGEST_BLOCK,
            num_warps=4,
        )
        self.corpus_head.copy_(self.ingest_out[0:1])
        self.doc_seq.copy_(self.ingest_out[1:2])

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run or num_reqs == 0:
            return self.drafts[:num_reqs]

        token_ids = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        idx_mapping = input_batch.idx_mapping
        k_po2 = max(1, triton.next_power_of_2(self.num_speculative_steps))

        if self.enable_adaptive_verification:
            if self.new_slots:
                slots = async_tensor_h2d(self.new_slots, self.device, torch.int64)
                self.last_bucket.index_fill_(0, slots, -1)
                self.new_slots.clear()
            _suffix_observe_kernel[(self.accept_stats.shape[0],)](
                idx_mapping,
                num_sampled,
                num_rejected,
                self.accept_stats,
                self.last_bucket,
                self.last_valid,
                num_reqs,
                REQS_PO2=triton.next_power_of_2(num_reqs),
                K=self.num_speculative_steps,
                K_PO2=k_po2,
                num_warps=4,
            )
            self.num_steps += 1
            if self.num_steps % _STATS_HALF_LIFE == 0:
                self.accept_stats.bitwise_right_shift_(1)

        _suffix_local_scan_kernel[(num_reqs, self.n_blocks)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            total_len,
            num_sampled,
            self.scratch,
            self.scratch.stride(0),
            self.max_model_len + 1,
            MAX_DEPTH=self.max_depth,
            BLOCK=self.block_l,
            num_warps=4,
        )
        _suffix_finalize_kernel[(num_reqs,)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            total_len,
            num_sampled,
            last_sampled.view(-1),
            self.scratch,
            self.scratch.stride(0),
            self.n_blocks,
            self.max_model_len + 1,
            self.corpus,
            self.corpus.shape[0],
            self.table,
            self.drafts,
            self.num_valid,
            self.accept_stats,
            self.last_bucket,
            self.last_valid,
            self.draft_token_confidence_probs,
            HAS_CORPUS=self.has_corpus,
            ADAPTIVE=self.enable_adaptive_verification,
            HASH_MUL=_HASH_MUL,
            NUM_BUCKETS=self.num_buckets,
            SLOTS=_BUCKET_SLOTS,
            MAX_DEPTH=self.max_depth,
            DEPTH_PO2=triton.next_power_of_2(self.max_depth),
            K=self.num_speculative_steps,
            K_PO2=k_po2,
            NB_PO2=max(1, triton.next_power_of_2(self.n_blocks)),
            num_warps=4,
        )
        return self.drafts[:num_reqs]
