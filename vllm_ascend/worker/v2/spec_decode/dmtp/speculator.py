# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dmtp.batch import NativeMTPBatch
from vllm_ascend.worker.v2.spec_decode.dmtp.native_mtp import NativeMTP, NativeMTPCache
from vllm_ascend.worker.v2.utils import communicator_switch


def _greedy_suffix(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Walk selector edges, conditioning each row on the preceding winning candidate."""
    batch_size, width, _ = candidate_ids.shape
    batch = torch.arange(batch_size, device=candidate_ids.device)
    previous = torch.zeros_like(batch)
    tokens = []
    for position in range(width):
        previous = scores[batch, position, previous].argmax(dim=-1)
        tokens.append(candidate_ids[batch, position, previous])
    return torch.stack(tokens, dim=1)


class AscendDMTPSpeculator(AscendDFlashSpeculator):
    """Native MTP supplies T1; DFlash2 supplies T2 through T7 in one pass."""

    _speculator_name = "DMTP"
    # Native T1 argmax uses the ordinary tensor-parallel vocabulary shards.
    _lmhead_tp_sample_draft_supported = False

    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        self._batched_native = not self.vllm_config.model_config.enforce_eager
        self.native_mtp = NativeMTP.from_target(self.vllm_config, target_model, paged=self._batched_native)
        self._native_caches: dict[int, NativeMTPCache] = {}
        self._t1_hidden = self.hidden_states.new_zeros(self.max_num_reqs, self.hidden_size)
        self._t1 = torch.zeros(self.max_num_reqs, dtype=torch.long, device=self.device)
        if self._batched_native:
            self.native_batch = NativeMTPBatch(
                self.native_mtp, self.max_num_reqs, max(self.max_num_tokens, self.max_num_reqs * self.num_query_per_req)
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        if self._batched_native:
            # Keep this manager directly on the speculator so upstream's graph
            # memory profiling also releases/recreates it with the draft graphs.
            self.native_cudagraph_manager = CudaGraphManager(
                self.vllm_config,
                self.device,
                self.query_cudagraph_manager.cudagraph_mode,
                decode_query_lens={self.num_query_per_req: self.max_num_reqs},
            )

    def capture(self) -> None:
        if self._batched_native:

            def create_forward(desc, warmup):
                self.native_batch.prepare_dummy(desc.num_reqs, desc.num_tokens)
                metadata = self.native_batch.attention_metadata(
                    desc.num_reqs, desc.num_tokens, self.num_query_per_req, persistent=True
                )

                def forward(mode):
                    self.native_batch.forward(
                        desc.num_reqs, desc.num_tokens, metadata, self.model, self._t1_hidden, self._t1
                    )

                return forward

            with communicator_switch():
                self.native_cudagraph_manager.capture(create_forward, "Capturing DMTP native ACL graphs")
        super().capture()

    def _advance_native_batch(self, batch, hidden, num_sampled, num_rejected, last_sampled, next_prefill_tokens):
        # Query boundaries already have a CPU scheduler mirror. Rejection counts
        # and exact context lengths remain on device throughout native drafting.
        query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
        uniform = self.num_query_per_req if (query_lens == self.num_query_per_req).all() else None
        desc = self.native_cudagraph_manager.dispatch(
            batch.num_reqs, batch.num_tokens, uniform_token_count=uniform, num_active_loras=0
        )
        n = desc.num_reqs if desc.cg_mode == CUDAGraphMode.FULL else batch.num_reqs
        t = desc.num_tokens
        self.native_batch.prepare(batch, hidden, num_sampled, num_rejected, last_sampled, next_prefill_tokens, n)
        metadata = self.native_batch.attention_metadata(
            n,
            t,
            self.num_query_per_req if uniform else int(query_lens.max()),
            persistent=desc.cg_mode == CUDAGraphMode.FULL,
        )
        if desc.cg_mode == CUDAGraphMode.FULL:
            self.native_cudagraph_manager.run_fullgraph(desc)
        else:
            self.native_batch.forward(n, t, metadata, self.model, self._t1_hidden, self._t1)

    def add_request(self, req_index: int) -> None:
        # MRV2 calls this on admission, preemption/recomputation and slot reuse.
        self._native_caches.pop(req_index, None)

    def reset_state(self) -> None:
        self._native_caches.clear()

    def _advance_native(
        self,
        batch: InputBatch,
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        if getattr(self, "_batched_native", False):
            self._advance_native_batch(
                batch, last_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens
            )
            return
        n = batch.num_reqs
        starts = batch.query_start_loc[:n].long()
        anchors = torch.where(
            num_sampled[:n] > 0,
            last_sampled.view(-1)[batch.idx_mapping[:n]],
            next_prefill_tokens[0, batch.idx_mapping[:n]],
        )
        # Eager native attention has one variable-length committed prefix per
        # request. Read the offsets/counts together, once per batch.
        spans = torch.stack((batch.positions[starts], num_rejected[:n].long())).T.cpu().tolist()
        offsets = batch.query_start_loc_np
        for row, (first_pos, rejected) in enumerate(spans):
            slot = int(batch.idx_mapping_np[row])
            start, end = int(offsets[row]), int(offsets[row + 1]) - rejected
            if end <= start:
                raise ValueError("DMTP requires at least one accepted target input row per request.")
            cache = self._native_caches.get(slot)
            if cache is None or first_pos == 0:
                cache = self.native_mtp.empty_cache(slot)
            if cache.length != first_pos:
                raise ValueError(f"DMTP native prefix length {cache.length} does not match target offset {first_pos}.")
            # Row j consumes the successor of target input j and the target
            # hidden at j. The verifier supplies the final successor (the next
            # prompt token for a partial prefill). Rejected rows never enter KV.
            ids = torch.cat((batch.input_ids[start + 1 : end], anchors[row : row + 1]))
            cache, hidden = self.native_mtp.append(
                cache, ids, last_hidden_states[start:end], batch.positions[start:end] + 1
            )
            self._native_caches[slot] = cache
            self._t1_hidden[row].copy_(hidden[-1])
        self._t1[:n] = self.model.logits_processor.get_top_tokens(self.model.lm_head, self._t1_hidden[:n])

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        if not dummy_run:
            self._advance_native(
                input_batch, last_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens
            )
        elif is_profile:
            # KV backing storage is already resident. Measure prefill workspace
            # with a fresh handle before a real request can own this slot.
            count = min(self.max_model_len, last_hidden_states.shape[0])
            ids = torch.zeros(count, dtype=torch.long, device=self.device)
            positions = torch.arange(1, count + 1, device=self.device)
            if self._batched_native:
                self.native_batch.prepare_dummy(1, count)
                metadata = self.native_batch.attention_metadata(1, count, count)
                self.native_batch.forward(1, count, metadata, self.model, self._t1_hidden, self._t1)
            else:
                self.native_mtp.append(self.native_mtp.empty_cache(0), ids, last_hidden_states[:count], positions)
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
            num_speculative_tokens,
        )

    def _generate_draft(
        self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode
    ) -> None:
        # The inherited context projection, slot mappings and verify widths
        # use A + 7 rows. Only the second query input and six scored rows differ.
        width = self.num_query_per_req
        input_ids = self.input_buffers.input_ids[:num_tokens_padded].view(-1, width)
        input_ids[:num_reqs, 1] = self._t1[:num_reqs].to(input_ids.dtype)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens_padded),
            is_padding=self.input_buffers.is_padding[:num_tokens_padded],
        ):
            embeds = self.model.embed_dmtp_inputs(input_ids, self._t1_hidden[: input_ids.shape[0]])
            hidden = self.model(
                input_ids=None,
                positions=self.input_buffers.positions[:num_tokens_padded],
                inputs_embeds=embeds.flatten(0, 1),
            )
        suffix_hidden = hidden.view(-1, width, self.hidden_size)[:num_reqs, 2:]
        candidate_ids, unary = self.model.compute_candidates(suffix_hidden.flatten(0, 1))
        ids = candidate_ids.view(num_reqs, width - 2, -1)
        scores = self.model.model.candidate_selector(ids, unary.view_as(ids), suffix_hidden, self._t1[:num_reqs])
        self.draft_tokens[:num_reqs, 0] = self._t1[:num_reqs]
        self.draft_tokens[:num_reqs, 1:] = _greedy_suffix(ids, scores)
