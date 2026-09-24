# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Ascend Gemma4 MTP speculator for Model Runner V2.

Beyond the wiring inherited from the upstream ``Gemma4Speculator``, this
subclass restores the execution semantics of the v1 ``Gemma4Proposer`` that
the V2 port dropped. Four defects cost ~5.3pp of draft acceptance on
Ascend (89.3% -> 94.8%, matching MRv1 bit-for-bit):

1. The draft prefill window reused the target's ``PrefillNoCache``
   metadata, running pure in-batch attention with no KV-cache reads while
   MRv1's draft reads the target's int8 KV cache — a different numeric
   path that changes the window sample.
2. The decode window was the rejection-shrunk prefix of EAGLE-shifted ids;
   MRv1 keeps the full-width verify rows with the sequence's post-commit
   values (accepted prefix, recovery token at the rejected slot, stale
   tail rows as in-batch context).
3. The window head hidden must be the target hidden of the position right
   before the window (a cross-propose stash), with the other hiddens
   right-shifted one row.
4. The KV view for the window and the decode steps must sit at the
   committed boundary (not the post-update seq_lens, which also counts
   the unverified bonus and the next round's drafts).
"""

import importlib
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)

_vllm_ar_speculator = importlib.import_module(
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"
)
_vllm_draft_speculator = importlib.import_module(
    "vllm.v1.worker.gpu.spec_decode.speculator"
)


@contextmanager
def _inject_seq_lens_np(seq_lens_np):
    """Inject the real per-request seq_lens into the Ascend metadata build.

    The Ascend ``build_attn_metadata`` takes ``seq_lens_np`` for its CPU-side
    seq_lens, but the upstream draft base does not pass it, so it falls back
    to ``max_seq_len`` — an upper bound that breaks the committed-boundary KV
    view. Inject it the same way ``build_draft_attn_metadata_factory``
    injects ``positions``: wrap the module symbol for the duration of one
    build, so every tiling decision is made from the real values.
    """
    raw = _vllm_draft_speculator.build_attn_metadata

    def wrapped(*args, **kwargs):
        kwargs["seq_lens_np"] = seq_lens_np
        return raw(*args, **kwargs)

    _vllm_draft_speculator.build_attn_metadata = wrapped
    try:
        yield
    finally:
        _vllm_draft_speculator.build_attn_metadata = raw


@contextmanager
def _gemma4_prefill_inputs(spec, input_batch, num_sampled, num_rejected):
    """Route the upstream ``prepare_prefill_inputs`` through the Gemma4
    window rebuild for the duration of one ``propose`` call.

    The upstream autoregressive loop builds its draft window with EAGLE
    semantics (ids shifted one left, the freshly sampled bonus appended at
    the tail, positions of the hidden rows). Gemma4 MTP needs the full-width
    verify rows with post-commit ids, right-shifted hiddens with a stashed
    head, and the KV boundary at the committed edge. Rewriting the buffers
    right after the stock kernel keeps every other bookkeeping write
    (query_start_loc, last_token_indices, padding) intact.
    """
    orig = _vllm_ar_speculator.prepare_prefill_inputs

    def patched(*args, **kwargs):
        result = orig(*args, **kwargs)
        _rebuild_gemma4_windows(spec, input_batch, num_sampled, num_rejected)
        return result

    _vllm_ar_speculator.prepare_prefill_inputs = patched
    try:
        yield
    finally:
        _vllm_ar_speculator.prepare_prefill_inputs = orig


def _rebuild_gemma4_windows(spec, input_batch, num_sampled, num_rejected):
    """Rebuild the Gemma4 MTP draft windows in the speculator buffers.

    Decode-continue rounds get the MRv1-shaped window over the FULL width
    of scheduled verify rows; the stock kernel only fills the
    rejection-shrunk prefix, leaving tail rows stale even though the
    forward runs full-width. True (chunked) prefill rounds keep the stock
    window — it matches MRv1 byte-for-byte there.
    """
    num_reqs = input_batch.num_reqs
    if num_reqs == 0:
        return
    try:
        # Dummy/warmup proposes carry all-zero positions and fake sampling
        # parameters; leave the stock window untouched for them.
        if int(input_batch.positions.max()) == 0:
            return
        qsl = input_batch.query_start_loc_np
        num_sampled_h = num_sampled[:num_reqs].tolist()
        idx_slots = input_batch.idx_mapping[:num_reqs].tolist()
        is_prefilling = input_batch.is_prefilling_np

        ids_buf = spec.input_buffers.input_ids
        pos_buf = spec.input_buffers.positions
        hidden = spec.hidden_states
        stash = spec._g4_stash
        committed = []
        for i in range(num_reqs):
            qs = int(qsl[i])
            full_len = int(qsl[i + 1]) - qs
            qe = qs + full_len
            if full_len <= 0:
                committed.append(0)
                continue
            slot = idx_slots[i]
            if num_sampled_h[i] > 0 and not is_prefilling[i]:
                head_hidden = stash[slot].clone()
                # The stash for the NEXT round = the target's hidden at the
                # last committed position = the pre-shift hidden[qs].
                stash[slot].copy_(hidden[qs])
                shifted = hidden[qs : qe - 1].clone()
                hidden[qs].copy_(head_hidden)
                hidden[qs + 1 : qe].copy_(shifted)
                pos_buf[qs:qe].copy_(input_batch.positions[qs:qe])
                # Window ids = the SEQUENCE values at each row's position
                # (post-commit: accepted values, the recovery token at the
                # rejected slot, stale tail values beyond). The verify input
                # rows would show the rejected draft at the recovery slot.
                tok = spec.model_state.req_states.all_token_ids.gpu
                pos = input_batch.positions[qs:qe].long()
                ids_buf[qs:qe].copy_(tok[slot].gather(0, pos))
                # KV boundary = one past the window's last position.
                committed_count = int(pos_buf[qe - 1].item()) + 1
                spec.input_buffers.seq_lens[i] = committed_count
                committed.append(committed_count)
                spec._g4_window_dirty = True
            else:
                # True (chunked) prefill: the stock window matches MRv1.
                committed.append(int(input_batch.positions[qe - 1].item()) + 1)
        spec._g4_committed = np.asarray(committed, dtype=np.int32)
    except Exception:
        logger.exception("[gemma4] window rebuild failed; keeping stock window")


class AscendGemma4Speculator(AscendAutoRegressiveSpeculator, Gemma4Speculator):
    """``_create_draft_vllm_config`` and ``load_draft_model`` are the only
    methods defined by both bases, so this subclass recombines them."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Gemma4 MTP decode-continue windows need the target hidden of the
        # position right before the window head, which is not part of the
        # current verify batch. Stash each request's last target hidden row
        # (indexed by req-state slot) across propose calls.
        self._g4_stash = torch.zeros(
            self.max_num_reqs,
            self.hidden_states.shape[1],
            dtype=self.hidden_states.dtype,
            device=device,
        )
        # Set when the latest propose built at least one decode-continue
        # window, so `_prefill` knows to build fresh draft attention
        # metadata instead of reusing the target's.
        self._g4_window_dirty = False
        self._g4_committed = None

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states=None,
        num_sampled=None,
        num_rejected=None,
        last_sampled=None,
        next_prefill_tokens=None,
        temperature=None,
        seeds=None,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=None,
        dp_sync=None,
    ):
        self._g4_window_dirty = False
        self._g4_committed = None
        with _gemma4_prefill_inputs(self, input_batch, num_sampled, num_rejected):
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
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                dp_sync=dp_sync,
            )

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs=None,
    ):
        if self._g4_committed is not None:
            # Build the draft window's own attention metadata instead of
            # reusing the target's. For prefill rounds the target metadata is
            # PrefillNoCache (no KV-cache reads — a different numeric path
            # from MRv1's cache-based draft); for decode rounds its seq_lens
            # already counts the unverified bonus and the next drafts.
            # Shrinking the target metadata after the fact violates the FIA
            # tiling invariants, so rebuild from scratch at the committed
            # boundary (same recipe as the replicated-PCP prefill path).
            attn_metadata, slot_mappings = self._build_gemma4_prefill_attn(
                num_reqs, num_tokens, input_batch=self.input_batch
            )
        super()._prefill(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

    def _build_gemma4_prefill_attn(self, num_reqs, num_tokens, input_batch):
        """Build the draft window's own cache-based attention metadata."""
        self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=num_reqs,
        )
        slot_mappings_tensor = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            self.input_buffers.positions,
            num_tokens_padded=num_tokens,
        )
        slot_mappings = build_slot_mappings_by_layer(
            slot_mappings_tensor, self.kv_cache_config
        )
        # input_buffers.seq_lens was already rewritten to the committed
        # boundary by the window rebuild, so it seeds the GPU seq_lens,
        # the CPU upper bound, and — via _inject_seq_lens_np — the Ascend
        # seq_lens_np that would otherwise default to max_seq_len.
        seq_lens_upper = self.input_buffers.seq_lens[:num_reqs].to(
            torch.int32
        ).cpu()
        # The window is a multi-row query per request, so it must run as a
        # Prefill-state batch like the target verify metadata it replaces;
        # the Ascend draft override forces DecodeOnly (single-row query
        # assumption), so call the upstream base builder directly under the
        # Ascend wrapper + factory with is_prefilling=True.
        is_prefilling_true = torch.ones(num_reqs, dtype=torch.bool)
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions, num_tokens, is_prefilling_true
            ),
            _inject_seq_lens_np(seq_lens_upper.numpy()),
        ):
            attn_metadata = DraftModelSpeculator._build_draft_attn_metadata(
                self,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs,
                num_tokens_padded=num_tokens,
                seq_lens_cpu_upper_bound=seq_lens_upper,
                step=0,
                query_start_loc_np=input_batch.query_start_loc_np,
            )
        return attn_metadata, slot_mappings

    def _build_draft_attn_metadata(  # type: ignore[override]
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int = 1,
        causal: bool = True,
        query_start_loc_np=None,
    ):
        """Pin the decode-step KV view at the committed boundary.

        The stock step metadata sizes its KV from the target's post-update
        seq_lens and advances it per step — for Gemma4 MTP (constant
        positions, shared target KV) that reads stale KV entries left by
        the previous verify pass. MRv1 keeps the KV view constant at the
        window's committed boundary instead.
        """
        committed = self._g4_committed
        if committed is not None and step >= 1:
            committed_t = torch.from_numpy(committed)
            self.input_buffers.seq_lens[:num_reqs] = committed_t.to(
                self.input_buffers.seq_lens.dtype
            ).to(self.input_buffers.seq_lens.device)
            with _inject_seq_lens_np(committed):
                metadata = super()._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    num_tokens_padded=num_tokens_padded,
                    seq_lens_cpu_upper_bound=committed_t,
                    step=0,
                    num_query_per_req=num_query_per_req,
                    causal=causal,
                    query_start_loc_np=query_start_loc_np,
                )
                if metadata:
                    # The Ascend override this super() chain goes through
                    # forces DecodeOnly, dispatching the step rows to a
                    # different attention kernel; MRv1 runs its step rows
                    # through the same fused-FIA kernel as the window.
                    # Match it.
                    for md in metadata.values():
                        if md is not None:
                            md.attn_state = AscendAttentionState.SpecDecoding
                return metadata
        return super()._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            num_query_per_req=num_query_per_req,
            causal=causal,
            query_start_loc_np=query_start_loc_np,
        )

    def _update_decode_attn_metadata(self, attn_metadata, step, num_reqs=None):
        """Keep the per-step KV view constant for Gemma4 MTP.

        The Ascend base advances ``seq_lens_cpu``/``seq_lens_list`` by the
        draft step; with positions fixed, that would grow the KV view into
        stale entries. MRv1 keeps both at the committed boundary.
        """
        committed = self._g4_committed
        if committed is not None and attn_metadata:
            attn_meta = next(iter(attn_metadata.values()))
            num_reqs_padded = attn_meta.seq_lens_cpu.shape[0]
            if num_reqs is None:
                num_reqs = num_reqs_padded
            query_lens_list = list(range(1, num_reqs_padded + 1))
            seq_lens_list = [
                int(committed[i]) if i < len(committed) else 0
                for i in range(num_reqs_padded)
            ]
            for metadata in attn_metadata.values():
                if metadata is None:
                    continue
                decode_metadata = (
                    metadata.decode if self.attn_architecture == "MLA" else metadata
                )
                decode_metadata.seq_lens_list = seq_lens_list
                decode_metadata.actual_seq_lengths_q = query_lens_list
                metadata.seq_lens_cpu.copy_(
                    torch.tensor(seq_lens_list, dtype=metadata.seq_lens_cpu.dtype)
                )
            return
        super()._update_decode_attn_metadata(attn_metadata, step, num_reqs)

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        # The draft is dense even for a MoE target, and Gemma4's heterogeneous
        # head dimensions require the target's forced attention backend.
        draft_vllm_config = replace(
            draft_vllm_config,
            parallel_config=replace(
                draft_vllm_config.parallel_config,
                prefill_context_parallel_size=1,
                enable_expert_parallel=False,
                enable_eplb=False,
            ),
        )
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is None:
            return draft_vllm_config
        return replace(
            draft_vllm_config,
            attention_config=replace(
                draft_vllm_config.attention_config,
                backend=target_backend,
            ),
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        # The Ascend base chains into Gemma4Speculator.load_draft_model.
        draft_model = super().load_draft_model(target_model, target_attn_layer_names)
        self._sync_kv_sharing_target_to_impl(draft_model)
        return draft_model

    def _sync_kv_sharing_target_to_impl(self, draft_model: nn.Module) -> None:
        """Copy the late-bound KV-sharing target onto the Ascend attention impls.

        ``AscendAttentionBackendImpl`` snapshots ``kv_sharing_target_layer_name``
        when it is constructed and uses it to skip writing its own KV, but
        ``_setup_gemma4_kv_sharing`` sets that attribute afterwards, on the vLLM
        ``Attention`` wrapper.
        """
        synced = 0
        total = 0
        for layer in getattr(draft_model.model, "layers", []):
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attn is None:
                continue
            total += 1
            target = getattr(attn, "kv_sharing_target_layer_name", None)
            impl = getattr(attn, "impl", None)
            if target is not None and impl is not None:
                impl.kv_sharing_target_layer_name = target
                synced += 1
        logger.info(
            "Gemma4 MTP: propagated KV-sharing target to %d/%d draft layers.",
            synced,
            total,
        )
