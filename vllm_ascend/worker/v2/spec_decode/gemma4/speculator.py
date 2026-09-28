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
"""Ascend Gemma4 MTP speculator for Model Runner V2."""

import importlib
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)

_vllm_ar_speculator = importlib.import_module("vllm.v1.worker.gpu.spec_decode.autoregressive.speculator")
_vllm_draft_speculator = importlib.import_module("vllm.v1.worker.gpu.spec_decode.speculator")


@contextmanager
def _gemma4_prefill_inputs(spec, input_batch, num_sampled, num_rejected):
    """Run the Gemma4 window retiming right after the stock prepare_prefill_inputs."""
    orig = _vllm_ar_speculator.prepare_prefill_inputs  # type: ignore[attr-defined]

    def patched(*args, **kwargs):
        result = orig(*args, **kwargs)
        _rebuild_gemma4_windows(spec, input_batch, num_sampled, num_rejected)
        return result

    _vllm_ar_speculator.prepare_prefill_inputs = patched  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _vllm_ar_speculator.prepare_prefill_inputs = orig  # type: ignore[attr-defined]


def _rebuild_gemma4_windows(spec, input_batch, num_sampled, num_rejected):
    """Retime the decode-continue windows; pure-prefill batches report the committed boundary."""
    num_reqs = input_batch.num_reqs
    if num_reqs == 0:
        return
    try:
        # Dummy/warmup proposes carry all-zero positions; keep the stock window.
        if int(input_batch.positions.max()) == 0:
            return
        qsl = input_batch.query_start_loc_np
        num_sampled_h = num_sampled[:num_reqs].tolist()
        idx_slots = input_batch.idx_mapping[:num_reqs].tolist()
        is_prefilling = input_batch.is_prefilling_np
        saw_decode_continue = False

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
                # Decode-continue: retime rows only; ids and metadata stay stock.
                saw_decode_continue = True
                head_hidden = stash[slot].clone()
                # Stash for the next round = the pre-shift hidden[qs].
                stash[slot].copy_(hidden[qs])
                shifted = hidden[qs : qe - 1].clone()
                hidden[qs].copy_(head_hidden)
                hidden[qs + 1 : qe].copy_(shifted)
                pos_buf[qs:qe].copy_(input_batch.positions[qs:qe])
            else:
                # True (chunked) prefill: the stock window matches MRv1.
                committed.append(int(input_batch.positions[qe - 1].item()) + 1)
                # Seed the stash with the prompt-tail hidden.
                stash[slot].copy_(hidden[qe - 1])
        if not saw_decode_continue:
            # Pure-prefill batch: report the boundary so `_prefill` rebuilds metadata.
            spec._g4_committed = np.asarray(committed, dtype=np.int32)
    except Exception:
        logger.exception("[gemma4] window rebuild failed; keeping stock window")


class AscendGemma4Speculator(AscendAutoRegressiveSpeculator, Gemma4Speculator):
    """Recombines the two methods both bases define (draft config, draft load)."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Per-request stash of the pre-window target hidden, kept across propose calls.
        self._g4_stash = torch.zeros(
            self.max_num_reqs,
            self.hidden_states.shape[1],
            dtype=self.hidden_states.dtype,
            device=device,
        )
        # Committed boundary of the latest pure-prefill batch.
        self._g4_committed = None

    def propose(  # type: ignore[override]
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
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: Any = None,
    ):
        self._g4_committed = None
        with _gemma4_prefill_inputs(self, input_batch, num_sampled, num_rejected):
            return AscendAutoRegressiveSpeculator.propose(
                self,
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                temperature=temperature,
                seeds=seeds,
                dp_sync=dp_sync,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
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
            # Rebuild cache-based window metadata instead of reusing the target's.
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
        slot_mappings = build_slot_mappings_by_layer(slot_mappings_tensor, self.kv_cache_config)
        # Use the committed boundary directly; no host sync of seq_lens.
        seq_lens_upper = torch.from_numpy(self._g4_committed)
        # Multi-row window needs Prefill state; bypass the DecodeOnly-forcing hooks.
        batch_desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs)
        is_prefilling_true = torch.ones(num_reqs, dtype=torch.bool)
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                is_prefilling_true,
                seq_lens_cpu=seq_lens_upper,
                parallel_config=self.draft_vllm_config.parallel_config,
            ),
        ):
            attn_metadata = DraftModelSpeculator._build_attn_metadata(
                self,
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=input_batch.query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_upper,
                step=0,
            )
        return attn_metadata, slot_mappings

    def _build_attn_metadata(  # type: ignore[override]
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ):
        """Pin the decode-step KV view at the committed boundary."""
        committed = self._g4_committed
        if committed is not None and step >= 1:
            committed_t = torch.from_numpy(committed)
            self.input_buffers.seq_lens[:num_reqs] = committed_t.to(self.input_buffers.seq_lens.dtype).to(
                self.input_buffers.seq_lens.device
            )
            with build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                batch_desc.num_tokens,
                torch.from_numpy(self.input_batch.is_prefilling_np),
                seq_lens_cpu=committed_t,
                parallel_config=self.draft_vllm_config.parallel_config,
            ):
                metadata = DraftModelSpeculator._build_attn_metadata(
                    self,
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    query_start_loc_np=query_start_loc_np,
                    seq_lens_cpu_upper_bound=committed_t,
                    step=0,
                    causal=causal,
                    dcp_local_seq_lens=dcp_local_seq_lens,
                )
            if metadata:
                # Same fused kernel as the window, not DecodeOnly.
                for md in metadata.values():
                    if md is not None:
                        md.attn_state = AscendAttentionState.SpecDecoding
            return metadata
        return super()._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )

    def _update_decode_attn_metadata(self, attn_metadata, step, num_reqs=None):
        """Keep the per-step KV view constant at the committed boundary."""
        committed = self._g4_committed
        if committed is not None and attn_metadata:
            attn_meta = next(iter(attn_metadata.values()))
            num_reqs_padded = attn_meta.seq_lens_cpu.shape[0]
            if num_reqs is None:
                num_reqs = num_reqs_padded
            query_lens_list = list(range(1, num_reqs_padded + 1))
            seq_lens_list = [int(committed[i]) if i < len(committed) else 0 for i in range(num_reqs_padded)]
            for metadata in attn_metadata.values():
                if metadata is None:
                    continue
                decode_metadata = metadata.decode if self.attn_architecture == "MLA" else metadata
                decode_metadata.seq_lens_list = seq_lens_list
                decode_metadata.actual_seq_lengths_q = query_lens_list
                metadata.seq_lens_cpu.copy_(torch.tensor(seq_lens_list, dtype=metadata.seq_lens_cpu.dtype))
            return
        super()._update_decode_attn_metadata(attn_metadata, step, num_reqs)

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        # Dense draft even for a MoE target; keep the target's forced backend.
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
        """Propagate the late-bound KV-sharing target onto the Ascend attention impls."""
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
