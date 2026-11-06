# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Static Eagle3 Graph owner adapter for the MRV2 interfaces.

The ordinary eager owner and MTP base retain their own execution paths. This adapter reuses
existing proposal math and per-consumer banks; it does not own sampling, input
kernels, model weights, or the upstream autoregressive loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBucket
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
    bind_draft_window_read,
    copy_draft_attention_metadata,
)
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


class Eagle3GraphIntegrationMixin(Eagle3DraftWindowMixin, AscendTargetDependentARSpeculator, EagleSpeculator):
    decode_cudagraph_manager: Any
    _eagle3_window_next_step: int
    _eagle3_capture_call: Iterator[DraftWindowGraphBucket] | None

    def _configure_decode_metadata_update(self) -> None:
        super()._configure_decode_metadata_update()
        if getattr(self, "_draft_window_policy", None) is not None:
            # The fixed merged AR route stages a canonical view per consumer.
            self.supports_decode_metadata_update = False

    def init_cudagraph_manager(self, cudagraph_mode):
        super().init_cudagraph_manager(cudagraph_mode)
        provider = getattr(self, "_draft_window_graph_binding_provider", None)
        if provider is not None:
            self._eagle3_capture_call = None
            capacity = min(self.max_model_len, provider.window_size + provider.group.table_column_span - 1)
            for first, manager in ((True, self.prefill_cudagraph_manager), (False, self.decode_cudagraph_manager)):
                if self.update_stream is None or manager.update_stream is not self.update_stream:
                    raise RuntimeError("Eagle3 capture managers must share their owner UPDATE stream")
                q = self.num_speculative_steps + 1 if first else 1
                for descriptors in manager._capture_descs.values():
                    for desc in descriptors:
                        if desc.cg_mode.name != "FULL":
                            raise NotImplementedError("Eagle3 Window capture requires an actual FULL descriptor")
                        n = desc.num_reqs
                        if n is None or desc.num_tokens != n * q:
                            raise RuntimeError("Eagle3 startup descriptor does not match its request/query width")
                        for ordinal in range(1 if first else self.num_speculative_steps - 1):
                            bank = provider.register_capture_identity(descriptor=desc, consumer_ordinal=ordinal)
                            # Seed before GLOBAL capture; no H2D belongs in its callback.
                            bank.cpu_window_exact.fill_(capacity)
            if not provider._carriers:
                raise RuntimeError("Eagle3 Window requires a nonempty capture bucket inventory")

    @contextmanager
    def _eagle3_capture_scope(self, *, n: int, tokens: int):
        if self._eagle3_capture_call is not None:
            raise RuntimeError("nested Eagle3 startup capture invocation")
        self._eagle3_capture_call = iter(
            self._draft_window_graph_binding_provider.capture_banks_for(num_reqs=n, num_tokens=tokens)
        )
        try:
            yield
            if next(self._eagle3_capture_call, None) is not None:
                raise RuntimeError("Eagle3 capture did not visit every actual first-pass/AR model consumer")
        finally:
            self._eagle3_capture_call = None

    def _build_drafter_attn_metadata(
        self, input_batch, attn_metadata, slot_mappings, cg_mode=CUDAGraphMode.NONE, for_capture=False
    ):
        attn_metadata, slot_mappings = super()._build_drafter_attn_metadata(
            input_batch, attn_metadata, slot_mappings, cg_mode, for_capture=for_capture
        )
        if not self._eagle3_window_is_active() or cg_mode == CUDAGraphMode.FULL:
            return attn_metadata, slot_mappings
        if self._eagle3_window_next_step != 0 or cg_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            raise RuntimeError("Eagle3 Window decode uses its FULL Graph managers; no eager fallback")
        # Native first retains the target's query/write operands. Only its
        # private Draft header and the first reader's stable bank change.
        provider = self._draft_window_graph_binding_provider
        n = input_batch.num_reqs
        queries = input_batch.query_start_loc_np[1 : n + 1].tolist()
        layer = attn_metadata[provider.group.layers[0]]
        if (
            layer.actual_seq_lengths_q != queries
            or layer.num_actual_tokens != queries[-1]
            or queries[-1] > input_batch.num_tokens_after_padding
            or layer.block_tables.shape[0] < n
            or len(layer.seq_lens_list) < n
        ):
            raise NotImplementedError(
                "Eagle3 Window native prefill requires canonical effective query rows; "
                "additional padding requests are not qualified for this read bank"
            )
        bank = provider.capture_banks_for(num_reqs=n, num_tokens=n * (self.num_speculative_steps + 1))[0]
        self.block_tables.gather_block_tables(input_batch.idx_mapping, num_reqs_padded=n)
        bank = self._materialize_eagle3_step(bank, 0)
        attn_metadata = copy_draft_attention_metadata(attn_metadata, provider.group.layers)
        layer = attn_metadata[provider.group.layers[0]]
        bind_draft_window_read(layer, bank)
        layer.attn_state = AscendAttentionState.ChunkedPrefill
        return attn_metadata, slot_mappings

    def on_prefill_end(self, num_reqs):
        super().on_prefill_end(num_reqs)
        if self._eagle3_window_is_active() and self._eagle3_window_next_step == 0:
            # This releases Host preparation, not the device reader. Any later
            # proposal failure still quarantines the strongly owned banks.
            self._draft_window_graph_binding_provider.complete_native()
            self._eagle3_window_next_step = 1

    def _run_model(
        self,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        mm_inputs=None,
    ):
        capture_call = getattr(self, "_eagle3_capture_call", None)
        if capture_call is not None:
            provider = self._draft_window_graph_binding_provider
            bank = next(capture_call, None)
            if bank is None:
                raise RuntimeError("Eagle3 capture visited more consumers than its real descriptor")
            n = bank.descriptor.num_reqs
            attn_metadata = copy_draft_attention_metadata(attn_metadata, provider.group.layers)
            query = self.input_buffers.query_start_loc
            if query.dtype != torch.int32 or query.ndim != 1 or query.numel() < n + 1:
                raise RuntimeError("Eagle3 capture requires its owned int32 Full input query")
            layer = attn_metadata[provider.group.layers[0]]
            bind_draft_window_read(layer, bank)
            layer.causal = True
            # Use the owned Full query, not a builder's transient GPU copy.
            layer.query_start_loc = query[: n + 1]
        elif self._eagle3_window_is_active() and (
            self._draft_window_graph_binding_provider._active_bank is None
            or cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
        ):
            raise RuntimeError("Eagle3 Window native first requires its prepared Draft bank; no eager fallback")
        return super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

    def _multi_step_decode(
        self,
        num_reqs,
        skip_attn,
        batch_desc,
        num_tokens_across_dp,
        seq_lens_cpu_upper_bound=None,
        num_speculative_steps=None,
    ):
        if self._eagle3_window_is_active():
            provider = self._draft_window_graph_binding_provider
            padded = batch_desc.num_reqs or num_reqs
            if (
                self._eagle3_window_expected_steps == 1
                or self._eagle3_window_next_step != 1
                or num_reqs != self.input_batch.num_reqs
                or padded < num_reqs
                or batch_desc.cg_mode != CUDAGraphMode.FULL
                or skip_attn
                or num_speculative_steps not in (None, self.num_speculative_steps)
            ):
                raise RuntimeError("Eagle3 Window requires its actual static-K FULL AR descriptor; no eager fallback")
            # Rejection correction is already applied. Keep its sole D2H
            # snapshot local while staging all merged AR consumers.
            exact = self.input_buffers.seq_lens[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
            for step in range(1, self.num_speculative_steps):
                capture = provider.capture_banks_for(num_reqs=padded, num_tokens=batch_desc.num_tokens)[step - 1]
                self._materialize_eagle3_step(capture, step, exact.add(step - 1).clamp_(max=self.max_model_len))
                provider.bind_current(capture.descriptor)
                self._eagle3_window_next_step = step + 1
        return super()._multi_step_decode(
            num_reqs,
            skip_attn,
            batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )
