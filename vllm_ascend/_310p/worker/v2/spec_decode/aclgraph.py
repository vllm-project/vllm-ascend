# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P draft ACLGraph manager: SpecDecoding capture + per-step decode FULL.

K=1 draft-prefill FULL reuses target batch shape with ``q_len = 1+K = 2``.
Eager draft uses target SpecDecoding (splitfuse) metadata; if capture tags
``DecodeOnly`` via ``AscendInputBatch.make_dummy``, the graph records PA and
replay yields bad drafts (accept ~55% with intact final accuracy after
rejection). Mirror target ``ModelAclGraphManager310``: force SpecDecoding when
``num_tokens // num_reqs > 1`` during draft-prefill capture.

K>1 draft-decode cannot record CPU slot_mapping D2H/H2D inside an NPU GLOBAL
graph. Align with MRv1: capture a **single** decode step, recompute slots on
the host between steps, then ``run_fullgraph`` per step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.worker.v2.spec_utils import set_draft_step_host
from vllm_ascend.worker.v2.aclgraph_utils import model_capture_wrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import (
    AutoRegressiveAclGraphManager,
)
from vllm_ascend.worker.v2.utils import communicator_switch


class AutoRegressiveAclGraphManager310(AutoRegressiveAclGraphManager):
    """310P draft FULL: SpecDecoding prefill + per-step decode graphs."""

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        if not self.is_draft_model_prefill:
            self._capture_decode_per_step(
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                progress_bar_desc=progress_bar_desc,
            )
            return

        if not self.cudagraph_mode.has_full_cudagraphs():
            return super().capture(
                forward_fn,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                progress_bar_desc=progress_bar_desc,
            )

        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        orig_make_dummy = AscendInputBatch.make_dummy

        @classmethod
        def make_dummy_draft_prefill(
            cls,
            num_reqs: int,
            num_tokens: int,
            input_buffers_arg: Any,
            max_query_len: int | None = None,
        ) -> AscendInputBatch:
            kwargs: dict[str, Any] = {}
            if max_query_len is not None:
                kwargs["max_query_len"] = max_query_len
            batch = orig_make_dummy(num_reqs, num_tokens, input_buffers_arg, **kwargs)
            if num_reqs > 0 and (num_tokens // num_reqs) > 1:
                batch.attn_state = AscendAttentionState.SpecDecoding
            return batch

        AscendInputBatch.make_dummy = make_dummy_draft_prefill  # type: ignore[method-assign]
        try:
            logger.info(
                "Capturing 310P draft-prefill FULL with SpecDecoding make_dummy "
                "(q_len>1 → splitfuse, not DecodeOnly/PA)."
            )
            return super().capture(
                forward_fn,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                progress_bar_desc=progress_bar_desc,
            )
        finally:
            AscendInputBatch.make_dummy = orig_make_dummy  # type: ignore[method-assign]

    def _capture_decode_per_step(
        self,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str,
    ) -> None:
        """Capture one draft-decode step (q_len=1); runtime loops with host slots."""
        logger.info(
            "Capturing 310P draft-decode FULL as per-step graphs "
            "(CPU slot_mapping updated between steps, outside capture)."
        )

        with communicator_switch(), model_capture_wrapper(self.speculator, False):

            def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
                del warmup
                num_tokens = desc.num_tokens
                num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
                num_tokens_across_dp = (
                    torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                    if self.dp_size > 1
                    else None
                )
                # Runs outside torch.cuda.graph / acl capture — D2H/H2D allowed.
                prepare_inputs_to_capture(
                    num_reqs,
                    num_tokens,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    full_cudagraph=(desc.cg_mode == CUDAGraphMode.FULL),
                )
                seq_ub = input_buffers.seq_lens_cpu[:num_reqs]
                # Stable device slot buffer; content refreshed before each replay.
                slot_tensor = self.speculator.block_tables.get_dummy_slot_mappings(num_tokens)
                slot_by_layer = build_slot_mappings_by_layer(slot_tensor, self.speculator.kv_cache_config)
                attn_metadata = self.speculator._build_draft_attn_metadata(  # type: ignore[attr-defined]
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs,
                    num_tokens_padded=num_tokens,
                    seq_lens_cpu_upper_bound=seq_ub,
                    step=1,
                )
                # Move capture-stable attn tensors to NPU before graph begin
                # (pageable H2D inside capture is banned).
                device = self.speculator.device
                if attn_metadata is not None:
                    for meta in attn_metadata.values():
                        if meta is None:
                            continue
                        seq_lens = getattr(meta, "seq_lens", None)
                        if seq_lens is not None and seq_lens.device != device:
                            meta.seq_lens = seq_lens.to(device=device, non_blocking=False)
                torch.npu.current_stream().synchronize()
                self.speculator.current_draft_step.fill_(1)
                set_draft_step_host(1)

                def run(cg_mode: CUDAGraphMode) -> None:
                    del cg_mode
                    # Record model forward + sample + device-only draft write.
                    # Do NOT call update_draft_inputs_cpu / attn metadata H2D
                    # (pageable memcpy is banned under NPU GLOBAL capture).
                    self.speculator._prepare_eplb_forward(num_reqs)
                    last_hidden_states, _hidden_states = self.speculator._run_model(
                        num_tokens,
                        attn_metadata,
                        slot_by_layer,
                        num_tokens_across_dp,
                        CUDAGraphMode.NONE,
                    )
                    last_hidden_states = last_hidden_states[:num_reqs]
                    positions = self.speculator.input_buffers.positions[:num_reqs]
                    idx_mapping = self.speculator.idx_mapping[:num_reqs]
                    draft_tokens = self.speculator.sample_draft(
                        last_hidden_states,
                        positions,
                        idx_mapping,
                        self.speculator.temperature,
                        self.speculator.seeds,
                        self.speculator.current_draft_step,
                        self.speculator.draft_logits,
                    )
                    # Persist sampled drafts into the capture-stable buffer.
                    self.speculator.draft_tokens[:num_reqs, 1].copy_(draft_tokens[:num_reqs])

                return run

            CudaGraphManager.capture(self, create_forward_fn, progress_bar_desc=progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens = desc.num_tokens
        if self.is_draft_model_prefill:
            logger.info_once(
                "AutoRegressiveAclGraphManager310: draft prefill run_fullgraph with num_tokens=%s",
                num_tokens,
            )
        else:
            logger.info_once(
                "AutoRegressiveAclGraphManager310: draft decode per-step run_fullgraph with num_tokens=%s",
                num_tokens,
            )

        # Ensure H2D into capture-stable buffers is visible before replay.
        torch.npu.current_stream().synchronize()
        ms = self.speculator.model_state
        runtime_seq_lens = self.speculator.input_buffers.seq_lens
        refresh = getattr(ms, "_refresh_capture_seq_lens", None)
        if callable(refresh):
            refresh(runtime_seq_lens)

        if self.is_draft_model_prefill:
            return super().run_fullgraph(desc)

        # Per-step decode: parent builds multi-step FIA metadatas; 310P has no
        # FIA graph_task for that path. Replay the captured single-step graph
        # after host-side slot/seq updates (done by the speculator loop).
        pending_attn = getattr(self.speculator, "_pending_draft_attn_metadata", None)
        if pending_attn is not None:
            ms.attn_metadata = pending_attn
        return CudaGraphManager.run_fullgraph(self, desc)
