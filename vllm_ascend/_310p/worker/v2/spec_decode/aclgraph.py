# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P draft ACLGraph manager: SpecDecoding capture + seq_lens refresh (no FIA).

K=1 draft-prefill FULL reuses target batch shape with ``q_len = 1+K = 2``.
Eager draft uses target SpecDecoding (splitfuse) metadata; if capture tags
``DecodeOnly`` via ``AscendInputBatch.make_dummy``, the graph records PA and
replay yields bad drafts (accept ~55% with intact final accuracy after
rejection). Mirror target ``ModelAclGraphManager310``: force SpecDecoding when
``num_tokens // num_reqs > 1`` during draft-prefill capture.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import (
    AutoRegressiveAclGraphManager,
)


class AutoRegressiveAclGraphManager310(AutoRegressiveAclGraphManager):
    """310P draft FULL: SpecDecoding capture + in-place seq_lens refresh.

    310P attention has no FIA ``graph_task``; mainline
    ``update_full_graph_params`` is a no-op. Capture must match runtime
    SpecDecoding/splitfuse; replay refreshes capture-stable ``seq_lens``.
    """

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
        # Draft-decode (q_len=1 / K>1 multi-step) needs host CPU slot_mapping
        # (D2H). NPU GLOBAL capture forbids sync memcpy → skip decode graphs;
        # runtime stays on the eager ``_multi_step_decode`` path.
        if not self.is_draft_model_prefill:
            logger.info(
                "Skipping 310P draft-decode ACLGraph capture "
                "(CPU slot_mapping D2H incompatible with NPU graph capture)."
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

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens = desc.num_tokens
        if self.is_draft_model_prefill:
            logger.info_once(
                "AutoRegressiveAclGraphManager310: draft prefill run_fullgraph with num_tokens=%s",
                num_tokens,
            )
        else:
            logger.info_once(
                "AutoRegressiveAclGraphManager310: draft run_fullgraph with num_tokens=%s",
                num_tokens,
            )

        # Ensure H2D into capture-stable buffers is visible before replay.
        torch.npu.current_stream().synchronize()
        ms = self.speculator.model_state
        runtime_seq_lens = self.speculator.target_input_buffers.seq_lens
        refresh = getattr(ms, "_refresh_capture_seq_lens", None)
        if callable(refresh):
            refresh(runtime_seq_lens)
        return CudaGraphManager.run_fullgraph(self, desc)
