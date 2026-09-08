# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P ACLGraph helpers for MRv2 MTP FULL_DECODE_ONLY.

Target verify on 310P maps MTP uniform batches (q_len = 1+K) to SpecDecoding
(splitfuse), not DecodeOnly (PA). Upstream ``AscendInputBatch.make_dummy``
always tags DecodeOnly, so FULL capture would record the wrong attention path
and replay would diverge from runtime SpecDecoding. Wrap capture only on 310P.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class ModelAclGraphManager310(ModelAclGraphManager):
    """310P target ACLGraph manager: MTP capture uses SpecDecoding metadata."""

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        pcp_manager: Any = None,
    ) -> None:
        speculative = self.vllm_config.speculative_config is not None
        if not speculative or not self.cudagraph_mode.has_full_cudagraphs():
            return super().capture(
                model,
                model_state,
                input_buffers,
                intermediate_tensors,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora=has_lora,
                use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                lora_capture_hook=lora_capture_hook,
                progress_bar_desc=progress_bar_desc,
                pcp_manager=pcp_manager,
            )

        # Lazy import: attention_v1 pulls DeviceOperator and circularizes at module load.
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        orig_make_dummy = AscendInputBatch.make_dummy

        @classmethod
        def make_dummy_mtp(
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
            # Uniform MTP verify: q_len = 1+K (>1). Decode-only graphs stay PA.
            if num_reqs > 0 and (num_tokens // num_reqs) > 1:
                batch.attn_state = AscendAttentionState.SpecDecoding
            return batch

        AscendInputBatch.make_dummy = make_dummy_mtp  # type: ignore[method-assign]
        try:
            return super().capture(
                model,
                model_state,
                input_buffers,
                intermediate_tensors,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora=has_lora,
                use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                lora_capture_hook=lora_capture_hook,
                progress_bar_desc=progress_bar_desc,
                pcp_manager=pcp_manager,
            )
        finally:
            AscendInputBatch.make_dummy = orig_make_dummy  # type: ignore[method-assign]
