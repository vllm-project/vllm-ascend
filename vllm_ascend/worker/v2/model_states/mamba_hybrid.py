# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    preprocess_mamba_align_fused_kernel,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@dataclass
class AscendMambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class AscendMambaHybridModelState(AscendModelState):
    """Ascend v2 state for hybrid attention and Mamba models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )

        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.num_accepted_tokens_gpu[req_index] = 1
        if self._align_mode:
            self._mamba_state_idx_gpu[req_index] = (
                new_req_data.num_computed_tokens - 1
            ) // self.cache_config.block_size

    def _get_mamba_group_info(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            group_ids: list[int] = []
            specs: list[MambaSpec] = []
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, MambaSpec):
                    group_ids.append(group_id)
                    specs.append(group.kv_cache_spec)
            assert specs, "no mamba layers in the model"
            assert all(specs[0] == spec for spec in specs)
            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
        return self._mamba_group_ids, self._mamba_spec

    def _ensure_align_ctx(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> MambaSpecDecodeGPUContext:
        if self._mamba_ctx is None:
            copy_funcs = self.model.get_mamba_state_copy_func()
            if get_conv_copy_spec in copy_funcs and is_conv_state_dim_first():
                assert self.vllm_config.speculative_config is None, (
                    "DS conv state layout does not support mamba align state "
                    "copies with speculative decoding"
                )
            self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=kv_cache_config,
                num_state_types=len(copy_funcs),
                device=self.device,
                make_buffer=lambda n, dtype: CpuGpuBuffer(
                    n,
                    dtype=dtype,
                    device=self.device,
                ),
            )

        ctx = self._mamba_ctx
        if not ctx.is_initialized:
            ctx.initialize_from_forward_context(
                kv_cache_config,
                self.vllm_config.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[group_id] for group_id in mamba_group_ids],
            )
        return ctx

    def preprocess_state(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        if not self._align_mode or input_batch.num_reqs == 0:
            return

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(
            kv_cache_config,
            mamba_group_ids,
            block_tables,
        )
        block = 256
        grid = (triton.cdiv(input_batch.num_reqs, block),)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        ctx.run_fused_precopy(
            input_batch.num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = (
                self.num_accepted_tokens_gpu[input_batch.idx_mapping]
            )

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = (
                    input_batch.num_scheduled_tokens
                    == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(
                num_decode_draft_tokens_np
            )

        model_specific_metadata = AscendMambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_specific_metadata,
            for_cudagraph_capture=for_capture,
        )
        return self.attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if isinstance(num_sampled, int):
            self.num_accepted_tokens_gpu.index_fill_(
                0,
                idx_mapping,
                max(num_sampled, 1),
            )
        elif (num_reqs := idx_mapping.shape[0]) > 0:
            _scatter_num_accepted_kernel[(num_reqs,)](
                idx_mapping,
                num_sampled,
                self.num_accepted_tokens_gpu,
            )

        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
            and idx_mapping.shape[0] > 0
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                idx_mapping.shape[0],
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    num_accepted_ptr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))
