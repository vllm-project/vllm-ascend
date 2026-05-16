#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend._310p.block_table import MultiGroupBlockTable as MultiGroupBlockTable310
from vllm_ascend._310p.npu_input_batch import NPUInputBatch310 as NPUInputBatch
from vllm_ascend._310p.ops.rotary_embedding import prepare_mrope_cos_sin_slices_from_runner
from vllm_ascend._310p.sample.sampler import AscendSampler310
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, lmhead_tp_enable
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

_NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN = 1
_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


class NPUModelRunner310(NPUModelRunner):
    # Inherited from parent runner; annotated here to satisfy strict type checks.
    uniform_decode_query_len: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=self.input_batch.logitsprocs,
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )
        self._acl_format = ACL_FORMAT_FRACTAL_NZ
        self.sampler = AscendSampler310()
        if getattr(self, "rejection_sampler", None) is not None:
            self.rejection_sampler = RejectionSampler(self.sampler)
        if self.speculative_config is not None and self.speculative_config.method == "ngram":
            # 310P ngram requires decode-only graph shapes to be built with q_len=1.
            # Keep dispatcher's internal query_len in sync to avoid key-init assert.
            self.cudagraph_dispatcher.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN

    @contextmanager
    def temporary_modify_uniform_decode_query_len(self):
        # This is only needed for the 310P ngram path where dispatcher uses q_len=1
        # while runner's default uniform_decode_query_len remains 1 + num_spec_tokens.
        # TODO: remove this temporary override after upstream supports independent
        # decode capture query_len for backend-specific paths.
        if self.speculative_config is None or self.speculative_config.method != "ngram":
            yield
            return

        original_uniform_decode_query_len = self.uniform_decode_query_len
        self.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN
        try:
            yield
        finally:
            self.uniform_decode_query_len = original_uniform_decode_query_len

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ):
        if self.attn_state in (AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit):
            force_eager = True

        if force_uniform_decode is None and self.attn_state == AscendAttentionState.DecodeOnly:
            decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN
            if (
                max_num_scheduled_tokens == decode_query_len
                and num_tokens == max_num_scheduled_tokens * num_reqs
                and np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
            ):
                # Respect explicit caller override: only force when unset.
                force_uniform_decode = True

        return super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        # Keep this aligned with the dispatcher because batch_desc.num_reqs is
        # generated by dispatcher._create_padded_batch_descriptor().
        # For 310P ngram we intentionally set dispatcher q_len=1, while runner's
        # default uniform_decode_query_len may remain 1 + num_spec_tokens.
        uniform_decode_query_len = self.cudagraph_dispatcher.uniform_decode_query_len

        if num_tokens_padded == num_reqs_padded * uniform_decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = self.query_start_loc.np[num_reqs]
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        self.query_start_loc.copy_to_gpu()
        return num_reqs_padded

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
        """
        310P cannot use the Triton slot-mapping kernel or the generic NPU Add
        kernels used by the base runner for decode metadata. Keep those pieces
        on CPU and upload the prepared tensors.
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        self.input_batch.block_table.commit_block_table(num_reqs)

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = num_scheduled_tokens
        else:
            num_valid_tokens = np.array(
                [
                    scheduler_output.num_scheduled_tokens[i]
                    - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for i in self.input_batch.req_ids
                ],
                dtype=np.int32,
            )
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            self.query_pos.np[: cu_num_tokens[-1]],
            out=positions_np,
        )
        block_table = cast(MultiGroupBlockTable310, self.input_batch.block_table)
        block_table.compute_slot_mapping(
            req_indices,
            positions_np[:total_num_scheduled_tokens],
        )

        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
            )

        if self.speculative_config and self.use_cp:
            self.pcp_manager.generate_pcp_mtp_input(
                total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens,
                with_prefill,
                self.input_batch,
                self.arange_np,
                req_indices,
                positions_np,
                cu_num_tokens,
                self._draft_token_ids,  # type: ignore[has-type]
                scheduler_output,
                self.num_spec_tokens,
            )

        if self.pcp_size > 1:
            num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                num_scheduled_tokens[:num_reqs], self.arange_np
            )
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
            positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        if self.pcp_size > 1 and self.pcp_manager.pcp_use_hybrid_attn:
            assert self.pcp_manager.num_scheduled_tokens_padded is not None
            self.query_lens = torch.from_numpy(self.pcp_manager.num_scheduled_tokens_padded)
        else:
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
            )

        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                if self.pcp_size > 1:
                    req_positions_np = positions_np[output_idx : output_idx + num_sched]
                    dst_slice = self.inputs_embeds.cpu[output_idx : output_idx + num_sched]
                    self.pcp_manager.fill_prompt_embeds_for_pcp(
                        req_embeds=req_embeds,
                        req_positions_np=req_positions_np,
                        dst_slice=dst_slice,
                    )
                else:
                    start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                    if start_pos >= req_embeds.shape[0]:
                        output_idx += num_sched
                        continue

                    end_pos = start_pos + num_sched
                    actual_end = min(end_pos, req_embeds.shape[0])
                    actual_num_sched = actual_end - start_pos

                    if actual_num_sched > 0:
                        self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                            req_embeds[start_pos:actual_end]
                        )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
            self.gdn_query_start_loc.copy_to_gpu()

        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)

        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.gpu.copy_(
                self.mrope_positions.cpu,
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        tokens_original = None
        if self.pcp_size > 1:
            tokens_original = [scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids]
            original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
                tokens_original, dtype=np.int32
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[
                    np.where(new_mask, 0, prev_idx)
                ]
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = self.num_accepted_tokens.np[:num_reqs]
            else:
                self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        need_async_num_computed_update = (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        )
        if need_async_num_computed_update:
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                cpu_values,
            )
        else:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        self.positions[:total_num_scheduled_tokens].copy_(
            self._positions_cpu_buf[:total_num_scheduled_tokens],
            non_blocking=True,
        )
        if need_async_num_computed_update:
            self.seq_lens[:num_reqs] = self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        else:
            self.seq_lens[:num_reqs].copy_(
                self.optimistic_seq_lens_cpu[:num_reqs],
                non_blocking=True,
            )
        self.seq_lens[num_reqs:].fill_(0)

        if (
            self._needs_seq_lens_cpu_sync
            and self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            self.optimistic_seq_lens_cpu[:num_reqs].copy_(
                self.seq_lens[:num_reqs], non_blocking=True
            )
            if self._seq_lens_cpu_event is None:
                self._seq_lens_cpu_event = torch.npu.Event()
            self._seq_lens_cpu_event.record()
            self._seq_lens_cpu_event_pending = True
        else:
            self._seq_lens_cpu_event_pending = False

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            spec_decode_metadata = None
            num_draft_tokens = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
            if self.use_cp:
                logits_indices = self.pcp_manager.get_logits_indices(cu_num_tokens, num_reqs, tokens_original)
                logits_indices = logits_indices.pin_memory().to(self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (self.is_kv_consumer and req_id in new_schedule_reqs) or (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = draft_len
                else:
                    num_decode_draft_tokens[req_idx] = -1

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs] if self.pcp_size > 1 else None,
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        self.logits_indices = logits_indices

        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        if (
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            self.pcp_manager.cache_local_schedule_layout(
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=base_num_reqs,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
            )

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode=None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ):
        temporary_context = self.temporary_modify_uniform_decode_query_len() if uniform_decode else nullcontext()
        with temporary_context:
            return super()._dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention,
                uniform_decode=uniform_decode,
                is_profile=is_profile,
                create_mixed_batch=create_mixed_batch,
                allow_microbatching=allow_microbatching,
                skip_eplb=skip_eplb,
                remove_lora=remove_lora,
                is_graph_capturing=is_graph_capturing,
                num_active_loras=num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ):
        if self.uses_mrope:
            assert positions is not None
            prepare_mrope_cos_sin_slices_from_runner(self, positions)
        return super()._model_forward(
            num_tokens_padded,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends,
        kv_cache_groups,
    ) -> None:
        # 910B does not need this branch because runner/dispatcher query_len are
        # naturally consistent there. 310P ngram needs temporary alignment.
        with self.temporary_modify_uniform_decode_query_len():
            super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Override the base class method.
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # 310P limitation: KV transfer is not supported
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")
        if self.use_sparse:
            raise ValueError("Deepseek Sparse Attention is not supported for 310P.")
        if self.model_config.use_mla:
            raise ValueError("MLAAttention is not supported for 310P.")
        # Initialize the memory buffer for KV cache
        kv_caches = self._allocate_kv_cache_tensors(kv_cache_config)
        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache size. The buffer needs to be reshaped to the desired shape before being used by
        the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer.
        """
        # init kv cache tensors
        kv_cache: dict[str, list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]] = {}
        # get kv cache spec for each layer
        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            for layer_name in group_kv_cache_spec.layer_names:
                layer_kv_cache_spec[layer_name] = group_kv_cache_spec.kv_cache_spec
        # Allocate kv cache buffers according to the kv_cache_config and kv_cache_spec
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if layer_name in self.runner_only_attn_layers:
                    continue
                if "linear_attn" in layer_name and layer_name not in kv_cache:
                    cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(cache_spec, MambaSpec)
                    assert kv_cache_tensor.size % cache_spec.page_size_bytes == 0
                    num_blocks = kv_cache_tensor.size // cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks
                    raw_tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(cache_spec.shapes, cache_spec.dtypes):
                        target_shape = (num_blocks, *shape)
                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if "linear_attn" in layer_name_inner:
                            kv_cache[layer_name_inner] = state_tensors
                elif "attn" in layer_name and layer_name not in kv_cache:
                    kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(kv_cache_spec, AttentionSpec)
                    assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks
                    # Page attention operation on 310P limits block_size * head_size <= 128 * 128
                    supported_sizes = [
                        support_size
                        for support_size in self.attn_backend.get_supported_kernel_block_sizes()
                        if support_size * kv_cache_spec.head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
                    ]
                    if supported_sizes:
                        block_size = supported_sizes[0]
                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size
                        )
                    k_shape = kv_cache_shape[1:]
                    v_shape = k_shape
                    dtype = kv_cache_spec.dtype
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache[layer_name_inner] = (k_cache, v_cache)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache.keys()), "Some layers are not correctly initialized"
        return kv_cache

    # Override this function because of tensor.copy_(other) accuracy issue.
    # TODO: This override will be removed after tensor.copy_(other) accuracy issue is resolved.
    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the NPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(range(flattened_index - draft_len + 1, flattened_index + 1))
                start = prev_index * self.num_spec_tokens
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_common_tokens < total_without_spec:
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            return
        if indices_match and max_flattened_index == (num_common_tokens - 1):
            # NOTE: Override the copy_ function here
            indices = torch.arange(num_common_tokens, device=self.input_ids.gpu.device)
            source = self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0]
            self.input_ids.gpu.index_copy_(0, indices, source)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_common_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0],
        )
        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return
        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        self.kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                    backend = attn_groups[0].backend
                    # Page attention operation on 310P limits block_size * head_size <= 128 * 128
                    supported_sizes = [
                        support_size
                        for support_size in backend.get_supported_kernel_block_sizes()
                        if support_size * kv_cache_spec.head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
                    ]
                    kernel_block_size_list = supported_sizes if supported_sizes else [self.cache_config.block_size]
                except IndexError:
                    kernel_block_size_list = [self.cache_config.block_size]
                self.kernel_block_sizes.append(kernel_block_size_list)
            else:
                self.kernel_block_sizes.append([0])

        if block_sizes != [self.cache_config.block_size] or self.kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=self.kernel_block_sizes,
            )
