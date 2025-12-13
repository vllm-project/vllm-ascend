# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.worker.gpu.input_batch import (InputBatch,
                                            combine_sampled_and_draft_tokens,
                                            prepare_pos_seq_lens,
                                            prepare_prefill_inputs)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.aclgraph_utils import AclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

logger = init_logger(__name__)


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # release original object.
        del self.cudagraph_manager
        del self.req_states
        del self.input_buffers

        # NPU specific initializations can be added below.
        self.cudagraph_manager = AclGraphManager(vllm_config, device)
        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        self.req_states = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        self.input_buffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            inputs_embeds_size=self.inputs_embeds_size,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group(
        ).rank_in_group if self.pcp_size > 1 else 0

        self.attn_mask_builder = AttentionMaskBuilder(self.device)
        self.attn_mask = None
        self.spec_attn_mask = None
        self.attn_state = None

        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1

        # this is for async scheduling with speculative decoding.
        self.num_rejected_tokens_event = None
        self.num_rejectd_tokens_cpu = None
        self.num_rejected_token_stream = None
        self.req_ids: list[str] = []
        if self.use_async_scheduling and self.do_spec_decode:
            self.num_rejected_tokens_event = torch.npu.Event()
            self.num_rejected_token_stream = torch.npu.Stream()
            self.num_rejectd_tokens_cpu = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )

    def prepare_inputs(
        self,
        scheduler_output,
        num_tokens_after_padding,
    ) -> InputBatch:
        """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
        npu attention bakcends need seq_lens_cpu to work.
        so we need to prepare seq_lens_cpu here.
        """
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_reqs = len(scheduler_output.num_scheduled_tokens)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(
            scheduler_output.num_scheduled_tokens.keys(),
            key=lambda k: scheduler_output.num_scheduled_tokens[k],
        )

        # special handling for npu.
        self.req_ids = req_ids
        self._update_seq_lens_cpu(scheduler_output, req_ids)
        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids],
            dtype=np.int32)
        num_valid_tokens = num_scheduled_tokens
        if scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array([
                num_tokens -
                len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(num_scheduled_tokens, req_ids)
            ],
                                        dtype=np.int32)
        self.attn_state = self._build_attn_state(
            num_reqs,
            num_scheduled_tokens,
            num_valid_tokens,
        )
        self.attn_mask = self._make_attention_mask(self.attn_state)

        idx_mapping_list = [
            self.req_states.req_id_to_index[req_id] for req_id in req_ids
        ]
        idx_mapping = self.input_buffers.idx_mapping
        idx_mapping.np[:num_reqs] = idx_mapping_list
        idx_mapping_np = idx_mapping.np[:num_reqs]
        idx_mapping = idx_mapping.copy_to_gpu(num_reqs)

        # Get the number of draft tokens for each request.
        if not scheduler_output.scheduled_spec_decode_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits = torch.arange(num_reqs + 1,
                                         device=self.device,
                                         dtype=torch.int32)
        else:
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens = np.array(
                [
                    len(draft_tokens[req_id]) if req_id in draft_tokens else 0
                    for req_id in req_ids
                ],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            np.cumsum(
                num_draft_tokens + 1,
                out=self.input_buffers.cu_num_logits.np[1:num_reqs + 1],
            )
            cu_num_logits = self.input_buffers.cu_num_logits.copy_to_gpu(
                num_reqs + 1)

        # Block tables: num_kv_cache_groups x [num_reqs, max_num_blocks]
        block_tables = self.block_tables.gather_block_tables(idx_mapping)

        # Get query_start_loc.
        np.cumsum(
            num_scheduled_tokens,
            out=self.input_buffers.query_start_loc.np[1:num_reqs + 1],
        )
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        self.input_buffers.query_start_loc.np[num_reqs + 1:] = num_tokens
        self.input_buffers.query_start_loc.copy_to_gpu()
        query_start_loc_gpu = self.input_buffers.query_start_loc.gpu[:
                                                                     num_reqs +
                                                                     1]
        query_start_loc_cpu = self.input_buffers.query_start_loc.cpu[:
                                                                     num_reqs +
                                                                     1]
        query_start_loc_np = self.input_buffers.query_start_loc.np[:num_reqs +
                                                                   1]

        # Get prefill tokens.
        prepare_prefill_inputs(
            self.input_buffers.input_ids,
            self.req_states.next_prefill_tokens,
            idx_mapping,
            query_start_loc_gpu,
            self.req_states.prefill_token_ids.gpu,
            self.req_states.prefill_len.gpu,
            self.req_states.num_computed_tokens,
        )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc_gpu,
            self.req_states.num_computed_tokens,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc_gpu,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        # Compute slot mappings: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, self.input_buffers.positions[:num_tokens])

        # Layer name -> attention metadata.
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=self.input_buffers.seq_lens,
            seq_lens_cpu=self.input_buffers.seq_lens_cpu,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            num_computed_tokens_cpu=self.req_states.
            num_computed_tokens_cpu[idx_mapping],
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            decode_token_per_req=self.decode_token_per_req,
            attn_mask=self.attn_mask,
            spec_attn_mask=self.spec_attn_mask,
            attn_state=self.attn_state,
        )

        input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]
        positions = self.input_buffers.positions[:num_tokens_after_padding]
        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc_gpu,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_np=seq_lens_np,
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
        )

    def sample_tokens(self, grammar_output):
        with torch_cuda_wrapper():
            output = super().sample_tokens(grammar_output)
            self.num_rejected_tokens_event.synchronize()
            n_rejected = self.num_rejectd_tokens_cpu[:len(self.req_ids
                                                          )].tolist()
            for req_id, rejected in zip(self.req_ids, n_rejected):
                req_idx = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_idx] -= rejected
        return output

    def sample(
        self,
        hidden_states,
        input_batch,
        sampling_metadata,
        grammar_output,
    ):
        sampler_output, num_sampled, num_rejected = super().sample(
            hidden_states,
            input_batch,
            sampling_metadata,
            grammar_output,
        )
        if self.num_rejected_tokens_event is not None:
            # npu attention backend still need to use seq_lens_cpu,
            # when doing speculative decoding with async_scheduling,
            # we need to copy num_rejected_tokens back to cpu.
            default_stream = torch.cuda.current_stream()
            with torch.npu.stream(self.num_rejected_token_stream):
                self.num_rejected_token_stream.wait_stream(default_stream)
                self.num_rejectd_tokens_cpu.copy_(
                    num_rejected,
                    non_blocking=True,
                )
                self.num_rejected_tokens_event.record()
        return sampler_output, num_sampled, num_rejected

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[int],
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_computed_tokens in zip(
                scheduler_output.scheduled_cached_reqs.req_ids,
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            self.req_states.num_computed_tokens_cpu[
                req_index] = num_computed_tokens
        for i, req_id in enumerate(req_ids):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[
                req_index]
            self.input_buffers.seq_lens_cpu[
                i] = num_computed_token + num_scheduled_tokens[req_id]

    def _revert_num_computed_tokens_cpu(self):
        for req_id, req_index in self.req_states.req_id_to_index.items():
            self.req_states.num_computed_tokens_cpu[
                req_index] = self.req_states.num_computed_tokens[
                    req_index].item()

    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
                          num_valid_tokens):
        if self.model_config.runner_type == "pooling":
            if isinstance(
                    self.kv_cache_config.kv_cache_groups[0].kv_cache_spec,
                    EncoderOnlyAttentionSpec):
                attn_state = AscendAttentionState.PrefillNoCache
            else:
                attn_state = AscendAttentionState.PrefillCacheHit
        elif np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config and self.speculative_config.method == 'mtp':
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
        return attn_state

    def _make_attention_mask(
        self,
        attn_state: AscendAttentionState,
    ) -> torch.Tensor:
        # pcp situation.
        if self.attn_mask_builder is None:
            raise ValueError("Attn mask builder is None")
        # Pooling situation.
        if self.model_config.runner_type == "pooling":
            return self.attn_mask_builder.get_attn_mask(2048, torch.bool)

        if self.vllm_config.model_config.use_mla:
            if self.pcp_size > 1:
                return self.attn_mask_builder.get_pcp_mla_mask(self.dtype)
            # mla prefill
            if attn_state != AscendAttentionState.DecodeOnly:
                return self.attn_mask_builder.get_mla_mask(self.dtype)
        return self.attn_mask_builder.get_splitfuse_attn_mask()
