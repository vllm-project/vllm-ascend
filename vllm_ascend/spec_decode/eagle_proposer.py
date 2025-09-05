# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config,set_current_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.utils import is_pin_memory_available
from vllm.model_executor.model_loader import get_model,get_model_loader
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading, set_default_torch_dtype)
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.models.deepseek_mtp import CustomDeepSeekMTP
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm_ascend.torchair.torchair_attention import AscendTorchairMetadata
from vllm_ascend.torchair.torchair_mla import AscendMLATorchairMetadata
from vllm_ascend.torchair.utils import TorchairCommonAttentionMetadata
from vllm_ascend.utils import ProfileExecuteDuration, lmhead_tp_enable

PADDING_SLOT_ID = -1


class EagleProposer(Proposer):

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner=None):
        if vllm_config.speculative_config.method == "eagle":
            self.name = SpecDcodeType.EAGLE
        elif vllm_config.speculative_config.method == "eagle3":
            self.name = SpecDcodeType.EAGLE3
        else:
            self.name = SpecDcodeType.MTP
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

        self.block_size = vllm_config.cache_config.block_size
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size(
        )

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device)
        self.positions = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device)
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.token_arange_np = np.arange(self.max_num_tokens)
        self.hidden_states = torch.zeros(
            (self.vllm_config.scheduler_config.max_num_batched_tokens,
             self.hidden_size),
            dtype=self.vllm_config.model_config.dtype,
            device=device) if self.name == SpecDcodeType.EAGLE3 else torch.zeros(
            (self.runner.max_num_tokens,
             vllm_config.model_config.get_hidden_size()),
            dtype=self.runner.dtype,
            device=self.device)
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs +
                                   1,
                                   device=device,
                                   dtype=torch.int32)
        attn_mask_len = min(self.vllm_config.model_config.max_model_len,
                            int(os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)))
        self.attn_mask_builder = AttentionMaskBuilder(
            attn_mask_len, self.vllm_config.model_config.dtype)

    def load_model(self, model: nn.Module) -> None:
        if self.name != SpecDcodeType.MTP:
            target_attn_layer_names = set(
                get_layers_from_vllm_config(self.vllm_config, Attention).keys())

            self.model = get_model(vllm_config=self.vllm_config,
                                   model_config=self.vllm_config.speculative_config.draft_model_config)

            draft_attn_layer_names = (
                get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
                target_attn_layer_names)

            self.attn_layer_names = list(draft_attn_layer_names)
            self.attn_layer_name = next(iter(draft_attn_layer_names))

            # share embed_tokens with the target model if needed
            if get_pp_group().world_size == 1:
                logger.info(
                    "The EAGLE head shares the same vocab embedding" \
                    " with the target model."
                )
                self.model.model.embed_tokens = model.model.embed_tokens
            else:
                logger.info(
                    "Since PP > 1, the EAGLE head loaded its own vocab embedding" \
                    " weights instead of sharing them with the target model."
                )
        else:
            loader = get_model_loader(self.vllm_config.load_config)

            target_attn_layer_names = set(
                get_layers_from_vllm_config(self.vllm_config, Attention).keys())
            draft_model_config = \
                self.vllm_config.speculative_config.draft_model_config
            target_device = self.vllm_config.device_config.device

            with set_default_torch_dtype(
                    draft_model_config.dtype), set_current_vllm_config(
                self.vllm_config):
                self.model = CustomDeepSeekMTP(
                    vllm_config=self.vllm_config).to(target_device)

            draft_attn_layer_names = (
                    get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
                    target_attn_layer_names)

            assert len(draft_attn_layer_names) == 1
            self.attn_layer_name = next(iter(draft_attn_layer_names))

            self.model.load_weights(
                loader.get_all_weights(
                    self.vllm_config.speculative_config.draft_model_config,
                    self.model))
            process_weights_after_loading(self.model, draft_model_config,
                                          target_device)
        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.name == SpecDcodeType.EAGLE and hasattr(model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            if supports_multimodal(model):
                self.model.lm_head = model.get_language_model().lm_head
            else:
                self.model.lm_head = model.lm_head

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp: Optional[torch.Tensor] = None):
        if self.name == SpecDcodeType.MTP:
            # TODO: adapt enable_dbo later
            (num_tokens, num_tokens_across_dp, with_prefill,
             _) = self.runner._sync_metadata_across_dp(
                 num_tokens, with_prefill, False)
            if skip_attn:
                attn_metadata = None
            else:
                common_attn_metadata = TorchairCommonAttentionMetadata(
                    num_reqs=num_reqs,
                    num_actual_tokens=1,
                    actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                    attn_mask=self.runner.attn_mask,
                    spec_attn_mask=self.runner.spec_attn_mask,
                    decode_token_per_req=self.runner.decode_token_per_req,
                )
                attn_metadata = self.runner.attn_metadata_builder.build_torchair_graph_dummy(
                    common_attn_metadata)

            input_ids = self.input_ids[:num_tokens]
            positions = self.positions[:num_tokens]
            previous_hidden_states = self.hidden_states[:num_tokens]
            for _ in range(self.num_speculative_tokens):
                with set_ascend_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_tokens,
                        with_prefill=with_prefill,
                        num_tokens_across_dp=num_tokens_across_dp,
                        reserved_mc2_mask=self.runner.reserved_mc2_mask,
                        in_profile_run=self.runner.in_profile_run,
                        num_actual_tokens=0):
                    self.model(input_ids=input_ids,
                               positions=positions,
                               previous_hidden_states=previous_hidden_states)
                if with_prefill:
                    break
        else:
            with set_ascend_forward_context(None,
                                        self.vllm_config,
                                        num_tokens=num_tokens):
                self.model(
                    input_ids=self.input_ids[:num_tokens],
                    positions=self.positions[:num_tokens],
                    hidden_states=self.hidden_states[:num_tokens],
            )

    def generate_token_ids(self,
                           valid_sampled_token_ids: list[list[int]],
                           sampling_metadata: SamplingMetadata = None,
                           scheduler_output: SchedulerOutput = None,
                           spec_decode_metadata: SpecDecodeMetadata = None,
                           positions: torch.Tensor = None,
                           num_scheduled_tokens: int = 0,
                           hidden_states: torch.Tensor = None,
                           attn_metadata=None,
                           aux_hidden_states: torch.Tensor = None):
        if self.name == SpecDcodeType.EAGLE:
            raise NotImplementedError("Eagle Is Not Supported Yet.")

        attn_metadata = self._get_eagle_atten_dict(scheduler_output) if self.name == SpecDcodeType.EAGLE3 else attn_metadata
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(valid_sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = self.runner.input_batch.req_ids[i]
                req_state = self.runner.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])

                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.device)
        eagle_attn_metadata = attn_metadata[self.attn_layer_name] if self.name == SpecDcodeType.EAGLE3 else attn_metadata
        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.runner.input_ids[:num_scheduled_tokens]
            target_positions = positions[:num_scheduled_tokens]
            if self.name == SpecDcodeType.EAGLE3:
                target_hidden_states = torch.cat(
                    [h[:num_scheduled_tokens] for h in aux_hidden_states],
                    dim=-1)
            else:
                target_hidden_states = hidden_states[:num_scheduled_tokens]
            target_slot_mapping = eagle_attn_metadata.slot_mapping
            cu_num_tokens = eagle_attn_metadata.query_start_loc
        else:
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            num_rejected_tokens = [
                n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]
            num_rejected_tokens = torch.tensor(
                num_rejected_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            num_rejected_tokens_cpu = num_rejected_tokens.cpu()
            cu_num_tokens, token_indices = self._prepare_inputs(eagle_attn_metadata,num_rejected_tokens_cpu)
            target_token_ids = self.runner.input_ids[token_indices]
            target_positions = positions[token_indices]
            if self.name == SpecDcodeType.EAGLE3:
                target_hidden_states = torch.cat(
                    [h[token_indices] for h in aux_hidden_states], dim=-1)
            else:
                target_hidden_states = hidden_states[token_indices.to(self.device)]
            target_slot_mapping = eagle_attn_metadata.slot_mapping[
                token_indices]

        draft_token_ids = self._propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            target_slot_mapping=target_slot_mapping,
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=eagle_attn_metadata.block_tables,
        )
        spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    def _get_eagle_atten_dict(
        self,
        scheduler_output: "SchedulerOutput",
    ):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.runner.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.runner.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.runner.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)
        self.runner.query_lens = torch.from_numpy(num_scheduled_tokens)
        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.runner.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.runner.positions_np[:total_num_scheduled_tokens]
        np.add(self.runner.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.runner.uses_mrope:
            self.runner._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np +
            req_indices * self.runner.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.runner.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=self.runner.input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        # NOTE(Chen): there is exactly one KV cache group that contains all
        # attetnion layers in the model for now, so the current logic for
        # getting attn_metadata is not related to kv_cache_group information.
        # Will extend this part to support multiple KV cache groups later.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.runner.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table = self.runner.input_batch.block_table[
                kv_cache_group_id]
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_num_blocks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size`
            # here because M (max_model_len) is not necessarily divisible by
            # block_size.
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions_np // block_size)
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.runner.query_start_loc_np[0] = 0
        self.runner.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.runner.seq_lens_np[:num_reqs] = (
            self.runner.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self.runner.input_ids[:total_num_scheduled_tokens].copy_(
            self.runner.input_ids_cpu[:total_num_scheduled_tokens],
            non_blocking=True)
        if self.runner.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.runner.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.runner.
                mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.runner.positions[:total_num_scheduled_tokens].copy_(
                self.runner.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        self.runner.query_start_loc[:num_reqs + 1].copy_(
            self.runner.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.runner.seq_lens[:num_reqs].copy_(
            self.runner.seq_lens_cpu[:num_reqs], non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.runner.seq_lens[num_reqs:].fill_(0)
        self.runner.query_start_loc[num_reqs + 1:].fill_(-1)

        attn_metadata = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.runner.kv_cache_config.kv_cache_groups):
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.runner.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.runner.query_start_loc_cpu[:num_reqs +
                                                                    1],
                seq_lens_cpu=self.runner.seq_lens_cpu,
                num_reqs=num_reqs,
                max_query_len=max_num_scheduled_tokens,
                num_actual_tokens=total_num_scheduled_tokens,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].
                get_device_tensor(),
                slot_mapping_cpu=self.runner.slot_mapping_cpu,
                positions=self.runner.positions,
                attn_mask=self.runner.attn_mask,
                spec_attn_mask=self.runner.spec_attn_mask,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
            )
            attn_metadata_i = self.runner.attn_metadata_builder.build(
                common_attn_metadata, self.runner.get_model())
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.runner.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _propose(
            self,
            # [num_tokens]
            target_token_ids: torch.Tensor,
            # [num_tokens]
            target_positions: torch.Tensor,
            # [num_tokens, hidden_size]
            target_hidden_states: torch.Tensor,
            # [num_tokens]
            target_slot_mapping: torch.Tensor,
            # [batch_size]
            next_token_ids: torch.Tensor,
            # [batch_size + 1] starting with 0
            cu_num_tokens: torch.Tensor,
            # [batch_size, max_num_blocks_per_req]
            block_table: torch.Tensor,
    ) -> torch.Tensor:
        device = cu_num_tokens.device
        cu_num_tokens = cu_num_tokens.cpu()
        block_table = block_table.cpu()
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1
        if self.name == SpecDcodeType.EAGLE3:
            target_positions = target_positions.cpu()
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        seq_lens = target_positions[last_token_indices] + 1
        seq_lens = seq_lens.int()
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.runner.query_start_loc[
                            :batch_size + 1] if self.name != SpecDcodeType.MTP else cu_num_tokens[:batch_size + 1],
            query_start_loc_cpu=self.runner.query_start_loc_cpu[:batch_size +
                                                                1] if self.name != SpecDcodeType.MTP else cu_num_tokens[
                                                                                                        :batch_size + 1].cpu(),
            seq_lens_cpu=self.runner.seq_lens_cpu if self.name != SpecDcodeType.MTP else seq_lens.cpu(),
            max_query_len=max_query_len,
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=self.runner.input_batch.block_table[0].
            get_device_tensor(),
            slot_mapping_cpu=target_slot_mapping,
            positions=target_positions,
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            graph_pad_size=self.runner.graph_pad_size,
            decode_token_per_req=self.runner.decode_token_per_req,
        )
        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        attn_metadata = self.runner.attn_metadata_builder.build(
            common_attn_metadata, self.runner.model)
        if self.use_cuda_graph and \
                num_tokens <= self.cudagraph_batch_sizes[-1] and self.name != SpecDcodeType.MTP:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions.to(device)
        self.hidden_states[:num_tokens] = target_hidden_states
        if self.name == SpecDcodeType.MTP:
            (num_input_tokens, num_tokens_across_dp, with_prefill,
            _) = self.runner._sync_metadata_across_dp(
                num_tokens, self.runner.with_prefill, False)
            attn_metadata.slot_mapping = target_slot_mapping
            for step in range(self.num_speculative_tokens):
                with set_ascend_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_input_tokens,
                        with_prefill=with_prefill,
                        num_tokens_across_dp=num_tokens_across_dp,
                        reserved_mc2_mask=self.runner.reserved_mc2_mask,
                        in_profile_run=self.runner.in_profile_run,
                        num_actual_tokens=num_tokens):
                    with ProfileExecuteDuration().capture_async('mtp_forward'):
                        model_kwargs = {}
                        model_kwargs["attn_metadata"] = attn_metadata
                        hidden_states = self.model(
                            input_ids=self.input_ids[:num_input_tokens],
                            positions=self.positions[:num_input_tokens],
                            previous_hidden_states=self.
                                                hidden_states[:num_input_tokens],
                            kv_caches=self.runner.kv_caches[-1:])

                num_indices = last_token_indices.shape[0]
                if lmhead_tp_enable():
                    if not self.runner.with_prefill:
                        max_num_reqs_across_dp = num_input_tokens
                    else:
                        max_num_reqs_across_dp = self.vllm_config.scheduler_config.max_num_seqs
                    last_token_indices = nn.functional.pad(
                        last_token_indices,
                        (0, max_num_reqs_across_dp - num_indices))

                sample_hidden_states = hidden_states[last_token_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
                if lmhead_tp_enable() and num_indices < logits.shape[0]:
                    logits = logits[:num_indices]
                draft_token_ids = logits.argmax(dim=-1)

                if self.num_speculative_tokens == 1:
                    # [batch_size, 1]
                    return draft_token_ids.view(-1, 1)

                if step == 0:
                    draft_token_ids_list = [draft_token_ids]
                else:
                    draft_token_ids_list.append(draft_token_ids)

                # prepare next mtp inputs
                # mtp>1: prefill skip or decode skip last loop
                if with_prefill:
                    for _ in range(self.num_speculative_tokens - 1):
                        draft_token_ids_list.append(draft_token_ids)
                if step == self.num_speculative_tokens - 1 or with_prefill:
                    break

                if step == 0:
                    positions = target_positions[last_token_indices]
                    hidden_states = hidden_states[last_token_indices]
                    slot_mapping = attn_metadata.slot_mapping[last_token_indices]
                    attn_metadata.slot_mapping.fill_(-1)
                    attn_metadata.query_start_loc = self.arange[:batch_size + 1]
                    last_token_indices = self.arange[:batch_size]
                    if attn_metadata.num_decode_tokens != 0:
                        attn_metadata.num_decode_tokens = batch_size

                input_ids = draft_token_ids_list[-1].int()
                positions += 1

                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length. Since it is complex
                # to remove such requests from the batch, we keep them in the batch
                # but adjust the position ids and slot mappings to avoid the
                # out-of-range access during the model execution. The draft tokens
                # generated with this adjustment should be ignored.
                exceeds_max_model_len = positions >= self.runner.model_config.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(exceeds_max_model_len, 0,
                                                positions)
                # Increment the sequence lengths.
                attn_metadata.seq_lens[:batch_size] += 1
                # For the requests that exceed the max model length, we set the
                # sequence length to 1 to minimize their overheads in attention.
                exceeds_max_model_len_cpu = exceeds_max_model_len.to(
                    attn_metadata.seq_lens.device, non_blocking=True)
                attn_metadata.seq_lens[:batch_size].masked_fill_(
                    exceeds_max_model_len_cpu, 1)
                # Mask out the slot mappings that exceed the max model length.
                # Otherwise, the KV cache will be inadvertently updated with the
                # padding tokens.
                slot_mapping += 1
                slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

                # copy inputs to buffer for cudagraph
                self.input_ids[:batch_size] = input_ids
                self.positions[:batch_size] = clamped_positions
                self.hidden_states[:batch_size] = hidden_states
                attn_metadata.slot_mapping[:batch_size] = slot_mapping

                if attn_metadata.prefill is not None:
                    attn_metadata.prefill.seq_lens = attn_metadata.seq_lens
                    attn_metadata.prefill.context_lens = attn_metadata.seq_lens
                    attn_metadata.prefill.input_positions = self.positions[:
                                                                        num_input_tokens]
                    attn_metadata.prefill.max_seq_lens += 1
                    attn_metadata.prefill.max_seq_lens = min(
                        attn_metadata.prefill.max_seq_lens,
                        self.runner.model_config.max_model_len)
                if attn_metadata.decode is not None:
                    attn_metadata.decode.seq_lens = attn_metadata.seq_lens
                    attn_metadata.decode.input_positions = self.positions[:
                                                                        num_input_tokens]
                    attn_metadata.decode.max_seq_lens += 1
                    attn_metadata.decode.max_seq_lens = min(
                        attn_metadata.decode.max_seq_lens,
                        self.runner.model_config.max_model_len)

            # mtp>1: [batch_size, k]
            draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        else:
            attn_metadata.block_tables = block_table.to(device)
            with set_ascend_forward_context(attn_metadata,
                                        self.vllm_config,
                                        num_tokens=num_input_tokens):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                )
            sample_hidden_states = last_hidden_states[last_token_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
            draft_token_ids = logits.argmax(dim=-1)

            # Early exit if there is only one draft token to be generated.
            if self.vllm_config.speculative_config.num_speculative_tokens == 1:
                # [batch_size, 1]
                return draft_token_ids.view(-1, 1)

            # Generate the remaining draft tokens.
            draft_token_ids_tensor = torch.zeros(
                (self.vllm_config.speculative_config.num_speculative_tokens,
                *draft_token_ids.shape),
                dtype=draft_token_ids.dtype)
            draft_token_ids_tensor[0] = draft_token_ids

            positions_cpu = target_positions[last_token_indices].cpu().to(
                torch.int64)
            hidden_states = hidden_states[last_token_indices]
            if self.use_cuda_graph and \
                batch_size <= self.cudagraph_batch_sizes[-1]:
                input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
            else:
                input_batch_size = batch_size
            attn_metadata.num_actual_tokens = batch_size
            attn_metadata.max_query_len = 1
            attn_metadata.query_start_loc = self.arange[:batch_size + 1]

            if self.vllm_config.speculative_config.num_speculative_tokens > 2:
                raise ValueError("Speculative tokens > 2 are not supported yet.")

            attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
            for now_speculative in range(
                    self.vllm_config.speculative_config.num_speculative_tokens -
                    1):
                # Update the inputs.
                # cast to int32 is crucial when eagle model is compiled.
                # tensor.argmax() returns int64 by default.
                input_ids = draft_token_ids_tensor[now_speculative].to(device)
                positions_cpu += 1

                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length. Since it is complex
                # to remove such requests from the batch, we keep them in the batch
                # but adjust the position ids and slot mappings to avoid the
                # out-of-range access during the model execution. The draft tokens
                # generated with this adjustment should be ignored.
                exceeds_max_model_len = positions_cpu >= self.vllm_config.model_config.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions_cpu = torch.where(exceeds_max_model_len, 0,
                                                    positions_cpu)
                clamped_positions = clamped_positions_cpu.to(device)

                # TODO: Increment the sequence lengths.

                attn_metadata.seq_lens += 1
                # TODO: Consider max model length.
                # attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                #                                 self.max_model_len)
                # For the requests that exceed the max model length, we set the
                # TODO: sequence length to 1 to minimize their overheads in attention.

                # Compute the slot mapping.
                block_numbers = (clamped_positions_cpu // self.block_size)
                block_ids = block_table.gather(dim=1,
                                            index=block_numbers.view(-1, 1))
                block_ids = block_ids.view(-1)
                slot_mapping_cpu = (
                    block_ids * self.vllm_config.cache_config.block_size +
                    clamped_positions_cpu % self.block_size)

                # Mask out the slot mappings that exceed the max model length.
                # Otherwise, the KV cache will be inadvertently updated with the
                # padding tokens.
                slot_mapping_cpu.masked_fill_(exceeds_max_model_len,
                                            PADDING_SLOT_ID)
                # NOTE: ASCEND slot_mapping must on cpu
                attn_metadata.slot_mapping = slot_mapping_cpu.to(
                    torch.int32).to(device)
                # copy inputs to buffer for cudagraph
                self.input_ids[:batch_size] = input_ids
                self.positions[:batch_size] = clamped_positions
                self.hidden_states[:batch_size] = hidden_states
                attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask(
                    attn_metadata.seq_lens, positions_cpu,
                    self.vllm_config.model_config.dtype, self.device)
                attn_metadata.attn_mask = attn_mask
                attn_metadata.block_tables = block_table.to(device)
                # Run the model.
                with set_ascend_forward_context(attn_metadata,
                                                self.vllm_config,
                                                num_tokens=input_batch_size):
                    last_hidden_states, hidden_states = self.model(
                        input_ids=self.input_ids[:input_batch_size],
                        positions=self.positions[:input_batch_size],
                        hidden_states=self.hidden_states[:input_batch_size],
                    )
                hidden_states = hidden_states[:batch_size]
                logits = self.model.compute_logits(last_hidden_states[:batch_size],
                                                None)
                # TODO(wenlong): get more than one token for tree attention
                draft_token_ids = logits.argmax(dim=-1)

                draft_token_ids_tensor[now_speculative + 1] = draft_token_ids.cpu()
            # [batch_size, num_speculative_tokens]
            draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
        return draft_token_ids


    def _prepare_inputs(
        self,
        common_attn_metadata:Union[AscendMetadata, AscendMLAMetadata,
                             AscendTorchairMetadata,
                             AscendMLATorchairMetadata],
        # [batch_size]
        num_rejected_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc.cpu()
        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = (query_start_loc_cpu[1:] -
                                 query_start_loc_cpu[:-1])
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available())
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1],
                                                  new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = self.token_arange_np[:total_num_tokens] \
            - new_query_start_locs_expanded

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(
            device, non_blocking=True)

        return new_query_start_loc_cpu.to(device,non_blocking=True), token_indices