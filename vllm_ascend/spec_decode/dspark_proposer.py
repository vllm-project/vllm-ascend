from typing import Any

import torch
from copy import deepcopy

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid,
)
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class AscendDsparkProposer(AscendDflashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )

        #   Static buffers for graph mode.
        #   DSpark returns K draft tokens, but internally keeps seed + K tokens.
        blk = 1 + self.num_speculative_tokens
        self._dspark_seed_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self._dspark_draft_buffer = torch.zeros(
            (self.max_batch_size, blk),
            dtype=torch.int64,
            device=device,
        )
        self._dspark_confidence_logits_buffer = torch.zeros(
            (self.max_batch_size, self.num_speculative_tokens),
            dtype=torch.float32,
            device=device,
        )
        self._dspark_num_verify_tokens_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self._keep_lens = torch.zeros(
            (self.max_batch_size,),
            dtype=torch.int32,
            device=self.device,
        )
        self._dspark_updates = torch.ones(
            (self.max_batch_size * self.num_speculative_tokens,),
            dtype=torch.int32,
            device=self.device,
        )

    def initialize_cudagraph_keys(
        self,
        cudagraph_mode: CUDAGraphMode,
    ) -> None:
        """Initialize independent FULL graph keys for the DSpark drafter.
        Target width: K + 1
        DSpark width: K
        """
        dispatcher = self.cudagraph_dispatcher
        draft_query_len = int(self.num_speculative_tokens)

        #   Do not reuse the target model's CompilationConfig object directly.
        #   The target dispatcher needs [K+1, 2(K+1), ...], while DSpark needs [K, 2K, ...].
        draft_compilation_config = deepcopy(dispatcher.compilation_config)

        max_num_reqs = int(
            dispatcher.vllm_config.scheduler_config.max_num_seqs
        )

        target_max_capture_size = int(
            dispatcher.compilation_config.max_cudagraph_capture_size
        )

        draft_capture_sizes = [
            batch_size * draft_query_len
            for batch_size in range(1, max_num_reqs + 1)
            if batch_size * draft_query_len <= target_max_capture_size
        ]

        if not draft_capture_sizes:
            raise RuntimeError(
                "No valid DSpark cudagraph capture sizes: "
                f"draft_query_len={draft_query_len}, "
                f"max_num_reqs={max_num_reqs}, "
                f"max_capture_size={target_max_capture_size}"
            )

        draft_compilation_config.cudagraph_capture_sizes = draft_capture_sizes
        draft_compilation_config.max_cudagraph_capture_size = (
            draft_capture_sizes[-1]
        )

        #   avoid validation against target compile sizes such as 8, 16, ...
        if draft_compilation_config.compile_sizes:
            draft_compilation_config.compile_sizes = list(
                draft_capture_sizes
            )

        #   this dispatcher now owns a separate config from the target dispatcher.
        dispatcher.compilation_config = draft_compilation_config

        #    critical: _create_padded_batch_descriptor() reads this attribute,
        #    not the initialize_cudagraph_keys() argument.
        dispatcher.uniform_decode_query_len = draft_query_len

        #    robust if initialization is invoked more than once.
        for keys in dispatcher.cudagraph_keys.values():
            keys.clear()
        dispatcher.keys_initialized = False

        if self.speculative_config.enforce_eager:
            dspark_cudagraph_mode = CUDAGraphMode.NONE
        else:
            #   preserve FULL_DECODE_ONLY / PIECEWISE_AND_FULL semantics.
            #   do not replace this with cudagraph_mode.mixed_mode(), because
            #   doing so would discard the separate FULL decode routine.
            dspark_cudagraph_mode = cudagraph_mode

        dispatcher.initialize_cudagraph_keys(
            dspark_cudagraph_mode,
            uniform_decode_query_len=draft_query_len,
        )

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        #   DSpark: context width = K + 1 from target hidden states, query width   = K from draft query block

        batch_size = cad.num_reqs
        num_query_per_req = self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req
        num_context = target_token_ids.shape[0]

        #   store target hidden states as DSpark/DFlash context KV source.
        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        #   static seed buffer for graph replay.
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        if n < self._dspark_seed_buffer.shape[0]:
            self._dspark_seed_buffer[n:].fill_(0)

        #   use static token_indices buffer.
        token_indices_to_sample = self.token_indices_to_sample[:num_query_total]

        has_num_rejected = num_rejected_tokens_gpu is not None

        copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            context_slot_mapping_ptr=cad.slot_mapping,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            seq_lens_ptr=cad.seq_lens,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.kernel_block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            batch_size=batch_size,
            HAS_NUM_REJECTED=has_num_rejected,
            SAMPLE_FROM_ANCHOR=True,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = (
            self.arange_dflash[: batch_size + 1] * num_query_per_req
        )

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
            * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size

        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        dspark_num_context_tokens=None,
        drafter_context_tokens=None,
        **kwargs,
    ) -> None:
        #   DSpark graph capture:
        #   num_tokens / num_input_tokens = query tokens = B*K
        #   dspark_num_context_tokens     = context tokens = B*(K+1)

        #   Backward compatibility: support either kwarg name.
        #   Do NOT overwrite the real named argument with kwargs.pop().
        if dspark_num_context_tokens is None:
            dspark_num_context_tokens = drafter_context_tokens

        if dspark_num_context_tokens is None:
            dspark_num_context_tokens = kwargs.pop(
                "dspark_num_context_tokens",
                None,
            )
        else:
            kwargs.pop("dspark_num_context_tokens", None)

        num_query_per_req = self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req
        num_query_tokens = min(
            num_query_total if num_reqs > 0 else num_tokens,
            self.max_query_tokens,
        )

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_query_tokens,
            is_draft_model=True,
        )

        if dspark_num_context_tokens is None:
            dspark_num_context_tokens = num_input_tokens

        dspark_num_context_tokens = int(dspark_num_context_tokens)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[
            :dspark_num_context_tokens
        ]
        context_states = self.hidden_states[:dspark_num_context_tokens]

        #   Keep same invariant as runtime build_model_inputs_first_pass()
        self._dflash_num_context = dspark_num_context_tokens

        multi_steps_attn_metadata = []

        if (
            aclgraph_runtime_mode == CUDAGraphMode.FULL
            and len(self.draft_attn_groups) > 0
        ):
            builder = self.draft_attn_groups[0].get_metadata_builder()

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=(
                    self.arange_dflash[: num_reqs + 1] * num_query_per_req
                ),
                query_start_loc_cpu=(
                    torch.from_numpy(
                        self.token_arange_np[: num_reqs + 1]
                    ).clone()
                    * num_query_per_req
                ).to(torch.int32),
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                seq_lens=self.runner.seq_lens[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_input_tokens,
                max_query_len=num_query_per_req,
                max_seq_len=0,
                slot_mapping=self._slot_mapping_buffer[:num_query_total],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=False,
                is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
                block_table_tensor=(
                    self.runner.input_batch.block_table[self.kv_cache_gid]
                    .get_device_tensor()[:num_reqs]
                ),
            )

            attn_metadata_dspark = builder.build_for_graph_capture(
                common_attn_metadata,
                AscendAttentionState.ChunkedPrefill,
            )

            attn_metadata_dspark.attn_mask = None
            attn_metadata_dspark.attn_state = AscendAttentionState.ChunkedPrefill

            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata_dspark

            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        self.token_indices_to_sample.fill_(0)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = dspark_num_context_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=context_positions,
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if (
                forward_context is not None
                and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not _EXTRA_CTX.capturing
            ):
                self._update_full_graph_params(
                    forward_context,
                    num_query_tokens,
                    multi_steps_attn_metadata,
                )