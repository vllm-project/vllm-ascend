# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.acl_graph import update_full_graph_params
from vllm_ascend.sample.rejection_sampler import strict_rejection_sample_tensor
from vllm_ascend.sample.sampler import sample_with_runtime_state
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import enable_sp, enable_sp_by_pass


class _FusedModelWithMTP:
    """Wraps the main model forward together with ALL MTP steps.

    Used as the ``runnable`` of :class:`ACLGraphWrapper` so that both the
    main-model forward **and** all N MTP speculative steps are captured
    into a single ACLGraph.  On replay the entire fused graph runs in one
    launch on the main stream — no separate graph or stream-sync for MTP.

    Attribute access is transparently delegated to ``raw_model`` so that
    call-sites like ``self.model.compute_logits(...)`` keep working.
    """

    def __init__(self, raw_model: nn.Module, drafter: "SpecDecodeBaseProposer"):
        self.raw_model = raw_model
        self.drafter = drafter
        # Keep graph replay ordering conservative: MTP graph params are updated
        # asynchronously before replay, and stale params directly hurt acceptance.
        self.skip_aclgraph_replay_sync = False
        self.capture_mtp_enabled = True
        num_spec_tokens = drafter.num_speculative_tokens
        # Fused full-graph replay is captured by token count. Keep the wrapper
        # request capacity aligned with the graph request capacity so MTP=1
        # batch16 uses exactly 32 sample rows, not an extra slack row.
        max_num_reqs = drafter.runner.max_num_reqs
        max_num_tokens = drafter.runner.max_num_tokens
        device = drafter.device
        self.logits_indices_buf = torch.zeros(max_num_reqs * (1 + num_spec_tokens), dtype=torch.int64, device=device)
        self.sample_logits_indices_buf = torch.zeros(
            max_num_reqs * (1 + num_spec_tokens), dtype=torch.int64, device=device
        )
        self.sample_idx_mapping_buf = torch.zeros(
            max_num_reqs * (1 + num_spec_tokens), dtype=torch.int32, device=device
        )
        self.bonus_row_indices_buf = torch.zeros((max_num_reqs,), dtype=torch.int64, device=device)
        self.target_row_indices_buf = torch.zeros((max_num_reqs * num_spec_tokens,), dtype=torch.int64, device=device)
        self.num_reqs_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_actual_tokens_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_sample_rows_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_target_rows_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.cu_num_draft_tokens_buf = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
        self.target_logits_indices_buf = torch.full(
            (max_num_reqs, num_spec_tokens), -1, dtype=torch.int64, device=device
        )
        self.spec_decode_token_ids_buf = torch.zeros((max_num_reqs, num_spec_tokens), dtype=torch.int64, device=device)
        self.draft_token_ids_flat_buf = torch.zeros((max_num_reqs * num_spec_tokens,), dtype=torch.int64, device=device)
        self.backup_next_token_ids_buf = torch.zeros((max_num_reqs,), dtype=torch.int64, device=device)
        self.discarded_req_mask_buf = torch.zeros((max_num_reqs,), dtype=torch.bool, device=device)
        self.sampled_token_ids_buf = torch.full(
            (max_num_reqs, num_spec_tokens + 1), -1, dtype=torch.int32, device=device
        )
        self.draft_token_ids_buf = torch.zeros((max_num_reqs, num_spec_tokens), dtype=torch.int64, device=device)
        self.next_token_ids_buf = torch.zeros((max_num_reqs,), dtype=torch.int64, device=device)
        self.valid_sampled_tokens_count_buf = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
        self.mtp_last_hidden_states_buf = torch.zeros(
            (max_num_tokens, drafter.hidden_size), dtype=drafter.dtype, device=device
        )

    def __getattr__(self, key: str):
        return getattr(self.raw_model, key)

    def _prepare_next_token_ids_from_sampled(self) -> torch.Tensor:
        valid_mask = self.sampled_token_ids_buf != -1
        valid_mask &= ~self.discarded_req_mask_buf[:, None]
        max_num_reqs = self.sampled_token_ids_buf.shape[0]
        req_indices = torch.arange(max_num_reqs, device=self.sampled_token_ids_buf.device, dtype=torch.int64)
        active_req_mask = req_indices < self.num_reqs_buf[0].to(torch.int64)

        valid_sampled_tokens_count = valid_mask.sum(dim=1, dtype=torch.int32)
        self.valid_sampled_tokens_count_buf.copy_(valid_sampled_tokens_count)

        last_valid_indices = torch.clamp(
            valid_sampled_tokens_count.to(torch.long) - 1,
            min=0,
        )
        selected_tokens = torch.gather(
            self.sampled_token_ids_buf,
            1,
            last_valid_indices.unsqueeze(1),
        ).squeeze(1)
        next_token_ids = torch.where(
            valid_sampled_tokens_count > 0,
            selected_tokens.to(torch.int64),
            self.backup_next_token_ids_buf,
        )
        dummy_tokens = torch.zeros_like(next_token_ids)
        next_token_ids = torch.where(
            active_req_mask,
            next_token_ids,
            dummy_tokens,
        )
        self.next_token_ids_buf.copy_(next_token_ids)
        return self.next_token_ids_buf

    def _get_draft_hidden_states(self, raw_hidden: torch.Tensor, num_tokens: int) -> torch.Tensor:
        get_target_hidden_states = getattr(self.raw_model, "get_mtp_target_hidden_states", None)
        if get_target_hidden_states is None:
            return raw_hidden

        draft_hidden_states = get_target_hidden_states()
        if draft_hidden_states is None:
            return raw_hidden

        return draft_hidden_states[:num_tokens]

    def _update_token_indices_to_sample(
        self,
        active_req_mask: torch.Tensor,
        draft_counts: torch.Tensor,
    ) -> None:
        valid_counts = self.valid_sampled_tokens_count_buf.to(torch.int64)
        sample_counts = torch.where(
            active_req_mask,
            draft_counts + 1,
            torch.zeros_like(draft_counts),
        )
        has_valid_tokens = active_req_mask & (valid_counts > 0)
        sample_row_indices = self.bonus_row_indices_buf - (sample_counts - valid_counts)
        sample_row_indices = sample_row_indices.clamp(
            min=0,
            max=self.sample_logits_indices_buf.shape[0] - 1,
        )
        current_token_positions = self.sample_logits_indices_buf[sample_row_indices]
        num_reqs = self.bonus_row_indices_buf.shape[0]
        self.logits_indices_buf[:num_reqs] = torch.where(
            has_valid_tokens,
            current_token_positions,
            self.logits_indices_buf[:num_reqs],
        )

    def _run_rejection_verifier(
        self,
        sample_logits: torch.Tensor,
        runtime_positions: torch.Tensor,
    ) -> torch.Tensor:
        sampling_metadata = self.drafter.runner.input_batch.sampling_metadata
        device = sample_logits.device
        max_num_reqs, max_num_sampled_tokens = self.sampled_token_ids_buf.shape
        max_spec_len = max_num_sampled_tokens - 1

        req_indices = torch.arange(max_num_reqs, device=device, dtype=torch.int64)
        num_reqs = self.num_reqs_buf[0].to(torch.int64)
        active_req_mask = req_indices < num_reqs

        cu_num_draft_tokens = self.cu_num_draft_tokens_buf.to(torch.int64)
        cu_start = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=device),
                cu_num_draft_tokens[:-1],
            ]
        )
        draft_counts = torch.where(
            active_req_mask,
            cu_num_draft_tokens - cu_start,
            torch.zeros_like(cu_num_draft_tokens),
        )
        all_greedy = getattr(
            sampling_metadata,
            "all_greedy",
            sampling_metadata.temperature is None,
        )
        if all_greedy:
            sampled_row_token_ids = sample_logits.argmax(dim=-1)
        else:
            sample_positions = runtime_positions[self.sample_logits_indices_buf]
            sample_idx_mapping = self.sample_idx_mapping_buf
            sampled_row_token_ids = sample_with_runtime_state(
                sample_logits.to(torch.float32),
                sample_idx_mapping,
                sample_positions,
                sampling_metadata.temperature,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
                getattr(sampling_metadata, "seeds", None),
                all_greedy,
                getattr(sampling_metadata, "all_random", False),
            )
        target_token_ids = sampled_row_token_ids[self.target_row_indices_buf]
        bonus_token_ids = sampled_row_token_ids[self.bonus_row_indices_buf]
        target_token_ids = target_token_ids.to(torch.int32)
        bonus_token_ids = bonus_token_ids.to(torch.int32)
        if max_spec_len == 1:
            # Keep the hot MTP=1 graph path minimal; larger tensor construction
            # here has shown fragile ACL-graph behavior on NPU.
            draft_token_ids = self.spec_decode_token_ids_buf[:, 0].to(torch.int32)
            placeholder_token_ids = torch.full_like(target_token_ids, -1)
            has_draft_mask = active_req_mask & (draft_counts > 0)
            first_token_ids = torch.where(
                has_draft_mask,
                target_token_ids,
                bonus_token_ids,
            )
            accepted_bonus_mask = has_draft_mask & (draft_token_ids == target_token_ids)
            self.sampled_token_ids_buf[:, 0].copy_(
                torch.where(
                    active_req_mask,
                    first_token_ids,
                    placeholder_token_ids,
                )
            )
            self.sampled_token_ids_buf[:, 1].copy_(
                torch.where(
                    accepted_bonus_mask,
                    bonus_token_ids,
                    placeholder_token_ids,
                )
            )
        else:
            draft_token_ids = self.draft_token_ids_flat_buf.to(torch.int32)
            placeholder_req = torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device)

            # Keep graph replay off the generic Triton rejection kernel; its
            # cumulative-count boundary path is fragile under ACL graph.
            token_offsets = torch.arange(max_spec_len, device=device, dtype=torch.int64)
            flat_indices = (cu_start.unsqueeze(1) + token_offsets.unsqueeze(0)).clamp(
                max=max_num_reqs * max_spec_len - 1
            )
            target_token_matrix = target_token_ids[flat_indices]
            draft_token_matrix = draft_token_ids[flat_indices]

            prefix_accepted = active_req_mask
            sampled_columns = []
            for draft_idx in range(max_spec_len):
                has_draft = draft_counts > draft_idx
                emit_target = prefix_accepted & has_draft
                target_column = target_token_matrix[:, draft_idx]
                draft_column = draft_token_matrix[:, draft_idx]
                sampled_columns.append(
                    torch.where(
                        emit_target,
                        target_column,
                        placeholder_req,
                    )
                )
                prefix_accepted = prefix_accepted & (~has_draft | (draft_column == target_column))

            sampled_without_bonus = torch.cat(
                [
                    torch.stack(sampled_columns, dim=1),
                    placeholder_req.unsqueeze(1),
                ],
                dim=1,
            )
            output_columns = torch.arange(max_spec_len + 1, device=device, dtype=torch.int64)
            bonus_column = draft_counts.clamp(min=0, max=max_spec_len)
            bonus_mask = prefix_accepted.unsqueeze(1) & (output_columns.unsqueeze(0) == bonus_column.unsqueeze(1))
            sampled_token_ids = torch.where(
                bonus_mask,
                bonus_token_ids.unsqueeze(1),
                sampled_without_bonus,
            )
            self.sampled_token_ids_buf.copy_(sampled_token_ids)

        next_token_ids = self._prepare_next_token_ids_from_sampled()
        self._update_token_indices_to_sample(active_req_mask, draft_counts)
        return next_token_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.raw_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        forward_context = get_forward_context()
        is_capturing = getattr(forward_context, "capturing", False)
        if is_capturing and self.capture_mtp_enabled and not isinstance(hidden_states, IntermediateTensors):
            raw_hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
            if getattr(forward_context, "flash_comm_v1_enabled", False):
                from vllm.distributed import tensor_model_parallel_all_gather

                raw_hidden = tensor_model_parallel_all_gather(raw_hidden, 0)
                pad_size = getattr(forward_context, "pad_size", 0)
                if pad_size > 0:
                    raw_hidden = raw_hidden[:-pad_size, :]
            runtime_positions = positions[0] if positions.ndim > 1 else positions
            num_tokens = input_ids.shape[0]
            num_spec = self.drafter.num_speculative_tokens
            batch_size = max(num_tokens // (num_spec + 1), 1)
            sample_hs = raw_hidden[self.sample_logits_indices_buf]
            sample_logits = self.raw_model.compute_logits(sample_hs)
            next_token_ids = self._run_rejection_verifier(
                sample_logits,
                runtime_positions,
            )
            draft_hidden_states = self._get_draft_hidden_states(raw_hidden, num_tokens)
            all_draft_ids = self.drafter.propose_all_in_graph(
                hidden_states=draft_hidden_states,
                input_ids=input_ids,
                positions=positions,
                logits_indices=self.logits_indices_buf,
                next_token_ids=next_token_ids[:batch_size],
                num_tokens=num_tokens,
                num_reqs=self.num_reqs_buf,
                num_actual_tokens=self.num_actual_tokens_buf,
            )
            num_reqs = all_draft_ids.shape[0]
            self.draft_token_ids_buf[:num_reqs, : all_draft_ids.shape[1]].copy_(all_draft_ids)
        return hidden_states


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    """Ascend-specific Eagle/MTP proposer combining EagleProposer interface
    with AscendSpecDecodeBaseProposer common logic, plus MTP-specific methods.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(
            self, vllm_config, device, pass_hidden_states_to_model=True, runner=runner
        )

    # ------------------------------------------------------------------
    # MTP-specific methods
    # ------------------------------------------------------------------

    def _draft_graph_capture_enabled(self) -> bool:
        return self.use_cuda_graph or getattr(self, "fused_with_main_graph", False)

    def build_graph_capture_attn_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> list[dict]:
        multi_steps_attn_metadata: list[dict] = []
        if not self._draft_graph_capture_enabled():
            return multi_steps_attn_metadata
        if aclgraph_runtime_mode != CUDAGraphMode.FULL:
            return multi_steps_attn_metadata
        if len(self.draft_attn_groups) == 0:
            return multi_steps_attn_metadata

        num_reqs = self._prepare_graph_capture_query_start_loc(num_tokens, num_reqs)
        num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
            seq_lens_cpu=self.runner.seq_lens.cpu,
            seq_lens=self.runner.seq_lens.gpu[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            max_query_len=self.num_speculative_tokens + 1,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=(self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs]),
            slot_mapping=(self.runner.input_batch.block_table[0].slot_mapping.gpu),
            positions=self.runner.positions.gpu,
            positions_cpu=self.runner.positions.cpu,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            max_seq_len=0,
        )
        if self.pcp_size * self.dcp_size > 1:
            common_attn_metadata.prefill_context_parallel_metadata = self.runner.pcp_manager.long_seq_metadata
            common_attn_metadata.block_table_tensor = self.runner.input_batch.block_table[0].get_device_tensor()[
                : num_reqs * self.decode_threshold
            ]

        builder = self.draft_attn_groups[0].get_metadata_builder()
        for draft_step in range(self.num_speculative_tokens):
            common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
            common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]
            attn_metadata_eagle = builder.build_for_graph_capture(
                common_attn_metadata,
                AscendAttentionState.SpecDecoding if self.method == "mtp" else AscendAttentionState.ChunkedPrefill,
            )
            per_layer_attn_metadata = dict()
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata_eagle
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        return multi_steps_attn_metadata

    def _uses_mtp_full_graph_padding(
        self,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> bool:
        return self.method == "mtp" and aclgraph_runtime_mode == CUDAGraphMode.FULL

    def _get_mtp_eager_padded_num_input_tokens(
        self,
        num_input_tokens: int,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> int:
        if self.method != "mtp" or aclgraph_runtime_mode != CUDAGraphMode.NONE:
            return num_input_tokens

        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", 1)
        try:
            tp_size = int(tp_size)
        except (TypeError, ValueError):
            return num_input_tokens

        if tp_size <= 1 or num_input_tokens <= 0:
            return num_input_tokens

        # MTP eager may see very small decode batches after warmup. Keep its
        # token extent consistent with graph capture/SP padding so TP kernels
        # never receive a zero-token shard.
        if num_input_tokens < tp_size or enable_sp(self.vllm_config) or enable_sp_by_pass():
            return round_up(num_input_tokens, tp_size)
        return num_input_tokens

    def _uses_mtp_eager_input_padding(
        self,
        num_tokens: int,
        num_input_tokens: int,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> bool:
        return self.method == "mtp" and aclgraph_runtime_mode == CUDAGraphMode.NONE and num_input_tokens > num_tokens

    def _graph_padding_slot_id(
        self,
        aclgraph_runtime_mode: CUDAGraphMode,
    ) -> int:
        if self._uses_mtp_full_graph_padding(aclgraph_runtime_mode):
            return 0
        return PADDING_SLOT_ID

    def _pad_mtp_full_graph_inputs(
        self,
        num_tokens: int,
        num_input_tokens: int,
        aclgraph_runtime_mode: CUDAGraphMode,
        force: bool = False,
    ) -> None:
        if not force and not self._uses_mtp_full_graph_padding(aclgraph_runtime_mode):
            return
        if num_input_tokens <= num_tokens:
            return

        start = int(num_tokens)
        end = int(num_input_tokens)
        self.input_ids[start:end].zero_()
        if self.uses_mrope:
            self.mrope_positions[:, start:end].zero_()
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, start:end].zero_()
        else:
            self.positions[start:end].zero_()

        self.hidden_states[start:end].zero_()
        for slot_mapping in getattr(self, "slot_mapping_group", ()):
            slot_mapping[start:end].fill_(0)

    def _prepare_graph_capture_query_start_loc(
        self,
        num_tokens: int,
        num_reqs: int,
    ) -> int:
        """Mirror the original MTP _propose padding on drafter buffers.

        The fused full-graph path must not derive MTP request padding from
        ``num_tokens // decode_threshold``. The eager/reference MTP path keeps
        the real requests and appends one dummy request spanning graph padding.
        Using a different request layout changes draft attention context and
        can lower acceptance.
        """
        actual_num_reqs = num_reqs
        self.query_start_loc.cpu[: actual_num_reqs + 1].copy_(self.runner.query_start_loc.cpu[: actual_num_reqs + 1])

        uniform_decode_query_len = getattr(self.runner, "uniform_decode_query_len", self.decode_threshold)
        if not isinstance(uniform_decode_query_len, int):
            uniform_decode_query_len = self.decode_threshold
        if num_tokens != actual_num_reqs * uniform_decode_query_len:
            self.query_start_loc.cpu[actual_num_reqs + 1] = num_tokens
            num_reqs = actual_num_reqs + 1

        self.query_start_loc.copy_to_gpu()
        return num_reqs

    def propose_all_in_graph(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_tokens: int,
        num_reqs: torch.Tensor | None = None,
        num_actual_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ALL MTP steps inside the main model's ACLGraph.

        Called by :class:`_FusedModelWithMTP` so that main model forward
        plus all N MTP steps are recorded/replayed as a **single** graph.
        Returns ``draft_token_ids`` of shape ``(batch_size, num_speculative_tokens)``.
        """
        batch_size = max(num_tokens // (self.num_speculative_tokens + 1), 1)
        raw_model = self.get_model()
        next_token_ids = next_token_ids.to(self.input_ids.dtype)
        if num_actual_tokens is None:
            actual_num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=self.device)
        else:
            actual_num_tokens = num_actual_tokens[0].to(torch.int32)
        if hasattr(self, "arange"):
            token_offsets = self.arange[:num_tokens].to(torch.int32)
        else:
            token_offsets = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        active_token_mask = token_offsets < actual_num_tokens
        if num_reqs is None:
            decode_query_len = self.num_speculative_tokens + 1
            active_batch_size = torch.div(
                actual_num_tokens.to(torch.int64) + decode_query_len - 1,
                decode_query_len,
                rounding_mode="floor",
            )
        else:
            active_batch_size = num_reqs[0].to(torch.int64)
        active_batch_size = torch.clamp(active_batch_size, min=0, max=batch_size)
        if hasattr(self, "arange"):
            batch_offsets = self.arange[:batch_size].to(torch.int64)
        else:
            batch_offsets = torch.arange(batch_size, dtype=torch.int64, device=self.device)
        active_batch_mask = batch_offsets < active_batch_size
        raw_logits_indices = logits_indices[:batch_size].to(torch.int64)
        safe_logits_indices = torch.where(
            active_batch_mask,
            raw_logits_indices,
            torch.zeros_like(raw_logits_indices),
        )

        def select_draft_token_ids(logits: torch.Tensor) -> torch.Tensor:
            return logits.argmax(dim=-1)

        current_input_ids = self.input_ids[:num_tokens]
        shifted_input_ids = current_input_ids.clone()
        shifted_input_ids[: num_tokens - 1].copy_(input_ids[1:num_tokens])
        shift_mask = token_offsets < torch.clamp(actual_num_tokens - 1, min=0)
        current_input_ids.copy_(torch.where(shift_mask, shifted_input_ids, torch.zeros_like(current_input_ids)))
        next_token_ids = next_token_ids[:batch_size]
        token_offsets_i64 = token_offsets.to(torch.int64)
        bonus_write_mask = (
            raw_logits_indices.unsqueeze(1) == token_offsets_i64.unsqueeze(0)
        ) & active_batch_mask.unsqueeze(1)
        bonus_update_mask = bonus_write_mask.any(dim=0)
        bonus_values = (bonus_write_mask.to(next_token_ids.dtype) * next_token_ids.unsqueeze(1)).sum(dim=0)
        current_input_ids.copy_(
            torch.where(bonus_update_mask, bonus_values.to(current_input_ids.dtype), current_input_ids)
        )

        current_positions = self._get_positions(num_tokens)
        if positions.ndim > 1:
            incoming_positions = positions[:, :num_tokens]
            position_mask = active_token_mask.unsqueeze(0)
        else:
            incoming_positions = positions[:num_tokens]
            position_mask = active_token_mask
        self._set_positions(
            num_tokens,
            torch.where(position_mask, incoming_positions, torch.zeros_like(current_positions)),
        )

        current_hidden_states = self.hidden_states[:num_tokens]
        current_hidden_states.copy_(
            torch.where(
                active_token_mask.unsqueeze(-1), hidden_states[:num_tokens], torch.zeros_like(current_hidden_states)
            )
        )

        forward_context = get_forward_context() if is_forward_context_available() else None

        draft_attn_metadatas = getattr(forward_context, "draft_attn_metadatas", None)

        @contextmanager
        def draft_forward_context(attn_metadata=None):
            if forward_context is None:
                yield
                return

            saved_attn_metadata = getattr(forward_context, "attn_metadata", None)
            saved_num_tokens = getattr(forward_context, "num_tokens", None)
            saved_num_accept_tokens = getattr(forward_context, "num_accept_tokens", None)
            saved_is_draft_model = getattr(forward_context, "is_draft_model", False)
            saved_moe_layer_index = getattr(forward_context, "moe_layer_index", 0)

            forward_context.num_tokens = num_tokens
            forward_context.num_accept_tokens = batch_size
            forward_context.is_draft_model = True
            forward_context.moe_layer_index = 0
            forward_context.attn_metadata = attn_metadata
            try:
                yield
            finally:
                forward_context.attn_metadata = saved_attn_metadata
                if saved_num_tokens is not None:
                    forward_context.num_tokens = saved_num_tokens
                if saved_num_accept_tokens is not None:
                    forward_context.num_accept_tokens = saved_num_accept_tokens
                forward_context.is_draft_model = saved_is_draft_model
                forward_context.moe_layer_index = saved_moe_layer_index

        first_step_attn_metadata = draft_attn_metadatas[0] if draft_attn_metadatas else None

        with draft_forward_context(first_step_attn_metadata):
            model_input_ids = self.input_ids[:num_tokens]
            model_positions = self._get_positions(num_tokens)
            model_hidden_states = self.hidden_states[:num_tokens]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(model_hidden_states, model_positions)

            model_kwargs: dict[str, torch.Tensor] = {
                "input_ids": model_input_ids,
                "positions": model_positions,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = model_hidden_states
                if self.method == "mtp":
                    model_kwargs["positions"] = model_positions

            ret_hidden_states = raw_model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states_out = last_hidden_states
            else:
                last_hidden_states, hidden_states_out = ret_hidden_states

            last_hidden_states, model_positions, hidden_states_out = self.maybe_all_gather_and_unpad(
                last_hidden_states, model_positions, hidden_states_out
            )

            sample_hs = last_hidden_states[safe_logits_indices]
            logits = raw_model.compute_logits(sample_hs)
            draft_token_ids = select_draft_token_ids(logits)

        if self.num_speculative_tokens == 1:
            return draft_token_ids.view(-1, 1)

        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *draft_token_ids.shape), dtype=draft_token_ids.dtype, device=self.device
        )
        draft_token_ids_tensor[0] = draft_token_ids

        step_positions = self.positions[safe_logits_indices]
        step_hidden_states = hidden_states_out[safe_logits_indices]
        token_indices_to_sample = self.arange[:batch_size]

        for draft_step in range(self.num_speculative_tokens - 1):
            step_input_ids = draft_token_ids_tensor[draft_step].to(self.input_ids.dtype)
            step_positions = step_positions + 1

            exceeds_max_model_len = step_positions >= self.vllm_config.model_config.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, step_positions)

            self.input_ids[:batch_size] = step_input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = step_hidden_states

            model_input_ids = self.input_ids[:num_tokens]
            model_positions = self._get_positions(num_tokens)
            model_hidden_states = self.hidden_states[:num_tokens]
            step_attn_metadata = (
                draft_attn_metadatas[draft_step + 1]
                if draft_attn_metadatas and draft_step + 1 < len(draft_attn_metadatas)
                else None
            )

            with draft_forward_context(step_attn_metadata):
                model_hidden_states, model_positions = self.maybe_pad_and_reduce(model_hidden_states, model_positions)

                model_kwargs = {
                    "input_ids": model_input_ids,
                    "positions": model_positions,
                }
                if self.pass_hidden_states_to_model:
                    model_kwargs["hidden_states"] = model_hidden_states

                ret_hidden_states = raw_model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states_out = last_hidden_states
                else:
                    last_hidden_states, hidden_states_out = ret_hidden_states

                last_hidden_states, model_positions, hidden_states_out = self.maybe_all_gather_and_unpad(
                    last_hidden_states, model_positions, hidden_states_out
                )

                sample_hs = last_hidden_states[token_indices_to_sample]
                logits = raw_model.compute_logits(sample_hs)
                draft_token_ids = select_draft_token_ids(logits)
            draft_token_ids_tensor[draft_step + 1] = draft_token_ids
            step_hidden_states = hidden_states_out[:batch_size]

        return draft_token_ids_tensor.swapaxes(0, 1)


# Backward-compatible symbol export for older tests/call-sites that import
# SpecDecodeBaseProposer from this module and expect MTP-specific methods.
SpecDecodeBaseProposer = AscendEagleProposer
