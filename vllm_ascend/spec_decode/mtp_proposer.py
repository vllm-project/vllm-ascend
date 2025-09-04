import types

import torch
import torch.nn as nn
import torchair
import vllm.envs as envs_vllm
from torchair import patch_for_hcom
from vllm.attention.layer import Attention
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading, set_default_torch_dtype)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.models.deepseek_mtp import CustomDeepSeekMTP
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm_ascend.torchair.models.torchair_deepseek_mtp import \
    TorchairDeepSeekMTP
from vllm_ascend.torchair.utils import (TORCHAIR_CACHE_DIR,
                                        TorchairCommonAttentionMetadata)
from vllm_ascend.utils import ProfileExecuteDuration, lmhead_tp_enable

PADDING_SLOT_ID = -1


class MtpProposer(Proposer):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device,
        runner,
    ):
        self.name = SpecDcodeType.MTP
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

        # persistent buffers for graph
        self.input_ids = torch.zeros(self.runner.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.runner.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.hidden_states = torch.zeros(
            (self.runner.max_num_tokens,
             vllm_config.model_config.get_hidden_size()),
            dtype=self.runner.dtype,
            device=self.device)
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        self.torchair_graph_enabled = get_ascend_config(
        ).torchair_graph_config.enabled
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs +
                                   1,
                                   device=self.runner.device,
                                   dtype=torch.int32)

    def load_model(self, model) -> None:
        loader = get_model_loader(self.vllm_config.load_config)

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_device = self.vllm_config.device_config.device

        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            if self.torchair_graph_enabled:
                self.model = TorchairDeepSeekMTP(
                    vllm_config=self.vllm_config).to(target_device)
            else:
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

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp=None) -> None:
        if not self.torchair_graph_enabled:
            # TODO: adapt enable_dbo later
            (num_tokens, num_tokens_across_dp, with_prefill,
             _) = self.runner._sync_metadata_across_dp(num_tokens,
                                                       with_prefill, False)
        is_running_torchair = self.torchair_graph_enabled and \
            not with_prefill

        if is_running_torchair:
            skip_attn = False
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
                if is_running_torchair:
                    assert attn_metadata is not None
                    torch._dynamo.mark_static(input_ids)
                    torch._dynamo.mark_static(positions)
                    torch._dynamo.mark_static(previous_hidden_states)
                    torch._dynamo.mark_static(attn_metadata.decode.block_table)
                    torch._dynamo.mark_static(
                        attn_metadata.decode.input_positions)
                    if hasattr(attn_metadata.decode, "sin"):
                        torch._dynamo.mark_static(attn_metadata.decode.sin)
                        torch._dynamo.mark_static(attn_metadata.decode.cos)
                    torch._dynamo.mark_static(get_forward_context().mc2_mask)
                    torch._dynamo.mark_static(attn_metadata.slot_mapping)
                    torch._dynamo.mark_static(attn_metadata.decode.attn_mask)
                    torchair_compiled_model = self._get_torchair_lazy_compiled_model(
                        num_tokens)
                    torchair_compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        previous_hidden_states=previous_hidden_states,
                        inputs_embeds=None,
                        intermediate_tensors=None,
                        attn_metadata=attn_metadata,
                        kv_caches=self.runner.kv_caches[-1:],
                        spec_step_idx=0)
                else:
                    self.model(input_ids=input_ids,
                               positions=positions,
                               previous_hidden_states=previous_hidden_states)
            if with_prefill:
                break

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
        accepted_token_indices = None
        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.runner.input_ids[:num_scheduled_tokens]
            target_positions = positions[:num_scheduled_tokens]
            target_hidden_states = hidden_states[:num_scheduled_tokens]
            target_slot_mapping = attn_metadata.slot_mapping
            cu_num_tokens = attn_metadata.query_start_loc
        else:
            # TODO(woosuk): Refactor this.
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
            cu_num_tokens, accepted_token_indices, target_token_ids, \
                target_positions, target_hidden_states, target_slot_mapping = self._prepare_inputs(
                attn_metadata.query_start_loc,
                num_rejected_tokens,
                self.runner.input_ids[:num_scheduled_tokens],
                positions[:num_scheduled_tokens],
                hidden_states[:num_scheduled_tokens],
                attn_metadata.slot_mapping[:num_scheduled_tokens],
                is_torchair_graph=self.runner._build_drafter_prepare_inputs_torchair_param(),
            )

        draft_token_ids = self._propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            target_slot_mapping=target_slot_mapping,
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=attn_metadata.block_tables,
            sampling_metadata=sampling_metadata,
            token_indices=accepted_token_indices)
        spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    def _prepare_inputs(
        self,
        # [batch_size + 1]
        cu_target_query_lens: torch.Tensor,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_torchair_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]
        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = (cu_target_query_lens[1:] -
                             cu_target_query_lens[:-1])
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens
        if is_torchair_graph:
            cu_num_tokens = cu_target_query_lens
            relative_index = query_len_per_req - num_rejected_tokens - 1
            token_indices = cu_num_tokens[:-1] + relative_index
            # the seq len of each bath is padded to 1+num_speculative_tokens, thus input is same as the main model
            target_token_ids = token_ids
            target_positions = positions
            target_hidden_states = hidden_states
            target_slot_mapping = slot_mapping
        else:
            cu_num_tokens = torch.empty_like(cu_target_query_lens)
            torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
            cu_num_tokens[0] = 0

            # FIXME(woosuk): Avoid synchronization.
            num_tokens = cu_num_tokens[-1].item()
            token_indices = torch.zeros(
                num_tokens,
                dtype=torch.int32,
                device=cu_num_tokens.device,
            )

            BLOCK_SIZE = 1024
            self._prepare_input_kernel(
                token_indices,
                cu_target_query_lens,
                cu_num_tokens,
                block_size=BLOCK_SIZE,
            )
            target_token_ids = token_ids[token_indices]
            target_positions = positions[token_indices]
            target_hidden_states = hidden_states[token_indices]
            target_slot_mapping = slot_mapping[token_indices]
        return cu_num_tokens, token_indices, target_token_ids, target_positions, target_hidden_states, target_slot_mapping

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
            sampling_metadata: SamplingMetadata,
            token_indices=None) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        if token_indices is not None and self.torchair_graph_enabled:
            last_token_indices = token_indices

        self.input_ids[last_token_indices] = next_token_ids

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        # FIXME: reorder_batch() needs to be called before build()
        # because fields of attn_metadata_builder needs to be updated.
        # However, currently reorder_batch() takes input_batch and
        # scheduler_output as arguments, we should probably refactor
        # the method to use new data structures which are independent
        # from input_batch and scheduler_output.
        # self.runner.attn_metadata_builder.reorder_batch(
        #     input_batch=self.runner.input_batch,
        #     scheduler_output=self.runner.scheduler_output,
        # )
        is_running_torchair = self.torchair_graph_enabled and \
            not self.runner.with_prefill

        if is_running_torchair:
            num_input_tokens = self.runner.graph_pad_size
        else:
            num_input_tokens = num_tokens

        seq_lens = target_positions[last_token_indices] + 1
        seq_lens = seq_lens.int()
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=cu_num_tokens[:batch_size + 1],
            query_start_loc_cpu=cu_num_tokens[:batch_size + 1].cpu(),
            seq_lens_cpu=seq_lens.cpu(),
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
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
        attn_metadata = self.runner.attn_metadata_builder.build(
            common_attn_metadata, self.runner.get_model())

        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        if not self.torchair_graph_enabled:
            # torch mode need to update num_tokens_across_dp
            # TODO: adapt enable_dbo later
            (num_input_tokens, num_tokens_across_dp, with_prefill,
             _) = self.runner._sync_metadata_across_dp(
                 num_tokens, self.runner.with_prefill, False)
            attn_metadata.slot_mapping = target_slot_mapping
        else:
            # torchair mode can reuse self.runner.num_tokens_across_dp
            num_tokens_across_dp = self.runner.num_tokens_across_dp
            with_prefill = self.runner.with_prefill

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
                    if self.torchair_graph_enabled:
                        model_kwargs["kv_caches"] = self.runner.kv_caches[-1:]
                    if is_running_torchair:
                        torchair_compiled_model = self._get_torchair_lazy_compiled_model(
                            num_input_tokens)
                        hidden_states = torchair_compiled_model(
                            input_ids=self.input_ids[:num_input_tokens],
                            positions=self.positions[:num_input_tokens],
                            previous_hidden_states=self.
                            hidden_states[:num_input_tokens],
                            inputs_embeds=None,
                            intermediate_tensors=None,
                            spec_step_idx=0,
                            **model_kwargs)
                    else:
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
            if with_prefill and self.torchair_graph_enabled:
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
                if is_running_torchair:
                    attn_metadata.num_actual_tokens = batch_size
                    attn_metadata.query_lens = [1] * batch_size

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
        return draft_token_ids

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.runner.torchair_graph_batch_sizes[
                -1]:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_graph_batch_sizes:{self.runner.torchair_graph_batch_sizes[-1]}"
            )

        compiled_model = self.torchair_compiled_models.get(
            batch_size
        ) if self.runner.use_cached_npu_graph else self.torchair_compiled_model

        if compiled_model:
            return compiled_model

        patch_for_hcom()
        config = torchair.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        config.experimental_config.enable_view_optimize = \
        get_ascend_config().torchair_graph_config.enable_view_optimize
        torch.npu.set_compile_mode(jit_compile=False)
        if not self.runner.use_cached_npu_graph:
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.torchair_compiled_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=npu_backend)
            return self.torchair_compiled_model
        else:
            # Generate a new forward proxy code object to prevent the invalidation of
            # compilation cache caused by dynamo retracing
            forward_proxy_name = f"{self.model.__class__.__name__}_forward_with_batch_size_{batch_size}"
            forward_fn = self.model.forward
            code = forward_fn.__code__
            # Mark code object with a new proxy name
            modified_code = code.replace(co_name=forward_proxy_name, )

            modified_func = types.FunctionType(modified_code,
                                               forward_fn.__globals__,
                                               name=forward_proxy_name,
                                               argdefs=forward_fn.__defaults__)

            self.model.__dict__[forward_proxy_name] = modified_func.__get__(
                self.model, nn.Module)
            self.torchair_compiled_models[
                batch_size] = torchair.inference.cache_compile(
                    self.model.__dict__[forward_proxy_name],
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    cache_dir=TORCHAIR_CACHE_DIR,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]

    # TODO Using torch instead of triton may result in poor performance
    def _prepare_input_kernel(self, out_ptr: torch.Tensor,
                              cu_query_lens: torch.Tensor,
                              cu_num_tokens: torch.Tensor, block_size: int):
        device = cu_query_lens.device
        dtype = out_ptr.dtype

        offsets = torch.arange(block_size, device=device, dtype=dtype)
        start_pos = cu_num_tokens[:-1]
        end_pos = cu_num_tokens[1:]
        num_tokens = end_pos - start_pos

        global_indices = (start_pos.view(-1, 1) + offsets.view(1, -1))
        values = (cu_query_lens[:-1].view(-1, 1) + offsets.view(1, -1))

        mask = (offsets.view(1, -1) < num_tokens.view(-1, 1))

        global_indices_flat = global_indices[mask]
        values_flat = values[mask]
        out_ptr[global_indices_flat] = values_flat
