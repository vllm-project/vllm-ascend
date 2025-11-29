import importlib
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import (CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config, set_current_vllm_config)
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import \
    process_weights_after_loading
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import (ACLGraphWrapper,
                                               set_mtp_graph_params,
                                               update_mla_attn_params)
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm_ascend.utils import (ProfileExecuteDuration, lmhead_tp_enable,
                               prefill_context_parallel_enable)

if prefill_context_parallel_enable():
    from vllm.distributed import get_pcp_group

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

PADDING_SLOT_ID = -1

_MTP_MODELS = {
    "DeepseekV3ForCausalLM":
    ("vllm.model_executor.models.deepseek_mtp", "DeepSeekMTP"),
    "DeepseekV32ForCausalLM":
    ("vllm.model_executor.models.deepseek_mtp", "DeepSeekMTP"),
    "Qwen3NextForCausalLM":
    ("vllm_ascend.models.qwen3_next_mtp", "CustomQwen3NextMTP")
}

_DEFAULT_FIRST_LAYER = 'model.layers.0.self_attn.attn'

_FIRST_LAYERS = {"Qwen3NextForCausalLM": 'model.layers.3.self_attn.attn'}


def _load_model(architecture):
    if architecture not in _MTP_MODELS:
        raise ValueError("Invalid architecture for mtp.")
    module_name, model_name = _MTP_MODELS[architecture]
    module = importlib.import_module(module_name)
    model = getattr(module, model_name)
    return model


class MtpProposer(Proposer):

    # TODO: Find out why ModelRunner does not this explicit typing?
    model: Union[nn.Module, ACLGraphWrapper]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device,
        runner,
    ):
        self.name = SpecDcodeType.MTP
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.decode_threshold = 1 + self.num_speculative_tokens
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()

        self.pcp_size = self.runner.pcp_size
        self.dcp_size = self.runner.dcp_size
        self.pcp_rank = self.runner.pcp_rank

        self.attn_metadata_builder: Optional[AttentionMetadataBuilder] = None
        self.draft_indexer_metadata_builder: Optional[
            AttentionMetadataBuilder] = None
        self.attn_layer_names: list[str] = []
        self.indexer_layer_names: list[str] = []

        self.use_aclgraph = self.runner._use_aclgraph()

        self.cudagraph_batch_sizes = (list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))
                                      if self.use_aclgraph else [])

        # persistent buffers for aclgraph graph
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=device)
        self.uses_mrope = self.vllm_config.model_config.uses_mrope
        if self.uses_mrope:
            # M-RoPE need (3, max_num_tokens)
            self.mrope_positions = torch.zeros((3, self.max_num_tokens),
                                               dtype=torch.int64,
                                               device=device)
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device=device)
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)
        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size +
            self.pcp_size * self.dcp_size * self.runner.max_num_reqs)

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(max_num_slots_for_arange,
                                   device=device,
                                   dtype=torch.int32)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        self.use_sparse = hasattr(vllm_config.model_config.hf_config,
                                  "index_topk")

    def load_model(self, model) -> None:
        loader = get_model_loader(self.vllm_config.load_config)

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        AttentionLayerBase).keys())
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        DeepseekV32IndexerCache).keys())
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_device = self.vllm_config.device_config.device

        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            self._init_mtp_model()
        draft_attn_layer_names = (get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase).keys() -
                                  target_attn_layer_names)
        indexer_layers = get_layers_from_vllm_config(self.vllm_config,
                                                     DeepseekV32IndexerCache)
        draft_indexer_layer_names = indexer_layers.keys(
        ) - target_indexer_layer_names
        # NOTE: Currently we don't have specific attention backend and attention metadata
        # for deepseek v3.2 indexer, so we just exclude the indexer layers here.
        draft_attn_layer_names = draft_attn_layer_names - draft_indexer_layer_names

        assert len(draft_attn_layer_names) == 1
        self.attn_layer_name = list(draft_attn_layer_names)

        self.model.load_weights(
            loader.get_all_weights(
                self.vllm_config.speculative_config.draft_model_config,
                self.model))
        process_weights_after_loading(self.model, draft_model_config,
                                      target_device)

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(
        ):
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            set_mtp_graph_params(
                self.vllm_config.compilation_config.cudagraph_capture_sizes)
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp=None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None) -> None:

        (
            num_tokens,
            num_tokens_across_dp,
            with_prefill,
        ) = self.runner._sync_metadata_across_dp(num_tokens, with_prefill)

        moe_comm_type = self.runner._select_moe_comm_method(num_tokens)

        if skip_attn:
            attn_metadata = None
        elif aclgraph_runtime_mode == CUDAGraphMode.FULL:
            if len(self.runner.attn_groups) > 0:
                num_computed_tokens_cpu = (
                    self.runner.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs])
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.runner.query_start_loc[:num_reqs + 1],
                    query_start_loc_cpu=self.runner.
                    query_start_loc_cpu[:num_reqs + 1],
                    seq_lens_cpu=self.runner.seq_lens_cpu,
                    seq_lens=self.runner.seq_lens_cpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=self.num_speculative_tokens + 1,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                    block_table_tensor=self.runner.input_batch.block_table[0].
                    get_device_tensor()[:num_reqs],
                    slot_mapping=self.runner.input_batch.block_table[0].
                    slot_mapping,
                    positions=self.runner.positions,
                    attn_mask=self.runner.attn_mask,
                    spec_attn_mask=self.runner.spec_attn_mask,
                    attn_state=self.runner.attn_state,
                    decode_token_per_req=self.runner.decode_token_per_req,
                    cos=self.runner.cos,
                    sin=self.runner.sin,
                )

                builder = self.runner.attn_groups[0][0].get_metadata_builder()
                attn_metadata_mtp = builder.build_for_graph_capture(
                    common_attn_metadata, AscendAttentionState.SpecDecoding,
                    self.runner.get_model())
                attn_metadata = {}
                for layer_name in self.attn_layer_name:
                    attn_metadata[layer_name] = attn_metadata_mtp
            else:
                attn_metadata = None
        else:
            attn_metadata = None

        input_ids = self.input_ids[:num_tokens]
        positions = self.positions[:num_tokens]
        previous_hidden_states = self.hidden_states[:num_tokens]
        for i in range(self.num_speculative_tokens):
            if i > 0:
                aclgraph_runtime_mode = CUDAGraphMode.NONE
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    with_prefill=with_prefill,
                    num_tokens_across_dp=num_tokens_across_dp,
                    reserved_mc2_mask=self.runner.reserved_mc2_mask,
                    moe_comm_type=moe_comm_type,
                    in_profile_run=self.runner.in_profile_run,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    is_mtp_model=True):
                self.model(input_ids=input_ids,
                           positions=positions,
                           hidden_states=previous_hidden_states)
                forward_context = get_forward_context()
                if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and \
                    not forward_context.capturing:
                    if self.vllm_config.model_config.use_mla:
                        update_mla_attn_params(
                            self.update_stream, forward_context,
                            positions.shape[0],
                            self.vllm_config.speculative_config)
            if with_prefill:
                break

    def generate_token_ids(self,
                           sampled_token_ids: Union[torch.Tensor,
                                                    list[np.ndarray]],
                           sampling_metadata: SamplingMetadata = None,
                           scheduler_output: SchedulerOutput = None,
                           spec_decode_metadata: SpecDecodeMetadata = None,
                           positions: torch.Tensor = None,
                           num_scheduled_tokens: int = 0,
                           hidden_states: torch.Tensor = None,
                           attn_metadata=None,
                           aux_hidden_states: torch.Tensor = None):
        common_attn_metadata = self.runner.spec_decode_common_attn_metadata
        attn_metadata = self._get_attn_metadata(attn_metadata)

        if self.speculative_config.disable_padded_drafter_batch:
            # When padded-batch is disabled, the sampled_token_ids should be
            # the cpu-side list[list[int]] of valid sampled tokens for each
            # request, with invalid requests having empty lists.
            assert isinstance(sampled_token_ids, list), \
                "sampled_token_ids should be a python list when" \
                "padded-batch is disabled."
            next_token_ids = self.prepare_next_token_ids_cpu(
                sampled_token_ids, self.runner.requests,
                self.runner.input_batch, scheduler_output.num_scheduled_tokens)
        else:
            # When using padded-batch, the sampled_token_ids should be
            # the gpu tensor of sampled tokens for each request, of shape
            # (num_reqs, num_spec_tokens + 1) with rejected tokens having
            # value -1.
            assert isinstance(sampled_token_ids, torch.Tensor), \
                "sampled_token_ids should be a torch.Tensor when" \
                "padded-batch is enabled."
            next_token_ids, valid_sampled_tokens_count = \
                self.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.runner.requests,
                    self.runner.input_batch,
                    self.runner.discard_request_indices.gpu,
                    self.runner.num_discarded_requests
                )

        req_scheduled_tokens = scheduler_output.num_scheduled_tokens
        if self.pcp_size > 1:
            long_seq_metadata = self.runner.long_seq_metadata
            input_ids_pcp_full = self.runner.input_ids_pcp_full
            query_start_loc_pcp_full = self.runner.query_start_loc_pcp_full
            query_start_loc_pcp_full_cpu = self.runner.query_start_loc_pcp_full_cpu
            num_reqs = self.runner.input_batch.num_reqs
            ori_query_lens = query_start_loc_pcp_full_cpu[1:num_reqs+1] - \
                query_start_loc_pcp_full_cpu[:num_reqs]
            num_prefill_reqs = (ori_query_lens
                                > self.decode_threshold).sum().item()
            num_decode_reqs = num_reqs - num_prefill_reqs
        else:
            long_seq_metadata = None
            num_prefill_reqs = 0
            num_decode_reqs = 0
        if spec_decode_metadata is None:
            # update pcp related params
            if self.pcp_size > 1:
                token_indices_to_sample = \
                    query_start_loc_pcp_full_cpu[1:num_reqs + 1] - 1
                target_token_ids = input_ids_pcp_full[:num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                target_hidden_states = hidden_states
            else:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.runner.input_ids[:num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                target_hidden_states = hidden_states[:num_scheduled_tokens]
        else:
            if self.pcp_size > 1:
                common_attn_metadata.query_start_loc_cpu = \
                    query_start_loc_pcp_full_cpu[:num_reqs + 1]
                common_attn_metadata.query_start_loc = \
                    query_start_loc_pcp_full[:num_reqs + 1]
            if self.speculative_config.disable_padded_drafter_batch:
                assert isinstance(sampled_token_ids, list)
                # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                token_indices_to_sample = None
                common_attn_metadata, token_indices =\
                    self._prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens)
            else:
                common_attn_metadata, token_indices, \
                    token_indices_to_sample =\
                        self.prepare_inputs_padded(
                            common_attn_metadata,
                            spec_decode_metadata,
                            valid_sampled_tokens_count)
            if self.pcp_size > 1:
                target_token_ids = input_ids_pcp_full[token_indices]
                target_positions = positions
                target_hidden_states = hidden_states
            else:
                target_token_ids = self.runner.input_ids[token_indices]
                target_positions = positions[token_indices]
                target_hidden_states = hidden_states[token_indices]

        draft_token_ids = self._propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            last_token_indices=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
            scheduler_output=scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        return draft_token_ids

    def _init_mtp_model(self):
        architecture = self.vllm_config.model_config.architecture
        target_device = self.vllm_config.device_config.device
        model = _load_model(architecture)
        self.model = model(vllm_config=self.vllm_config).to(target_device)

    def _get_attn_metadata(self, attn_metadata):
        if attn_metadata is not None and isinstance(attn_metadata, dict):
            architecture = self.vllm_config.model_config.architecture
            layer_name = _FIRST_LAYERS.get(architecture, _DEFAULT_FIRST_LAYER)
            attn_metadata = attn_metadata[layer_name]

        return attn_metadata

    def _prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[np.ndarray],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
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

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens,
                                           dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[
            1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
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
        token_offests = (self.token_arange_np[:total_num_tokens] -
                         new_query_start_locs_expanded)

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

        common_attn_metadata.slot_mapping[:token_indices.shape[0]].copy_(
            common_attn_metadata.slot_mapping[token_indices])
        common_attn_metadata.slot_mapping[token_indices.shape[0]:].fill_(-1)

        # NOTE: Currently positions and seq_lens are not used in mla_v1 forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device,
                                                       non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            positions=common_attn_metadata.positions[token_indices],
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            graph_pad_size=self.runner.graph_pad_size,
            decode_token_per_req=self.runner.decode_token_per_req,
        )
        return spec_common_attn_metadata, token_indices

    def _propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: Optional[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: Optional[tuple[list[torch.Tensor],
                                        torch.Tensor]] = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput = None,
        num_scheduled_tokens: int = 0,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
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

        # update pcp related params
        if self.pcp_size > 1:
            assert long_seq_metadata is not None
            common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
            # 1. preprocess decode/prefill input_ids & target_hidden_states
            # decode input_ids: keep unchanged
            # decode target_hidden_states: remove padding
            # prefill input_ids: add padding and pcp split
            # prefill target_hidden_states: pcp split
            num_tokens_d = num_decode_reqs * self.decode_threshold
            num_tokens_d_padded = num_tokens_d * self.pcp_size
            input_ids_d = self.input_ids[:num_tokens_d]
            input_ids_p = self.input_ids[num_tokens_d:num_tokens]
            target_hidden_states_d_padded = \
                target_hidden_states[:num_tokens_d_padded]
            if num_tokens_d:
                # remove padding (from pcp all-gather) in decode part
                target_hidden_states_d = target_hidden_states_d_padded.reshape(
                    [
                        num_decode_reqs, self.decode_threshold * self.pcp_size,
                        -1
                    ])[:, :self.decode_threshold, :].reshape(
                        [num_tokens_d, -1])
            else:
                target_hidden_states_d = target_hidden_states_d_padded
            target_hidden_states_p = target_hidden_states[num_tokens_d_padded:]
            req_scheduled_tokens_p = {}
            for i, req_id in enumerate(self.runner.input_batch.req_ids):
                if i >= num_decode_reqs:
                    req_scheduled_tokens_p[req_id] = \
                        req_scheduled_tokens[req_id]
            (num_tokens_p, input_ids_p, target_hidden_states_p,
             max_query_len_p, seq_lens_p, cu_num_tokens_p) = \
                self._split_pcp_input(
                    req_scheduled_tokens_p, input_ids_p, target_hidden_states_p)
            num_tokens = num_tokens_d + num_tokens_p
            target_positions = target_positions[:num_tokens]
            self.input_ids[:num_tokens].copy_(
                torch.cat([input_ids_d, input_ids_p], dim=0))
            target_hidden_states = torch.cat(
                [target_hidden_states_d, target_hidden_states_p], dim=0)
            # 2. update attn_metadata params that may be influenced by pcp
            common_attn_metadata.num_actual_tokens = num_tokens
            common_attn_metadata.max_query_len = max(self.decode_threshold,
                                                     max_query_len_p)
            common_attn_metadata.seq_lens[num_decode_reqs:] = seq_lens_p
            common_attn_metadata.seq_lens_cpu[num_decode_reqs:] = seq_lens_p
            query_start_loc_p = cu_num_tokens_p[1:] + \
                common_attn_metadata.query_start_loc[num_decode_reqs].item()
            common_attn_metadata.query_start_loc[num_decode_reqs + 1:] = \
                query_start_loc_p
            common_attn_metadata.query_start_loc_cpu[num_decode_reqs + 1:] = \
                query_start_loc_p
            # 3. update sample_indices according to main model
            if num_decode_reqs:
                last_token_indices[:num_decode_reqs] = \
                    self.runner.logits_indices[last_token_indices[:num_decode_reqs]]
            if num_prefill_reqs:
                last_token_indices[-num_prefill_reqs:] = \
                    self.runner.logits_indices[-num_prefill_reqs:]

        assert self.runner is not None

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(
        ) and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        elif self.use_aclgraph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            # Acl graph mode, add padding to the batch size
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            # Eager mode, no padding needed
            num_input_tokens = num_tokens

        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states
        # eager/acl piecewise mode need to update num_tokens_across_dp
        (num_input_tokens, num_tokens_across_dp,
         with_prefill) = self.runner._sync_metadata_across_dp(
             num_input_tokens, self.runner.with_prefill)

        moe_comm_type = self.runner._select_moe_comm_method(num_input_tokens)

        if scheduler_output:
            max_query_len = common_attn_metadata.max_query_len
            uniform_decode = (max_query_len in list(
                range(1, self.num_speculative_tokens +
                      2))) and (scheduler_output.total_num_scheduled_tokens
                                == self.runner.input_batch.num_reqs *
                                (self.num_speculative_tokens + 1))
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
        else:
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=False)
        aclgraph_runtime_mode, batch_descriptor = \
            self.runner.aclgraph_dispatcher.dispatch(batch_descriptor)

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(
        ) and aclgraph_runtime_mode == CUDAGraphMode.FULL:
            graph_pad_size = num_input_tokens
        else:
            # Currently, runner.graph_pad_size will always be -1.
            graph_pad_size = self.runner.graph_pad_size

        # If use fullgraph and disable_padded_drafter_batch=True, We need to
        # update the graph_pad_size in common_attn_metadata, to tell the
        # builder padding some elements.
        common_attn_metadata.graph_pad_size = graph_pad_size
        builder = self.runner.attn_groups[0][0].get_metadata_builder()
        attn_metadata_mtp = builder.build(0, common_attn_metadata,
                                          self.runner.get_model())
        attn_metadata = {}
        for layer_name in self.attn_layer_name:
            attn_metadata[layer_name] = attn_metadata_mtp

        for step in range(self.num_speculative_tokens):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    with_prefill=with_prefill,
                    num_tokens_across_dp=num_tokens_across_dp,
                    reserved_mc2_mask=self.runner.reserved_mc2_mask,
                    moe_comm_type=moe_comm_type,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    in_profile_run=self.runner.in_profile_run,
                    num_actual_tokens=num_tokens,
                    is_mtp_model=True):
                with ProfileExecuteDuration().capture_async('mtp_forward'):
                    model_kwargs = {}
                    model_kwargs["attn_metadata"] = attn_metadata

                    hidden_states = self.model(
                        input_ids=self.input_ids[:num_input_tokens],
                        positions=self.positions[:num_input_tokens],
                        hidden_states=self.hidden_states[:num_input_tokens])
                    forward_context = get_forward_context()
                    if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                        if self.vllm_config.model_config.use_mla:
                            update_mla_attn_params(
                                self.update_stream, forward_context,
                                num_input_tokens,
                                self.vllm_config.speculative_config)

            num_indices = last_token_indices.shape[0]
            if lmhead_tp_enable():
                if not self.runner.with_prefill:
                    max_num_reqs_across_dp = num_input_tokens
                else:
                    max_num_reqs_across_dp = self.vllm_config.scheduler_config.max_num_seqs
                last_token_indices = nn.functional.pad(
                    last_token_indices,
                    (0, max_num_reqs_across_dp - num_indices))

            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(hidden_states, 0)
                hidden_states = torch.index_select(
                    hidden_states, 0, self.runner.
                    pcp_allgather_restore_idx[:hidden_states.shape[0]])

            sample_hidden_states = hidden_states[last_token_indices]
            logits = self.model.compute_logits(sample_hidden_states)
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

            attn_metadata_i = attn_metadata[self.attn_layer_name[0]]

            if step == 0:
                positions = target_positions[last_token_indices]
                hidden_states = hidden_states[last_token_indices]
                slot_mapping = attn_metadata_i.slot_mapping[last_token_indices]
                attn_metadata_i.slot_mapping.fill_(-1)
                attn_metadata_i.query_start_loc = self.arange[:batch_size + 1]
                last_token_indices = self.arange[:batch_size]
                if attn_metadata_i.num_decode_tokens != 0:
                    attn_metadata_i.num_decode_tokens = batch_size

            input_ids = draft_token_ids_list[-1].int()
            positions += 1

            decode_meta = getattr(attn_metadata_i, "decode", None)
            prefill_meta = getattr(attn_metadata_i, "prefill", None)
            # When disable_padded_drafter_batch=False, it should not to be updating these params, maybe.
            if decode_meta is not None and (self.speculative_config.disable_padded_drafter_batch or \
                    aclgraph_runtime_mode != CUDAGraphMode.FULL):
                attn_metadata_i.decode.actual_seq_lengths_q = attn_metadata_i.query_start_loc[
                    1:batch_size + 1].tolist()
                if aclgraph_runtime_mode == CUDAGraphMode.FULL:
                    attn_metadata_i.decode.actual_seq_lengths_q = \
                        builder.pad_actual_seq_len_q_mtp_disable_pad(
                            graph_pad_size - batch_size,
                            batch_size,
                            attn_metadata_i.decode.actual_seq_lengths_q)
                attn_metadata_i.decode.cos = builder.cos_cache[
                    positions].unsqueeze(1).unsqueeze(2)
                attn_metadata_i.decode.sin = builder.sin_cache[
                    positions].unsqueeze(1).unsqueeze(2)
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
            attn_metadata_i.seq_lens[:batch_size] += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            exceeds_max_model_len_cpu = exceeds_max_model_len.to(
                attn_metadata_i.seq_lens.device, non_blocking=True)
            attn_metadata_i.seq_lens[:batch_size].masked_fill_(
                exceeds_max_model_len_cpu, 1)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping += 1
            slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:hidden_states.shape[0]] = hidden_states
            attn_metadata_i.slot_mapping[:batch_size] = slot_mapping
            if self.speculative_config.disable_padded_drafter_batch:
                self.positions[batch_size:num_input_tokens] = 0
                self.input_ids[batch_size:num_input_tokens] = 0
                self.hidden_states[batch_size:num_input_tokens].fill_(0)

            if prefill_meta is not None:
                attn_metadata_i.prefill.seq_lens = attn_metadata_i.seq_lens
                attn_metadata_i.prefill.seq_lens_list = attn_metadata_i.prefill.seq_lens.tolist(
                )
                attn_metadata_i.prefill.context_lens = attn_metadata_i.seq_lens
                attn_metadata_i.prefill.input_positions = self.positions[:
                                                                         num_input_tokens]
                attn_metadata_i.prefill.max_seq_lens += 1
                attn_metadata_i.prefill.max_seq_lens = min(
                    attn_metadata_i.prefill.max_seq_lens,
                    self.runner.model_config.max_model_len)
            if decode_meta is not None:
                attn_metadata_i.decode.seq_lens = attn_metadata_i.seq_lens
                attn_metadata_i.decode.seq_lens_list = attn_metadata_i.decode.seq_lens.tolist(
                )
                decode_seq_lens_list = attn_metadata_i.decode.seq_lens_list
                if aclgraph_runtime_mode == CUDAGraphMode.FULL and \
                        self.speculative_config.disable_padded_drafter_batch:
                    attn_metadata_i.decode.seq_lens_list = decode_seq_lens_list + [
                        0
                    ] * (graph_pad_size - len(decode_seq_lens_list))
                attn_metadata_i.decode.input_positions = self.positions[:
                                                                        num_input_tokens]
                attn_metadata_i.decode.max_seq_lens += 1
                attn_metadata_i.decode.max_seq_lens = min(
                    attn_metadata_i.decode.max_seq_lens,
                    self.runner.model_config.max_model_len)

        # mtp>1: [batch_size, k]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

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

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[np.ndarray],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids.shape[0] > 0:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[
                    req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id.item())
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.input_ids.device)
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """
        # TODO(Ben): Combine this into a custom fused kernel

        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array([
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                common_attn_metadata.seq_lens_cpu[i].item())
            for i in range(num_reqs)
        ])
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = discard_request_indices[:
                                                                     num_discarded_requests]

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(
            0, discard_sampled_tokens_req_indices, -1)

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1,
            last_valid_indices_safe.unsqueeze(1)).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_draft_tokens_gpu = torch.cat([
            spec_decode_metadata.cu_num_draft_tokens[0:1],
            spec_decode_metadata.cu_num_draft_tokens[1:] -
            spec_decode_metadata.cu_num_draft_tokens[:-1],
        ])

        num_rejected_tokens_gpu = torch.where(
            num_draft_tokens_gpu > 0,
            num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens_gpu),
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[
            1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        # NOTE: Currently positions and seq_lens are not used in mla_v1 forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            positions=common_attn_metadata.positions,
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            graph_pad_size=self.runner.graph_pad_size,
            decode_token_per_req=self.runner.decode_token_per_req,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            seq_lens=common_attn_metadata.seq_lens)

        token_indices_to_sample = (common_attn_metadata.query_start_loc[1:] -
                                   1 - num_rejected_tokens_gpu)

        return spec_common_attn_metadata, token_indices, token_indices_to_sample

    def _split_pcp_input(self, req_scheduled_tokens, input_ids,
                         target_hidden_states):
        """
        Split prefill input_ids and target_hidden_states in pcp group.
        1. input_ids padding: [t0, t1, t2, t3, t4, t5] -> [t0, t1, t2, t3, t4, t5, pad, pad]
        2. split input_ids: pcp0 [t0, t1, pad, pad], pcp1 [t2, t3, t4, t5]
        3. split target_hidden_states (already include pcp padding):
        [h0, h1, h2, h3, h4, h5, pad, pad] -> pcp0 [h0, h1, pad, pad], pcp1 [h2, h3, h4, h5]
        4. also update max_query_len, seq_lens, cu_num_tokens according to pcp split.
        """
        if len(req_scheduled_tokens) == 0:
            # no prefill inputs to split, return empty result
            return (
                0,
                torch.zeros([0], device='npu'),
                torch.zeros([0, target_hidden_states.size(1)], device='npu'),
                0,
                torch.zeros([0]),
                torch.tensor([0], dtype=torch.int32),
            )

        def _pcp_pad_and_split(num_tokens):
            num_pcp_padded_scheduled_tokens = cdiv(
                num_tokens, 2 * self.pcp_size) * 2 * self.pcp_size
            pcp_pad = num_pcp_padded_scheduled_tokens - num_tokens
            chunk_size = num_pcp_padded_scheduled_tokens // (2 * self.pcp_size)

            # split position_ids (and use split position_ids to split input_ids afterwards)
            req_position_cp: list[int] = []
            req_position_cp.extend(
                self.full_indices[self.pcp_rank *
                                  chunk_size:(self.pcp_rank + 1) * chunk_size])
            req_position_cp.extend(
                self.full_indices[num_pcp_padded_scheduled_tokens -
                                  (self.pcp_rank + 1) *
                                  chunk_size:num_pcp_padded_scheduled_tokens -
                                  self.pcp_rank * chunk_size])

            return req_position_cp, num_pcp_padded_scheduled_tokens, pcp_pad

        num_pcp_scheduled_tokens = []
        ori_start_index = 0
        pad_start_index = 0
        pcp_split_input_ids_list = []
        pcp_split_hidden_states_list = []
        for ori_num_tokens in req_scheduled_tokens.values():
            req_position_pcp, num_pcp_padded_scheduled_tokens, num_pcp_pad = \
                _pcp_pad_and_split(ori_num_tokens)
            actual_num_tokens = len(req_position_pcp)
            num_pcp_scheduled_tokens.append(actual_num_tokens)
            pad_input_ids = F.pad(
                input_ids[ori_start_index:ori_start_index + ori_num_tokens],
                (0, num_pcp_pad))
            ori_start_index += ori_num_tokens
            pcp_chunk_indices = [
                pad_start_index + pos for pos in req_position_pcp
            ]
            pcp_split_input_ids = pad_input_ids[req_position_pcp]
            pcp_split_hidden_states = target_hidden_states[pcp_chunk_indices]
            pcp_split_input_ids_list.append(pcp_split_input_ids)
            pcp_split_hidden_states_list.append(pcp_split_hidden_states)
            pad_start_index += num_pcp_padded_scheduled_tokens
        num_tokens = sum(num_pcp_scheduled_tokens)
        input_ids = torch.cat(pcp_split_input_ids_list)
        target_hidden_states = torch.cat(pcp_split_hidden_states_list, dim=0)
        max_query_len = max(num_pcp_scheduled_tokens)
        seq_lens = torch.tensor(num_pcp_scheduled_tokens, dtype=torch.int32)
        cu_num_tokens = torch.tensor(
            np.insert(np.cumsum(np.array(num_pcp_scheduled_tokens)), 0, 0))
        return num_tokens, input_ids, target_hidden_states, max_query_len, seq_lens, cu_num_tokens
