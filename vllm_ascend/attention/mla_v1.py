from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backends.mla.common import MLACommonMetadataBuilder
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import (
    AscendPCPMetadata, CPChunkedContextMetadata)
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         enable_cp,
                                         maybe_save_kv_layer_to_connector,
                                         split_decodes_and_prefills,
                                         trans_rope_weight, transdata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params, get_graph_params,
    update_draft_graph_params_workspaces, update_graph_params_workspaces)
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer, post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, maybe_trans_nz,
                               weak_ref_tensors)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1


class AscendMLABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_MLA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        if enable_cp():
            from vllm_ascend.attention.context_parallel.mla_cp import \
                AscendMlaCPMetadataBuilder
            return AscendMlaCPMetadataBuilder
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_impl_cls() -> Type["MLAAttentionImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.mla_cp import \
                AscendMlaCPImpl
            return AscendMlaCPImpl
        return AscendMLAImpl


@dataclass
class ChunkedContextMetadata:
    # New for MLA (compared to FlashAttention)
    # For handling chunked prefill
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    workspace: torch.Tensor
    chunk_seq_lens: torch.Tensor
    chunk_seq_lens_npu: torch.Tensor


@dataclass
class AscendMLAPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""
    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: list[int]
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_lens: int
    chunked_context: Optional[ChunkedContextMetadata
                              | CPChunkedContextMetadata] = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    pcp_metadata: Optional[AscendPCPMetadata] = None


@dataclass
class AscendMLADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_lens: int
    seq_lens_list: list[int]
    actual_seq_lengths_q: Optional[list[int]] = None
    attn_mask: Optional[torch.Tensor] = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    cp_seq_len: torch.Tensor = None
    batch_seq_mask: torch.Tensor = None


@dataclass
class AscendMLAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens_pcp_padded: int
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: Optional[list[int]] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Optional[AscendMLADecodeMetadata] = None
    prefill: Optional[AscendMLAPrefillMetadata] = None
    reshape_cache_event: torch.npu.Event = None

    def __post_init__(self):
        pass
        # supported_head_sizes = AscendMLABackend.get_supported_head_sizes()
        # if self.head_dim is not None and self.head_dim \
        #         not in supported_head_sizes:
        #     raise ValueError(
        #         f"Only {supported_head_sizes} are supported for head_dim,",
        #         f"received {self.head_dim}.")


M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMLAMetadataBuilder(MLACommonMetadataBuilder[AscendMLAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendMLAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.metadata_cls = (metadata_cls if metadata_cls is not None else
                             AscendMLAMetadata)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.reorder_batch_threshold = self.decode_threshold
        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(8 * self.model_config.max_model_len,
                    4 * scheduler_config.max_num_seqs * self.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.chunked_prefill_workspace_size >= \
                   scheduler_config.max_num_seqs * self.block_size
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 self.model_config.get_head_size()),
                dtype=self.model_config.dtype,
                device=device,
            )
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.chunk_seq_lens: torch.Tensor = None
        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_chunks: torch.Tensor = None
        self.max_context_chunk = 0
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.context_lens_cpu: torch.Tensor = None
        self.num_actual_tokens: Optional[int] = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendMLAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def pad_actual_seq_len_q_mtp_enable_pad(self, num_reqs_pad_size, num_reqs,
                                            actual_seq_lengths_q,
                                            common_attn_metadata):
        """
        Pads actual_seq_lengths_q evenly to not exceed 16 tokens per request
        in order to meet the requirement of npu_fused_infer_attention_score.

        In Torchair scenario, the lengths of the queries must be padded to the same length.
        And npu_fused_infer_attention_score constraint requires the last element must equal to batch_size(num_tokens).

        For example:
        batch_size=36, num_reqs_pad_size=2, num_reqs=16
        By default, each request should have inference 2 token, which means actual_seq_lengths_q should be
        [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36].

        However, mtp torchair + PD scenario, the actual_seq_lengths_q may be
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] before padding, since the first decode request only has 1 token.
        In order to meet the requirement of npu_fused_infer_attention_score, we need to pad actual_seq_lengths_q evenly to not exceed 16 tokens per request.
        after padding actual_seq_lengths_q should be similar to [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,36]
        """
        FIA_SEQ_LEN_LIMIT = 16
        need_padding = num_reqs_pad_size != 0 and \
                       len(common_attn_metadata.actual_seq_lengths_q) > num_reqs and \
                       common_attn_metadata.actual_seq_lengths_q[num_reqs] - actual_seq_lengths_q[
                           -1] > FIA_SEQ_LEN_LIMIT
        if need_padding:
            padding_seq_len_q = common_attn_metadata.actual_seq_lengths_q[
                num_reqs:num_reqs + num_reqs_pad_size]
            start_val = actual_seq_lengths_q[-1]
            end_val = padding_seq_len_q[-1]

            num_step = len(padding_seq_len_q)
            interpolated = np.round(
                np.linspace(start_val, end_val,
                            num_step + 1)[1:]).astype(int).tolist()
            assert interpolated[-1] == end_val
            assert len(interpolated) == len(padding_seq_len_q)
            actual_seq_lengths_q = actual_seq_lengths_q + interpolated
        else:
            actual_seq_lengths_q = actual_seq_lengths_q + common_attn_metadata.actual_seq_lengths_q[
                num_reqs:num_reqs + num_reqs_pad_size]

        return actual_seq_lengths_q

    def pad_actual_seq_len_q_mtp_disable_pad(self, num_reqs_pad_size, num_reqs,
                                             actual_seq_lengths_q):
        """
        Only use for acl full graph mode.
        Pad the last element of the actual_seq_lengths_q equal to the TND(T) and
        the num of dimensions equal to the batch_size of main model.

        For example:
        batch_size = 8, num_reqs = 4, num_speculative_tokens = 1
        input actual_seq_lengths_q = [1, 2, 4, 5]  (the 3rd req was accept a token)
        After padding the actual_seq_lengths_q will be similar to [1, 2, 4, 5, 6, 6, 7, 8]
        """
        need_padding = num_reqs_pad_size > 0
        if need_padding:
            start_val = actual_seq_lengths_q[-1]
            end_val = num_reqs + num_reqs_pad_size
            num_step = num_reqs_pad_size
            interpolated = np.round(
                np.linspace(start_val, end_val,
                            num_step + 1)[1:]).astype(int).tolist()
            assert interpolated[-1] == end_val
            assert len(interpolated) == num_reqs_pad_size
            actual_seq_lengths_q = actual_seq_lengths_q + interpolated
        return actual_seq_lengths_q

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMLAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        self.set_num_actual_tokens(common_attn_metadata)
        assert self.num_decodes + self.num_prefills == num_reqs
        assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        self.slot_mapping = common_attn_metadata.slot_mapping[:self.
                                                              num_actual_tokens]

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]
        self.seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:
                                                                   block_table_size]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(
                common_prefix_len, common_attn_metadata)

        decode_metadata = None
        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(
                common_prefix_len, common_attn_metadata)
        return self.metadata_cls(  # type: ignore
            num_actual_tokens_pcp_padded=self.num_actual_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens.tolist(),
            slot_mapping=self.slot_mapping,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=self.attn_mask_builder.get_final_mla_mask(
                self.model_config),
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=self.block_table,
            seq_lens=self.seq_lens,
        )

    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        if not self.chunked_prefill_enabled:
            return None
        num_reqs = common_attn_metadata.num_reqs

        num_computed_tokens_cpu = (self.seq_lens - self.query_lens)
        reqs_start = self.num_decodes  # prefill_start

        self.context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
        max_context_len_cpu = self.context_lens_cpu.max().item()
        if not max_context_len_cpu > 0:
            return None
        num_prefills_with_context_cpu = (self.context_lens_cpu
                                         > 0).sum().item()
        self.max_context_chunk = (self.chunked_prefill_workspace_size //
                                  num_prefills_with_context_cpu)
        self.max_context_chunk = round_down(self.max_context_chunk,
                                            self.block_size)

        assert self.max_context_chunk > 0
        self.num_chunks = cdiv(max_context_len_cpu, self.max_context_chunk)
        chunk_starts = torch.arange(self.num_chunks, dtype=torch.int32) \
                           .unsqueeze(1).expand(-1, self.num_prefills) * self.max_context_chunk
        chunk_ends = torch.min(self.context_lens_cpu.unsqueeze(0),
                               chunk_starts + self.max_context_chunk)
        self.chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
        self.cu_seq_lens_cpu = torch.zeros(self.num_chunks,
                                           self.num_prefills + 1,
                                           dtype=torch.int32,
                                           pin_memory=True)
        torch.cumsum(self.chunk_seq_lens,
                     dim=1,
                     out=self.cu_seq_lens_cpu[:, 1:],
                     dtype=torch.int32)
        return ChunkedContextMetadata(
            cu_seq_lens=self.cu_seq_lens_cpu.pin_memory().to(
                self.device, non_blocking=True),
            starts=chunk_starts.pin_memory().to(self.device,
                                                non_blocking=True),
            seq_tot=self.chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=self.chunk_seq_lens.max(dim=1).values.tolist(),
            chunk_seq_lens=self.chunk_seq_lens,
            chunk_seq_lens_npu=self.chunk_seq_lens.npu(),
            workspace=self.chunked_prefill_workspace,
        )

    def get_block_table_size(
            self, common_attn_metadata: AscendCommonAttentionMetadata,
            build_metadata_step: int):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            # If graph_pad_size > -1, mean is running in fullgraph mode.
            # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
            if self.graph_pad_size > common_attn_metadata.num_reqs and self.speculative_config.disable_padded_drafter_batch:
                return self.graph_pad_size
            return common_attn_metadata.num_reqs
        return self.num_decodes

    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLAPrefillMetadata:
        query_start_loc = common_attn_metadata.query_start_loc

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )

        chunked_context_metadata = self.build_chunked_metadata(
            common_prefix_len, common_attn_metadata)
        reqs_start = self.num_decodes  # prefill_start
        tokens_start = self.num_decode_tokens
        max_query_len = self.query_lens[reqs_start:].max().item()
        max_seq_lens = self.seq_lens[reqs_start:].max().item()
        prefill_query_start_loc = query_start_loc[
            reqs_start:] - query_start_loc[reqs_start]

        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_mla(prefill_input_positions)
        return AscendMLAPrefillMetadata(
            attn_mask=self.attn_mask_builder.get_final_mla_mask(
                self.model_config),
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens,
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            chunked_context=chunked_context_metadata,
            sin=sin,
            cos=cos,
        )

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLADecodeMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        input_positions = common_attn_metadata.positions[:self.
                                                         num_actual_tokens].long(
                                                         )

        # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
        actual_seq_lengths_q = query_start_loc_cpu[1:self.num_decodes +
                                                   1].tolist()
        max_seq_lens = self.seq_lens[:self.num_decodes].max().item()
        self.seq_lens = self.seq_lens[:self.num_decodes]
        input_positions = input_positions[:self.num_decode_tokens]

        block_table_size = self.get_block_table_size(
            common_attn_metadata, BUILD_METADATA_STEP_DECODE)
        self.block_table = self.block_table[:block_table_size]

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
        if self.graph_pad_size > self.num_decodes and \
                self.speculative_config.disable_padded_drafter_batch:
            self.block_table = self.block_table[:self.graph_pad_size, ...]
        seq_lens_list = self.seq_lens.tolist()

        cp_seq_len, batch_seq_mask = None, None

        if self.graph_pad_size > num_reqs:
            if self.speculative_config.disable_padded_drafter_batch:
                num_reqs_pad_size = self.graph_pad_size - num_reqs
                actual_seq_lengths_q = self.pad_actual_seq_len_q_mtp_disable_pad(
                    num_reqs_pad_size, num_reqs, actual_seq_lengths_q)
                seq_lens_list = seq_lens_list + [0] * (self.graph_pad_size -
                                                       self.num_decodes)
                num_block_pad_size = self.graph_pad_size - self.block_table.shape[
                    0]
                if num_block_pad_size > 0:
                    block_table_padding = torch.zeros(
                        (num_block_pad_size, ) + self.block_table.shape[1:],
                        dtype=self.block_table.dtype,
                        device=self.block_table.device)
                    self.block_table = torch.cat(
                        [self.block_table, block_table_padding], dim=0)
            else:
                num_token_pad_size = self.graph_pad_size - self.num_decode_tokens
                num_reqs_pad_size = (
                    self.graph_pad_size //
                    common_attn_metadata.decode_token_per_req - num_reqs)
                num_block_table_pad_size = (
                    self.graph_pad_size //
                    common_attn_metadata.decode_token_per_req -
                    self.num_decodes)
                seq_lens_list = self.seq_lens.tolist() + [0
                                                          ] * num_reqs_pad_size
                slot_padding = torch.full((num_token_pad_size, ),
                                          PAD_SLOT_ID,
                                          dtype=self.slot_mapping.dtype,
                                          device=self.slot_mapping.device)
                self.slot_mapping = torch.cat(
                    [self.slot_mapping, slot_padding])
                block_table_padding = torch.zeros(
                    (num_block_table_pad_size, ) + self.block_table.shape[1:],
                    dtype=self.block_table.dtype,
                    device=self.block_table.device)
                self.block_table = torch.cat(
                    [self.block_table, block_table_padding], dim=0)
                position_padding = torch.zeros(num_token_pad_size,
                                               dtype=input_positions.dtype,
                                               device=input_positions.device)
                input_positions = torch.cat(
                    [input_positions, position_padding])
                actual_seq_lengths_q = self.pad_actual_seq_len_q_mtp_enable_pad(
                    num_reqs_pad_size, num_reqs, actual_seq_lengths_q,
                    common_attn_metadata)

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=True)
        decode_metadata = AscendMLADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table,
            seq_lens=self.seq_lens,
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
            actual_seq_lengths_q=actual_seq_lengths_q,
            sin=sin[:self.num_decode_tokens, ...],
            cos=cos[:self.num_decode_tokens, ...],
            cp_seq_len=cp_seq_len,
            batch_seq_mask=batch_seq_mask)
        return decode_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        }:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class DecodeMLAPreprocessResult(NamedTuple):
    ql_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    decode_q_wo_k_up: Optional[torch.Tensor] = None


class PrefillMLAPreprocessResult(NamedTuple):
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


class AscendMLAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ):
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args
        self.q_lora_rank = kwargs['q_lora_rank']
        self.kv_lora_rank = kwargs['kv_lora_rank']
        self.qk_nope_head_dim = kwargs['qk_nope_head_dim']
        self.qk_rope_head_dim = kwargs['qk_rope_head_dim']
        self.qk_head_dim = kwargs['qk_head_dim']
        self.v_head_dim = kwargs['v_head_dim']
        self.rotary_emb = kwargs['rotary_emb']
        self.fused_qkv_a_proj = kwargs.get('fused_qkv_a_proj', None)
        self.q_proj = kwargs['q_proj'] if self.q_lora_rank is None else kwargs[
            'q_b_proj']
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.vllm_config = get_current_vllm_config()
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.q_a_layernorm = kwargs.get('q_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_prefetch = ascend_config.weight_prefetch_config.enabled
        self.enable_kv_nz = ascend_config.enable_kv_nz

        self.ring_mla_mask_size = 512

        self.speculative_config = self.vllm_config.speculative_config
        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        self.is_kv_producer = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        self.layer_sharding_kwargs = []
        for layer_name in (get_ascend_config().layer_sharding or []):
            if layer_name in kwargs:
                self.layer_sharding_kwargs.append(kwargs[layer_name])
            else:
                logger.warning_once(
                    f"Layer '{layer_name}' not found in kwargs for layer sharding, skipping sharding configuration"
                )
        register_all_layers_to_shard_weight_series(self.layer_sharding_kwargs)

    def _v_up_proj(self, x):
        # Convert from (N, B, L)/(N, B, 1, L) to (N, B, L)
        x = x.view(self.num_heads, -1, self.kv_lora_rank)
        # Multiply (N, B, L) x (N, L, V) -> (B, N, V)
        x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_y=(1, 0, 2))
        # Convert from (B, N, V) to (B, N * V)
        x = x.reshape(-1, self.num_heads * self.v_head_dim)
        return x

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = self.q_proj(x)[0] \
            .view(-1, self.num_heads, self.qk_head_dim) \
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(
            self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        if self.enable_mlapo:
            # Currently mlapo only supports W8A8 quantization in MLA scenario
            # TODO(whx): modify this limitation when mlapo supports floating point
            if self.fused_qkv_a_proj is None or not isinstance(
                    getattr(self.fused_qkv_a_proj.quant_method, 'quant_method',
                            None), AscendW8A8LinearMethod):
                self.enable_mlapo = False
                logger.warning_once(
                    "Currently mlapo only supports W8A8 quantization in MLA scenario."
                    "Some layers in your model are not quantized with W8A8,"
                    "thus mlapo is disabled for these layers.")
        if self.enable_mlapo:
            self._process_weights_for_fused_mlapo(act_dtype)
        else:
            # if mlapo, W_UK_T can't trans nz
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        for layer in (self.layer_sharding_kwargs or []):
            if is_hidden_layer(layer):
                post_process_after_loading_for_shard_weight_series(layer)

    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        kv_a_proj_wt = self.fused_qkv_a_proj.weight.data[
            ..., self.q_lora_rank:].contiguous()
        q_a_proj_wt = self.fused_qkv_a_proj.weight.data[
            ..., :self.q_lora_rank].contiguous()
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, self.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv,
                           block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

        kv_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[
            self.q_lora_rank:].contiguous()
        q_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[:self.
                                                           q_lora_rank].contiguous(
                                                           )
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(
            self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl,
                                              self.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(
            self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl),
                                       dim=-1).contiguous()

        kv_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[
            self.q_lora_rank:].contiguous()
        q_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[:self.
                                                            q_lora_rank].contiguous(
                                                            )
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(
            self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias,
                                              self.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(
            self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias),
                                        dim=-1).contiguous()

        wu_q = self.q_proj.weight.data
        wu_q = wu_q.t().reshape(self.num_heads,
                                self.qk_nope_head_dim + self.qk_rope_head_dim,
                                -1)
        wu_q = trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
            -1)
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wu_q = torch_npu.npu_format_cast(wu_q, 29)

        qb_deq_scl = self.q_proj.deq_scale.data
        qb_deq_scl = qb_deq_scl.reshape(
            self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, self.qk_rope_head_dim)
        self.qb_deq_scl = qb_deq_scl.reshape(
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        qb_qt_bias = self.q_proj.quant_bias.data
        qb_qt_bias = qb_qt_bias.reshape(
            self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, self.qk_rope_head_dim)
        self.qb_qt_bias = qb_qt_bias.reshape(
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        device = self.q_proj.weight.device
        self.gamma1 = self.q_a_layernorm.weight.data
        self.beta1 = torch.zeros_like(self.gamma1) if (
            _bias := self.q_a_layernorm.bias) is None else _bias.data
        self.gamma2 = self.kv_a_layernorm.weight.data
        self.quant_scale0 = self.fused_qkv_a_proj.input_scale.data
        self.quant_offset0 = self.fused_qkv_a_proj.input_offset.data
        self.quant_scale1 = self.q_proj.input_scale.data
        self.quant_offset1 = self.q_proj.input_offset.data
        self.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        self.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

        # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
        # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
        # referenced, so drop them to save memory.
        ascend_config = get_ascend_config()
        if self.vllm_config.kv_transfer_config is not None and \
                self.vllm_config.kv_transfer_config.is_kv_consumer and \
                ascend_config.recompute_scheduler_enable:
            self.fused_qkv_a_proj.weight = None
            self.fused_qkv_a_proj.deq_scale = None
            self.fused_qkv_a_proj.quant_bias = None
            self.q_proj.weight = None
            self.q_proj.deq_scale = None
            self.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def get_context_seq_len_npu(self, index: int,
                                attn_metadata: AscendMLAMetadata):
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.chunk_seq_lens_npu is not None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        assert 0 <= index < iters
        return prefill_metadata.chunked_context.chunk_seq_lens_npu[index]

    def _reorg_kvcache(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        chunked_context: CPChunkedContextMetadata,
        chunk_idx: int,
        toks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kv_c_normed, k_pe

    def _compute_prefill_context(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        rope_dim: int,
        attn_metadata: AscendMLAMetadata,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
    ):
        assert len(kv_c_and_k_pe_cache) > 1
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata is None or prefill_metadata.chunked_context is None:
            return prefix_output, prefix_lse

        iters = len(prefill_metadata.chunked_context.seq_tot)

        current_seq_len = torch.tensor(prefill_metadata.query_lens,
                                       dtype=torch.int32)
        cache_kv_c = kv_c_and_k_pe_cache[0]
        cache_k_pe = kv_c_and_k_pe_cache[1]
        num_heads = cache_k_pe.size(2)
        latent_kv_dim = kv_c_and_k_pe_cache[0].size(-1)
        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            # chunk_seq_lens will be padded when pcp&dcp
            context_seq_len = prefill_metadata.chunked_context.chunk_seq_lens[
                i]
            seq_len = torch.stack([current_seq_len, context_seq_len])
            context_seq_len_npu = self.get_context_seq_len_npu(
                i, attn_metadata)
            kv_c_normed = torch.empty(toks,
                                      num_heads,
                                      latent_kv_dim,
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            k_pe = torch.empty(toks,
                               num_heads,
                               rope_dim,
                               dtype=q_nope.dtype,
                               device=q_nope.device)

            torch_npu.atb.npu_paged_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                context_seq_len_npu,
                seq_starts=prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )
            kv_c_normed, k_pe = self._reorg_kvcache(
                kv_c_normed,
                k_pe,
                chunked_context=prefill_metadata.chunked_context,
                chunk_idx=i,
                toks=toks,
            )
            kv_c_normed = kv_c_normed.squeeze()
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope \
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))

            mask = attn_metadata.attn_mask
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=v,
                mask=mask,
                seqlen=seq_len,
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=prefix_output,
                prev_lse=prefix_lse,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                input_layout="type_bsnd",
                calc_type="calc_type_default",
                output=prefix_output,
                softmax_lse=prefix_lse)
        return prefix_output, prefix_lse

    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None
        assert len(kv_c_and_k_pe_cache) > 1
        num_tokens = q_nope.size(0)
        attn_output = torch.empty(num_tokens,
                                  self.num_heads,
                                  self.v_head_dim,
                                  dtype=q_nope.dtype,
                                  device=q_nope.device)
        attn_lse = torch.empty(self.num_heads,
                               num_tokens,
                               dtype=torch.float32,
                               device=q_nope.device)
        torch_npu.atb.npu_ring_mla(q_nope=q_nope,
                                   q_rope=q_pe,
                                   k_nope=k_nope,
                                   k_rope=k_pe,
                                   value=value,
                                   mask=attn_metadata.attn_mask,
                                   seqlen=attn_metadata.prefill.query_lens,
                                   head_num=self.num_heads,
                                   kv_head_num=self.num_heads,
                                   pre_out=None,
                                   prev_lse=None,
                                   qk_scale=self.scale,
                                   kernel_type="kernel_type_high_precision",
                                   mask_type="mask_type_triu",
                                   input_layout="type_bsnd",
                                   calc_type="calc_type_first_ring",
                                   output=attn_output,
                                   softmax_lse=attn_lse)
        attn_output, attn_lse = self._compute_prefill_context(
            q_nope, q_pe, kv_c_and_k_pe_cache, self.qk_rope_head_dim,
            attn_metadata, attn_output, attn_lse)

        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim])
        return attn_output

    def exec_kv_decode(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"
        k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
        )
        return k_pe, k_nope

    def exec_kv_prefill(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"
        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
            is_output_kv=True,
        )
        return k_pe, k_nope

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_nope.size(0)
        # shape of knope/k_pe for npu graph mode should be:
        # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
        actual_seq_lengths = None
        if self.enable_kv_nz:
            nz_fmt_last_dim = 16
            k_nope = k_nope.view(-1, self.num_kv_heads,
                                 self.kv_lora_rank // nz_fmt_last_dim,
                                 block_size, nz_fmt_last_dim)
            k_pe = k_pe.view(-1, self.num_kv_heads,
                             self.qk_rope_head_dim // nz_fmt_last_dim,
                             block_size, nz_fmt_last_dim)
        else:
            k_nope = k_nope.view(-1, self.num_kv_heads, block_size,
                                 self.kv_lora_rank)
            k_pe = k_pe.view(-1, self.num_kv_heads, block_size,
                             self.qk_rope_head_dim)

        attn_output_shape: tuple | None = None
        if attn_metadata.attn_state in [
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.DecodeOnly,
        ] and self.speculative_config is not None:
            # The right part layout indicates the layout of the attention
            # output. It is set to NTD to avoid the need for a transpose
            # operation after attention.
            input_layout = "TND_NTD"
            # TODO: If the driver is upgraded later, the contiguous function can be deleted.
            # Input shape: [num_tokens, num_heads, dim]
            q_nope = q_nope.view(num_tokens, self.num_heads, -1).contiguous()
            q_pe = q_pe.view(num_tokens, self.num_heads, -1)
            # Output shape: [num_heads, num_tokens, dim]
            attn_output_shape = (self.num_heads, num_tokens, self.kv_lora_rank)
            sparse_mode = 3
            attn_mask = attn_metadata.decode.attn_mask  # type:ignore
            actual_seq_lengths = decode_meta.actual_seq_lengths_q
        else:
            # The output layout is set to NBSD to eliminate the need for a
            # transpose operation after attention.
            if self.enable_kv_nz:
                # Input shape: [num_tokens, seq_len, num_heads, dim]
                input_layout = "BSND_NBSD"
                q_nope = q_nope.view(num_tokens, 1, self.num_heads,
                                     -1).contiguous()
                q_pe = q_pe.view(num_tokens, 1, self.num_heads, -1)
            else:
                # Input shape: [num_tokens, num_heads, seq_len, dim]
                input_layout = "BNSD_NBSD"
                q_nope = q_nope.view(num_tokens, self.num_heads, 1,
                                     -1).contiguous()
                q_pe = q_pe.view(num_tokens, self.num_heads, 1, -1)
            # Output shape: [num_heads, num_tokens, seq_len, dim]
            attn_output_shape = (self.num_heads, num_tokens, 1,
                                 self.kv_lora_rank)
            sparse_mode = 0
            attn_mask = None

        common_kwargs = {
            'query_rope': q_pe,
            'key_rope': k_pe,
            'num_heads': self.num_heads,
            'num_key_value_heads': self.num_kv_heads,
            'input_layout': input_layout,
            'atten_mask': attn_mask,
            'sparse_mode': sparse_mode,
            'scale': self.scale,
            'antiquant_mode': 0,
            'antiquant_scale': None,
            'block_table': decode_meta.block_table,
            'block_size': block_size,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": decode_meta.seq_lens_list,
        }
        forward_context: ForwardContext = get_forward_context()
        if forward_context.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        if forward_context.capturing:
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)

            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope, k_nope, k_nope, **common_kwargs)
                if forward_context.is_draft_model:
                    update_draft_graph_params_workspaces(num_tokens, workspace)
                else:
                    update_graph_params_workspaces(num_tokens, workspace)

            attn_output = torch.empty(attn_output_shape,
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            softmax_lse = torch.empty(num_tokens,
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)

            graph_params.attn_params[num_tokens].append(
                (weak_ref_tensors(q_nope), weak_ref_tensors(k_nope),
                 weak_ref_tensors(q_pe), weak_ref_tensors(k_pe),
                 self.num_heads, self.num_kv_heads, input_layout,
                 weak_ref_tensors(attn_mask) if attn_mask is not None else
                 None, sparse_mode, self.scale, decode_meta.block_table,
                 block_size, decode_meta.seq_lens_list, actual_seq_lengths,
                 weak_ref_tensors(attn_output), weak_ref_tensors(softmax_lse)))

            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                k_nope,
                **common_kwargs,
                workspace=workspace,
                out=[attn_output, softmax_lse])
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                q_nope, k_nope, k_nope, **common_kwargs)

        return self._v_up_proj(attn_output)

    def reorg_decode_q(self, decode_q_nope, decode_q_pe):
        return decode_q_nope, decode_q_pe

    def _mla_preprocess_only_decode(self, hidden_states, kv_cache,
                                    attn_metadata):
        bsz = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:bsz]

        cos_shape = attn_metadata.decode.cos.shape
        cos = attn_metadata.decode.cos.view(cos_shape[0], cos_shape[-1])
        sin = attn_metadata.decode.sin.view(cos_shape[0], cos_shape[-1])

        decode_k_nope, decode_k_pe = kv_cache[0], kv_cache[1]
        decode_q_nope = torch.empty(
            (hidden_states.shape[0], self.W_UK_T.shape[0],
             decode_k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        decode_q_pe = torch.empty(
            (hidden_states.shape[0], self.W_UK_T.shape[0],
             decode_k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            self.wd_qkv,
            self.deq_scale_qkv,
            self.gamma1,
            self.beta1,
            self.wu_q,
            self.qb_deq_scl,
            self.gamma2,
            cos,
            sin,
            self.W_UK_T,
            decode_k_nope,
            decode_k_pe,
            attn_metadata.slot_mapping[:bsz],
            quant_scale0=self.quant_scale0,
            quant_offset0=self.quant_offset0,
            bias0=self.quant_bias_qkv,
            quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1,
            bias1=self.qb_qt_bias,
            ctkv_scale=self.ctkv_scale,
            q_nope_scale=self.q_nope_scale,
            cache_mode="nzcache" if self.enable_kv_nz else "krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            q_out0=decode_q_nope,
            kv_cache_out0=decode_k_nope,
            q_out1=decode_q_pe,
            kv_cache_out1=decode_k_pe,
            enable_inner_out=False,
            inner_out=torch.tensor([], device=hidden_states.device))
        decode_q_nope = decode_q_nope.view(bsz, self.num_heads,
                                           self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads, -1)

        decode_q_nope, decode_q_pe = self.reorg_decode_q(
            decode_q_nope, decode_q_pe)

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        return decode_preprocess_res, None

    def mla_preprocess_prefill(self, q_c, kv_no_split, kv_cache,
                               attn_metadata):
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        prefill_kv_no_split = kv_no_split[num_decode_tokens:num_actual_tokens]
        prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
        prefill_q = self.q_proj(prefill_q_c)[0] \
            .view(-1, self.num_heads, self.qk_head_dim)
        prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
        prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
        cos = attn_metadata.prefill.cos
        sin = attn_metadata.prefill.sin
        prefill_slots = attn_metadata.slot_mapping[
            num_decode_tokens:num_actual_tokens]
        prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event = torch.npu.Event()
        prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(
            prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event.record()
        prefill_k_nope, prefill_value = self.kv_b_proj(
            prefill_k_c_normed)[0].view(
                -1, self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim).split(
                    [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0],
                                         self.num_kv_heads, -1)
        prefill_k_pe = prefill_k_pe.expand((*prefill_k_nope.shape[:-1], -1))
        return PrefillMLAPreprocessResult(prefill_q_nope, prefill_q_pe,
                                          prefill_k_nope, prefill_k_pe,
                                          prefill_value)

    def mla_preprocess_decode(self, q_c, kv_no_split, kv_cache, attn_metadata):
        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_q_c = q_c[:num_decode_tokens]
        cos = attn_metadata.decode.cos
        sin = attn_metadata.decode.sin
        decode_ql_nope, decode_q_pe = \
            self._q_proj_and_k_up_proj(decode_q_c)
        decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
        decode_slots = attn_metadata.slot_mapping[:num_decode_tokens:1]
        decode_kv_no_split = kv_no_split[:num_decode_tokens]
        decode_k_pe, decode_k_nope = self.exec_kv_decode(
            decode_kv_no_split, cos, sin, kv_cache, decode_slots)
        return DecodeMLAPreprocessResult(decode_ql_nope, decode_q_pe,
                                         decode_k_nope, decode_k_pe)

    def _mla_preprocess(self, layer_name, hidden_states, kv_cache,
                        attn_metadata, need_gather_q_kv):
        # MLA Preprocess:
        # 1. Perform fused_qkv_a_proj and q_a_layernorm to obtain q_c and kv_no_split
        # or
        #    Perform kv_a_proj_with_mqa to obtain kv_no_split
        # 2. If need_gather_q_kv, perform all_gather.
        # 3. Preprocess decode tokens, write kv cache and get:
        # decode_ql_nope, decode_q_pe, decode_k_pe, decode_k_nope
        # 4. Preprocess prefill tokens, write kv cache and get:
        # prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        if self.fused_qkv_a_proj is not None:
            maybe_npu_prefetch(inputs=self.fused_qkv_a_proj.weight,
                               dependency=hidden_states,
                               enabled=self.enable_prefetch)
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            # allgather need contiguous data
            kv_no_split = kv_no_split.contiguous()
        else:
            q_c = hidden_states
            kv_no_split = self.kv_a_proj_with_mqa(hidden_states)[0]

        # Process for Flash Comm V1
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c.contiguous(), need_gather_q_kv)
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split.contiguous(), need_gather_q_kv)

        for layer in (self.layer_sharding_kwargs or []):
            if is_hidden_layer(layer):
                reach_layer_for_shard_weight_series(layer)

        decode_preprocess_res = None
        prefill_preprocess_res = None
        if has_prefill:
            wait_for_kv_layer_from_connector(layer_name)
        # Preprocess for decode tokens
        if has_decode:
            decode_preprocess_res = self.mla_preprocess_decode(
                q_c, kv_no_split, kv_cache, attn_metadata)
        # Preprocess for prefill tokens
        if has_prefill:
            prefill_preprocess_res = self.mla_preprocess_prefill(
                q_c, kv_no_split, kv_cache, attn_metadata)
        return decode_preprocess_res, prefill_preprocess_res

    def get_num_actual_tokens(self, attn_metadata: M):
        return attn_metadata.num_actual_tokens

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            for layer in (self.layer_sharding_kwargs or []):
                if is_hidden_layer(layer):
                    reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        forward_context = get_forward_context()
        num_actual_tokens = self.get_num_actual_tokens(attn_metadata)
        assert attn_metadata.num_decodes is not None and \
               attn_metadata.num_prefills is not None and \
               attn_metadata.num_decode_tokens is not None

        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        o_proj_input_shape = (forward_context.num_tokens,
                              self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        # MLA Preprocess
        if self.enable_mlapo and not has_prefill:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv)
            decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess_only_decode(
                hidden_states, kv_cache, attn_metadata)
        else:
            decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
                layer_name, hidden_states, kv_cache, attn_metadata,
                need_gather_q_kv)
        if decode_preprocess_res is not None:
            # MLA Preprocess for decoding
            output_decode = self._forward_decode(decode_preprocess_res.ql_nope,
                                                 decode_preprocess_res.q_pe,
                                                 decode_preprocess_res.k_nope,
                                                 decode_preprocess_res.k_pe,
                                                 kv_cache[0].shape[1],
                                                 attn_metadata)

            o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            output_prefill = self._forward_prefill(
                prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                prefill_preprocess_res.value, kv_cache, attn_metadata)

            o_proj_input[num_decode_tokens:num_actual_tokens] = output_prefill
        # O proj
        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
        maybe_npu_prefetch(inputs=self.o_proj.weight,
                           dependency=o_proj_input,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=self.enable_prefetch)

        output[...] = self.o_proj(o_proj_input,
                                  is_prefill=prefill_preprocess_res
                                  is not None)[0]

        del o_proj_input

        if has_prefill:
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output_padded
