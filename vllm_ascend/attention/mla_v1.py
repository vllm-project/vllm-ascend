from dataclasses import dataclass
from typing import (TYPE_CHECKING, ClassVar, List, NamedTuple, Optional, Tuple,
                    Type, TypeVar)

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (get_dcp_group,
                              get_decode_context_model_parallel_rank,
                              get_decode_context_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backends.utils import AttentionCGSupport

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         maybe_save_kv_layer_to_connector,
                                         split_decodes_and_prefills,
                                         trans_rope_weight, transdata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (get_graph_params,
                                               get_mtp_graph_params,
                                               update_graph_params_workspaces)
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               is_enable_nz, prefill_context_parallel_enable,
                               weak_ref_tensors)
from vllm_ascend.worker.npu_input_batch import InputBatch

# isort: off
if prefill_context_parallel_enable():
    from vllm.distributed import (get_pcp_group,
                                  get_prefill_context_model_parallel_rank,
                                  get_prefill_context_model_parallel_world_size
                                  )
# isort: on
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024


class AscendMLABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA"

    @staticmethod
    def get_builder_cls():
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> Type["MLAAttentionImpl"]:
        return AscendMLAImpl


@dataclass
class AscendPCPMetadata:
    q_head_idx: torch.Tensor = None
    q_tail_idx: torch.Tensor = None
    kv_with_q_head_nomask_idx: torch.Tensor = None
    kv_with_q_head_mask_idx: torch.Tensor = None
    kv_with_q_tail_nomask_idx: torch.Tensor = None
    kv_with_q_tail_mask_idx: torch.Tensor = None
    attn_mask_seqlens: torch.Tensor = None
    head_attn_nomask_seqlens: torch.Tensor = None
    tail_attn_nomask_seqlens: torch.Tensor = None
    q_full_idx: torch.Tensor = None
    pcp_prefill_mask: torch.Tensor = None
    pcp_allgather_restore_idx: Optional[list[int]] = None


@dataclass
class AscendMLAPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""

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
        # for mla DCP & PCP
        padded_chunk_seq_lens_npu: torch.Tensor = None
        padded_local_chunk_seq_lens: Optional[list[list[int]]] = None
        local_context_lens_allranks: Optional[list[list[int]]] = None
        padded_local_cu_seq_lens: torch.Tensor = None
        cu_seq_lens_lst: Optional[list[list[int]]] = None
        chunk_size: Optional[int] = None

    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: list[int]
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_lens: int
    chunked_context: Optional[ChunkedContextMetadata] = None
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

    def __post_init__(self):
        pass
        # supported_head_sizes = AscendMLABackend.get_supported_head_sizes()
        # if self.head_dim is not None and self.head_dim \
        #         not in supported_head_sizes:
        #     raise ValueError(
        #         f"Only {supported_head_sizes} are supported for head_dim,",
        #         f"received {self.head_dim}.")


M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMLAMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # _attn_mask_builder = None
    def __init__(self,
                 kv_cache_spec,
                 layer_names,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 metadata_cls: Optional[AscendMLAMetadata] = None):
        self.metadata_cls: Optional[AscendMLAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendMLAMetadata  # type: ignore
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled

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

        self.pcp_size = get_prefill_context_model_parallel_world_size(
        ) if prefill_context_parallel_enable() else 1
        self.pcp_rank = get_prefill_context_model_parallel_rank(
        ) if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0
        self.cp_local_block_size = vllm_config.parallel_config.cp_kv_cache_interleave_size if prefill_context_parallel_enable(
        ) else 1
        self.cp_virtual_block_size = self.cp_local_block_size * self.dcp_size * self.pcp_size
        decode_max_num_seqs = getattr(scheduler_config, 'decode_max_num_seqs',
                                      0)
        max_num_seqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)
        self.batch_seq_mask_buf = torch.empty(max_num_seqs *
                                              self.decode_threshold,
                                              dtype=torch.uint8,
                                              device=device)

    def reorder_batch(self, input_batch: "InputBatch",
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
            common_attn_metadata.actual_seq_lengths_q[num_reqs] - actual_seq_lengths_q[-1] > FIA_SEQ_LEN_LIMIT
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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendMLAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata

        num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded if long_seq_metadata else None
        num_computed_tokens_of_pcp_dcp = long_seq_metadata.num_computed_tokens_of_pcp_dcp if long_seq_metadata else None

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device

        # If graph_pad_size > -1, mean is running in fullgraph mode.
        graph_pad_size = common_attn_metadata.graph_pad_size
        # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
        if graph_pad_size > num_reqs and self.speculative_config.disable_padded_drafter_batch:
            block_table = (
                common_attn_metadata.block_table_tensor[:graph_pad_size])
        else:
            block_table = (common_attn_metadata.block_table_tensor[:num_reqs])
        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        if self.pcp_size > 1:
            num_decodes_flatten = num_decodes * self.decode_threshold
            block_table = common_attn_metadata.block_table_tensor[:
                                                                  num_decodes_flatten
                                                                  +
                                                                  num_prefills]
        if num_actual_tokens_pcp_padded is None:
            num_actual_tokens_pcp_padded = num_actual_tokens

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens_pcp_padded]
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens_pcp_padded].long(
                                                         )

        if self.cos_cache is None:
            self.cos_cache = model.model.layers[
                model.model.start_layer].self_attn.rotary_emb.cos_cached
            self.sin_cache = model.model.layers[
                model.model.start_layer].self_attn.rotary_emb.sin_cached
        if self.cos_cache.dtype != self.model_config.dtype:  # type: ignore
            self.cos_cache = self.cos_cache.to(  # type: ignore
                self.model_config.dtype)  # type: ignore
            self.sin_cache = self.sin_cache.to(  # type: ignore
                self.model_config.dtype)  # type: ignore

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        query_lens = query_seq_lens_cpu[:num_reqs]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        num_computed_tokens_cpu = (seq_lens - query_lens)

        prefill_metadata = None
        chunked_context_metadata = None
        if num_prefills > 0:
            pcp_metadata = None
            common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
            if common_long_seq_metadata is not None:
                pcp_metadata = AscendPCPMetadata(
                    q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
                    q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
                    kv_with_q_head_nomask_idx=common_long_seq_metadata.
                    kv_with_q_head_nomask_idx_tensor,
                    kv_with_q_head_mask_idx=common_long_seq_metadata.
                    kv_with_q_head_mask_idx_tensor,
                    kv_with_q_tail_nomask_idx=common_long_seq_metadata.
                    kv_with_q_tail_nomask_idx_tensor,
                    kv_with_q_tail_mask_idx=common_long_seq_metadata.
                    kv_with_q_tail_mask_idx_tensor,
                    attn_mask_seqlens=common_long_seq_metadata.
                    attn_mask_seqlens,
                    head_attn_nomask_seqlens=common_long_seq_metadata.
                    head_attn_nomask_seqlens,
                    tail_attn_nomask_seqlens=common_long_seq_metadata.
                    tail_attn_nomask_seqlens,
                    q_full_idx=common_long_seq_metadata.q_full_idx,
                    pcp_prefill_mask=common_long_seq_metadata.pcp_prefill_mask
                    if long_seq_metadata else None,
                    pcp_allgather_restore_idx=long_seq_metadata.
                    pcp_allgather_restore_idx if long_seq_metadata else None)

            reqs_start = num_decodes  # prefill_start
            tokens_start = num_decode_tokens
            max_query_len = query_lens[reqs_start:].max().item()
            max_seq_lens = seq_lens[reqs_start:].max().item()
            prefill_query_start_loc = query_start_loc[
                reqs_start:] - query_start_loc[reqs_start]

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            if self.chunked_prefill_enabled and max_context_len_cpu > 0:
                max_context_chunk = (self.chunked_prefill_workspace_size //
                                     num_prefills_with_context_cpu)
                max_context_chunk = round_down(max_context_chunk,
                                               self.block_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)
                chunk_starts = torch.arange(num_chunks, dtype=torch.int32) \
                    .unsqueeze(1).expand(-1, num_prefills) * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
                cu_seq_lens_cpu = torch.zeros(num_chunks,
                                              num_prefills + 1,
                                              dtype=torch.int32,
                                              pin_memory=True)
                torch.cumsum(chunk_seq_lens,
                             dim=1,
                             out=cu_seq_lens_cpu[:, 1:],
                             dtype=torch.int32)

                if self.dcp_size * self.pcp_size > 1:
                    if num_computed_tokens_of_pcp_dcp is not None:
                        local_context_lens_allranks = torch.tensor(
                            num_computed_tokens_of_pcp_dcp[reqs_start:num_reqs]
                        ).reshape(-1, self.dcp_size * self.pcp_size)
                    # Note(qcs): The max local context lengths
                    # padded to `cp_local_block_size`.
                    padded_local_context_lens_cpu = (cdiv(
                        context_lens_cpu,
                        self.cp_virtual_block_size,
                    ) * self.cp_local_block_size)
                    padded_local_max_context_chunk_across_ranks = (cdiv(
                        max_context_chunk,
                        self.cp_virtual_block_size,
                    ) * self.cp_local_block_size)
                    local_chunk_starts = (
                        torch.arange(num_chunks,
                                     dtype=torch.int32).unsqueeze(1).expand(
                                         -1, num_prefills) *
                        padded_local_max_context_chunk_across_ranks)
                    local_chunk_ends = torch.min(
                        padded_local_context_lens_cpu.unsqueeze(0),
                        local_chunk_starts +
                        padded_local_max_context_chunk_across_ranks,
                    )
                    padded_local_chunk_seq_lens = (local_chunk_ends -
                                                   local_chunk_starts).clamp(
                                                       min=0)
                    padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
                        num_chunks,
                        num_prefills + 1,
                        dtype=torch.int32,
                        pin_memory=True)
                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )
                    chunked_context_metadata = \
                    AscendMLAPrefillMetadata.ChunkedContextMetadata(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=local_chunk_starts.to(device, non_blocking=True),
                        seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        chunk_seq_lens=chunk_seq_lens,
                        chunk_seq_lens_npu=chunk_seq_lens.npu(),
                        workspace=self.chunked_prefill_workspace,
                        padded_chunk_seq_lens_npu=padded_local_chunk_seq_lens.npu(),
                        padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
                        local_context_lens_allranks=local_context_lens_allranks.tolist(),
                        padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.to(
                            device, non_blocking=True
                        ),
                        cu_seq_lens_lst=cu_seq_lens_cpu.tolist(),
                        chunk_size=padded_local_max_context_chunk_across_ranks,
                    )
                else:
                    chunked_context_metadata = \
                        AscendMLAPrefillMetadata.ChunkedContextMetadata(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=chunk_starts.to(device, non_blocking=True),
                        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        chunk_seq_lens=chunk_seq_lens,
                        chunk_seq_lens_npu=chunk_seq_lens.npu(),
                        workspace=self.chunked_prefill_workspace,
                    )
            prefill_input_positions = input_positions[tokens_start:]
            cos = self.cos_cache[
                prefill_input_positions].unsqueeze(  # type: ignore
                    1).unsqueeze(2)
            sin = self.sin_cache[
                prefill_input_positions].unsqueeze(  # type: ignore
                    1).unsqueeze(2)
            prefill_metadata = AscendMLAPrefillMetadata(
                attn_mask=common_attn_metadata.attn_mask,
                query_lens=query_lens[reqs_start:].to(torch.int32),
                seq_lens=seq_lens,
                context_lens=seq_lens[reqs_start:],
                input_positions=prefill_input_positions,
                block_table=block_table[reqs_start:, ...],
                max_query_len=max_query_len,
                max_seq_lens=max_seq_lens,
                query_start_loc=prefill_query_start_loc,
                chunked_context=chunked_context_metadata,
                sin=sin,
                cos=cos,
                pcp_metadata=pcp_metadata,
            )
            if self.pcp_size > 1:
                prefill_metadata.block_table = block_table[
                    num_decodes_flatten:, ...]

        decode_metadata = None
        if num_decodes > 0:
            cos = common_attn_metadata.cos
            sin = common_attn_metadata.sin
            # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
            actual_seq_lengths_q = query_start_loc[1:num_decodes + 1].tolist()
            max_seq_lens = seq_lens[:num_decodes].max().item()
            seq_lens = seq_lens[:num_decodes]
            input_positions = input_positions[:num_decode_tokens]
            if self.pcp_size > 1:
                # For pcp + spec decode, we flatten seq_lens and block_table
                # to avoid irregular spec_attn_mask shape
                block_table = block_table[:num_decodes_flatten, ...]
            else:
                block_table = block_table[:num_decodes, ...]
            # NOTE: Currently, MTP-fullgraph is incompatibility pcp
            # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
            if graph_pad_size > num_decodes and \
                    self.speculative_config.disable_padded_drafter_batch:
                block_table = block_table[:graph_pad_size, ...]
            seq_lens_list = seq_lens.tolist()

            if num_computed_tokens_of_pcp_dcp is not None:
                # [bs, pcp_size, dcp_size]
                num_computed_tokens_of_cp_dcp_array = np.array(
                    num_computed_tokens_of_pcp_dcp)[:num_decodes *
                                                    self.decode_threshold]

                cp_seq_len = num_computed_tokens_of_cp_dcp_array[:,
                                                                 self.pcp_rank,
                                                                 self.dcp_rank]
                cp_seq_len = torch.tensor(cp_seq_len, dtype=torch.int32)
                batch_seq_mask = (cp_seq_len == 0)
                self.batch_seq_mask_buf[:batch_seq_mask.shape[0]].copy_(
                    batch_seq_mask, non_blocking=True)
                batch_seq_mask = self.batch_seq_mask_buf[:batch_seq_mask.
                                                         shape[0]]
                cp_seq_len = torch.where(cp_seq_len == 0, 1, cp_seq_len)
            else:
                cp_seq_len, batch_seq_mask = None, None

            if graph_pad_size > num_reqs:
                if self.speculative_config.disable_padded_drafter_batch:
                    num_reqs_pad_size = graph_pad_size - num_reqs
                    actual_seq_lengths_q = self.pad_actual_seq_len_q_mtp_disable_pad(
                        num_reqs_pad_size, num_reqs, actual_seq_lengths_q)
                    seq_lens_list = seq_lens_list + [0] * (graph_pad_size - \
                                                           num_decodes)
                    num_block_pad_size = graph_pad_size - block_table.shape[0]
                    if num_block_pad_size > 0:
                        block_table_padding = torch.zeros(
                            (num_block_pad_size, ) + block_table.shape[1:],
                            dtype=block_table.dtype,
                            device=block_table.device)
                        block_table = torch.cat(
                            [block_table, block_table_padding], dim=0)
                else:
                    num_token_pad_size = graph_pad_size - num_decode_tokens
                    num_reqs_pad_size = (
                        graph_pad_size //
                        common_attn_metadata.decode_token_per_req - num_reqs)
                    num_block_table_pad_size = (
                        graph_pad_size //
                        common_attn_metadata.decode_token_per_req -
                        num_decodes)
                    seq_lens_list = seq_lens.tolist() + [0] * num_reqs_pad_size
                    slot_padding = torch.full((num_token_pad_size, ),
                                              PAD_SLOT_ID,
                                              dtype=slot_mapping.dtype,
                                              device=slot_mapping.device)
                    slot_mapping = torch.cat([slot_mapping, slot_padding])
                    block_table_padding = torch.zeros(
                        (num_block_table_pad_size, ) + block_table.shape[1:],
                        dtype=block_table.dtype,
                        device=block_table.device)
                    block_table = torch.cat([block_table, block_table_padding],
                                            dim=0)
                    position_padding = torch.zeros(
                        num_token_pad_size,
                        dtype=input_positions.dtype,
                        device=input_positions.device)
                    input_positions = torch.cat(
                        [input_positions, position_padding])
                    actual_seq_lengths_q = self.pad_actual_seq_len_q_mtp_enable_pad(
                        num_reqs_pad_size, num_reqs, actual_seq_lengths_q,
                        common_attn_metadata)

            # TODO: After the fullgraph supports MTP, the if branch needs to deleted
            assert self.cos_cache is not None
            assert self.sin_cache is not None
            if cos is None and sin is None:
                cos = self.cos_cache[
                    input_positions].unsqueeze(  # type: ignore
                        1).unsqueeze(2)
                sin = self.sin_cache[
                    input_positions].unsqueeze(  # type: ignore
                        1).unsqueeze(2)

                decode_metadata = AscendMLADecodeMetadata(
                    input_positions=input_positions,
                    block_table=block_table,
                    seq_lens=seq_lens,
                    seq_lens_list=seq_lens_list,
                    max_seq_lens=max_seq_lens,
                    attn_mask=common_attn_metadata.spec_attn_mask,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    sin=sin,
                    cos=cos,
                    cp_seq_len=cp_seq_len,
                    batch_seq_mask=batch_seq_mask)
            else:
                cos[:num_decode_tokens,
                    ...] = self.cos_cache[input_positions].unsqueeze(
                        1).unsqueeze(2)
                sin[:num_decode_tokens,
                    ...] = self.sin_cache[input_positions].unsqueeze(
                        1).unsqueeze(2)

                decode_metadata = AscendMLADecodeMetadata(
                    input_positions=input_positions,
                    block_table=block_table,
                    seq_lens=seq_lens,
                    seq_lens_list=seq_lens_list,
                    max_seq_lens=max_seq_lens,
                    attn_mask=common_attn_metadata.spec_attn_mask,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    sin=sin[:num_decode_tokens, ...],
                    cos=cos[:num_decode_tokens, ...],
                    cp_seq_len=cp_seq_len,
                    batch_seq_mask=batch_seq_mask)

        return self.metadata_cls(  # type: ignore
            num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            query_lens=query_lens.tolist(),
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=common_attn_metadata.attn_mask,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=block_table,
            seq_lens=seq_lens,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        }:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                model=model,
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
    ) -> None:
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
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.q_a_layernorm = kwargs.get('q_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_prefetch = ascend_config.weight_prefetch_config.enabled
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz

        vllm_config = get_current_vllm_config()
        self.ring_mla_mask_size = 512
        self.prefill_mask = None

        self.speculative_config = vllm_config.speculative_config
        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        self.pcp_size = get_prefill_context_model_parallel_world_size(
        ) if prefill_context_parallel_enable() else 1
        self.pcp_rank = get_prefill_context_model_parallel_rank(
        ) if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group(
        ).device_group if self.pcp_size > 1 else None

        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group(
        ).device_group if self.dcp_size > 1 else None

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group(
        ).device_group if self.tp_size > 1 else None

    def _v_up_proj(self, x):
        if x.dtype in [torch.float16, torch.bfloat16] \
                and hasattr(torch.ops._C_ascend, "batch_matmul_transpose") \
                and not self.dcp_size * self.pcp_size > 1:
            x = x.view(-1, self.num_heads, self.kv_lora_rank)
            b, _, _ = x.shape
            res = torch.empty((b, self.num_heads, self.v_head_dim),
                              dtype=x.dtype,
                              device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return x

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = self.q_proj(x)[0]\
            .view(-1, self.num_heads, self.qk_head_dim)\
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            # Weight will be reshaped next. To be on the safe side, the format
            # of the weight should be reverted to FRACTAL_AND.
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_ND)
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
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

        # Function `get_and_maybe_dequant_weights` will cast the weights to
        # FRACTAL_AND. So we need to cast to FRACTAL_NZ again.
        if is_enable_nz():
            self.kv_b_proj.weight.data = torch_npu.npu_format_cast(
                self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_NZ)

        # Waiting for BMM NZ support
        # self.W_UV.data = torch_npu.npu_format_cast(self.W_UV.data, 29)
        # self.W_UK_T.data = torch_npu.npu_format_cast(self.W_UK_T.data, 29)

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
        self.beta1 = self.q_a_layernorm.bias.data
        self.gamma2 = self.kv_a_layernorm.weight.data
        self.quant_scale0 = self.fused_qkv_a_proj.input_scale.data
        self.quant_offset0 = self.fused_qkv_a_proj.input_offset.data
        self.quant_scale1 = self.q_proj.input_scale.data
        self.quant_offset1 = self.q_proj.input_offset.data
        self.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        self.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

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
            context_seq_len_npu = prefill_metadata.chunked_context.chunk_seq_lens_npu[
                i]
            seq_len = torch.stack([current_seq_len, context_seq_len])
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

            if self.dcp_size * self.pcp_size > 1:
                context_seq_len_npu = prefill_metadata.chunked_context.padded_chunk_seq_lens_npu[
                    i]

            torch_npu.atb.npu_paged_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                context_seq_len_npu,
                seq_starts=prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )

            cache_kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
            if self.dcp_size > 1:
                cache_kv_c_k_pe = get_dcp_group().all_gather(
                    cache_kv_c_k_pe, 0)

            if self.pcp_size > 1:
                cache_kv_c_k_pe = get_pcp_group().all_gather(
                    cache_kv_c_k_pe, 0)

            if self.dcp_size * self.pcp_size > 1:
                allgatered_kv_c_normed, allgatered_k_pe = cache_kv_c_k_pe.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed, k_pe = self._reorg_kvcache(
                    allgatered_kv_c_normed,
                    allgatered_k_pe,
                    padded_local_chunk_seq_lens_lst=prefill_metadata.
                    chunked_context.padded_local_chunk_seq_lens[i],
                    local_context_lens_allranks=prefill_metadata.
                    chunked_context.local_context_lens_allranks,
                    sum_seq_len=prefill_metadata.chunked_context.
                    cu_seq_lens_lst[i][-1],
                    max_seq_len=prefill_metadata.chunked_context.
                    max_seq_lens[i],
                    chunk_size=prefill_metadata.chunked_context.chunk_size,
                    chunk_idx=i,
                    toks=toks,
                )

            kv_c_normed = kv_c_normed.squeeze()
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))

            if self.pcp_size > 1:
                mask = attn_metadata.prefill.pcp_metadata.pcp_prefill_mask
            else:
                mask = self.prefill_mask
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
        if self.prefill_mask is None:
            if q_nope.dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            prefill_mask = torch.triu(
                torch.ones(self.ring_mla_mask_size,
                           self.ring_mla_mask_size,
                           device=q_nope.device,
                           dtype=q_nope.dtype), 1)
            self.prefill_mask = torch.where(prefill_mask == 1, mask_value,
                                            0).to(q_nope.dtype)
        torch_npu.atb.npu_ring_mla(q_nope=q_nope,
                                   q_rope=q_pe,
                                   k_nope=k_nope,
                                   k_rope=k_pe,
                                   value=value,
                                   mask=self.prefill_mask,
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
        attn_output, attn_lse = self._compute_prefill_context( \
            q_nope, q_pe, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse)

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
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"
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
            k_nope = k_nope.view(-1, self.num_kv_heads,
                                 self.kv_lora_rank // 16, block_size, 16)
            k_pe = k_pe.view(-1, self.num_kv_heads,
                             self.qk_rope_head_dim // 16, block_size, 16)
            input_layout = "BSND"
        else:
            k_nope = k_nope.view(-1, self.num_kv_heads, block_size,
                                 self.kv_lora_rank)
            k_pe = k_pe.view(-1, self.num_kv_heads, block_size,
                             self.qk_rope_head_dim)
            input_layout = "BNSD"

        if attn_metadata.attn_state in [
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.DecodeOnly,
        ] and self.speculative_config is not None:
            # Use TND layout for pure SpecDecoding and SpecDecoding in ChunkedPrefill
            input_layout = "TND"
            # [bs * q_seq_len, num_heads_per_rank, dim]
            # TODO: If the driver is upgraded later, the contiguous function can be deleted.
            q_nope = q_nope.view(num_tokens, self.num_heads, -1).contiguous()
            q_pe = q_pe.view(num_tokens, self.num_heads, -1)
            sparse_mode = 3
            spec_attn_mask = attn_metadata.decode.attn_mask  # type:ignore
            actual_seq_lengths = decode_meta.actual_seq_lengths_q
        else:
            if self.enable_kv_nz:
                q_nope = q_nope.view(num_tokens, 1, self.num_heads,
                                     -1).contiguous()
                q_pe = q_pe.view(num_tokens, 1, self.num_heads, -1)
            else:
                q_nope = q_nope.view(num_tokens, self.num_heads, 1,
                                     -1).contiguous()
                q_pe = q_pe.view(num_tokens, self.num_heads, 1, -1)
            sparse_mode = 0
            spec_attn_mask = None

        common_kwargs = {
            'query_rope': q_pe,
            'key_rope': k_pe,
            'num_heads': self.num_heads,
            'num_key_value_heads': self.num_kv_heads,
            'input_layout': input_layout,
            'atten_mask': spec_attn_mask,
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
        if forward_context.is_mtp_model:
            graph_params = get_mtp_graph_params()
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
                update_graph_params_workspaces(num_tokens, workspace)

            attn_output = torch.empty_like(q_nope)
            softmax_lse = torch.empty(num_tokens,
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)

            graph_params.attn_params[num_tokens].append(
                (weak_ref_tensors(q_nope), weak_ref_tensors(k_nope),
                 weak_ref_tensors(q_pe), weak_ref_tensors(k_pe),
                 self.num_heads, self.num_kv_heads, input_layout,
                 weak_ref_tensors(spec_attn_mask) if spec_attn_mask is not None
                 else None, sparse_mode, self.scale, decode_meta.block_table,
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

    def _mla_decode_preprocess(self, hidden_states, kv_cache, attn_metadata):
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
            attn_metadata.slot_mapping[:bsz].flatten(),
            quant_scale0=self.quant_scale0,
            quant_offset0=self.quant_offset0,
            bias0=self.quant_bias_qkv,
            quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1,
            bias1=self.qb_qt_bias,
            ctkv_scale=self.ctkv_scale,
            q_nope_scale=self.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            q_out0=decode_q_nope,
            kv_cache_out0=decode_k_nope,
            q_out1=decode_q_pe,
            kv_cache_out1=decode_k_pe,
        )
        decode_q_nope = decode_q_nope.view(bsz, self.num_heads,
                                           self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads, -1)

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        return decode_preprocess_res, None

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
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
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

        decode_preprocess_res = None
        prefill_preprocess_res = None
        if has_prefill:
            wait_for_kv_layer_from_connector(layer_name)
        # Preprocess for decode tokens
        if has_decode:
            decode_q_c = q_c[:num_decode_tokens]
            cos = attn_metadata.decode.cos
            sin = attn_metadata.decode.sin
            decode_ql_nope, decode_q_pe = \
                self._q_proj_and_k_up_proj(decode_q_c)
            if self.dcp_size > 1:
                decode_q_no_split = torch.cat([decode_ql_nope, decode_q_pe],
                                              dim=-1)
                decode_q_no_split = get_dcp_group().all_gather(
                    decode_q_no_split, 1)
                decode_ql_nope, decode_q_pe = decode_q_no_split.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
            decode_slots = attn_metadata.slot_mapping[:num_decode_tokens *
                                                      self.pcp_size:self.
                                                      pcp_size]
            decode_kv_no_split = kv_no_split[:num_decode_tokens]
            decode_k_pe, decode_k_nope = self.exec_kv_decode(
                decode_kv_no_split, cos, sin, kv_cache, decode_slots)
            decode_preprocess_res = DecodeMLAPreprocessResult(
                decode_ql_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        # Preprocess for prefill tokens
        if has_prefill:
            if self.pcp_size > 1:
                num_actual_tokens = (attn_metadata.num_actual_tokens_pcp_padded
                                     - self.pcp_size * num_decode_tokens
                                     ) // self.pcp_size + num_decode_tokens
            prefill_kv_no_split = kv_no_split[
                num_decode_tokens:num_actual_tokens]
            prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
            prefill_q = self.q_proj(prefill_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
            if self.pcp_size > 1:
                cos = attn_metadata.prefill.cos[:num_actual_tokens -
                                                num_decode_tokens]
                sin = attn_metadata.prefill.sin[:num_actual_tokens -
                                                num_decode_tokens]
            else:
                cos = attn_metadata.prefill.cos
                sin = attn_metadata.prefill.sin
            prefill_slots = attn_metadata.slot_mapping[
                num_decode_tokens:num_actual_tokens]
            prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
            if self.pcp_size > 1:
                prefill_kv_no_split = kv_no_split[:num_actual_tokens]
                kv_c, k_pe = prefill_kv_no_split.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
                assert len(
                    kv_cache
                ) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
                kv_c_normed = kv_c_normed.view(
                    [num_actual_tokens, self.num_kv_heads, -1])
                k_pe = k_pe.unsqueeze(1)
                prefill_k_pe = k_pe
                prefill_k_pe[
                    num_decode_tokens:num_actual_tokens] = self.rope_single(
                        prefill_k_pe[num_decode_tokens:num_actual_tokens], cos,
                        sin)
                prefill_k_c_normed = kv_c_normed[:num_actual_tokens]
                prefill_kv_c_k_pe = torch.cat(
                    [prefill_k_c_normed, prefill_k_pe], dim=-1)
                prefill_kv_c_k_pe = get_pcp_group().all_gather(
                    prefill_kv_c_k_pe, 0)
                prefill_kv_c_k_pe = torch.index_select(
                    prefill_kv_c_k_pe, 0, attn_metadata.prefill.pcp_metadata.
                    pcp_allgather_restore_idx)
                prefill_kv_c_k_pe = prefill_kv_c_k_pe[num_decode_tokens *
                                                      self.pcp_size:]
                prefill_k_c_normed, prefill_k_pe = prefill_kv_c_k_pe.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed, k_pe = prefill_k_c_normed, prefill_k_pe
                prefill_k_c_normed = prefill_k_c_normed.squeeze()
                slot_mapping = attn_metadata.slot_mapping[self.pcp_size *
                                                          num_decode_tokens:]
                torch_npu._npu_reshape_and_cache(key=kv_c_normed,
                                                 value=k_pe,
                                                 key_cache=kv_cache[0],
                                                 value_cache=kv_cache[1],
                                                 slot_indices=slot_mapping)
            else:
                prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(
                    prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
            prefill_k_nope, prefill_value = self.kv_b_proj(
                prefill_k_c_normed)[0].view(
                    -1, self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim).split(
                        [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            if not self.pcp_size > 1:
                prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0],
                                                 self.num_kv_heads, -1)
            prefill_k_pe = prefill_k_pe.expand(
                (*prefill_k_nope.shape[:-1], -1))
            prefill_preprocess_res = PrefillMLAPreprocessResult(
                prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe,
                prefill_value)
        return decode_preprocess_res, prefill_preprocess_res

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
            return output.fill_(0)
        if self.pcp_size > 1:
            num_actual_tokens = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
        else:
            num_actual_tokens = attn_metadata.num_actual_tokens
        assert attn_metadata.num_decodes is not None and \
        attn_metadata.num_prefills is not None and \
        attn_metadata.num_decode_tokens is not None
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        o_proj_input_shape = (get_forward_context().num_tokens,
                              self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        # MLA Preprocess
        forward_context = get_forward_context()
        if (self.enable_mlapo and
            (attn_metadata is None or not forward_context.with_prefill)):
            decode_preprocess_res, prefill_preprocess_res = self._mla_decode_preprocess(
                hidden_states, kv_cache, attn_metadata)
        else:
            decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
                layer_name, hidden_states, kv_cache, attn_metadata,
                need_gather_q_kv)

        if decode_preprocess_res is not None:
            # MLA Preprocess for decoding
            if self.pcp_size * self.dcp_size > 1:
                output_decode = self._forward_decode_pcp_dcp(
                    decode_preprocess_res.ql_nope,
                    decode_preprocess_res.q_pe,
                    decode_preprocess_res.k_nope,
                    decode_preprocess_res.k_pe,
                    kv_cache[0].shape[1],
                    attn_metadata,
                )
            else:
                output_decode = self._forward_decode(
                    decode_preprocess_res.ql_nope, decode_preprocess_res.q_pe,
                    decode_preprocess_res.k_nope, decode_preprocess_res.k_pe,
                    kv_cache[0].shape[1], attn_metadata)

            o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            if self.pcp_size > 1:
                output_prefill = self._forward_prefill_cp(
                    prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                    prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                    prefill_preprocess_res.value, kv_cache, attn_metadata)
            else:
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

        has_prefill = attn_metadata.num_prefills > 0
        if has_prefill:
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output_padded

    def _forward_prefill_cp(
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
        assert attn_metadata.prefill.pcp_metadata is not None
        num_tokens = q_nope.size(0)
        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.pcp_metadata.q_head_idx
        q_tail_idx = attn_metadata.prefill.pcp_metadata.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx
        attn_mask_seqlens = attn_metadata.prefill.pcp_metadata.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens
        mask = attn_metadata.prefill.pcp_metadata.pcp_prefill_mask
        output_head, lse_head = self._attention_with_mask_and_nomask(
            q_nope=torch.index_select(q_nope, 0, q_head_idx),
            q_pe=torch.index_select(q_pe, 0, q_head_idx),
            k_nope=k_nope,
            k_pe=k_pe,
            value=value,
            kv_mask_idx=kv_with_q_head_mask_idx,
            kv_nomask_idx=kv_with_q_head_nomask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            attn_nomask_seqlens=head_attn_nomask_seqlens,
            mask=mask)

        output_tail, lse_tail = self._attention_with_mask_and_nomask(
            q_nope=torch.index_select(q_nope, 0, q_tail_idx),
            q_pe=torch.index_select(q_pe, 0, q_tail_idx),
            k_nope=k_nope,
            k_pe=k_pe,
            value=value,
            kv_mask_idx=kv_with_q_tail_mask_idx,
            kv_nomask_idx=kv_with_q_tail_nomask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            attn_nomask_seqlens=tail_attn_nomask_seqlens,
            mask=mask)

        q_full_idx = attn_metadata.prefill.pcp_metadata.q_full_idx
        attn_output = torch.index_select(
            torch.cat([output_head, output_tail], dim=0), 0, q_full_idx)
        attn_lse = torch.index_select(torch.cat([lse_head, lse_tail], dim=1),
                                      1, q_full_idx)

        output, _ = self._compute_prefill_context( \
            q_nope, q_pe, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse)

        output = output.reshape([num_tokens, self.num_heads * self.v_head_dim])

        return output

    def _attention_with_mask_and_nomask(
            self, q_nope: torch.Tensor, q_pe: torch.Tensor,
            k_nope: torch.Tensor, k_pe: torch.Tensor, value: torch.Tensor,
            kv_mask_idx: torch.Tensor, kv_nomask_idx: torch.Tensor,
            attn_mask_seqlens: torch.Tensor, attn_nomask_seqlens: torch.Tensor,
            mask: torch.Tensor):
        attn_output = torch.empty(q_nope.shape[0],
                                  self.num_heads,
                                  self.v_head_dim,
                                  dtype=k_pe.dtype,
                                  device=k_pe.device)
        attn_lse = torch.empty(self.num_heads,
                               q_pe.shape[0],
                               dtype=torch.float32,
                               device=k_pe.device)
        # mask
        k_nope_mask = torch.index_select(k_nope, 0, kv_mask_idx)
        value_mask = torch.index_select(value, 0, kv_mask_idx)
        k_pe_mask = torch.index_select(k_pe, 0, kv_mask_idx)
        torch_npu.atb.npu_ring_mla(q_nope=q_nope,
                                   q_rope=q_pe,
                                   k_nope=k_nope_mask,
                                   k_rope=k_pe_mask,
                                   value=value_mask,
                                   mask=mask,
                                   seqlen=attn_mask_seqlens,
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

        # nomask
        if kv_nomask_idx.shape[0] == 0:
            return attn_output, attn_lse

        k_nope_nomask = torch.index_select(k_nope, 0, kv_nomask_idx)
        value_nomask = torch.index_select(value, 0, kv_nomask_idx)
        k_pe_nomask = torch.index_select(k_pe, 0, kv_nomask_idx)
        torch_npu.atb.npu_ring_mla(q_nope=q_nope,
                                   q_rope=q_pe,
                                   k_nope=k_nope_nomask,
                                   k_rope=k_pe_nomask,
                                   value=value_nomask,
                                   mask=mask,
                                   seqlen=attn_nomask_seqlens,
                                   head_num=self.num_heads,
                                   kv_head_num=self.num_heads,
                                   pre_out=attn_output,
                                   prev_lse=attn_lse,
                                   qk_scale=self.scale,
                                   kernel_type="kernel_type_high_precision",
                                   mask_type="no_mask",
                                   input_layout="type_bsnd",
                                   calc_type="calc_type_default",
                                   output=attn_output,
                                   softmax_lse=attn_lse)
        return attn_output, attn_lse

    def _forward_decode_pcp_dcp(
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
        if self.dcp_size > 1:
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        k_nope = k_nope.view(-1, block_size, self.num_kv_heads,
                             self.kv_lora_rank)
        k_pe = k_pe.view(-1, block_size, self.num_kv_heads,
                         self.qk_rope_head_dim)
        q_nope = q_nope.view(num_tokens, num_heads, -1)
        q_pe = q_pe.view(num_tokens, num_heads, -1)
        # use pcp & dcp split computed token nums from scheduler to compute actual seq_len and seq_mask
        seq_len = decode_meta.cp_seq_len

        common_kwargs = {
            "return_lse": True,
            "calc_type": "calc_type_ring",
        }
        graph_params = get_graph_params()
        forward_context: ForwardContext = get_forward_context()
        if forward_context.capturing:
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu.atb._npu_multi_head_latent_attention_get_workspace(
                    q_nope, q_pe, k_nope, k_pe, decode_meta.block_table,
                    seq_len, num_heads, self.scale, self.num_kv_heads,
                    **common_kwargs)
                update_graph_params_workspaces(num_tokens, workspace)
            attn_output = torch.empty_like(q_nope)
            softmax_lse = torch.empty((num_tokens, num_heads, 1),
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            graph_params.attn_params[num_tokens].append(
                (weak_ref_tensors(q_nope), weak_ref_tensors(q_pe),
                 weak_ref_tensors(k_nope), weak_ref_tensors(k_pe),
                 decode_meta.block_table, seq_len, num_heads, self.scale,
                 self.num_kv_heads, weak_ref_tensors(attn_output),
                 weak_ref_tensors(softmax_lse)))
            torch.npu.graph_task_group_begin(stream)
            torch_npu.atb.npu_multi_head_latent_attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                decode_meta.block_table,
                seq_len,
                num_heads,
                self.scale,
                self.num_kv_heads,
                **common_kwargs,
                workspace=workspace,
                output=attn_output,
                lse=softmax_lse)
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_output = torch.empty_like(q_nope)
            softmax_lse = torch.empty((num_tokens, num_heads, 1),
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            torch_npu.atb.npu_multi_head_latent_attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                decode_meta.block_table,
                seq_len,
                num_heads,
                self.scale,
                self.num_kv_heads,
                return_lse=True,
                calc_type="calc_type_ring",
                output=attn_output,
                lse=softmax_lse)

        # Update out&lse
        attn_out_lse_list = self._process_attn_out_lse(attn_output,
                                                       softmax_lse,
                                                       decode_meta)
        attn_output = self._npu_attention_update(attn_out_lse_list)
        return self._v_up_proj(attn_output)

    def _npu_attention_update(
            self, attn_out_lse_list: List[torch.Tensor]) -> torch.Tensor:
        attn_out_split_cp = []
        attn_lse_split_cp = []

        for attn_out_lse in attn_out_lse_list:
            attn_out_allgather, attn_lse_allgather = self._out_lse_reshape(
                *torch.split(attn_out_lse, [self.kv_lora_rank, 1], dim=-1))
            attn_out_split_cp.append(attn_out_allgather)
            attn_lse_split_cp.append(attn_lse_allgather)
        attn_out, _ = torch_npu.npu_attention_update(attn_lse_split_cp,
                                                     attn_out_split_cp, 0)
        attn_out = attn_out.view(-1, attn_out_lse_list[0].shape[1],
                                 self.kv_lora_rank)
        return attn_out

    def _out_lse_reshape(self, attn_out: torch.Tensor,
                         attn_lse: torch.Tensor) -> torch.Tensor:
        attn_out = attn_out.contiguous().view(
            attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
        attn_lse = attn_lse.contiguous().view(
            attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
        return attn_out, attn_lse

    def _process_attn_out_lse(
        self,
        attn_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        decode_meta: AscendMLADecodeMetadata,
    ) -> List[torch.Tensor]:
        attn_out_lse_list = []
        out_mask = decode_meta.batch_seq_mask[:, None,
                                              None].expand_as(attn_output)
        attn_output = torch.where(out_mask, 0, attn_output)
        lse_mask = decode_meta.batch_seq_mask[:, None,
                                              None].expand_as(softmax_lse)
        softmax_lse = torch.where(lse_mask, -torch.inf, softmax_lse)

        softmax_lse = softmax_lse.to(torch.float32)
        attn_output = attn_output.to(torch.float32)
        # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
        attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
        if self.dcp_size > 1:
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   attn_out_lse,
                                   group=self.dcp_group)
            # permute: [num_heads, v_head_dim+1, bs] -> [bs, num_heads, v_head_dim+1]
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            if self.pcp_size > 1:
                attn_out_lse = attn_out_lse_all2all.contiguous()
            attn_out_lse_list = list(
                torch.chunk(attn_out_lse_all2all, self.dcp_size, dim=1))

        if self.pcp_size > 1:
            # AllGather out&lse within PCP group
            attn_out_lse_list = [
                torch.empty_like(attn_out_lse) for _ in range(self.pcp_size)
            ]
            dist.all_gather(attn_out_lse_list,
                            attn_out_lse,
                            group=self.pcp_group)
        if self.dcp_size > 1 and self.pcp_size > 1:
            attn_out_lse_list_pcp_dcp = []
            for s in attn_out_lse_list:
                attn_out_lse_list_split = list(
                    torch.chunk(s, self.dcp_size, dim=1))
                attn_out_lse_list_pcp_dcp += attn_out_lse_list_split
            attn_out_lse_list = attn_out_lse_list_pcp_dcp

        return attn_out_lse_list

    def _reorg_kvcache(
        self,
        allgatered_kv_c_normed: torch.Tensor,
        allgatered_k_pe: torch.Tensor,
        padded_local_chunk_seq_lens_lst: list[int],
        local_context_lens_allranks: list[list[int]],
        sum_seq_len: int,
        max_seq_len: int,
        chunk_size: int,
        chunk_idx: int,
        toks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        reorg and unpad kvcache after cp local gather to tp layout for attn kernel.
        e.g.
        kv_c_normed in rank0 = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...]
        kv_c_normed in rank1 = [T0_4, T0_5, pad, pad, T1_2, pad, ...]
        allgatered_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,
                                T0_4, T0_5, pad, pad, T1_2, pad, ...]
        -> reorganized_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                                    T1_0, T1_1, T1_2, ...]
        Args:
            padded_local_chunk_seq_lens_lst: local chunk context lengths
                under current CP rank.
            local_context_lens_allranks: local context lengths on each CP rank.
            sum_seq_len: the sum of cp_chunk_seq_lens_lst.
            max_seq_len: the max value of cp_chunk_seq_lens_lst.
            chunk_size: the local padded max context chunk from
                chunked_context_metadata building.
            chunk_idx: chunk idx of chunked_prefill.
            toks: the number of tokens for local gather cache.
        """
        kv_c_segments = []
        k_pe_segments = []
        src_token_idx = 0
        max_seq_len_check = 0
        for padded_local_chunk_seq_len, local_context_lens in zip(
                padded_local_chunk_seq_lens_lst, local_context_lens_allranks):
            cur_seq_len = 0
            for rank, local_context_len in enumerate(local_context_lens):
                # Note(qcs): We split the context into multiple chunks,
                # depending on the size of the workspace.
                # local_context in dcp0:   |-----------------|
                # local_context in dcp1:   |--------------|
                # n*padded_local_chunk:    |-----|-----|-----|
                # local_chunk_len in dcp1: |-----|-----|--|
                # so we need update the last chunk length in dcp1.
                local_chunk_len = min(
                    max(0, local_context_len - chunk_idx * chunk_size),
                    padded_local_chunk_seq_len,
                )
                if local_chunk_len != 0:
                    kv_c_segment = allgatered_kv_c_normed[rank * toks +
                                                          src_token_idx:rank *
                                                          toks +
                                                          src_token_idx +
                                                          local_chunk_len]
                    k_pe_segment = allgatered_k_pe[rank * toks +
                                                   src_token_idx:rank * toks +
                                                   src_token_idx +
                                                   local_chunk_len]
                    kv_c_segments.append(kv_c_segment)
                    k_pe_segments.append(k_pe_segment)
                    cur_seq_len += local_chunk_len
            max_seq_len_check = max(max_seq_len_check, cur_seq_len)
            src_token_idx += padded_local_chunk_seq_len
        reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
        reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
        assert reorganized_kv_c_normed.shape[0] == sum_seq_len
        assert reorganized_k_pe.shape[0] == sum_seq_len
        assert max_seq_len_check == max_seq_len
        return reorganized_kv_c_normed, reorganized_k_pe
