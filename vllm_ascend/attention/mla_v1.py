from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionLayer,
                                              AttentionMetadata,
                                              MLAAttentionImpl)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.utils import cdiv, round_down
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              get_tp_group,
                              get_context_model_parallel_world_size,
                              get_context_model_parallel_rank,
                              get_cp_group)
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.multistream.base import MSAttentionMetadataSplitConfig
from vllm_ascend.multistream.context import get_multistream_comm_context
from vllm_ascend.multistream.ms_split import model_input_split_v1_mla_attn
from vllm_ascend.ops.attention import vanilla_chunked_prefill_mla
from vllm_ascend.torchair.utils import npu_stream_switch, npu_wait_tensor
from vllm_ascend.utils import npu_prefetch
from vllm_ascend.worker.npu_input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendMLABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AscendMLAMetadata

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

    attn_mask: torch.Tensor
    query_lens: list[int]
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
    cp_kv_recover_idx: list[int] = None
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
    cp_prefill_mask: torch.Tensor = None


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
    num_computed_tokens_of_cp_sp: list[list[list[int]]] = None


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
    enable_dbo_across_dp: bool = False

    def __post_init__(self):
        pass
        # supported_head_sizes = AscendMLABackend.get_supported_head_sizes()
        # if self.head_dim is not None and self.head_dim \
        #         not in supported_head_sizes:
        #     raise ValueError(
        #         f"Only {supported_head_sizes} are supported for head_dim,",
        #         f"received {self.head_dim}.")

    def split_metadata_for_multistream(
        self,
        ms_split_config: MSAttentionMetadataSplitConfig,
    ) -> list["AscendMLAMetadata"]:
        """Split metadata for multi-stream with AscendMLAMetadata"""
        return model_input_split_v1_mla_attn(
            ms_split_config=ms_split_config,
            attn_metadata=self,
            _metadata_cls=AscendMLAMetadata,
        )


M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMLAMetadataBuilder:
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # _attn_mask_builder = None
    def __init__(self,
                 runner,
                 metadata_cls: Optional[AscendMLAMetadata] = None):
        self.metadata_cls: Optional[AscendMLAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendMLAMetadata  # type: ignore
        self.runner = runner
        scheduler_config = runner.scheduler_config
        model_config = runner.model_config
        self.block_size = runner.block_size
        self.chunked_prefill_enabled = runner.chunked_prefill_enabled
        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(8 * model_config.max_model_len,
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
                 model_config.get_head_size()),
                dtype=model_config.dtype,
                device=runner.device,
            )
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.rope_dim = self.runner.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

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
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            # For torch air graph mode we treat spec decoding as decode.
            if self.torchair_graph_enabled:
                if num_tokens - num_spec_tokens == 1:
                    decodes.append(i)
                    num_decode_tokens += num_tokens
                else:
                    prefills.append(i)
                    num_prefill_tokens += num_tokens
            # For eager mode we treat spec decoding as chunked prefill.
            else:
                if num_tokens == 1:
                    decodes.append(i)
                    num_decode_tokens += num_tokens
                else:
                    prefills.append(i)
                    num_prefill_tokens += num_tokens

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
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def _get_graph_runner_block_tables(
            self, num_seqs: int, block_tables: torch.Tensor) -> torch.Tensor:

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs, f"max_batch_size: {max_batch_size} should be bigger than cur_num_seqs: {num_seqs}"

        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_batch_size, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype)

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_seqs, :
                               num_blocks] = block_tables[:num_seqs, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_seqs, :
                               max_blocks] = block_tables[:num_seqs, :
                                                          max_blocks]

        return graph_block_tables[:num_seqs, :max_blocks]

    def build_torchair_graph_dummy(
            self, num_reqs: int, num_actual_tokens: int) -> AscendMLAMetadata:
        device = self.runner.device
        _, max_blocks = self.runner.graph_block_tables.shape
        block_table = torch.zeros((num_reqs, max_blocks),
                                  dtype=torch.int32,
                                  device=device)
        block_table = self._get_graph_runner_block_tables(
            num_reqs, block_table)
        num_tokens = num_reqs * self.runner.decode_token_per_req
        seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        seq_lens_list = [0] * num_reqs
        input_positions = torch.zeros(num_tokens,
                                      dtype=torch.int32,
                                      device=device).long()
        slot_mapping = torch.full((num_tokens, ),
                                  PAD_SLOT_ID,
                                  dtype=torch.int32,
                                  device=device)
        query_start_loc = torch.full((num_reqs, ),
                                     -1,
                                     dtype=torch.int32,
                                     device=device)
        sin = torch.ones(num_tokens,
                         1,
                         1,
                         self.rope_dim,
                         dtype=self.runner.dtype,
                         device=device)
        cos = torch.ones(num_tokens,
                         1,
                         1,
                         self.rope_dim,
                         dtype=self.runner.dtype,
                         device=device)
        if self.runner.speculative_config is not None and\
            self.runner.speculative_config.method == 'deepseek_mtp':
            attn_state = AscendAttentionState.SpecDecoding
            num_decode_tokens = 2
        else:
            attn_state = AscendAttentionState.DecodeOnly
            num_decode_tokens = 1
        decode_metadata = AscendMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens_list,
            max_seq_lens=1,
            attn_mask=self.runner.spec_attn_mask,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q[:num_reqs],
            sin=sin,
            cos=cos,
        )
        return self.metadata_cls(  # type: ignore
            num_input_tokens=num_actual_tokens,
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            head_dim=self.runner.model_config.get_head_size(),
            num_decodes=1,
            num_decode_tokens=num_decode_tokens,
            num_prefills=0,
            attn_mask=self.runner.attn_mask,
            attn_state=attn_state,
            prefill=None,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=block_table,
        )

    def build(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        max_query_len: int,
        graph_pad_size: int = -1,
        query_start_loc: torch.Tensor = None,
        enable_dbo_across_dp: bool = False,
        cp_kv_recover_idx: list[int] = None,
        num_actual_tokens_cp_full: int = None,
        num_computed_tokens_of_cp_sp: list[list[list[int]]] = None,
        q_head_idx_tensor = None,
        q_tail_idx_tensor = None,
        kv_with_q_head_nomask_idx_tensor = None,
        kv_with_q_head_mask_idx_tensor = None,
        kv_with_q_tail_nomask_idx_tensor = None,
        kv_with_q_tail_mask_idx_tensor = None,
        attn_mask_seqlens = None,
        head_attn_nomask_seqlens = None,
        tail_attn_nomask_seqlens = None,
        q_full_idx = None,
        cp_prefill_mask = None,
        *args,
        **kwargs,
    ) -> AscendMLAMetadata:
        assert self._num_decodes + self._num_prefills == num_reqs

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.runner.device

        block_table = (self.runner.input_batch.block_table[0].
                       get_device_tensor()[:num_reqs])
        if num_actual_tokens_cp_full is None:
            num_actual_tokens_cp_full = num_actual_tokens
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens_cp_full].to(
            device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True).long()

        seq_lens_cpu = self.runner.seq_lens_cpu[:num_reqs]
        query_lens = seq_lens_cpu - self.runner.input_batch.num_computed_tokens_cpu_tensor[:
                                                                                           num_reqs]
        seq_lens = seq_lens_cpu
        max_query_len = query_lens.max().item()
        max_seq_lens = seq_lens.max().item()
        if self.cos_cache is None:
            self.cos_cache = self.runner.get_model(
            ).model.layers[0].self_attn.rotary_emb.cos_cached
            self.sin_cache = self.runner.get_model(
            ).model.layers[0].self_attn.rotary_emb.sin_cached
        if self.cos_cache.dtype != self.runner.dtype:  # type: ignore
            self.cos_cache = self.cos_cache.to(  # type: ignore
                self.runner.dtype)  # type: ignore
            self.sin_cache = self.sin_cache.to(  # type: ignore
                self.runner.dtype)  # type: ignore

        prefill_metadata = None
        chunked_context_metadata = None
        if self._num_prefills > 0:
            reqs_start = self._num_decodes  # prefill_start
            tokens_start = self._num_decode_tokens
            max_query_len = query_lens[tokens_start:].max().item()
            max_seq_lens = seq_lens[tokens_start:].max().item()
            prefill_query_start_loc = query_start_loc[
                reqs_start:] - query_start_loc[reqs_start]

            context_lens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[
                reqs_start:num_reqs]
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
                    .unsqueeze(1).expand(-1, self._num_prefills) * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
                cu_seq_lens_cpu = torch.zeros(num_chunks,
                                              self._num_prefills + 1,
                                              dtype=torch.int32,
                                              pin_memory=True)
                torch.cumsum(chunk_seq_lens,
                             dim=1,
                             out=cu_seq_lens_cpu[:, 1:],
                             dtype=torch.int32)
                chunked_context_metadata = \
                    AscendMLAPrefillMetadata.ChunkedContextMetadata(
                    cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                    starts=chunk_starts.to(device, non_blocking=True),
                    seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                    max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                    chunk_seq_lens=chunk_seq_lens,
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
                attn_mask=self.runner.attn_mask,
                query_lens=query_lens[tokens_start:],
                seq_lens=seq_lens,
                context_lens=seq_lens[tokens_start:],
                input_positions=prefill_input_positions,
                block_table=block_table[reqs_start:, ...],
                max_query_len=max_query_len,
                max_seq_lens=max_seq_lens,
                query_start_loc=prefill_query_start_loc,
                chunked_context=chunked_context_metadata,
                sin=sin,
                cos=cos,
                cp_kv_recover_idx=cp_kv_recover_idx,
                q_head_idx=q_head_idx_tensor,
                q_tail_idx=q_tail_idx_tensor,
                kv_with_q_head_nomask_idx=kv_with_q_head_nomask_idx_tensor,
                kv_with_q_head_mask_idx=kv_with_q_head_mask_idx_tensor,
                kv_with_q_tail_nomask_idx=kv_with_q_tail_nomask_idx_tensor,
                kv_with_q_tail_mask_idx=kv_with_q_tail_mask_idx_tensor,
                attn_mask_seqlens=attn_mask_seqlens,
                head_attn_nomask_seqlens=head_attn_nomask_seqlens,
                tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
                q_full_idx=q_full_idx,
                cp_prefill_mask=cp_prefill_mask
            )

        decode_metadata = None
        use_torchair_graph = graph_pad_size != -1
        if self._num_decodes > 0:
            actual_seq_lengths_q = query_start_loc[1:].tolist()
            max_seq_lens = seq_lens[:self._num_decodes].max().item()
            seq_lens = seq_lens[:self._num_decode_tokens]
            input_positions = input_positions[:self._num_decode_tokens]
            block_table = block_table[:self._num_decode_tokens, ...]
            if use_torchair_graph and self.runner.attn_state in [
                    AscendAttentionState.DecodeOnly,
                    AscendAttentionState.SpecDecoding
            ]:
                num_reqs_pad_size = 0
                num_token_pad_size = 0
                if graph_pad_size != 0:
                    pad_value = 0
                    num_token_pad_size = graph_pad_size - self._num_decode_tokens
                    num_reqs_pad_size = (
                        graph_pad_size // self.runner.decode_token_per_req -
                        num_reqs)
                    padded_seq_lens = seq_lens.tolist(
                    ) + [pad_value] * num_reqs_pad_size
                else:
                    padded_seq_lens = seq_lens.tolist()

                seq_lens = torch.from_numpy(
                    np.array(padded_seq_lens).astype(np.int32))
                seq_lens_list = padded_seq_lens
                slot_padding = torch.full((num_token_pad_size, ),
                                          PAD_SLOT_ID,
                                          dtype=slot_mapping.dtype,
                                          device=slot_mapping.device)
                slot_mapping = torch.cat([slot_mapping, slot_padding])
                block_table_padding = torch.zeros(
                    (num_reqs_pad_size, ) + block_table.shape[1:],
                    dtype=block_table.dtype,
                    device=block_table.device)
                block_table = torch.cat([block_table, block_table_padding],
                                        dim=0)
                block_table = self._get_graph_runner_block_tables(
                    num_reqs + num_reqs_pad_size, block_table)
                position_padding = torch.zeros(num_token_pad_size,
                                               dtype=input_positions.dtype,
                                               device=input_positions.device)
                input_positions = torch.cat(
                    [input_positions, position_padding])
                actual_seq_lengths_q = query_start_loc[1:].tolist(
                ) + self.runner.actual_seq_lengths_q[num_reqs:num_reqs +
                                                     num_reqs_pad_size]
            else:
                seq_lens_list = seq_lens.tolist()
            # mtp torchair + PD scenario, last element of actual_seq_lengths_q must equal to batch_size(num_tokens)
            batch_size = slot_mapping.size(0)
            if actual_seq_lengths_q[-1] != batch_size \
                and self.runner.attn_state == AscendAttentionState.SpecDecoding:
                actual_seq_lengths_q[-1] = batch_size

            cos = self.cos_cache[input_positions].unsqueeze(  # type: ignore
                1).unsqueeze(2)
            sin = self.sin_cache[input_positions].unsqueeze(  # type: ignore
                1).unsqueeze(2)

            decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                seq_lens_list=seq_lens_list,
                max_seq_lens=max_seq_lens,
                attn_mask=self.runner.spec_attn_mask,
                actual_seq_lengths_q=actual_seq_lengths_q,
                sin=sin,
                cos=cos,
                num_computed_tokens_of_cp_sp=num_computed_tokens_of_cp_sp,
            )

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_actual_tokens,
            query_lens=query_lens.tolist(),
            slot_mapping=slot_mapping,
            head_dim=self.runner.model_config.get_head_size(),
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=block_table,
            seq_lens=seq_lens,
            enable_dbo_across_dp=enable_dbo_across_dp,
        )


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
        self.q_proj = kwargs['q_proj']
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.kv_b_proj_full = kwargs['kv_b_proj_full']
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.enable_sp = kwargs.get('enable_sp', False)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        # Adapt torch air graph mode with spec decoding.
        speculative_config = get_current_vllm_config().speculative_config
        if speculative_config is not None:
            self.spec_token_num = speculative_config.num_speculative_tokens
            assert self.spec_token_num > 0

        self.cp_size = get_context_model_parallel_world_size()
        self.cp_rank = get_context_model_parallel_rank()
        self.cp_group = get_cp_group().device_group
        self.sp_size = get_tensor_model_parallel_world_size() if self.enable_sp else 1
        self.sp_rank = get_tensor_model_parallel_rank() if self.enable_sp else 0
        self.sp_group = get_tp_group().device_group

    def _v_up_proj_and_o_proj(self, x, enable_multistream_mla: bool = False):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        if hasattr(self, "running_in_graph") and not self.running_in_graph:
            return x
        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024  # 16MB
        npu_prefetch(self.o_proj.weight,
                     x,
                     max_size=MAX_O_PROJ_PREFETCH_SIZE,
                     enabled=enable_multistream_mla)
        return self.o_proj(x, is_prefill=False)[0]

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

        # Waiting for BMM NZ support
        # self.W_UV.data = torch_npu.npu_format_cast(self.W_UV.data, 29)
        # self.W_UK_T.data = torch_npu.npu_format_cast(self.W_UK_T.data, 29)

    def _compute_prefill_context(
        self,
        query: torch.Tensor,
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
        q_pe = query[..., self.qk_nope_head_dim:]
        q_nope = query[..., :self.qk_nope_head_dim]

        seq_len1 = torch.tensor(prefill_metadata.query_lens, dtype=torch.int32)
        cache_kv_c = kv_c_and_k_pe_cache[0]
        cache_k_pe = kv_c_and_k_pe_cache[1]
        num_heads = cache_k_pe.size(2)
        latent_kv_dim = kv_c_and_k_pe_cache[0].size(-1)
        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]

            seq_len2 = prefill_metadata.chunked_context.chunk_seq_lens[i]
            seq_len = torch.stack([seq_len1, seq_len2])
            kv_c_normed = torch.empty(toks,
                                      num_heads,
                                      latent_kv_dim,
                                      dtype=query.dtype,
                                      device=query.device)
            k_pe = torch.empty(toks,
                               num_heads,
                               rope_dim,
                               dtype=query.dtype,
                               device=query.device)

            torch_npu.atb.npu_paged_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                seq_len2.to(query.device),
                seq_starts=prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )

            kv_c_normed = kv_c_normed.squeeze()
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
            mask = torch.triu(
                torch.ones(512, 512, device=query.device, dtype=query.dtype),
                1)
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
        query: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None
        assert len(kv_c_and_k_pe_cache) > 1

        num_tokens = query.size(0)
        attn_output = torch.empty(num_tokens,
                                  self.num_heads,
                                  self.v_head_dim,
                                  dtype=query.dtype,
                                  device=query.device)
        k_nope, value = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
        # Here is only 2 possibility of input, ChunkedPrefill or PrefillNoCache
        ascend_config = get_ascend_config()

        if attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ] and not ascend_config.chunked_prefill_for_mla:
            attn_output_torch = torch.empty(num_tokens,
                                            self.num_heads * self.v_head_dim,
                                            dtype=query.dtype,
                                            device=query.device)
            # current requests is chunked in prefill, disable flash attention with chunked prefill
            vanilla_chunked_prefill_mla(
                output=attn_output_torch,
                query=query,
                kv_cache=kv_c_and_k_pe_cache,
                block_tables=attn_metadata.prefill.block_table,
                query_lens=attn_metadata.prefill.query_lens,
                context_lens=attn_metadata.prefill.context_lens,
                kv_b_proj=self.kv_b_proj,
                max_query_len=attn_metadata.prefill.max_query_len,
                max_context_len=attn_metadata.prefill.max_seq_lens,
                nope_dim=self.qk_nope_head_dim,
                rope_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                scale=self.scale,
                alibi_slopes=None,
                causal=True)
        elif attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ]:
            attn_lse = torch.empty(self.num_heads,
                                   num_tokens,
                                   dtype=torch.float32,
                                   device=query.device)
            q_pe = query[..., self.qk_nope_head_dim:]
            q_nope = query[..., :self.qk_nope_head_dim]
            mask = torch.triu(
                torch.ones(512, 512, device=query.device, dtype=query.dtype),
                1)  # 512: mask only support 512
            if attn_metadata.num_prefills > 1:
                mask = mask.unsqueeze(0).repeat(attn_metadata.num_prefills, 1,
                                                1)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=value,
                mask=mask,
                seqlen=torch.tensor(attn_metadata.prefill.query_lens,
                                    dtype=torch.int32),
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
                query, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse)

        elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = torch.cat((k_nope, k_pe), dim=-1)
            torch_npu._npu_flash_attention(
                query=query,
                key=key,
                value=value,
                mask=attn_metadata.attn_mask,
                seq_len=attn_metadata.prefill.context_lens,
                scale_value=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                out=attn_output)
            attn_output = attn_output.view(-1, self.num_heads, self.v_head_dim)
        else:
            raise RuntimeError(
                "Unexpected path reached, AscendMLAImpl should only have PrefillNoCache, PrefillCacheHit, ChunkedPrefill and SpecDecoding scenario in forward prefill, please file a bug to vllm-ascend !"
            )
        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim])
        if attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ] and not ascend_config.chunked_prefill_for_mla:
            attn_output = attn_output_torch

        return attn_output

    def _forward_prefill_cp(
        self,
        query: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        num_tokens = query.size(0) # 单卡的token数
        # 构造存放attention的，分别存储requests以及单block的attention的值
        # attn_output_requests = []
        k_nope, value = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
        q_pe = query[..., self.qk_nope_head_dim:]
        q_nope = query[..., :self.qk_nope_head_dim]

        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.q_head_idx
        q_tail_idx = attn_metadata.prefill.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.kv_with_q_tail_mask_idx
        attn_mask_seqlens = attn_metadata.prefill.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.tail_attn_nomask_seqlens
        mask = attn_metadata.prefill.cp_prefill_mask

        # 1. 负载均衡中，Q前半段的Attention计算
        # cp_rank0: Q0*KV0
        # cp_rank1: Q1*KV0 + Q1*KV1
        output_head = self._attention_with_mask_and_nomask(
            q_nope=torch.index_select(q_nope, 0, q_head_idx),
            q_pe=torch.index_select(q_pe, 0, q_head_idx),
            k_nope=k_nope,
            k_pe=k_pe,
            value=value,
            kv_mask_idx=kv_with_q_head_mask_idx,
            kv_nomask_idx=kv_with_q_head_nomask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            attn_nomask_seqlens=head_attn_nomask_seqlens,
            mask=mask
        )

        # 2. 负载均衡中，Q后半段的Attention计算
        # cp_rank0: Q3*KV0~KV2 + Q3*KV3
        # cp_rank1: Q2*KV0~KV1 + Q2*KV2
        output_tail = self._attention_with_mask_and_nomask(
            q_nope=torch.index_select(q_nope, 0, q_tail_idx),
            q_pe=torch.index_select(q_pe, 0, q_tail_idx),
            k_nope=k_nope,
            k_pe=k_pe,
            value=value,
            kv_mask_idx=kv_with_q_tail_mask_idx,
            kv_nomask_idx=kv_with_q_tail_nomask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            attn_nomask_seqlens=tail_attn_nomask_seqlens,
            mask=mask
        )

        # 3. 合并前半段和后半段的输出
        q_full_idx = attn_metadata.prefill.q_full_idx
        output = torch.index_select(torch.cat([output_head, output_tail], dim=0), 0, q_full_idx)

        # 后处理过程
        output = output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim])

        return output


    def _attention_with_mask_and_nomask(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_mask_idx: torch.Tensor,
        kv_nomask_idx: torch.Tensor,
        attn_mask_seqlens: torch.Tensor,
        attn_nomask_seqlens: torch.Tensor,
        mask: torch.Tensor
    ):
        attn_output = torch.empty(q_nope.shape[0], # 长度现在是每个req的cp_block求和
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
        torch_npu.atb.npu_ring_mla(
            q_nope=q_nope,
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
            softmax_lse=attn_lse
        )

        # nomask
        if kv_nomask_idx.shape[0] == 0:
            return attn_output

        k_nope_nomask = torch.index_select(k_nope, 0, kv_nomask_idx)
        value_nomask = torch.index_select(value, 0, kv_nomask_idx)
        k_pe_nomask = torch.index_select(k_pe, 0, kv_nomask_idx)
        torch_npu.atb.npu_ring_mla(
            q_nope=q_nope,
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
            softmax_lse=attn_lse
        )
        return attn_output

    def exec_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):

        B = hidden_states.shape[0]
        N = self.num_kv_heads
        S = 1
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv = kv.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"
        k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
        )
        return k_pe, k_nope, kv

    def exec_kv_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):

        B = hidden_states.shape[0]
        N = self.num_kv_heads
        S = 1
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv = kv.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_BLK_NZ" if self.enable_kv_nz else "PA"
        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
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
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
        enable_multistream_mla: bool = False,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_nope.size(0)
        if self.running_in_graph or self.running_chunkprefilll_with_torchair:
            # shape of knope/k_pe for npu graph mode should be:
            # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
            block_size = kv_c_and_k_pe_cache[0].shape[1]
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

            if attn_metadata.attn_state == AscendAttentionState.SpecDecoding:
                assert num_tokens % self.spec_token_num == 0
                input_layout = "TND"
                # [bs * q_seq_len, num_heads_per_rank, dim]
                q_nope = q_nope.view(num_tokens, self.num_heads, -1)
                q_pe = q_pe.view(num_tokens, self.num_heads, -1)
                sparse_mode = 3
                spec_attn_mask = attn_metadata.decode.attn_mask  # type:ignore
                actual_seq_lengths = decode_meta.actual_seq_lengths_q
            else:
                if self.enable_kv_nz:
                    q_nope = q_nope.view(num_tokens, 1, self.num_heads, -1)
                    q_pe = q_pe.view(num_tokens, 1, self.num_heads, -1)
                else:
                    q_nope = q_nope.view(num_tokens, self.num_heads, 1, -1)
                    q_pe = q_pe.view(num_tokens, self.num_heads, 1, -1)
                sparse_mode = 0
                spec_attn_mask = None

            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout=input_layout,
                atten_mask=spec_attn_mask,
                sparse_mode=sparse_mode,
                scale=self.scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=decode_meta.block_table,
                block_size=block_size,
                actual_seq_lengths_kv=decode_meta.seq_lens_list,
                actual_seq_lengths=actual_seq_lengths)
        else:
            # The MLA_PA path will be used as default path in the future, `_npu_paged_attention_mla` will
            # be removed after the torch_npu contains `torch_npu.atb.npu_multi_head_latent_attention` become
            # public available
            assert len(kv_c_and_k_pe_cache) > 1
            if envs_ascend.VLLM_ASCEND_MLA_PA:
                attn_output = torch_npu.atb.npu_multi_head_latent_attention(
                    q_nope, q_pe, kv_c_and_k_pe_cache[0],
                    kv_c_and_k_pe_cache[1], attn_metadata.decode.block_table,
                    attn_metadata.decode.seq_lens, self.num_heads, self.scale,
                    self.num_kv_heads)
            else:
                q = torch.cat([q_nope, q_pe], dim=-1)
                attn_output = torch.empty(
                    [num_tokens, self.num_heads, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device)
                k_cache = torch.cat(
                    [kv_c_and_k_pe_cache[0], kv_c_and_k_pe_cache[1]], dim=-1)
                torch_npu._npu_paged_attention_mla(
                    query=q,
                    key_cache=k_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.decode.
                    block_table,  # type:ignore
                    context_lens=attn_metadata.decode.seq_lens,  # type:ignore
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output)
        current_ms_metadata = get_multistream_comm_context()
        # attn_output: [seq, num_heads(16/tp), kv_lora_rank(512)]
        # v_up: [seq, num_heads(16/tp), v_head_dim(128)]
        # o_proj: [seq, hidden_size(2048)]
        if current_ms_metadata is None:
            return self._v_up_proj_and_o_proj(attn_output,
                                              enable_multistream_mla)
        else:
            current_ms_metadata.before_comm_event.record()
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait()
                return self._v_up_proj_and_o_proj(attn_output)

    def _forward_decode_sp(
        self,
        # q_nope: torch.Tensor,
        # q_pe: torch.Tensor,
        # k_nope: torch.Tensor,
        # k_pe: torch.Tensor,
        q_wo_k_up: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
        # enable_multistream_mla: bool = False,
    ) -> torch.Tensor:
        """
        Since new page_attn op with softmax_lse return is not ready yet,
        use npu_ring_mla instead, temporarily.
        Get kv_cache manually by npu_paged_cache_load, other process is same to prefill.
        TODO After new op is ready we should abandon this and use the same process as _forward_decode
        TODO Delete redundant comments before pr
        """
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_wo_k_up.size(0)

        q_pe = q_wo_k_up[..., self.qk_nope_head_dim:]
        q_nope = q_wo_k_up[..., :self.qk_nope_head_dim]
        decode_metadata = attn_metadata.decode

        # use cp & sp splited computed token nums from scheduler to compute actual seq_len and seq_mask
        num_computed_tokens_of_cp_sp = np.array(decode_metadata.num_computed_tokens_of_cp_sp) # [bs, cp_size, sp_size]
        seq_mask_cp = torch.where(torch.tensor(num_computed_tokens_of_cp_sp.sum(2)) == 0, 0, 1).to(torch.uint8).to(q_pe.device)
        seq_mask_sp = torch.where(torch.tensor(num_computed_tokens_of_cp_sp[:, self.cp_rank, :]) == 0, 0, 1).to(torch.uint8).to(q_pe.device)
        seq_len = num_computed_tokens_of_cp_sp[:, self.cp_rank, self.sp_rank]

        seq_len = torch.tensor(seq_len, dtype=torch.int32)
        batch_size = seq_len.size(0)

        if torch.sum(seq_len).item() == 0:
            # Case that no kv_cache has been stored on this rank, no need to do following computation.
            attn_output = torch.zeros([batch_size, self.num_heads * self.sp_size, self.v_head_dim], dtype=q_pe.dtype, device=q_pe.device)
            softmax_lse = torch.zeros([self.num_heads * self.sp_size, batch_size], dtype=torch.float32, device=q_pe.device)
        else:
            # Normal case.
            cache_kv_c = kv_c_and_k_pe_cache[0]
            cache_k_pe = kv_c_and_k_pe_cache[1]

            kv_c_normed, k_pe = torch_npu.atb.npu_paged_cache_load(
                cache_kv_c,
                cache_k_pe,
                decode_metadata.block_table,
                seq_len.to(q_nope.device),
                seq_starts=torch.zeros([batch_size], dtype=torch.int32).to(q_nope.device),
            )
            # kv_c_normed: [seq, 1, kv_lora_rank(512)], k_pe: [seq, 1, qk_rope_head_dim(64)]

            kv_c_normed = kv_c_normed.squeeze() # [seq, kv_lora_rank(512)]
            if self.sp_size > 1:
                # sp > 1, kv splited in sequence dimension, should not do tp here
                kv_nope = self.kv_b_proj_full(kv_c_normed)[0].view( \
                    -1, self.num_heads * self.sp_size, self.qk_nope_head_dim + self.v_head_dim)
                # [seq, kv_lora_rank(512)] -> [seq, num_heads_full * (qk_nope_head_dim + v_head_dim)] -> view to [seq, num_heads_full(16), qk_nope_head_dim + v_head_dim (256)]
            else:
                # sp = 1, kv full in sequence dimension, use original tp kv_b_proj
                kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                    -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                # [seq, kv_lora_rank(512)] -> [seq, num_heads * (qk_nope_head_dim + v_head_dim)] -> view to [seq, num_heads(16/tp), qk_nope_head_dim + v_head_dim (256)]

            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k_nope: [seq, num_heads_full(16), qk_nope_head_dim(128)], v: [seq, num_heads_full(16), v_head_dim(128)]
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
            mask = torch.triu(
                torch.ones(512, 512, device=q_nope.device, dtype=q_nope.dtype),
                1)
            seq_len_q = torch.ones([batch_size], dtype=torch.int32)
            seq_len_kv = seq_len
            seq_len_all = torch.stack([seq_len_q, seq_len_kv])

            attn_output, softmax_lse = torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,          # [batch_size, num_heads_full(16), qk_nope_head_dim(128)]
                q_rope=q_pe,            # [batch_size, num_heads_full(16), qk_rope_head_dim(64)]
                k_nope=k_nope,          # [kv_seq_len, num_heads_full(16), qk_nope_head_dim(128)]
                k_rope=k_pe,            # [kv_seq_len, num_heads_full(16), qk_rope_head_dim(64)]
                value=v,                # [kv_seq_len, num_heads_full(16), v_head_dim(128)]
                mask=mask,
                seqlen=seq_len_all,     # seq_q + seq_kv, for example: [[1, 1, 1, 1], [9, 7, 7, 7]]
                head_num=self.num_heads * self.sp_size,
                kv_head_num=self.num_heads * self.sp_size,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                input_layout="type_bsnd",
                calc_type="calc_type_first_ring",
            )
        # attn_output: [bs, num_heads_full(16), v_head_dim(128)], softmax_lse: [num_heads_full(16), bs]
        
        # TODO use update op to replace this
        def _update_out_and_lse(
            out: torch.Tensor,
            lse: torch.Tensor,
            block_out: torch.Tensor,
            block_lse: torch.Tensor,
            mask: torch.Tensor=None,
        ):
            if out is None:
                out = block_out.to(torch.float32)
                lse = block_lse
            else:
                if mask is None:
                    mask = torch.ones([block_out.size(0)], dtype=torch.uint8, device=block_out.device)
                out_mask = mask[:, None, None].expand_as(block_out)
                lse_mask = mask[:, None, None].expand_as(block_lse)
                block_out = block_out.to(torch.float32)
                out_without_update = out.clone()
                lse_without_update = lse.clone()

                # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
                # out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
                # lse = new_lse

                # is equal to above
                out = out - F.sigmoid(block_lse - lse) * (out - block_out)
                lse = lse - F.logsigmoid(lse - block_lse)

                # mask
                out = torch.where(out_mask, out, out_without_update)
                lse = torch.where(lse_mask, lse, lse_without_update)
            return out, lse

        # lse: [num_heads,bs] -> [bs,num_heads] -> [bs,num_heads,1]
        softmax_lse = softmax_lse.permute([1, 0]).unsqueeze(dim=-1)

        if self.sp_size > 1:
            # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
            attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all, attn_out_lse, group=self.sp_group)
            # permute: [num_heads, v_head_dim+1, bs] -> [bs, num_heads, v_head_dim+1]
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            attn_out_lse_split_on_seq = list(torch.chunk(attn_out_lse_all2all, self.sp_size, dim=1))
            # Update out&lse
            attn_out_g = None
            attn_lse_g = None
            for i, attn_out_lse_l in enumerate(attn_out_lse_split_on_seq):
                attn_out_l, attn_lse_l = torch.split(attn_out_lse_l, [self.v_head_dim, 1], dim=-1)
                attn_out_g, attn_lse_g = _update_out_and_lse(attn_out_g, attn_lse_g, attn_out_l, attn_lse_l, seq_mask_sp[:, i])
            attn_output = attn_out_g
            softmax_lse = attn_lse_g

        if self.cp_size > 1:
            # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
            attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
            # AllGather out&lse within CP group
            attn_out_lse_list = [torch.empty_like(attn_out_lse) for _ in range(self.cp_size)]
            dist.all_gather(attn_out_lse_list, attn_out_lse, group=self.cp_group)
            # Update out&lse
            attn_out_g = None
            attn_lse_g = None
            for i, attn_out_lse_l in enumerate(attn_out_lse_list):
                attn_out_l, attn_lse_l = torch.split(attn_out_lse_l, [self.v_head_dim, 1], dim=-1)
                attn_out_g, attn_lse_g = _update_out_and_lse(attn_out_g, attn_lse_g, attn_out_l, attn_lse_l, seq_mask_cp[:, i])
            attn_output = attn_out_g

        attn_output = attn_output.reshape([num_tokens, self.num_heads * self.v_head_dim]).to(torch.bfloat16)
        # out = self.o_proj(attn_output, is_prefill=False)[0]
        out = attn_output
        return out

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        hidden_states_or_kv_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: M,
        output: Optional[torch.Tensor] = None,
        enable_multistream_mla: bool = False,
        ckq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output
        self.running_in_graph = self.torchair_graph_enabled and attn_metadata.attn_state in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]
        self.running_chunkprefilll_with_torchair = self.torchair_graph_enabled and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
        num_actual_toks = attn_metadata.num_actual_tokens
        if k_pe is None and not self.running_in_graph:
            kv_c, k_pe = self.kv_a_proj_with_mqa(
                hidden_states_or_kv_c_normed)[0].split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
        else:
            kv_c_normed = hidden_states_or_kv_c_normed
        assert attn_metadata.num_decodes is not None and \
        attn_metadata.num_prefills is not None and \
        attn_metadata.num_decode_tokens is not None
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        if not self.running_in_graph:
            # Inputs and outputs may be padded for CUDA graphs
            output_padded = output
            output = output[:num_actual_toks, ...]
            if not self.torchair_graph_enabled:
                kv_c_normed = kv_c_normed[:num_actual_toks, ...]
                prefill_k_c_normed = kv_c_normed[num_decode_tokens:]
        if not self.running_in_graph:
            hidden_states_or_q_c = hidden_states_or_q_c[:num_actual_toks, ...]
            prefill_hs_or_q_c = hidden_states_or_q_c[num_decode_tokens:]
            decode_hs_or_q_c = hidden_states_or_q_c[:num_decode_tokens]
            prefill_hs = hidden_states_or_kv_c_normed[num_decode_tokens:]
            # if not self.torchair_graph_enabled:
            k_pe = k_pe[:num_actual_toks, ...]
            k_pe = k_pe.unsqueeze(1)
            decode_k_pe = k_pe[:num_decode_tokens]
            prefill_k_pe = k_pe[num_decode_tokens:]
        else:
            decode_hs_or_q_c = hidden_states_or_q_c
        if has_decode:
            decode_k_nope = None
            assert attn_metadata.decode is not None
            if self.running_in_graph or self.running_chunkprefilll_with_torchair:
                cos = attn_metadata.decode.cos
                sin = attn_metadata.decode.sin
                if self.running_chunkprefilll_with_torchair:
                    decode_hs = (
                        hidden_states_or_kv_c_normed[:num_decode_tokens])
                    slots = attn_metadata.slot_mapping[:num_decode_tokens]
                    decode_k_pe, decode_k_nope, decode_kv = self.exec_kv(
                        decode_hs, cos, sin, kv_cache, slots)
                else:
                    with npu_stream_switch("mla_secondary",
                                           0,
                                           enabled=enable_multistream_mla):
                        npu_wait_tensor(hidden_states_or_kv_c_normed,
                                        ckq,
                                        enabled=enable_multistream_mla)
                        decode_k_pe, decode_k_nope, decode_kv = self.exec_kv(
                            hidden_states_or_kv_c_normed, cos, sin, kv_cache,
                            attn_metadata.slot_mapping)
                # Without explicitly controlling the order, IndexByTensor operations
                # would be placed after `matmul W_KV_T` hindering the overlapping of
                # KvRmsNormRopeCache and SingleRope.
                npu_wait_tensor(decode_hs_or_q_c,
                                cos,
                                enabled=enable_multistream_mla)
                npu_wait_tensor(decode_hs_or_q_c,
                                sin,
                                enabled=enable_multistream_mla)
                npu_wait_tensor(decode_hs_or_q_c,
                                decode_kv,
                                enabled=enable_multistream_mla)

            if self.cp_size * self.sp_size > 1:
                decode_q_wo_k_up = self.q_proj(decode_hs_or_q_c)[0]\
                    .view(-1, self.num_heads, self.qk_head_dim)
                # decode_hs_or_q_c: [seq, q_lora_rank/hidden_size(2048)],
                # decode_q_wo_k_up: [seq, num_heads(16/tp), qk_nope_head_dim(128) + qk_rope_head_dim(64)]
            else:
                decode_ql_nope, decode_q_pe = \
                    self._q_proj_and_k_up_proj(decode_hs_or_q_c)
                # decode_hs_or_q_c: [seq, q_lora_rank/hidden_size(2048)],
                # decode_ql_nope: [seq, num_heads(16/tp), kv_lora_rank(512)], decode_q_pe: [seq, num_heads(16/tp), qk_rope_head_dim(64)]
            if self.sp_size > 1:
                decode_q_wo_k_up = decode_q_wo_k_up.contiguous()
                decode_q_wo_k_up = get_tp_group().all_gather(decode_q_wo_k_up, 1)
            if self.running_in_graph:
                with npu_stream_switch("mla_secondary",
                                       0,
                                       enabled=enable_multistream_mla):
                    npu_wait_tensor(decode_q_pe,
                                    decode_k_pe,
                                    enabled=enable_multistream_mla)
                    decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
            elif self.running_chunkprefilll_with_torchair:
                decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
            else:
                if self.cp_size * self.sp_size > 1:
                    decode_q_wo_k_up_pe = decode_q_wo_k_up[..., self.qk_nope_head_dim:] # [seq, num_heads_full(16), qk_rope_head_dim(64)]
                    # decode_q_wo_k_up_nope = decode_q_wo_k_up[..., :self.qk_nope_head_dim] # [seq, num_heads_full(16), qk_nope_head_dim(128)]
                    decode_q_wo_k_up_pe[...], decode_k_pe[...] = self.rotary_emb(
                        attn_metadata.decode.input_positions,
                        decode_q_wo_k_up_pe.contiguous(),
                        decode_k_pe,
                        max_seq_len=attn_metadata.decode.max_seq_lens)
                else:
                    decode_q_pe[...], decode_k_pe[...] = self.rotary_emb(
                        attn_metadata.decode.input_positions,
                        decode_q_pe.contiguous(),
                        decode_k_pe,
                        max_seq_len=attn_metadata.decode.max_seq_lens)
        if has_prefill:
            assert attn_metadata.prefill is not None
            prefill_q = self.q_proj(prefill_hs_or_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
            if self.torchair_graph_enabled:
                num_tokens = prefill_hs_or_q_c.shape[0]
                cos = attn_metadata.prefill.cos
                sin = attn_metadata.prefill.sin

                prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
                prefill_k_pe, prefill_k_nope = self.exec_kv_prefill(
                    prefill_hs, cos, sin, kv_cache,
                    attn_metadata.slot_mapping[num_decode_tokens:])

                kv_c_normed = prefill_k_nope[:num_actual_toks, ...]
                prefill_k_c_normed = prefill_k_nope
                prefill_k_pe = prefill_k_pe.view(num_tokens, self.num_kv_heads,
                                                 -1)
                prefill_q = torch.cat([prefill_q_nope, prefill_q_pe], dim=-1)
            else:
                prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
                    attn_metadata.prefill.input_positions,
                    prefill_q_pe.contiguous(),
                    prefill_k_pe,
                    max_seq_len=attn_metadata.prefill.max_seq_lens)

        assert len(
            kv_cache
        ) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
        if self.torchair_graph_enabled:
            if kv_cache[0].numel() > 0 and has_prefill:
                slots = attn_metadata.slot_mapping
                # NOTE: Separate the kv cache in advance to avoid OOM or other issues
                torch_npu._npu_reshape_and_cache(
                    key=kv_c_normed.view(num_tokens, self.num_kv_heads, -1),
                    value=prefill_k_pe,
                    key_cache=kv_cache[0],
                    value_cache=kv_cache[1],
                    slot_indices=slots[num_decode_tokens:])
        else:
            kv_c_normed = kv_c_normed.view(
                [num_actual_toks, self.num_kv_heads, -1])

            # 添加cp allgather逻辑
            # 暂未考虑 prefill_k_c_normed != kv_c_normed (chunk_prefill) 场景
            if self.cp_size > 1 and attn_metadata.num_prefills > 0:
                prefill_k_c_normed = prefill_k_c_normed.unsqueeze(1)
                prefill_kv_c_k_pe = torch.cat([prefill_k_c_normed, prefill_k_pe], dim=-1)
                prefill_kv_c_k_pe = get_cp_group().all_gather(prefill_kv_c_k_pe, 0)
                prefill_kv_c_k_pe = torch.index_select(prefill_kv_c_k_pe, 0, attn_metadata.prefill.cp_kv_recover_idx)
                prefill_k_c_normed, prefill_k_pe = prefill_kv_c_k_pe.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed, k_pe = prefill_k_c_normed, prefill_k_pe
                prefill_k_c_normed = prefill_k_c_normed.squeeze()

            torch_npu._npu_reshape_and_cache(
                key=kv_c_normed,
                value=k_pe,
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                slot_indices=attn_metadata.slot_mapping)
        if not self.running_in_graph:
            o_proj_input_shape = (num_actual_toks,
                                  self.num_heads * self.v_head_dim)
            o_proj_input = torch.empty(o_proj_input_shape,
                                       dtype=hidden_states_or_q_c.dtype,
                                       device=hidden_states_or_q_c.device)
        if has_prefill:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            if self.cp_size > 1:
                output_prefill = self._forward_prefill_cp(prefill_q,
                                                    prefill_k_c_normed,
                                                    prefill_k_pe,
                                                    attn_metadata)
            else:
                output_prefill = self._forward_prefill(prefill_q,
                                                    prefill_k_c_normed,
                                                    prefill_k_pe, kv_cache,
                                                    attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                current_ms_metadata.before_comm_event.record()
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    current_ms_metadata.before_comm_event.wait()
                    o_proj_input[num_decode_tokens:] = output_prefill
            else:
                o_proj_input[num_decode_tokens:] = output_prefill

        if has_decode:
            if self.running_in_graph:
                return self._forward_decode(decode_ql_nope, decode_q_pe,
                                            decode_k_nope, decode_k_pe,
                                            kv_cache, attn_metadata,
                                            enable_multistream_mla)
            else:
                if self.cp_size * self.sp_size > 1:
                    output_decode = self._forward_decode_sp(decode_q_wo_k_up,
                                                            kv_cache,
                                                            attn_metadata)
                else:
                    output_decode = self._forward_decode(decode_ql_nope,
                                                        decode_q_pe,
                                                        decode_k_nope,
                                                        decode_k_pe, kv_cache,
                                                        attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    o_proj_input[:num_decode_tokens] = output_decode
            else:
                o_proj_input[:num_decode_tokens] = output_decode

        current_ms_metadata = get_multistream_comm_context()
        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024  # 16MB
        if current_ms_metadata is None:
            npu_prefetch(self.o_proj.weight,
                         o_proj_input,
                         max_size=MAX_O_PROJ_PREFETCH_SIZE,
                         enabled=enable_multistream_mla)

            output[...] = self.o_proj(
                o_proj_input,
                is_prefill=has_prefill, # TODO case that both has_prefill and has_decode
                is_force_scatter=self.enable_shared_expert_dp)[0]
        else:
            with torch.npu.stream(current_ms_metadata.comm_stream):
                npu_prefetch(self.o_proj.weight,
                             o_proj_input,
                             max_size=MAX_O_PROJ_PREFETCH_SIZE,
                             enabled=enable_multistream_mla)
                output[...] = self.o_proj(
                    o_proj_input,
                    is_prefill=True,
                    is_force_scatter=self.enable_shared_expert_dp)[0]
                current_ms_metadata.after_comm_event.record()
        del o_proj_input
        return output_padded
