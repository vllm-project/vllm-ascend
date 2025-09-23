from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata,
                                              MLAAttentionImpl)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.utils import cdiv, round_down

from vllm_ascend.utils import context_parallel_enable, sequence_parallel_enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         split_decodes_and_prefills)
from vllm_ascend.multistream.base import MSAttentionMetadataSplitConfig
from vllm_ascend.multistream.context import get_multistream_comm_context
from vllm_ascend.multistream.ms_split import model_input_split_v1_mla_attn
from vllm_ascend.ops.attention import vanilla_chunked_prefill_mla
from vllm_ascend.utils import npu_prefetch
from vllm_ascend.worker.npu_input_batch import InputBatch

if context_parallel_enable:
    from vllm.distributed import (get_context_model_parallel_rank,
                                  get_context_model_parallel_world_size,
                                  get_cp_group)
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
    cp_kv_recover_idx: Optional[list[int]] = None
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
    num_computed_tokens_of_cp_sp: Optional[list[Optional[list[Optional[
        list[int]]]]]] = None


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

        self.decode_threshold = 1

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

    def build(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendMLAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        long_seq_metadata = common_attn_metadata.common_long_seq_metadata
        cp_kv_recover_idx = long_seq_metadata.cp_kv_recover_idx if long_seq_metadata else None
        num_actual_tokens_cp_full = long_seq_metadata.num_actual_tokens_cp_full if long_seq_metadata else None
        num_computed_tokens_of_cp_sp = long_seq_metadata.num_computed_tokens_of_cp_sp if long_seq_metadata else None
        q_head_idx_tensor = long_seq_metadata.q_head_idx_tensor if long_seq_metadata else None
        q_tail_idx_tensor = long_seq_metadata.q_tail_idx_tensor if long_seq_metadata else None
        kv_with_q_head_nomask_idx_tensor = long_seq_metadata.kv_with_q_head_nomask_idx_tensor if long_seq_metadata else None
        kv_with_q_head_mask_idx_tensor = long_seq_metadata.kv_with_q_head_mask_idx_tensor if long_seq_metadata else None
        kv_with_q_tail_nomask_idx_tensor = long_seq_metadata.kv_with_q_tail_nomask_idx_tensor if long_seq_metadata else None
        kv_with_q_tail_mask_idx_tensor = long_seq_metadata.kv_with_q_tail_mask_idx_tensor if long_seq_metadata else None
        attn_mask_seqlens = long_seq_metadata.attn_mask_seqlens if long_seq_metadata else None
        head_attn_nomask_seqlens = long_seq_metadata.head_attn_nomask_seqlens if long_seq_metadata else None
        tail_attn_nomask_seqlens = long_seq_metadata.tail_attn_nomask_seqlens if long_seq_metadata else None
        q_full_idx = long_seq_metadata.q_full_idx if long_seq_metadata else None
        cp_prefill_mask = long_seq_metadata.cp_prefill_mask if long_seq_metadata else None

        # TODO(xyx): remove the if condition after mla supports torch mode speculative decoding
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device

        block_table = (common_attn_metadata.block_table_tensor[:num_reqs])
        if num_actual_tokens_cp_full is None:
            num_actual_tokens_cp_full = num_actual_tokens
        slot_mapping = common_attn_metadata.slot_mapping_cpu[:
                                                             num_actual_tokens_cp_full].to(
                                                                 device,
                                                                 non_blocking=
                                                                 True)
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens].long(
                                                         )

        if self.cos_cache is None:
            self.cos_cache = model.model.layers[
                0].self_attn.rotary_emb.cos_cached
            self.sin_cache = model.model.layers[
                0].self_attn.rotary_emb.sin_cached
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
                attn_mask=common_attn_metadata.attn_mask,
                query_lens=query_lens[reqs_start:],
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
                cp_prefill_mask=cp_prefill_mask)

        decode_metadata = None
        if num_decodes > 0:
            actual_seq_lengths_q = query_start_loc[1:num_decodes + 1].tolist()
            max_seq_lens = seq_lens[:num_decodes].max().item()
            seq_lens = seq_lens[:num_decode_tokens]
            input_positions = input_positions[:num_decode_tokens]
            block_table = block_table[:num_decode_tokens, ...]
            seq_lens_list = seq_lens.tolist()

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
                attn_mask=common_attn_metadata.spec_attn_mask,
                actual_seq_lengths_q=actual_seq_lengths_q,
                sin=sin,
                cos=cos,
                num_computed_tokens_of_cp_sp=num_computed_tokens_of_cp_sp,
            )

        return self.metadata_cls(  # type: ignore
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
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp,
        )


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
        self.q_proj = kwargs['q_proj']
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.q_a_proj = kwargs.get('q_a_proj', None)
        self.q_a_layernorm = kwargs.get('q_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_prefetch = ascend_config.enable_prefetch
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz
        self.chunked_prefill_for_mla = ascend_config.chunked_prefill_for_mla

        vllm_config = get_current_vllm_config()
        self.ring_mla_mask_size = 512
        self.prefill_mask = None

        # Adapt torch air graph mode with spec decoding.
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            self.spec_token_num = speculative_config.num_speculative_tokens
            assert self.spec_token_num > 0

        self.cp_size = get_context_model_parallel_world_size(
        ) if context_parallel_enable else 1
        self.cp_rank = get_context_model_parallel_rank(
        ) if self.cp_size > 1 else 0
        self.cp_group = get_cp_group(
        ).device_group if self.cp_size > 1 else None
        self.enable_sp = get_current_vllm_config(
        ).parallel_config.enable_sequence_parallel if sequence_parallel_enable else 0

        self.sp_size = get_tensor_model_parallel_world_size(
        ) if self.enable_sp else 1
        self.sp_group = get_tp_group(
        ).device_group if self.sp_size > 1 else None

    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
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
                seq_len2.to(q_nope.device),
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
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=v,
                mask=self.prefill_mask,
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
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            query = torch.cat((q_nope, q_pe), dim=-1)
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
        elif self.chunked_prefill_for_mla:
            attn_lse = torch.empty(self.num_heads,
                                   num_tokens,
                                   dtype=torch.float32,
                                   device=q_nope.device)
            if self.prefill_mask is None:
                self.prefill_mask = torch.triu(
                    torch.ones(self.ring_mla_mask_size,
                               self.ring_mla_mask_size,
                               device=q_nope.device,
                               dtype=q_nope.dtype), 1)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=value,
                mask=self.prefill_mask,
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
                q_nope, q_pe, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse)
        else:
            query = torch.cat((q_nope, q_pe), dim=-1)
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

        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim])
        if attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ] and not self.chunked_prefill_for_mla:
            attn_output = attn_output_torch
        return attn_output

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
        num_tokens = q_nope.size(0)
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
            mask=mask)

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
            mask=mask)

        q_full_idx = attn_metadata.prefill.q_full_idx
        output = torch.index_select(
            torch.cat([output_head, output_tail], dim=0), 0, q_full_idx)

        output = output.reshape([num_tokens, self.num_heads * self.v_head_dim])

        return output

    def _attention_with_mask_and_nomask(
            self, q_nope: torch.Tensor, q_pe: torch.Tensor,
            k_nope: torch.Tensor, k_pe: torch.Tensor, value: torch.Tensor,
            kv_mask_idx: torch.Tensor, kv_nomask_idx: torch.Tensor,
            attn_mask_seqlens: torch.Tensor, attn_nomask_seqlens: torch.Tensor,
            mask: torch.Tensor):
        attn_output = torch.empty(
            q_nope.shape[0],
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
            return attn_output

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
        cache_mode = "PA_BLK_NZ" if self.enable_kv_nz else "PA"
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

        if attn_metadata.attn_state == AscendAttentionState.SpecDecoding:
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

        current_ms_metadata = get_multistream_comm_context()
        if current_ms_metadata is None:
            return self._v_up_proj(attn_output)
        else:
            current_ms_metadata.before_comm_event.record()
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait()
                return self._v_up_proj(attn_output)

    def _mla_preprocess(self, hidden_states, kv_cache, attn_metadata,
                        need_gather_q_kv):
        # MLA Preprocess:
        # 1. Perform q_a_proj and q_a_layernorm to obtain q_c
        # 2. Perform kv_a_proj_with_mqa to obtain kv_no_split
        # 3. If need_gather_q_kv, perform all_gather.
        # 4. Preprocess decode tokens, write kv cache and get:
        # decode_ql_nope, decode_q_pe, decode_k_pe, decode_k_nope
        # 5. Preprocess prefill tokens, write kv cache and get:
        # prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        if self.enable_sp and has_prefill:
            need_gather_q_kv = True
        if self.q_a_proj is not None:
            npu_prefetch(self.q_a_proj.weight,
                         hidden_states,
                         enabled=self.enable_prefetch)
            ckq = self.q_a_proj(hidden_states)[0]
            q_c = self.q_a_layernorm(ckq)
        else:
            q_c = hidden_states

        kv_no_split = self.kv_a_proj_with_mqa(hidden_states)[0]
        # Process for shared_expert_dp
        if need_gather_q_kv:
            q_c = get_tp_group().all_gather(q_c, 0)
            q_c = q_c[:attn_metadata.num_input_tokens]
            kv_no_split = get_tp_group().all_gather(kv_no_split, 0)
            kv_no_split = kv_no_split[:attn_metadata.num_input_tokens]
        decode_preprocess_res = None
        prefill_preprocess_res = None
        # Preprocess for decode tokens
        if has_decode:
            decode_q_c = q_c[:num_decode_tokens]
            cos = attn_metadata.decode.cos
            sin = attn_metadata.decode.sin
            decode_ql_nope, decode_q_pe = \
                self._q_proj_and_k_up_proj(decode_q_c)
            if self.sp_size > 1:
                decode_q_no_split = torch.cat([decode_ql_nope, decode_q_pe],
                                              dim=-1)
                decode_q_no_split = get_tp_group().all_gather(
                    decode_q_no_split, 1)
                decode_ql_nope, decode_q_pe = decode_q_no_split.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
            decode_slots = attn_metadata.slot_mapping[:num_decode_tokens]
            decode_kv_no_split = kv_no_split[:num_decode_tokens]

            decode_k_pe, decode_k_nope = self.exec_kv_decode(
                decode_kv_no_split, cos, sin, kv_cache, decode_slots)

            decode_preprocess_res = DecodeMLAPreprocessResult(
                decode_ql_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        # Preprocess for prefill tokens
        if has_prefill:
            prefill_kv_no_split = kv_no_split[
                num_decode_tokens:num_actual_tokens]
            prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
            prefill_q = self.q_proj(prefill_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
            cos = attn_metadata.prefill.cos
            sin = attn_metadata.prefill.sin
            prefill_slots = attn_metadata.slot_mapping[
                num_decode_tokens:num_actual_tokens]
            prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
            if self.cp_size > 1:
                kv_c, k_pe = prefill_kv_no_split.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
                assert len(
                    kv_cache
                ) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
                kv_c_normed = kv_c_normed.view(
                    [num_actual_tokens, self.num_kv_heads, -1])
                k_pe = k_pe.unsqueeze(1)
                prefill_k_pe = k_pe[num_decode_tokens:]
                prefill_k_pe = self.rope_single(prefill_k_pe, cos, sin)
                prefill_k_c_normed = kv_c_normed[num_decode_tokens:]

                prefill_kv_c_k_pe = torch.cat(
                    [prefill_k_c_normed, prefill_k_pe], dim=-1)
                prefill_kv_c_k_pe = get_cp_group().all_gather(
                    prefill_kv_c_k_pe, 0)
                prefill_kv_c_k_pe = torch.index_select(
                    prefill_kv_c_k_pe, 0,
                    attn_metadata.prefill.cp_kv_recover_idx)
                prefill_k_c_normed, prefill_k_pe = prefill_kv_c_k_pe.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed, k_pe = prefill_k_c_normed, prefill_k_pe
                prefill_k_c_normed = prefill_k_c_normed.squeeze()
                torch_npu._npu_reshape_and_cache(
                    key=kv_c_normed,
                    value=k_pe,
                    key_cache=kv_cache[0],
                    value_cache=kv_cache[1],
                    slot_indices=attn_metadata.slot_mapping)
            else:
                prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(
                    prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
            prefill_k_nope, prefill_value = self.kv_b_proj(
                prefill_k_c_normed)[0].view(
                    -1, self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim).split(
                        [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            if not self.cp_size > 1:
                prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0],
                                                 self.num_kv_heads, -1)
            prefill_k_pe = prefill_k_pe.expand(
                (*prefill_k_nope.shape[:-1], -1))
            prefill_preprocess_res = PrefillMLAPreprocessResult(
                prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe,
                prefill_value)
        return decode_preprocess_res, prefill_preprocess_res

    def _forward_decode_sp(
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
        if self.enable_sp:
            num_heads = self.num_heads * self.tp_size
        else:
            num_heads = self.num_heads

        k_nope = k_nope.view(-1, block_size, self.num_kv_heads,
                             self.kv_lora_rank)
        k_pe = k_pe.view(-1, block_size, self.num_kv_heads,
                         self.qk_rope_head_dim)
        q_nope = q_nope.view(num_tokens, num_heads, -1)
        q_pe = q_pe.view(num_tokens, num_heads, -1)

        # use cp & sp split computed token nums from scheduler to compute actual seq_len and seq_mask
        num_computed_tokens_of_cp_sp = np.array(
            decode_meta.num_computed_tokens_of_cp_sp)  # [bs, cp_size, sp_size]
        seq_mask_cp = torch.where(
            torch.tensor(num_computed_tokens_of_cp_sp.sum(2)) == 0, 0,
            1).to(torch.uint8).to(q_pe.device)
        seq_mask_sp = torch.where(
            torch.tensor(num_computed_tokens_of_cp_sp[:,
                                                      self.cp_rank, :]) == 0,
            0, 1).to(torch.uint8).to(q_pe.device)
        seq_len = num_computed_tokens_of_cp_sp[:, self.cp_rank, self.sp_rank]
        seq_len = torch.tensor(seq_len, dtype=torch.int32)

        if torch.sum(seq_len).item() == 0:
            # Case that no kv_cache has been stored on this rank, no need to do following computation.
            attn_output = torch.zeros(
                [num_tokens, num_heads, self.kv_lora_rank],
                dtype=q_nope.dtype,
                device=q_nope.device)
            softmax_lse = torch.full((num_tokens, num_heads, 1),
                                     float('-inf'),
                                     dtype=q_nope.dtype,
                                     device=q_nope.device)
        else:
            attn_output, softmax_lse = torch_npu.atb.npu_multi_head_latent_attention(
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
                calc_type="calc_type_ring")

        # TODO use update op to replace this
        def _update_out_and_lse(
            out: torch.Tensor,
            lse: torch.Tensor,
            block_out: torch.Tensor,
            block_lse: torch.Tensor,
            mask: torch.Tensor = None,
        ):
            if out is None:
                out = block_out.to(torch.float32)
                lse = block_lse
            else:
                if mask is None:
                    mask = torch.ones([block_out.size(0)],
                                      dtype=torch.uint8,
                                      device=block_out.device)
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
        if self.sp_size > 1:
            # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
            attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   attn_out_lse,
                                   group=self.sp_group)
            # permute: [num_heads, v_head_dim+1, bs] -> [bs, num_heads, v_head_dim+1]
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            attn_out_lse_split_on_seq = list(
                torch.chunk(attn_out_lse_all2all, self.sp_size, dim=1))
            # Update out&lse
            attn_out_g = None
            attn_lse_g = None
            for i, attn_out_lse_l in enumerate(attn_out_lse_split_on_seq):
                attn_out_l, attn_lse_l = torch.split(attn_out_lse_l,
                                                     [self.kv_lora_rank, 1],
                                                     dim=-1)
                attn_out_g, attn_lse_g = _update_out_and_lse(
                    attn_out_g, attn_lse_g, attn_out_l, attn_lse_l,
                    seq_mask_sp[:, i])
            attn_output = attn_out_g
            softmax_lse = attn_lse_g

        if self.cp_size > 1:
            # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
            attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
            # AllGather out&lse within CP group
            attn_out_lse_list = [
                torch.empty_like(attn_out_lse) for _ in range(self.cp_size)
            ]
            dist.all_gather(attn_out_lse_list,
                            attn_out_lse,
                            group=self.cp_group)
            # Update out&lse
            attn_out_g = None
            attn_lse_g = None
            for i, attn_out_lse_l in enumerate(attn_out_lse_list):
                attn_out_l, attn_lse_l = torch.split(attn_out_lse_l,
                                                     [self.kv_lora_rank, 1],
                                                     dim=-1)
                attn_out_g, attn_lse_g = _update_out_and_lse(
                    attn_out_g, attn_lse_g, attn_out_l, attn_lse_l,
                    seq_mask_cp[:, i])
            attn_output = attn_out_g
        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.kv_lora_rank])
        # out = self.o_proj(attn_output, is_prefill=False)[0]
        current_ms_metadata = get_multistream_comm_context()
        if current_ms_metadata is None:
            return self._v_up_proj(attn_output)
        else:
            current_ms_metadata.before_comm_event.record()
            with torch.npu.stream(current_ms_metadata.comm_stream):
                current_ms_metadata.before_comm_event.wait()
                return self._v_up_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output
        num_actual_tokens = attn_metadata.num_actual_tokens
        assert attn_metadata.num_decodes is not None and \
        attn_metadata.num_prefills is not None and \
        attn_metadata.num_decode_tokens is not None
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_tokens, ...]
        o_proj_input_shape = (num_actual_tokens,
                              self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        # MLA Preprocess
        decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
            hidden_states, kv_cache, attn_metadata, need_gather_q_kv)

        if decode_preprocess_res is not None:
            # MLA Preprocess for decoding
            self.sp_size = get_tensor_model_parallel_world_size(
            ) if self.enable_sp else 1
            self.sp_rank = get_tensor_model_parallel_rank(
            ) if self.enable_sp else 0
            if self.cp_size * self.sp_size > 1:
                output_decode = self._forward_decode_sp(
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
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    o_proj_input[:num_decode_tokens] = output_decode
                    current_ms_metadata.after_comm_event.record()
            else:
                o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            if self.cp_size > 1:
                output_prefill = self._forward_prefill_cp(
                    prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                    prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                    prefill_preprocess_res.value, kv_cache, attn_metadata)
            else:
                output_prefill = self._forward_prefill(
                    prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                    prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                    prefill_preprocess_res.value, kv_cache, attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    o_proj_input[num_decode_tokens:] = output_prefill
                    current_ms_metadata.after_comm_event.record()
            else:
                o_proj_input[num_decode_tokens:] = output_prefill
        # O proj
        current_ms_metadata = get_multistream_comm_context()
        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
        if current_ms_metadata is None:
            npu_prefetch(self.o_proj.weight,
                         o_proj_input,
                         max_size=MAX_O_PROJ_PREFETCH_SIZE,
                         enabled=self.enable_prefetch)

            output[...] = self.o_proj(o_proj_input)[0]
        else:
            with torch.npu.stream(current_ms_metadata.comm_stream):
                npu_prefetch(self.o_proj.weight,
                             o_proj_input,
                             max_size=MAX_O_PROJ_PREFETCH_SIZE,
                             enabled=self.enable_prefetch)
                output[...] = self.o_proj(
                    o_proj_input,
                    is_prefill=prefill_preprocess_res is not None,
                    is_force_scatter=self.enable_shared_expert_dp)[0]
                current_ms_metadata.after_comm_event.record()
        del o_proj_input
        return output_padded
