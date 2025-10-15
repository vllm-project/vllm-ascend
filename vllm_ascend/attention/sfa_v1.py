from dataclasses import dataclass
from typing import (TYPE_CHECKING, ClassVar, NamedTuple, Optional, Tuple, Type,
                    TypeVar)

import torch
import torch_npu
from torch import nn
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata,
                                              MLAAttentionImpl)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.utils import cdiv, round_down
from vllm.v1.attention.backends.utils import AttentionCGSupport

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         split_decodes_and_prefills)
from vllm_ascend.multistream.base import MSAttentionMetadataSplitConfig
from vllm_ascend.multistream.ms_split import model_input_split_v1_mla_attn
from vllm_ascend.worker.npu_input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendSFABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AscendSFAMetadata

    @staticmethod
    def get_builder_cls():
        return AscendSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> Type["AscendSFAImpl"]:
        return AscendSFAImpl


@dataclass
class AscendSFAPrefillMetadata:
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
    sin: torch.Tensor
    cos: torch.Tensor
    chunked_context: Optional[ChunkedContextMetadata] = None


@dataclass
class AscendSFADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_lens: int
    seq_lens_list: list[int]
    actual_seq_lengths_q: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    attn_mask: Optional[torch.Tensor] = None


@dataclass
class AscendSFAMetadata:
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

    decode: Optional[AscendSFADecodeMetadata] = None
    prefill: Optional[AscendSFAPrefillMetadata] = None
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
    ) -> list["AscendSFAMetadata"]:
        """Split metadata for multi-stream with AscendSFAMetadata"""
        return model_input_split_v1_mla_attn(
            ms_split_config=ms_split_config,
            attn_metadata=self,
            _metadata_cls=AscendMLAMetadata,
        )


M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.NEVER
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
                 metadata_cls: Optional[AscendSFAMetadata] = None):
        self.metadata_cls: Optional[AscendSFAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendSFAMetadata  # type: ignore
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
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device

        block_table = (common_attn_metadata.block_table_tensor[:num_reqs])
        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens].to(
                                                             device,
                                                             non_blocking=True)
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens].long(
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
                    AscendSFAPrefillMetadata.ChunkedContextMetadata(
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
            actual_query_lens = torch.tensor(query_lens[reqs_start:],
                                             dtype=torch.int32).npu()
            query_lens_prefill_sfa = torch.cumsum(actual_query_lens,
                                                  dim=0).to(torch.int32)
            seq_lens_prefill_sfa = seq_lens[reqs_start:].to(torch.int32).npu()
            prefill_metadata = AscendSFAPrefillMetadata(
                attn_mask=common_attn_metadata.attn_mask,
                query_lens=query_lens_prefill_sfa,
                seq_lens=seq_lens_prefill_sfa,
                context_lens=seq_lens[reqs_start:],
                input_positions=prefill_input_positions,
                block_table=block_table[reqs_start:, ...],
                max_query_len=max_query_len,
                max_seq_lens=max_seq_lens,
                query_start_loc=prefill_query_start_loc,
                chunked_context=chunked_context_metadata,
                sin=sin,
                cos=cos,
            )

        decode_metadata = None
        if num_decodes > 0:
            # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
            actual_seq_lengths_q = query_start_loc[1:num_decodes + 1].to(
                torch.int32).npu()
            max_seq_lens = seq_lens[:num_decodes].max().item()
            seq_lens = seq_lens[:num_decodes].to(torch.int32).npu()
            input_positions = input_positions[:num_decode_tokens]
            block_table = block_table[:num_decodes, ...]
            seq_lens_list = seq_lens.tolist()

            cos = self.cos_cache[input_positions].unsqueeze(  # type: ignore
                1).unsqueeze(2)
            sin = self.sin_cache[input_positions].unsqueeze(  # type: ignore
                1).unsqueeze(2)

            decode_metadata = AscendSFADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                seq_lens_list=seq_lens_list,
                max_seq_lens=max_seq_lens,
                attn_mask=common_attn_metadata.spec_attn_mask,
                actual_seq_lengths_q=actual_seq_lengths_q,
                sin=sin,
                cos=cos)

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


class PrefillSFAPreprocessResult(NamedTuple):
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    query_states: Optional[torch.Tensor] = None
    key_states: Optional[torch.Tensor] = None


class DecodeSFAPreprocessResult(NamedTuple):
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    # nope_cache: Optional[torch.Tensor] = None
    # rope_cache: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    query_states: Optional[torch.Tensor] = None
    key_states: Optional[torch.Tensor] = None
    bsz: Optional[int] = None


class AscendSFAImpl(MLAAttentionImpl):
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
        self.indexer = kwargs['indexer']
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.q_a_proj = kwargs.get('q_a_proj', None)
        self.q_a_layernorm = kwargs.get('q_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = self.num_heads // self.tp_size
        if self.q_a_proj is not None:
            self.q_b_proj = self.q_proj
        else:
            self.q_b_proj = None

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_prefetch = ascend_config.enable_prefetch
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz

        vllm_config = get_current_vllm_config()
        self.ring_mla_mask_size = 512
        self.prefill_mask = None

        # indexer param
        self.dim = self.indexer.dim
        self.n_heads: int = self.indexer.n_heads  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.index_topk: int = self.indexer.index_topk  # 2048
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm
        self.softmax_scale = self.indexer.softmax_scale

        # Adapt torch air graph mode with spec decoding.
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            self.spec_token_num = speculative_config.num_speculative_tokens
            assert self.spec_token_num > 0

        self.cp_size = 1

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

        self.kv_b_proj_w_k, self.kv_b_proj_w_v = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.kv_b_proj_w_v = self.kv_b_proj_w_v.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        self.kv_b_proj_w_k = self.kv_b_proj_w_k.permute(1, 2, 0).contiguous()

        # Waiting for BMM NZ support
        # self.W_UV.data = torch_npu.npu_format_cast(self.W_UV.data, 29)
        # self.W_UK_T.data = torch_npu.npu_format_cast(self.W_UK_T.data, 29)

    def _sfa_preprocess(self, hidden_states, kv_cache, attn_metadata,
                        need_gather_q_kv):
        # SFA Preprocess:
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
        if need_gather_q_kv:
            # q_c = get_tp_group().all_gather(q_c, 0)
            # kv_no_split = get_tp_group().all_gather(kv_no_split, 0)
            hidden_states = get_tp_group().all_gather(hidden_states, 0)
        # hidden_states_decode = hidden_states[:num_decode_tokens]
        # if self.q_a_proj is not None:
        #     npu_prefetch(self.q_a_proj.weight,
        #                  hidden_states,
        #                  enabled=self.enable_prefetch)
        #     ckq = self.q_a_proj(hidden_states) # q down
        #     q_c = self.q_a_layernorm(ckq)  # q down layernorm
        # else:
        #     q_c = hidden_states

        # kv_no_split = self.kv_a_proj_with_mqa(hidden_states) # c_kv
        # Process for shared_expert_dp

        decode_preprocess_res = None
        prefill_preprocess_res = None
        # Preprocess for decode tokens
        if has_decode:
            q_len = 1
            hidden_states_decode = hidden_states[:num_decode_tokens]
            decode_kq = self.q_a_proj(hidden_states_decode)  # q down
            decode_q_c = self.q_a_layernorm(decode_kq)  # q down layernorm
            decode_kv_no_split = self.kv_a_proj_with_mqa(
                hidden_states_decode)  # c_kv

            # decode_q_c = q_c[:num_decode_tokens]
            decode_slot_mapping = attn_metadata.slot_mapping[:
                                                             num_decode_tokens]
            # decode_kv_no_split = decode_kv_no_split[:num_decode_tokens]

            decode_q = self.q_b_proj(decode_q_c)
            bsz, _ = decode_q.shape
            decode_q = decode_q.view(bsz, self.num_heads, 1, self.qk_head_dim)
            decode_q_nope, decode_q_pe = torch.split(
                decode_q, [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1)
            decode_q_nope = decode_q_nope.view(
                -1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
            decode_q_nope = (torch.matmul(decode_q_nope,
                                          self.kv_b_proj_w_k).transpose(
                                              1,
                                              0).view(bsz, q_len,
                                                      self.num_heads,
                                                      self.kv_lora_rank))

            # stream2 kv
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            cos = attn_metadata.decode.cos
            sin = attn_metadata.decode.sin
            cos_q, sin_q = cos, sin
            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            decode_kv_no_split = decode_kv_no_split.unsqueeze(1).unsqueeze(1)
            decode_k_rope, decode_k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                decode_kv_no_split,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                decode_slot_mapping.to(torch.int64),
                value_cache,
                key_cache,
                c_kv_scale=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode='PA')  # adapter NZ
            # nz_block_size = 16
            # KVCACHE_NZ_DIM = 16
            # decode_k_nope = decode_k_nope.view(block_num, 1, self.kv_lora_rank // nz_block_size, block_size, nz_block_size)
            # decode_k_rope = decode_k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

            decode_q_pe = torch_npu.npu_interleave_rope(decode_q_pe, cos,
                                                        sin)  # BNSD

            decode_q_nope = decode_q_nope.view(bsz, self.num_heads,
                                               self.kv_lora_rank)
            decode_q_pe = decode_q_pe.view(bsz, self.num_heads, -1)

            topk_indices = self.indexer_select(hidden_states_decode,
                                               decode_q_c,
                                               attn_metadata=attn_metadata,
                                               kv_cache=kv_cache)

            query_states = (decode_q_nope, decode_q_pe)
            key_states = (decode_k_nope, decode_k_rope)
            decode_preprocess_res = DecodeSFAPreprocessResult(
                q_nope=decode_q_nope,
                q_pe=decode_q_pe,
                # nope_cache = nope_cache,
                # rope_cache = rope_cache,
                topk_indices=topk_indices,
                query_states=query_states,
                key_states=key_states,
                bsz=bsz,
            )

        # Preprocess for prefill tokens
        if has_prefill:
            bsz = 1

            hidden_states_prefill = hidden_states[
                num_decode_tokens:num_actual_tokens]
            prefill_kq = self.q_a_proj(hidden_states_prefill)  # q down
            prefill_q_c = self.q_a_layernorm(prefill_kq)  # q down layernorm
            prefill_kv_no_split = self.kv_a_proj_with_mqa(
                hidden_states_prefill)  # c_kv

            # prefill_q_c = q_c[
            #     num_decode_tokens:num_actual_tokens]
            prefill_slot_mapping = attn_metadata.slot_mapping[
                num_decode_tokens:num_actual_tokens]
            # decode_kv_no_split = decode_kv_no_split[:num_decode_tokens]

            prefill_slot_mapping = attn_metadata.slot_mapping[
                num_decode_tokens:num_actual_tokens]
            # prefill_kv_no_split = kv_no_split[
            #     num_decode_tokens:num_actual_tokens]
            # prefill_qr = prefill_q_c[num_decode_tokens:num_actual_tokens]
            prefill_qr = prefill_q_c
            prefill_q = self.q_b_proj(prefill_qr)
            prefill_q = prefill_q.view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_nope, prefill_q_pe = torch.split(
                prefill_q, [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1)
            prefill_q_nope = prefill_q_nope.view(
                -1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
            prefill_q_nope = (torch.matmul(prefill_q_nope,
                                           self.kv_b_proj_w_k).transpose(
                                               1,
                                               0).view(-1, self.num_heads,
                                                       self.kv_lora_rank))
            prefill_q_pe = prefill_q_pe.unsqueeze(2)

            # stream2 kv

            nope_cache = kv_cache[0]
            rope_cache = kv_cache[1]
            cos = attn_metadata.prefill.cos
            sin = attn_metadata.prefill.sin
            cos_q, sin_q = cos, sin

            # cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            # sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            prefill_q_pe = torch_npu.npu_interleave_rope(
                prefill_q_pe, cos_q, sin_q)  # BNSD
            prefill_q_pe = prefill_q_pe.squeeze(2)  #BSH
            # q[..., self.qk_nope_head_dim:] = prefill_q_pe # TODO:????

            prefill_latent_cache = prefill_kv_no_split  # (B,S,N,D)
            prefill_k_pe, prefill_k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                prefill_latent_cache.view(
                    -1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                prefill_slot_mapping.to(torch.int64),
                rope_cache,
                nope_cache,
                k_rope_scale=None,
                c_kv_scale=None,
                k_rope_offset=None,
                c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA")

            topk_indices = self.indexer_select(x=hidden_states_prefill,
                                               qr=prefill_qr,
                                               kv_cache=kv_cache,
                                               attn_metadata=attn_metadata)
            query_states = (prefill_q_nope, prefill_q_pe)
            key_states = (prefill_k_nope, prefill_k_pe)
            prefill_preprocess_res = PrefillSFAPreprocessResult(
                q_nope=prefill_q_nope,
                q_pe=prefill_q_pe,
                topk_indices=topk_indices,
                k_nope=prefill_k_nope,
                k_pe=prefill_k_pe,
                query_states=query_states,
                key_states=key_states,
            )

        return decode_preprocess_res, prefill_preprocess_res

    def forward(
        self,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
        output = output[:num_actual_tokens, ...]
        o_proj_input_shape = (num_actual_tokens,
                              self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        # SFA Preprocess
        decode_preprocess_res, prefill_preprocess_res = self._sfa_preprocess(
            hidden_states, kv_cache, attn_metadata, need_gather_q_kv)

        if decode_preprocess_res is not None:
            # bsz, q_len, _, _ = query_states[0].shape
            decode_attn_output = self.apply_attention_fusion(
                query_states=decode_preprocess_res.query_states,
                key_states=decode_preprocess_res.key_states,
                attn_metadata=attn_metadata,
                topk_indices=decode_preprocess_res.topk_indices)
            o_proj_input[:num_decode_tokens] = decode_attn_output

        if prefill_preprocess_res is not None:
            prefill_attn_output = self.apply_attention_fusion(
                query_states=prefill_preprocess_res.query_states,
                key_states=prefill_preprocess_res.key_states,
                attn_metadata=attn_metadata,
                topk_indices=prefill_preprocess_res.topk_indices)
            o_proj_input[num_decode_tokens:] = prefill_attn_output

        output[...] = self.mla_epilog(o_proj_input, absorb=True)
        return output

    def apply_attention_fusion(self, query_states, key_states, topk_indices,
                               attn_metadata: M):
        # repeat k/v heads if n_kv_heads < n_heads
        q_nope, q_pe = query_states
        k_nope, k_rope = key_states

        if attn_metadata.prefill is not None:

            prefill_metadata = attn_metadata.prefill

            slc_fa_fusion = torch.ops.custom.npu_sparse_flash_attention(
                query=q_nope,
                key=k_nope,
                value=k_nope,
                sparse_indices=topk_indices,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=prefill_metadata.block_table,
                actual_seq_lengths_query=prefill_metadata.query_lens,
                actual_seq_lengths_kv=prefill_metadata.seq_lens,
                query_rope=q_pe,
                key_rope=k_rope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )

        elif attn_metadata.decode is not None:
            decode_metadata = attn_metadata.decode

            slc_fa_fusion = torch.ops.custom.npu_sparse_flash_attention(
                query=q_nope,
                key=k_nope,
                value=k_nope,
                sparse_indices=topk_indices,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=attn_metadata.decode.block_table,
                actual_seq_lengths_query=decode_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=decode_metadata.seq_lens,
                query_rope=q_pe,
                key_rope=k_rope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
            slc_fa_fusion = slc_fa_fusion.squeeze(1)

        slc_fa_fusion = slc_fa_fusion.transpose(0, 1)

        # input shape [N//attn_tp_size, T(bs*q_len), D]
        # output shape [T(bs*q_len), N//attn_tp_size, D]
        attn_output = torch.matmul(slc_fa_fusion,
                                   self.kv_b_proj_w_v).transpose(1, 0).reshape(
                                       -1, self.num_heads * self.v_head_dim)
        # Note: Considering the fusion rules of TBMM, attn_output shape requires a 3-dim shape, and
        # with appropriate tensor stride for the later 'view' operation if oproj_tp_size > 1.
        # after reshape: [T(bs*q_len), 1, N//attn_tp_size*D]
        # attn_output = attn_output.reshape(-1, self.num_heads * self.v_head_dim)

        return attn_output

    def mla_epilog(self,
                   attn_output: torch.Tensor = None,
                   absorb: bool = False):
        # TODO: need to check
        attn_output = self.o_proj(attn_output.reshape(attn_output.shape[0],
                                                      -1),
                                  is_prefill=True,
                                  is_force_scatter=False)

        return attn_output

    def indexer_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
    ):
        if attn_metadata.prefill is not None:
            cos = attn_metadata.prefill.cos
            sin = attn_metadata.prefill.sin
            actual_seq_lengths_query = attn_metadata.prefill.query_lens
            actual_seq_lengths_key = attn_metadata.prefill.seq_lens
            block_table = attn_metadata.prefill.block_table
        elif attn_metadata.decode is not None:
            cos = attn_metadata.decode.cos
            sin = attn_metadata.decode.sin
            actual_seq_lengths_query = attn_metadata.decode.actual_seq_lengths_q
            actual_seq_lengths_key = attn_metadata.decode.seq_lens
            block_table = attn_metadata.decode.block_table

        cos_q, sin_q = cos, sin
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        # q process in new stream
        q = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q = q.view(-1, self.n_heads, self.head_dim)  # [b,s,64,128]
        q_pe, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64,64+64]

        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]

        k_proj = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k = self.k_norm(k_proj).unsqueeze(1)
        k_pe, k_nope = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1)  # [b,s,64+64]

        k_pe = k_pe.unsqueeze(2)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
        k_pe = k_pe.squeeze(2)

        k = torch.cat([k_pe, k_nope], dim=-1)  # [b*s,128]

        if kv_cache is not None:
            torch_npu.npu_scatter_nd_update_(kv_cache[2].view(-1, k.shape[-1]),
                                             attn_metadata.slot_mapping.view(
                                                 -1, 1),
                                             k.view(-1,
                                                    k.shape[-1]))  # b, s, n, d

        weights = self.weights_proj(x)

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q,
            key=kv_cache[2],
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3)
        return topk_indices
