# mypy: ignore-errors
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


class RequestStage(Enum):
    Prefill = 0
    Decode = 1
    PdMerged = 2


class AscendHSTUAttentionBackend(AttentionBackend):
    """HSTU 注意力后端"""

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendHSTUAttentionBackendImpl"]:
        return AscendHSTUAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendHSTUAttentionMetadata"]:
        return AscendHSTUAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AscendHSTUAttentionMetadataBuilder"]:
        return AscendHSTUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [64]


@dataclass
class AscendHSTUAttentionMetadata:
    """HSTU Attention Metadata"""
    attn_mask: Optional[torch.Tensor] = None
    attn_state: AscendAttentionState = AscendAttentionState.PrefillNoCache
    num_actual_tokens: int = 0
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None
    actual_seq_lengths_q: List[int] = None
    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    max_query_len: Optional[int] = None
    block_tables: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    additional_metadata: Optional[Dict[str, Any]] = None


class AscendHSTUAttentionMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendHSTUAttentionBackend.get_supported_block_size()[0])

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def _build_decode_metadata(self, attn_metadata, indices, is_dummy):
        """Build decode-style metadata for a subset of requests identified by indices."""
        sub_seq_lens = attn_metadata.seq_lens[indices].npu()
        sub_query_lens = attn_metadata.query_lens[indices].npu()
        sub_block_table = attn_metadata.block_tables[indices]
        npu_device = sub_seq_lens.device
        if sub_block_table.device != npu_device:
            sub_block_table = sub_block_table.to(npu_device)
        block_size = self.vllm_config.cache_config.block_size
        num_sub = indices.shape[0]

        deal_seq_lens = sub_seq_lens - sub_query_lens
        num_blocks_per_seq = torch.ceil(deal_seq_lens / block_size).to(
            torch.int64)

        page_offsets = torch.zeros(num_sub + 1,
                                   dtype=torch.int64,
                                   device=npu_device)
        torch.cumsum(num_blocks_per_seq, dim=0, out=page_offsets[1:])

        last_page_len_tensor = (deal_seq_lens - 1) % block_size + 1

        seq_offset_k = torch.zeros(num_sub + 1,
                                   dtype=torch.int64,
                                   device=npu_device)
        torch.cumsum(sub_seq_lens, dim=0, out=seq_offset_k[1:])

        sub_seq_lens_list = [
            attn_metadata.seq_lens_list[i] for i in indices.tolist()
        ]
        max_seq_len_k = max(sub_seq_lens_list) - 1 if is_dummy else max(
            sub_seq_lens_list)

        page_ids_list = []
        for i in range(num_sub):
            num_blocks_for_req = page_offsets[i + 1] - page_offsets[i]
            valid_blocks = sub_block_table[i, :num_blocks_for_req]
            page_ids_list.append(valid_blocks)
        page_ids = torch.cat(page_ids_list).to(torch.int64)

        query_start_sub = torch.zeros(num_sub + 1,
                                      dtype=torch.int64,
                                      device=npu_device)
        torch.cumsum(sub_query_lens, dim=0, out=query_start_sub[1:])

        return AscendHSTUAttentionMetadata(
            attn_state=AscendAttentionState.DecodeOnly,
            max_query_len=attn_metadata.max_query_len,
            query_start_loc=query_start_sub,
            query_lens=sub_query_lens,
            seq_lens=sub_seq_lens,
            block_tables=sub_block_table,
            additional_metadata={
                "max_seq_len_k":
                max_seq_len_k if not is_dummy else max_seq_len_k + 1,
                "seq_offset_k":
                seq_offset_k,
                "page_offsets":
                page_offsets,
                "page_ids":
                page_ids,
                "last_page_len":
                last_page_len_tensor,
            })

    def _build_prefill_metadata(self, attn_metadata, indices, num_candidates):
        """Build prefill-style metadata for a subset of requests identified by indices."""
        sub_query_lens = attn_metadata.query_lens[indices]
        num_sub = indices.shape[0]
        npu_device = num_candidates.device
        if sub_query_lens.device != npu_device:
            sub_query_lens = sub_query_lens.to(npu_device)

        query_start_sub = torch.zeros(num_sub + 1,
                                      dtype=torch.int64,
                                      device=npu_device)
        torch.cumsum(sub_query_lens, dim=0, out=query_start_sub[1:])

        return AscendHSTUAttentionMetadata(
            attn_state=AscendAttentionState.PrefillNoCache,
            max_query_len=attn_metadata.max_query_len,
            query_start_loc=query_start_sub,
            query_lens=sub_query_lens,
            additional_metadata={
                "num_candidates": num_candidates[indices],
            })

    def _enhance_metadata(
        self,
        attn_metadata,
        requests: Optional[dict] = None,
        scheduler_output: Optional["SchedulerOutput"] = None,
        is_dummy=False,
        num_reqs=0,
    ):
        if not is_dummy:
            reqs = scheduler_output.scheduled_new_reqs
            cached_reqs = scheduler_output.scheduled_cached_reqs
            uids = []
            candidate_nums = []
            if reqs:
                request_stage = reqs[0].sampling_params.extra_args[
                    "request_stage"]
                for req in reqs:
                    uids += req.sampling_params.extra_args["uid"]
                    candidate_nums += req.sampling_params.extra_args[
                        "candidate_num"]
            elif cached_reqs:
                tmp_request = requests.get(cached_reqs.req_ids[0], None)
                if tmp_request:
                    request_stage = tmp_request.sampling_params.extra_args[
                        "request_stage"]
            if cached_reqs:
                for request_id in cached_reqs.req_ids:
                    request = requests.get(request_id, None)
                    if request:
                        uids += request.sampling_params.extra_args["uid"]
                        candidate_nums += request.sampling_params.extra_args[
                            "candidate_num"]
            if request_stage != RequestStage.Decode.value:
                num_candidates = torch.tensor(candidate_nums,
                                              dtype=torch.int64,
                                              device=self.device)
        else:
            request_stage = RequestStage.Decode.value
            uids = [0]

        seq_lens = attn_metadata.seq_lens
        batch_lens = attn_metadata.query_lens
        batch_size = seq_lens.shape[0]
        block_size = self.vllm_config.cache_config.block_size
        attn_metadata.additional_metadata = {}
        is_pd_merge = request_stage == RequestStage.PdMerged.value

        if is_pd_merge:
            # PdMerge: batch may contain both cached (seq_len > batch_len) and
            # fresh (seq_len == batch_len) requests.  Split based on per-request
            # equality, build separate metadata for each subset, and store the
            # pre-built metadata so forward() can use them directly.
            cached_mask = seq_lens != batch_lens  # CPU bool tensor
            cached_indices = cached_mask.nonzero(as_tuple=True)[0].to(
                torch.int64)
            fresh_indices = (~cached_mask).nonzero(as_tuple=True)[0].to(
                torch.int64)

            attn_metadata.additional_metadata["is_pd_merge"] = True
            attn_metadata.additional_metadata["uids"] = uids
            attn_metadata.additional_metadata["cached_mask"] = cached_mask
            attn_metadata.additional_metadata[
                "cached_indices"] = cached_indices
            attn_metadata.additional_metadata["fresh_indices"] = fresh_indices

            # Pre-build subset metadata for forward() — avoids rebuilding per layer.
            if cached_indices.numel() > 0:
                attn_metadata.additional_metadata["cached_metadata"] = \
                    self._build_decode_metadata(attn_metadata, cached_indices, is_dummy)
                attn_metadata.attn_state = AscendAttentionState.PdMergedCacheHit
            if fresh_indices.numel() > 0:
                attn_metadata.additional_metadata["fresh_metadata"] = \
                    self._build_prefill_metadata(attn_metadata, fresh_indices, num_candidates)
                attn_metadata.attn_state = AscendAttentionState.PdMergedNoCache

            # num_candidates for model-side postprocessing: cached requests
            # get num_candidates = query_lens so postprocess_pd keeps all
            # their tokens (decode-style output is already the right shape).
            modified_num_candidates = num_candidates.clone()
            modified_num_candidates[cached_indices] = batch_lens[
                cached_indices].to(dtype=modified_num_candidates.dtype,
                                   device=modified_num_candidates.device)
            attn_metadata.additional_metadata[
                "num_candidates"] = modified_num_candidates
            attn_metadata.additional_metadata[
                "fresh_num_candidates"] = num_candidates[fresh_indices]
        elif request_stage == RequestStage.Decode.value:
            seq_lens = seq_lens.npu()
            batch_lens = batch_lens.npu()
            block_table = attn_metadata.block_tables
            deal_seq_lens = seq_lens - batch_lens
            num_blocks_per_seq = torch.ceil(deal_seq_lens / block_size).to(
                torch.int64)

            page_offsets = torch.zeros(batch_size + 1,
                                       dtype=torch.int64,
                                       device=block_table.device)
            torch.cumsum(num_blocks_per_seq, dim=0, out=page_offsets[1:])

            last_page_len_tensor = (deal_seq_lens - 1) % block_size + 1

            seq_offset_k = torch.zeros(seq_lens.shape[0] + 1,
                                       dtype=torch.int64,
                                       device=block_table.device)
            torch.cumsum(seq_lens, dim=0, out=seq_offset_k[1:])

            max_seq_len_k = max(
                attn_metadata.seq_lens_list) - 1 if is_dummy else max(
                    attn_metadata.seq_lens_list)

            current_block_tables = block_table[:batch_size]

            page_ids_list = []
            for i in range(batch_size):
                num_blocks_for_req = page_offsets[i + 1] - page_offsets[i]
                valid_blocks = current_block_tables[i, :num_blocks_for_req]
                page_ids_list.append(valid_blocks)

            page_ids = torch.cat(page_ids_list).to(torch.int64)
            attn_metadata.seq_lens = attn_metadata.seq_lens.npu()
            attn_metadata.query_lens = attn_metadata.query_lens.npu()
            attn_metadata.attn_state = AscendAttentionState.DecodeOnly
            attn_metadata.additional_metadata[
                "max_seq_len_k"] = max_seq_len_k if not is_dummy else max_seq_len_k + 1
            attn_metadata.additional_metadata["seq_offset_k"] = seq_offset_k
            attn_metadata.additional_metadata["page_offsets"] = page_offsets
            attn_metadata.additional_metadata["page_ids"] = page_ids
            attn_metadata.additional_metadata[
                "last_page_len"] = last_page_len_tensor
            attn_metadata.additional_metadata["decode_table_offset"] = [
                block_table[i][0].item() for i in range(batch_size)
            ]
            attn_metadata.additional_metadata["uids"] = uids
            attn_metadata.additional_metadata[
                "num_candidates"] = attn_metadata.query_lens
        else:
            attn_metadata.seq_lens = attn_metadata.seq_lens.npu()
            attn_metadata.query_lens = attn_metadata.query_lens.npu()
            attn_metadata.additional_metadata["uids"] = uids
            attn_metadata.attn_state = AscendAttentionState.PrefillNoCache
            attn_metadata.additional_metadata[
                "num_candidates"] = num_candidates
            attn_metadata.additional_metadata["is_pd_merge"] = is_pd_merge
        if requests is not None:
            query_lens = attn_metadata.query_lens
            start_pos = attn_metadata.seq_lens - query_lens
            position_ids_list = []
            for i in range(query_lens.shape[0]):
                start = int(start_pos[i].item())
                length = int(query_lens[i].item())
                position_ids_list.append(
                    torch.arange(start, start + length).npu())
            position_ids = torch.cat(position_ids_list, dim=0).npu()
        else:
            position_ids = None
        attn_metadata.additional_metadata["position_ids"] = position_ids
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        requests: Optional[dict] = None,
        scheduler_output: Optional["SchedulerOutput"] = None,
        is_dummy: bool = False,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]


        if attn_state == AscendAttentionState.DecodeOnly and \
            common_attn_metadata.num_input_tokens > num_actual_tokens:
            padded_num_tokens = common_attn_metadata.num_input_tokens - num_actual_tokens
            seq_lens = torch.cat([
                seq_lens,
                torch.ones(padded_num_tokens,
                           dtype=seq_lens.dtype,
                           device=seq_lens.device)
            ])
            block_table_padding = torch.zeros(
                (padded_num_tokens, ) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding], dim=0)
            query_start_loc_cpu = torch.cat([
                query_start_loc_cpu,
                torch.arange(query_start_loc_cpu[-1] + 1,
                             query_start_loc_cpu[-1] + padded_num_tokens,
                             dtype=query_start_loc_cpu.dtype,
                             device=query_start_loc_cpu.device)
            ])

        query_start_loc = query_start_loc_cpu.to(self.device,
                                                 non_blocking=True)

        attn_metadata = AscendHSTUAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            additional_metadata={})

        attn_metadata = self._enhance_metadata(
            attn_metadata=attn_metadata,
            requests=requests,
            scheduler_output=scheduler_output,
            is_dummy=is_dummy,
            num_reqs=num_reqs)

        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                is_dummy=True,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendHSTUAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        assert attn_type == AttentionType.DECODER
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendHSTUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        output = torch.ops.mxrec.hstu_jagged(
            q=query,
            k=key,
            v=value,
            mask=None,
            attn_bias=None,
            mask_type=0,
            max_seq_len=attn_metadata.max_query_len,
            silu_scale=1.0 / (self.hidden_size * 100),
            seq_offset=attn_metadata.query_start_loc,
            num_context=None,
            num_target=attn_metadata.additional_metadata["num_candidates"],
            target_group_size=1,
        ).view(-1, self.num_heads, self.head_size)
        assert output is not None
        return output[:num_tokens, :, :]

    def _forward_decode_only(
        self,
        query: torch.Tensor,
        attn_metadata: AscendHSTUAttentionMetadata,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block_size = self.key_cache.shape[1]
        output = torch.ops.mxrec.hstu_paged(
            q=query,
            k=key,
            v=value,
            k_cache=self.key_cache.view(self.key_cache.shape[0], block_size,
                                        self.num_kv_heads, self.head_size),
            v_cache=self.value_cache.view(self.value_cache.shape[0],
                                          block_size, self.num_kv_heads,
                                          self.head_size),
            mask=None,
            attn_bias=None,
            mask_type=0,
            max_seq_len=attn_metadata.max_query_len,
            max_seq_len_k=attn_metadata.additional_metadata["max_seq_len_k"],
            target_group_size=1,
            silu_scale=1.0 / (self.hidden_size * 100),
            seq_offset=attn_metadata.query_start_loc,
            seq_offset_k=attn_metadata.additional_metadata["seq_offset_k"],
            seq_offset_t=attn_metadata.query_start_loc,
            page_offsets=attn_metadata.additional_metadata["page_offsets"],
            page_ids=attn_metadata.additional_metadata["page_ids"],
            last_page_len=attn_metadata.additional_metadata["last_page_len"],
            num_target=attn_metadata.query_lens,
            deterministic=True,
        )
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendHSTUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)
        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size).fill_(0)
            num_actual_tokens = attn_metadata.num_actual_tokens
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if len(kv_cache) > 1 and attn_metadata.attn_state != \
               AscendAttentionState.DecodeOnly:
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_actual_tokens],
                    value=value[:num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots[:key[:num_actual_tokens].shape[0]])
            # V0-Style scheduler situation.
            if attn_metadata.attn_state == \
                AscendAttentionState.PrefillNoCache:
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)
            elif attn_metadata.attn_state == \
                AscendAttentionState.DecodeOnly:
                output = self._forward_decode_only(query, attn_metadata, key,
                                                   value, output)
            elif attn_metadata.attn_state == AscendAttentionState.PdMergedNoCache or \
                attn_metadata.attn_state == AscendAttentionState.PdMergedCacheHit:
                cached_indices = attn_metadata.additional_metadata[
                    "cached_indices"]
                fresh_indices = attn_metadata.additional_metadata[
                    "fresh_indices"]
                query_start = attn_metadata.query_start_loc

                # Per-token masks for cached / fresh subsets
                cached_token_mask = torch.zeros(num_tokens,
                                                dtype=torch.bool,
                                                device=query.device)
                fresh_token_mask = torch.zeros(num_tokens,
                                               dtype=torch.bool,
                                               device=query.device)
                for idx in cached_indices.tolist():
                    cached_token_mask[query_start[idx]:query_start[idx +
                                                                   1]] = True
                for idx in fresh_indices.tolist():
                    fresh_token_mask[query_start[idx]:query_start[idx +
                                                                  1]] = True

                # Process cached subset via paged attention (reads KV cache)
                cached_meta = attn_metadata.additional_metadata.get(
                    "cached_metadata")
                if cached_meta is not None:
                    cached_query = query[cached_token_mask]
                    cached_key = key[cached_token_mask]
                    cached_value = value[cached_token_mask]
                    cached_out = self._forward_decode_only(
                        cached_query, cached_meta, cached_key, cached_value)
                    output[cached_token_mask] = cached_out

                # Process fresh subset via jagged attention (no KV cache read)
                fresh_meta = attn_metadata.additional_metadata.get(
                    "fresh_metadata")
                if fresh_meta is not None:
                    fresh_query = query[fresh_token_mask]
                    fresh_key = key[fresh_token_mask]
                    fresh_value = value[fresh_token_mask]
                    fresh_out = self._forward_prefill_no_cache(
                        fresh_query,
                        fresh_key,
                        fresh_value,
                        fresh_meta,
                        num_tokens=fresh_query.shape[0])
                    output[fresh_token_mask] = fresh_out
            else:
                raise NotImplementedError(
                    f"attn_state {attn_metadata.attn_state.name}"
                    "is not implemented for "
                    "AscendHSTUAttentionBackendImpl")
        ori_output[:num_tokens, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)
