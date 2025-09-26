# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List, Type, TypeVar, Dict
import itertools
import numpy as np
import torch
import torch_npu

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    MLAAttentionImpl,
    AttentionType
)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import ModelConfig
from vllm.distributed import get_world_group
from vllm.platforms import current_platform
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

from omni.models.common.config.model_config import model_extra_config
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner
from omni.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder
from omni.accelerators.cache import OmniAttentionSpec, compute_omni_attn_metadata
from omni.adaptors.vllm.utils import get_attr_by_names

def group_request_list(seq_lens, query_lens, block_tables, threshold):
    s_lens_result = []
    q_lens_result = []
    blocks_result = []
    s_lens_current_group = []
    q_lens_current_group = []
    blocks_current_group = []
    current_sum = 0
    for seq_len, query_len, block_table in zip(seq_lens, query_lens, block_tables):
        if current_sum + seq_len > threshold and len(s_lens_current_group) > 0:
            s_lens_result.append(s_lens_current_group)
            q_lens_result.append(q_lens_current_group)
            blocks_result.append(blocks_current_group)
            s_lens_current_group = []
            q_lens_current_group = []
            blocks_current_group = []
            current_sum = 0
        s_lens_current_group.append(seq_len)
        q_lens_current_group.append(query_len)
        blocks_current_group.append(block_table)
        current_sum += seq_len
    if q_lens_current_group:
        s_lens_result.append(s_lens_current_group)
        q_lens_result.append(q_lens_current_group)
        blocks_result.append(blocks_current_group)
    return s_lens_result, q_lens_result, blocks_result

class AscendMLABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "VLLM_ASCEND_MLA"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AscendMLAMetadata

    @staticmethod
    def get_builder_cls():
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, 1, 512 + 64)

    @staticmethod
    def get_impl_cls() -> Type["MLAAttentionImpl"]:
        return AscendMLAImpl

    @staticmethod
    def init_kv_cache_each_layer(kv_cache_shape, dtype, device, model_config: "ModelConfig", enable_graph_mode) -> tuple[torch.Tensor, ...]:
        # KVCache needs to store the shape of the reduced dimension as [num_blocks, block_size, 1, kv_lora_rank] [num_blocks, block_size, 1, rope_dim]
        # The shape of the augmented dimension is [num_blocks, block_size, head_num, head_dim]
        kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
        qk_rope_dim_names = ['attention_qk_rope_dim', 'qk_rope_head_dim']
        kv_lora_dim = get_attr_by_names(model_config.hf_text_config, kv_lora_dim_names, 0)
        qk_rope_dim = get_attr_by_names(model_config.hf_text_config, qk_rope_dim_names, 0)
        layer_kv_cache_nope = torch.zeros(
                        kv_cache_shape[:-2] +
                        (1, kv_lora_dim, ),
                        dtype=dtype if not model_extra_config.operator_opt_config.fa_quant else torch.int8,
                        pin_memory=True,
                        device=device)
        layer_kv_cache_pe = torch.zeros(
                            kv_cache_shape[:-2] +
                            (1, qk_rope_dim, ),
                            dtype=dtype,
                            pin_memory=True,
                            device=device)
        if device != 'cpu':
            # force tensor format to ND
            layer_kv_cache_nope = torch_npu.npu_format_cast(layer_kv_cache_nope, 2)
            layer_kv_cache_pe = torch_npu.npu_format_cast(layer_kv_cache_pe, 2)
        return (layer_kv_cache_nope, layer_kv_cache_pe)

    @staticmethod
    def swap_blocks(
            src_kv_cache: List[torch.Tensor],
            dst_kv_cache: List[torch.Tensor],
            src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

@dataclass
class AscendMLAPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""
    attn_mask: torch.Tensor
    query_lens: list[int]
    seq_lens: list[int]
    input_positions: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int

    # adaptor for chunk-prefill & prefix-caching use
    seq_qlen_group: Optional[list] = None
    seq_kvlen_group: Optional[list] = None
    kv_index_list: Optional[list] = None

    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None

@dataclass
class AscendMLADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    mc2_mask: Optional[torch.Tensor] = None
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    best_topk: Optional[torch.Tensor] = None


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

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Optional[AscendMLADecodeMetadata] = None
    prefill: Optional[AscendMLAPrefillMetadata] = None

    def __post_init__(self):
        pass

    @staticmethod
    def advance_step(metadata, positions, block_size, pad_mask, model_layer):
        block_table = metadata.decode.block_table
        block_indices = block_table.gather(dim=1, index=(positions // block_size).reshape(-1, 1)).view(-1)
        block_offsets = positions % block_size
        metadata.slot_mapping[:] = torch.where(
            pad_mask,
            metadata.slot_mapping,
            block_indices * block_size + block_offsets)
        metadata.decode.seq_lens[:] = (positions + 1).to(metadata.decode.seq_lens.dtype)
        cos, sin = model_layer.self_attn.rotary_emb.get_cos_sin(metadata.decode.input_positions)
        metadata.decode.cos = cos
        metadata.decode.sin = sin

M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMLAMetadataBuilder(DummyAttentionMetadataBuilder):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self,
                 runner: "NPUModelRunner",
                 kv_cache_spec: AttentionSpec = None,
                 block_table: BlockTable = None,
                 metadata_cls: Optional[AscendMLAMetadata] = None):
        self.metadata_cls: Optional[AscendMLAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendMLAMetadata  # type: ignore
        self.runner = runner
        scheduler_config = runner.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.block_size = self.runner.block_size
        self.base_index = np.array(list(range(0, self.block_size)))
        self.base_block = self.block_size * np.ones([1, self.block_size])
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table
        self.decode_gear_list = model_extra_config.operator_opt_config.decode_gear_list
        if self.decode_gear_list:
            self.mc2_mask = torch.zeros(self.decode_gear_list[-1], dtype=torch.bool, device=current_platform.device_type)
        self.already_mark_static = False

    def generate_activate_mask(self, actual_seqs_num, batch_size):
        if len(self.decode_gear_list) > 1:
            gear = next((g for g in self.decode_gear_list if g >= batch_size), self.decode_gear_list[-1])
            self.mc2_mask = torch.zeros(gear, dtype=torch.bool, device=current_platform.device_type)
        else:
            self.mc2_mask.zero_()
        self.mc2_mask[:actual_seqs_num].fill_(True)

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # full attention layers reorder first, then other layers agree
        if isinstance(self.kv_cache_spec, OmniAttentionSpec):
            return False

        # In the case of prefill nodes in PD disaggregation, there is no need to
        # reorder the batch, since it's IMPOSSIBLE to have decode requests.
        # Besides, when APC is on, if a request has exactly one more token
        # than a previous one whose cache is hit, this request will be considered
        # as 'decode' mistakenly.
        kv_transfer_config = self.runner.vllm_config.kv_transfer_config
        if kv_transfer_config is not None and kv_transfer_config.kv_role in ["kv_producer", "kv_both"]:
            self._num_decodes = 0
            self._num_prefills = len(input_batch.req_ids)
            self._num_decode_tokens = 0
            self._num_prefill_tokens = scheduler_output.total_num_scheduled_tokens
            return False

        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            # Only in decode the spec tokens are scheduled
            if req_id in scheduler_output.scheduled_spec_decode_tokens or num_tokens == 1:
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
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def cal_best_topk(self, cur_batch_size):
        world_size = get_world_group().world_size
        batch_size = cur_batch_size * world_size
        top_k = self.runner.model.config.num_experts_per_tok
        step = batch_size // world_size * top_k
        global_rank = get_world_group().rank_in_group
        experts_tp_size = 1
        try:
            num_routed_experts = self.runner.model.config.n_routed_experts
        except:
            num_routed_experts = self.runner.model.config.num_routed_experts
        cur_topk_list = [
            i % num_routed_experts for i in range(
            global_rank // experts_tp_size * step, (global_rank // experts_tp_size + 1) * step)]
        return torch.Tensor(cur_topk_list).to(dtype=torch.int32, device=current_platform.device_type, non_blocking=True).view(batch_size // world_size, -1)

    def _get_graph_runner_block_tables(
            self, num_decode_tokens: int, block_tables: torch.Tensor) -> torch.Tensor:

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        if max_batch_size < num_decode_tokens:
            raise RuntimeError("max_batch_size must be greater than or equal to num_decode_tokens")

        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_batch_size, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype, non_blocking=True)

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_decode_tokens, :
                               num_blocks] = block_tables[:num_decode_tokens, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_decode_tokens, :
                               max_blocks] = block_tables[:num_decode_tokens, :
                                                          max_blocks]

        return graph_block_tables

    def get_kv_index(self, seq_lens, block_tables):
        kv_index = []
        for seq_len, block_table in zip(seq_lens, block_tables):
            index = self.base_index + np.expand_dims(block_table.cpu().numpy(), axis=-1) * self.base_block.repeat(block_table.shape[0], axis=0)
            kv_index.append(index.reshape(-1)[:seq_len])
        return torch.tensor(np.concatenate(kv_index, axis=0), dtype=torch.long, device="cpu").npu()

    def build(self,
              num_reqs: int,
              num_actual_tokens: int,
              max_query_len: int,
              common_prefix_len: Optional[int] = None,
              graph_pad_size: int = -1) -> AscendMLAMetadata:
        if isinstance(self.kv_cache_spec, OmniAttentionSpec):
            return self.build_omni_attn_metadata(num_reqs, num_actual_tokens, graph_pad_size)

        if self._num_decodes + self._num_prefills != num_reqs:
            raise RuntimeError("self._num_decodes + self._num_prefills must be equal to num_reqs")

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.runner.device
        block_table = self.block_table.get_device_tensor()[:num_reqs]

        slot_mapping = self.block_table.slot_mapping_cpu[:num_actual_tokens].to(
            device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True)

        # pad prefill to avoid error of operator's shape assert
        if graph_pad_size > 0:
            padding = torch.full((graph_pad_size, ),
                                    PAD_SLOT_ID,
                                    dtype=slot_mapping.dtype,
                                    device=device)
            padding_0 = torch.zeros(graph_pad_size,
                                    dtype=input_positions.dtype,
                                    device=device)
            slot_mapping = torch.cat([slot_mapping, padding])
            input_positions = torch.cat([input_positions, padding_0])

        prefill_metadata = None
        if self._num_prefills > 0:
            seq_lens_list = self.runner.seq_lens_cpu[:num_reqs].tolist()
            query_lens_list = (self.runner.seq_lens_np[:num_reqs] - self.runner.input_batch.num_computed_tokens_cpu[:num_reqs]).tolist()

            reqs_start = self._num_decodes  # prefill_start
            tokens_start = self._num_decode_tokens

            # Group request for Chunk-Prefill
            seq_kvlen_group, seq_qlen_group, block_groups = group_request_list(
                seq_lens_list,
                query_lens_list,
                block_table,
                self.runner.max_num_tokens)

            # Prepare kv index for prefill get kv_latent from kv_cache
            kv_index_list = []
            if block_table is not None and block_table.numel() > 0:
                for seq_lens, block_tables in zip(seq_kvlen_group, block_groups):
                    kv_index = self.get_kv_index(seq_lens, block_tables)
                    kv_index_list.append(kv_index)

            seq_qlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_qlen_group]
            seq_kvlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_kvlen_group]

            tmp_input_position = input_positions[tokens_start:]
            cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(tmp_input_position)

            prefill_metadata = AscendMLAPrefillMetadata(
                attn_mask=self.runner.attn_mask,
                query_lens=query_lens_list[reqs_start:],
                seq_lens=seq_lens_list,
                input_positions=tmp_input_position,
                block_table=block_table[reqs_start:, ...],
                max_query_len=max_query_len,
                seq_qlen_group=seq_qlen_group,
                seq_kvlen_group=seq_kvlen_group,
                kv_index_list=kv_index_list,
                sin=sin,
                cos=cos
            )

        decode_metadata = None

        if self._num_decodes > 0:
            if self.runner.attn_state == AscendAttentionState.DecodeOnly:
                if self._num_decode_tokens % self._num_decodes != 0:
                    raise RuntimeError("self._num_decode_tokens must be divisible by self._num_decodes")
                num_tokens_per_req = self._num_decode_tokens // self._num_decodes
                seq_lens = (input_positions + 1).to(self.runner.seq_lens.dtype)
                block_table = block_table[:self._num_decodes, ...]
                # has speculative tokens
                if num_tokens_per_req > 1:
                    block_table = block_table.repeat_interleave(num_tokens_per_req, dim=0)
                block_table = torch.cat([block_table,
                                         torch.zeros(
                                            (graph_pad_size, ) + block_table.shape[1:],
                                            dtype=block_table.dtype,
                                            device=block_table.device)],
                                        dim=0)
                block_table = self._get_graph_runner_block_tables(
                    self._num_decode_tokens, block_table)

                self.generate_activate_mask(num_actual_tokens, num_actual_tokens + graph_pad_size)
                cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(input_positions)
                best_topk = None
                if model_extra_config.operator_opt_config.best_ep:
                    best_topk = self.cal_best_topk(num_actual_tokens + graph_pad_size)
            else:
                raise NotImplementedError("Chunked prefill mode is not supported currently.")

            decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=self.mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

    def build_omni_attn_metadata(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        graph_pad_size: int,
    ) -> AscendMLAMetadata:

        decode_metadata = None
        ref: AscendMLAMetadata = self.runner.full_attn_metadata
        num_decodes, num_decode_tokens, num_prefills, prefill_metadata, ref_d = \
            ref.num_decodes, ref.num_decode_tokens, ref.num_prefills, ref.prefill, ref.decode

        if ref_d is not None:
            omni_block_table, omni_slot_mapping, omni_seq_lens = compute_omni_attn_metadata(
                self.kv_cache_spec,
                self.block_table,
                num_actual_tokens,
                num_decodes,
                num_decode_tokens,
                num_prefills,
                self.runner.input_batch.num_prompt_tokens[:num_reqs],
                self.runner.seq_lens_np[:num_reqs] - self.runner.input_batch.num_computed_tokens_cpu[:num_reqs],
                self.runner.seq_lens_np[:num_reqs],
                self.runner.device,
                use_spec_decode=self.runner.use_spec_decode,
            )

            input_positions, cos, sin, best_topk = \
                ref_d.input_positions, ref_d.cos, ref_d.sin, ref_d.best_topk
            num_tokens_per_req = num_decode_tokens // num_decodes

            block_table = torch.zeros_like(ref_d.block_table)
            seq_lens = (input_positions + 1).to(dtype=torch.int64)
            slot_mapping = torch.full_like(ref.slot_mapping, PAD_SLOT_ID)
            slot_mapping[:num_actual_tokens].copy_(omni_slot_mapping, non_blocking=True)

            if num_tokens_per_req > 1:
                omni_block_table = omni_block_table.repeat_interleave(num_tokens_per_req, dim=0)
            m, n = omni_block_table.shape
            block_table[:m, :n].copy_(omni_block_table, non_blocking=True)

            if num_tokens_per_req == 1:
                seq_lens.clamp_(min=0, max=self.kv_cache_spec.max_compressed_len)
            else:
                seq_lens[:num_actual_tokens].copy_(omni_seq_lens, non_blocking=True)

            self.generate_activate_mask(num_actual_tokens, num_actual_tokens + graph_pad_size)
            decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=self.mc2_mask,
                cos=cos.clone(),
                sin=sin.clone(),
                best_topk=best_topk)
        else:
            raise RuntimeError(f"Full attention metadata should not be None!")

        return self.metadata_cls(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

    def build_dummy(self, num_tokens: int, max_pad_size:int = -1) -> AscendMLAMetadata:
        if max_pad_size == -1:
            max_pad_size = self.runner.max_batch_size
        input_positions = torch.zeros(max_pad_size,
                                  dtype=self.runner.positions_cpu.dtype,
                                  device=self.runner.device)
        slot_mapping = torch.zeros(max_pad_size,
                                dtype=self.runner.slot_mapping_cpu.dtype,
                                device=self.runner.device)
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_pad_size, self.runner.graph_block_tables.shape[1]))
        block_table = graph_block_tables.to(
            device=self.runner.device,
            dtype=self.runner.input_batch.block_table[0].get_device_tensor().dtype
        )

        seq_lens = torch.ones(max_pad_size, dtype=torch.long, device=self.runner.device, pin_memory=True) * 2
        cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(input_positions)
        best_topk = None
        self.generate_activate_mask(0, max_pad_size)
        if model_extra_config.operator_opt_config.best_ep:
            best_topk = self.cal_best_topk(max_pad_size)
        decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=self.mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)
        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_tokens,
            slot_mapping=slot_mapping,
            num_decodes=num_tokens,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=None,
            decode=decode_metadata,
        )

    def mark_static_for_attn_metadata(self, attn_metadata):
        if self.already_mark_static:
            return
        if attn_metadata.decode.cos is not None:
            torch._dynamo.mark_static(attn_metadata.decode.cos)
        if attn_metadata.decode.sin is not None:
            torch._dynamo.mark_static(attn_metadata.decode.sin)
        if attn_metadata.decode.mc2_mask is not None:
            torch._dynamo.mark_static(attn_metadata.decode.mc2_mask)
        if attn_metadata.decode.best_topk is not None:
            torch._dynamo.mark_static(attn_metadata.decode.best_topk)
        if attn_metadata.decode.block_table is not None:
            torch._dynamo.mark_static(attn_metadata.decode.block_table)
        if attn_metadata.decode.seq_lens is not None:
            torch._dynamo.mark_static(attn_metadata.decode.seq_lens)
        if attn_metadata.slot_mapping is not None:
            torch._dynamo.mark_static(attn_metadata.slot_mapping)
        self.already_mark_static = True


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
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.blocksparse_params = blocksparse_params
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        # This method should be implemented in the subclass
        raise NotImplementedError("AscendMLAImpl.forward is not implemented.")  

