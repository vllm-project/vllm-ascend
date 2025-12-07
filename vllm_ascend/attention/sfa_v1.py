from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type, TypeVar

import torch
import torch_npu
from torch import nn
from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import (LinearBase, ReplicatedLinear,
                                               UnquantizedLinearMethod)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.utils import AttentionCGSupport

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         trans_rope_weight, transdata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.ops.shared_weight_layer import (
    is_hidden_layer, post_process_after_loading_for_shared_weight_series,
    reach_layer_for_shared_weight_series,
    register_layer_to_shared_weight_series)
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               _round_up, dispose_layer, enable_sp,
                               is_enable_nz, replace_layer)
from vllm_ascend.worker.npu_input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendSFABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA"

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
class SfaCpContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    pad_size: int
    local_pad_size: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


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
    has_prefill: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    sfa_cp_context: Optional[SfaCpContext] = None


M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
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
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.enable_sfa_cp = enable_sp() and \
            hasattr(self.model_config.hf_config, "index_topk")

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        device = self.device

        block_table = (common_attn_metadata.block_table_tensor[:num_reqs])
        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens].to(
                                                             device,
                                                             non_blocking=True)
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens].long(
                                                         )
        query_start_loc = common_attn_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        has_prefill = any(query_lens > self.decode_threshold)

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

        cum_query_lens = query_start_loc_cpu[1:num_reqs + 1].to(
            torch.int32).to(device, non_blocking=True)
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs].to(
            torch.int32).to(device, non_blocking=True)

        cos = self.cos_cache[input_positions].unsqueeze(  # type: ignore
            1).unsqueeze(2)
        sin = self.sin_cache[input_positions].unsqueeze(  # type: ignore
            1).unsqueeze(2)

        sfa_cp_context = None
        if self.enable_sfa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_actual_tokens
            num_tokens_pad = _round_up(num_actual_tokens, global_tp_size)
            num_tokens_per_device = num_tokens_pad // global_tp_size
            pad_size = num_tokens_pad - num_tokens
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, num_actual_tokens)
            local_pad_size = local_end_with_pad - local_end

            if pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))
                slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size),
                                                 value=-1)
            cos = cos[local_start:local_end_with_pad]
            sin = sin[local_start:local_end_with_pad]
            slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

            actual_seq_lengths_query = torch.empty_like(cum_query_lens)
            actual_seq_lengths_key = torch.empty_like(seq_lens)
            num_segs = cum_query_lens.shape[0]
            last_token = 0
            cum = 0
            for i in range(0, num_segs):
                global_start = last_token
                global_end = cum_query_lens[i].item()
                last_token = global_end

                local_start = max(global_start, local_start)
                local_end = min(global_end, local_end_with_pad)
                num_local_tokens = local_end - local_start

                if num_local_tokens > 0:
                    cum += num_local_tokens
                    actual_seq_lengths_query[i] = cum

                    offset = global_end - local_end
                    actual_seq_lengths_key[i] = seq_lens[i].item() - offset
                else:
                    actual_seq_lengths_query[i] = cum
                    actual_seq_lengths_key[i] = 0

            sfa_cp_context = SfaCpContext(
                num_tokens=num_tokens,
                num_tokens_pad=num_tokens_pad,
                local_start=local_start,
                local_end=local_end,
                local_end_with_pad=local_end_with_pad,
                pad_size=pad_size,
                local_pad_size=local_pad_size,
                slot_mapping_cp=slot_mapping_cp,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )

        return self.metadata_cls(  # type: ignore
            has_prefill=has_prefill,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=common_attn_metadata.attn_mask,
            attn_state=common_attn_metadata.attn_state,
            block_tables=block_table,
            sin=sin,
            cos=cos,
            sfa_cp_context=sfa_cp_context)

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
                model=model,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


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
        self.q_proj = kwargs['q_proj'] if self.q_lora_rank is None else kwargs[
            'q_b_proj']
        self.fused_qkv_a_proj = kwargs.get('fused_qkv_a_proj', None)
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.indexer = kwargs['indexer']
        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        self.q_a_layernorm = kwargs.get('q_a_layernorm', None)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.q_b_proj = kwargs['q_b_proj']

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_prefetch = ascend_config.weight_prefetch_config.enabled
        self.enable_kv_nz = ascend_config.torchair_graph_config.enable_kv_nz
        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        assert self.indexer is not None, "Indexer is required for DSA."

        self.enable_sfa_cp = enable_sp()
        self.local_num_heads = self.num_heads
        self.vllm_config = get_current_vllm_config()
        if self.enable_sfa_cp:
            self.local_num_heads = self.num_heads * self.tp_size

            #TODO: Temporarily adapt sfa-cp, remove after adapting near PCP. --clrs97
            self._replace_linear_class_for_sfa_cp()
            from vllm_ascend.distributed.parallel_state import \
                get_shared_weight_group
            if is_hidden_layer(self.vllm_config, self.q_proj):
                register_layer_to_shared_weight_series(
                    series_name="q_proj",
                    group=get_shared_weight_group(),
                    layer=self.q_proj,
                    prefetch_step=1)
            if is_hidden_layer(self.vllm_config, self.o_proj):
                register_layer_to_shared_weight_series(
                    series_name="o_proj",
                    group=get_shared_weight_group(),
                    layer=self.o_proj,
                    prefetch_step=1)

        # indexer param
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm

        self.cp_size = 1

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                try:
                    return getattr(layer, attr)
                except AttributeError:
                    pass
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
            self.kv_lora_rank, self.local_num_heads *
            (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.local_num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
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
        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        dispose_layer(self.kv_b_proj)

        if self.enable_sfa_cp:
            if is_hidden_layer(self.vllm_config, self.q_proj):
                post_process_after_loading_for_shared_weight_series(
                    self.q_proj)
            if is_hidden_layer(self.vllm_config, self.o_proj):
                post_process_after_loading_for_shared_weight_series(
                    self.o_proj)

        if self.enable_mlapo:
            quant_method = getattr(
                getattr(self.fused_qkv_a_proj, "quant_method", None),
                "quant_method",
                None,
            )
            reasons = []
            if self.fused_qkv_a_proj is None or not isinstance(
                    quant_method, AscendW8A8LinearMethod):
                reasons.append(
                    "Currently mlapo only supports W8A8 quantization in MLA scenario."
                    "Some layers in your model are not quantized with W8A8,"
                    "thus mlapo is disabled for these layers.")
            if self.enable_sfa_cp:
                reasons.append("Currently mlapo does not support SFA with CP,"
                               "thus mlapo is disabled for these layers.")
            if reasons:
                self.enable_mlapo = False
                for msg in reasons:
                    logger.warning_once(msg)
            else:
                self._process_weights_for_fused_mlapo(act_dtype)

    def _v_up_proj(self, x):
        if self.W_UV.shape[0] * self.W_UV.shape[1] < 65536:
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            x = torch_npu.npu_transpose_batchmatmul(x,
                                                    self.W_UV,
                                                    perm_x1=[1, 0, 2],
                                                    perm_x2=[0, 1, 2],
                                                    perm_y=[1, 0, 2])
            x = x.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads,
                       self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0,
                            1).reshape(-1,
                                       self.local_num_heads * self.v_head_dim)
        return x

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = self.q_proj(x)[0]\
            .view(-1, self.local_num_heads, self.qk_head_dim)\
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
        slots_cp: Optional[torch.Tensor],
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"

        if self.enable_sfa_cp:
            assert slots_cp is not None
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                slots_cp.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            #TODO: Temporarily adapt SFA-CP and replace it later with PCP. --clrs97
            k_pe = get_tp_group().all_gather(k_pe, 0)
            k_nope = get_tp_group().all_gather(k_nope, 0)

            if kv_cache is not None:
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[0].view(-1, k_nope.shape[-1]), slots.view(-1, 1),
                    k_nope.view(-1, k_nope.shape[-1]))
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[1].view(-1, k_pe.shape[-1]), slots.view(-1, 1),
                    k_pe.view(-1, k_pe.shape[-1]))
        else:
            torch_npu.npu_kv_rmsnorm_rope_cache(
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

    # Processing the input parameters for MLAPO by reordering and transposing
    # QKV(and part of Q) weight, applying RoPE-related dimension transformations,
    # and handling quantization parameters.
    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        assert self.kv_a_proj_with_mqa is None
        assert self.fused_qkv_a_proj is not None

        kv_a_proj_wt = self.fused_qkv_a_proj.weight.data[
            ..., self.q_lora_rank:].contiguous()
        q_a_proj_wt = self.fused_qkv_a_proj.weight.data[
            ..., :self.q_lora_rank].contiguous()

        self.fused_qkv_a_proj.weight = None

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

        if self.vllm_config.kv_transfer_config is not None:
            self.fused_qkv_a_proj.deq_scale = None
            self.fused_qkv_a_proj.quant_bias = None
            self.q_proj.deq_scale = None
            self.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def _sfa_preprocessc_decode(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool,
        num_actual_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(), need_gather_q_kv)
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_actual_tokens, self.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_actual_tokens, self.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_actual_tokens, self.q_lora_rank),
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
            attn_metadata.cos,
            attn_metadata.sin,
            self.W_UK_T,
            k_nope,
            k_pe,
            attn_metadata.slot_mapping[:num_actual_tokens].flatten(),
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
            enable_inner_out=True,
            q_out0=ql_nope,
            kv_cache_out0=k_nope,
            q_out1=q_pe,
            kv_cache_out1=k_pe,
            inner_out=q_c,
        )
        return hidden_states, ql_nope, q_pe, q_c

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        forward_context = get_forward_context()
        if attn_metadata is None:
            # Profiling run.
            if self.enable_sfa_cp and not forward_context.in_profile_run:
                if is_hidden_layer(self.vllm_config, self.q_proj):
                    reach_layer_for_shared_weight_series(self.q_proj)
                if is_hidden_layer(self.vllm_config, self.o_proj):
                    reach_layer_for_shared_weight_series(self.o_proj)
            return output.fill_(0)
        has_prefill = attn_metadata.has_prefill
        num_actual_tokens = attn_metadata.num_actual_tokens
        cos = attn_metadata.cos
        sin = attn_metadata.sin
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
        hidden_states = hidden_states[:num_actual_tokens]
        if self.enable_sfa_cp:
            need_gather_q_kv = False
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_tokens]

        if self.enable_mlapo and not forward_context.with_prefill:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocessc_decode(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                need_gather_q_kv=need_gather_q_kv,
                num_actual_tokens=num_actual_tokens,
            )
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            maybe_npu_prefetch(inputs=self.fused_qkv_a_proj.weight,
                               dependency=hidden_states,
                               enabled=self.enable_prefetch)
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            # Process for Flash Comm V1
            if need_gather_q_kv:
                q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    q_c.contiguous(), need_gather_q_kv)
                kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    kv_no_split.contiguous(), need_gather_q_kv)

            if has_prefill:
                wait_for_kv_layer_from_connector(layer_name)

            slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
            slot_mapping_cp = None
            if self.enable_sfa_cp:
                assert attn_metadata.sfa_cp_context is not None
                slot_mapping_cp = attn_metadata.sfa_cp_context.slot_mapping_cp
                actual_seq_lengths_query = attn_metadata.sfa_cp_context.actual_seq_lengths_query
                actual_seq_lengths_key = attn_metadata.sfa_cp_context.actual_seq_lengths_key

            self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping,
                         slot_mapping_cp)

            if self.enable_sfa_cp and attn_metadata.sfa_cp_context is not None:
                if is_hidden_layer(self.vllm_config, self.q_proj):
                    reach_layer_for_shared_weight_series(self.q_proj)
                if is_hidden_layer(self.vllm_config, self.o_proj):
                    reach_layer_for_shared_weight_series(self.o_proj)

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)

        topk_indices = self.indexer_select(
            x=hidden_states,
            qr=q_c,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            need_gather_q_kv=need_gather_q_kv)
        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv_cache[0],
            value=kv_cache[0],
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=kv_cache[1],
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        attn_output = self._v_up_proj(attn_output)
        maybe_npu_prefetch(inputs=self.o_proj.weight,
                           dependency=attn_output,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=self.enable_prefetch)
        output[...] = self.o_proj(attn_output)[0]
        return output_padded

    def indexer_select(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        need_gather_q_kv: bool = False,
    ):
        # q process in new stream
        q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q = q.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]

        k_proj, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_proj = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            k_proj, need_gather_q_kv)
        k = self.k_norm(k_proj).unsqueeze(1)
        k = k.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            q, k = rope_forward_triton(q,
                                       k,
                                       cos,
                                       sin,
                                       rope_dim=self.qk_rope_head_dim,
                                       is_neox_style=True)
        else:
            cos_q, sin_q = cos, sin
            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            q_pe, q_nope = torch.split(
                q,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1)  # [b,s,64,64+64]

            q_pe = q_pe.unsqueeze(2)
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
            q_pe = q_pe.squeeze(2)
            q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]

            k_pe, k_nope = torch.split(
                k,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1)  # [b,s,64+64]

            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)

            k = torch.cat([k_pe, k_nope], dim=-1)  # [b*s,128]

        if self.enable_sfa_cp:
            k = get_tp_group().all_gather(k, 0)

        if kv_cache is not None:
            torch_npu.npu_scatter_nd_update_(kv_cache[2].view(-1, k.shape[-1]),
                                             attn_metadata.slot_mapping.view(
                                                 -1, 1),
                                             k.view(-1,
                                                    k.shape[-1]))  # b, s, n, d

        weights, _ = self.weights_proj(x)
        weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            weights, need_gather_q_kv)

        block_table = attn_metadata.block_tables

        topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
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

    def _replace_linear_class_for_sfa_cp(self):

        vllm_config = get_current_vllm_config()
        # Dispose tensor from the original q_proj
        dispose_layer(self.q_proj)
        # Construct the new q_proj using ReplicatedLinear
        new_q_proj = ReplicatedLinear(self.q_lora_rank,
                                      self.local_num_heads * self.qk_head_dim,
                                      bias=False,
                                      quant_config=vllm_config.quant_config,
                                      prefix=self.q_proj.prefix)
        # Replace the q_proj with the new one
        replace_layer(self.q_proj, new_q_proj)

        # Dispose tensor from the original kv_b_proj
        dispose_layer(self.kv_b_proj)
        # Construct the new kv_b_proj using ReplicatedLinear
        new_kv_b_proj = ReplicatedLinear(
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=self.kv_b_proj.prefix)
        # Replace the kv_b_proj with the new one
        replace_layer(self.kv_b_proj, new_kv_b_proj)

        # Dispose tensor from the original o_proj
        dispose_layer(self.o_proj)
        # Construct the new o_proj using ReplicatedLinear
        config = vllm_config.model_config.hf_config
        new_o_proj = ReplicatedLinear(config.num_attention_heads *
                                      config.v_head_dim,
                                      config.hidden_size,
                                      bias=False,
                                      quant_config=vllm_config.quant_config,
                                      prefix=self.o_proj.prefix)
        # Replace the o_proj with the new one
        replace_layer(self.o_proj, new_o_proj)
