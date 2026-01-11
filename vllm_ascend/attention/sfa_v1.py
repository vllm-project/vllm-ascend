from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type, TypeVar

import torch
import torch_npu
import vllm.envs as envs_vllm
from torch import nn
from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.mla.common import MLACommonMetadataBuilder
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         trans_rope_weight, transdata,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer, post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, _round_up, dispose_layer,
                               enable_dsa_cp, maybe_trans_nz)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendSFABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

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
    reshape_cache_event: torch.npu.Event = None


M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # _attn_mask_builder = None
    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.metadata_cls = (metadata_cls if metadata_cls is not None else
                             AscendSFAMetadata)
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
        self.enable_sfa_cp = enable_dsa_cp()

        assert not (
            self.enable_sfa_cp
            and self.vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_DECODE_ONLY
        ), "FlashComm1 is not compatible with FULL_DECODE_ONLY. Please set graph_mode to 'piecewise' or disable FlashComm1."
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens

        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:
                                                         num_input_tokens].long(
                                                         )
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        has_prefill = any(query_lens_cpu > self.decode_threshold)

        cum_query_lens = common_attn_metadata.query_start_loc[1:num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        if has_prefill:
            cos, sin = get_cos_and_sin_mla(input_positions)
        else:
            cos, sin = get_cos_and_sin_mla(input_positions, True)

        sfa_cp_context = None
        if self.enable_sfa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_input_tokens
            num_tokens_pad = _round_up(num_tokens, global_tp_size)
            num_tokens_per_device = num_tokens_pad // global_tp_size
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, num_actual_tokens)

            pad_size = num_tokens_pad - cos.shape[0]
            assert cos.shape == sin.shape, \
                f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

            if pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

            pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
            if pad_size_slot > 0:
                slot_mapping = nn.functional.pad(slot_mapping,
                                                 (0, pad_size_slot),
                                                 value=-1)
            else:
                slot_mapping = slot_mapping[:num_tokens_pad]

            cos = cos[local_start:local_end_with_pad]
            sin = sin[local_start:local_end_with_pad]
            slot_mapping_cp = torch.full(size=(num_tokens_per_device, ),
                                         fill_value=-1,
                                         dtype=slot_mapping.dtype,
                                         device=slot_mapping.device)
            assert cos.shape[0] == num_tokens_per_device, \
                f"cos.shape[0] must be equal to num_tokens_per_device, \
                    got {cos.shape[0]} and {num_tokens_per_device}"
            assert slot_mapping_cp.shape[0] == num_tokens_per_device, \
                f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
                    got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
            assert slot_mapping.shape[0] == num_tokens_pad, \
                f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
                    got {slot_mapping.shape[0]} and {num_tokens_pad}"

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
            attn_mask=self.attn_mask_builder.get_attention_mask(
                self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_tables=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            sfa_cp_context=sfa_cp_context)

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
        self.enable_mlapo = envs.VLLM_ASCEND_ENABLE_MLAPO

        assert self.indexer is not None, "Indexer is required for DSA."

        self.enable_sfa_cp = enable_dsa_cp()
        self.local_num_heads = self.num_heads
        self.vllm_config = get_current_vllm_config()
        self.is_kv_producer = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        if self.enable_sfa_cp:
            self.local_num_heads = self.num_heads * self.tp_size
            self.layer_sharding_kwargs = []
            for layer_name in (get_ascend_config().layer_sharding or []):
                if layer_name in kwargs:
                    self.layer_sharding_kwargs.append(kwargs[layer_name])
                else:
                    logger.warning_once(
                        f"Layer '{layer_name}' not found in kwargs for layer sharding, skipping sharding configuration"
                    )
            register_all_layers_to_shard_weight_series(
                self.layer_sharding_kwargs)

        # indexer param
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm

        self.cp_size = 1

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(
            self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
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

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        dispose_layer(self.kv_b_proj)
        if self.enable_sfa_cp:
            for layer in (self.layer_sharding_kwargs or []):
                if is_hidden_layer(layer):
                    post_process_after_loading_for_shard_weight_series(layer)

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
                    "Currently mlapo only supports W8A8 quantization in SFA scenario."
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
        if not self.enable_mlapo:
            # if mlapo, W_UK_T can't trans nz
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

    def _v_up_proj(self, x, has_prefill: bool):
        # TODO(zzzzwwjj): We should not judge by whether `has_prefill` or not.
        # The true criteria for judgment is tensorA's shape[0] <= 1024 (num_tokens <= 1024).
        # This is a bug in the previous code.
        if x.dtype in [torch.float16, torch.bfloat16] \
                and hasattr(torch.ops._C_ascend, "batch_matmul_transpose") \
                and not self.enable_sfa_cp \
                and not has_prefill:
            x = x.view(-1, self.num_heads, self.kv_lora_rank)
            b, _, _ = x.shape
            res = torch.empty((b, self.num_heads, self.v_head_dim),
                              dtype=x.dtype,
                              device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.num_heads * self.v_head_dim)
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
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(
            B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"

        if self.enable_sfa_cp:
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
            return None, None

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

    def _sfa_preprocess_decode(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool,
        num_input_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(), need_gather_q_kv)
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_input_tokens, self.q_lora_rank),
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
            attn_metadata.slot_mapping,
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
                for layer in (self.layer_sharding_kwargs or []):
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        has_prefill = attn_metadata.has_prefill
        cos = attn_metadata.cos
        sin = attn_metadata.sin
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
        if self.enable_sfa_cp:
            need_gather_q_kv = False
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output

        # TODO(zzzzwwjj): In sfa, prefill and decode have the same calculation formula,
        # so `has_prefill` here is not necessary.
        if self.enable_mlapo and not has_prefill:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_decode(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                need_gather_q_kv=need_gather_q_kv,
                num_input_tokens=attn_metadata.num_input_tokens,
            )
            q, k = self.indexer_select_pre_process(
                x=hidden_states,
                qr=q_c,
                cos=cos,
                sin=sin,
                need_gather_q_kv=need_gather_q_kv)
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

            q, k = self.indexer_select_pre_process(
                x=hidden_states,
                qr=q_c,
                cos=cos,
                sin=sin,
                need_gather_q_kv=need_gather_q_kv)

            if has_prefill:
                wait_for_kv_layer_from_connector(layer_name)

            slot_mapping = attn_metadata.slot_mapping
            if self.enable_sfa_cp:
                assert attn_metadata.sfa_cp_context is not None
                slot_mapping = attn_metadata.sfa_cp_context.slot_mapping_cp
                actual_seq_lengths_query = attn_metadata.sfa_cp_context.actual_seq_lengths_query
                actual_seq_lengths_key = attn_metadata.sfa_cp_context.actual_seq_lengths_key

            k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache,
                                        slot_mapping)

            if self.enable_sfa_cp:
                assert k_pe is not None
                assert k_nope is not None
                # support all_gather kv async for communication calculation overlap
                fused_kv_no_split, kv_ag_handle = all_gather_async(
                    torch.cat([
                        k_pe.view(-1, k_pe.shape[-1]),
                        k_nope.view(-1, k_nope.shape[-1]),
                        k.view(-1, k.shape[-1])
                    ],
                              dim=1), get_tp_group())

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)

            if self.enable_sfa_cp:
                if kv_ag_handle is not None:
                    kv_ag_handle.wait()
                for layer in (self.layer_sharding_kwargs or []):
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)

                if kv_cache is not None:
                    assert fused_kv_no_split is not None
                    k_pe, k_nope, k = fused_kv_no_split.split([
                        self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim
                    ],
                                                              dim=-1)
                    slot_mapping = attn_metadata.slot_mapping.view(-1, 1)
                    torch_npu.npu_scatter_nd_update_(
                        kv_cache[0].view(-1, k_nope.shape[-1]), slot_mapping,
                        k_nope)
                    torch_npu.npu_scatter_nd_update_(
                        kv_cache[1].view(-1, k_pe.shape[-1]), slot_mapping,
                        k_pe)

        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            qr=q_c,
            q=q,
            k=k,
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

        attn_output = self._v_up_proj(attn_output, has_prefill)
        maybe_npu_prefetch(inputs=self.o_proj.weight,
                           dependency=attn_output,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=self.enable_prefetch)
        output[...] = self.o_proj(attn_output)[0]
        return output_padded

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        need_gather_q_kv: bool = False,
    ):
        k_proj, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_proj = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            k_proj, need_gather_q_kv)
        k = self.k_norm(k_proj).unsqueeze(1)
        k = k.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
            q = q.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]

            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            q, k = rope_forward_triton(q,
                                       k,
                                       cos,
                                       sin,
                                       rope_dim=self.qk_rope_head_dim,
                                       is_neox_style=True)
        else:
            k_pe, k_nope = torch.split(
                k,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1)  # [b,s,64+64]

            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)

            k = torch.cat([k_pe, k_nope], dim=-1)  # [b*s,128]
            q = None

        return q, k

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        q: Optional[torch.Tensor],
        k: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        need_gather_q_kv: bool = False,
    ):
        if q is None:
            q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
            q = q.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]

            cos_q, sin_q = cos, sin
            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            q_pe, q_nope = torch.split(
                q,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1)  # [b,s,64,64+64]

            q_pe = q_pe.unsqueeze(2)
            q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q)
            q_pe = q_pe.squeeze(2)
            q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]

        if kv_cache is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            torch_npu.npu_scatter_nd_update_(kv_cache[2].view(-1, k.shape[-1]),
                                             attn_metadata.slot_mapping.view(
                                                 -1, 1),
                                             k.view(-1,
                                                    k.shape[-1]))  # b, s, n, d
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()

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
