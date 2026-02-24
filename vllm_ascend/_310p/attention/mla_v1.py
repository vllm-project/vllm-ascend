#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MLAAttentionImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import ascend_chunked_prefill_workspace_size, split_decodes_and_prefills
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1


@dataclass
class AscendMLAMetadata310:
    """Metadata for MLA on 310P."""

    num_actual_tokens_pcp_padded: int
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    num_input_tokens: int = 0
    query_lens: list[int] | None = None
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Any = None
    prefill: Any = None


class AscendMLAMetadataBuilder310(MLACommonMetadataBuilder[AscendMLAMetadata310]):
    """Metadata builder for MLA on 310P."""

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendMLAMetadata310] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendMLAMetadata310,
            supports_dcp_with_varlen,
        )

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            self.decode_threshold += self.speculative_config.num_speculative_tokens

        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder310(self.device)

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendMLAMetadataBuilder310"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: Any, scheduler_output: SchedulerOutput) -> bool:
        return False

    def set_num_actual_tokens(self, common_attn_metadata: Any):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> AscendMLAMetadata310:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
        )
        self.set_num_actual_tokens(common_attn_metadata)

        self.slot_mapping = common_attn_metadata.slot_mapping[: self.num_actual_tokens]
        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]
        self.seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:block_table_size]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(common_attn_metadata)

        decode_metadata = None
        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(common_attn_metadata)

        return self.metadata_cls(
            num_actual_tokens_pcp_padded=self.num_actual_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens.tolist(),
            slot_mapping=self.slot_mapping,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=self.attn_mask_builder._get_causal_mask(self.model_config.max_model_len),
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=self.block_table,
            seq_lens=self.seq_lens,
        )

    def get_block_table_size(self, common_attn_metadata: Any, build_metadata_step: int):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            if (
                self.graph_pad_size > common_attn_metadata.num_reqs
                and self.speculative_config is not None
                and getattr(self.speculative_config, "disable_padded_drafter_batch", False)
            ):
                return self.graph_pad_size
            return common_attn_metadata.num_reqs
        return self.num_decodes

    def build_prefill_metadata(self, common_attn_metadata: Any) -> Any:
        query_start_loc = common_attn_metadata.query_start_loc
        input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()

        reqs_start = self.num_decodes
        tokens_start = self.num_decode_tokens
        max_query_len = self.query_lens[reqs_start:].max().item()
        max_seq_lens = self.seq_lens[reqs_start:].max().item()
        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]

        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_mla(prefill_input_positions)

        return {
            "attn_mask": self.attn_mask_builder._get_causal_mask(self.model_config.max_model_len),
            "query_lens": self.query_lens[reqs_start:].to(torch.int32),
            "seq_lens": self.seq_lens,
            "context_lens": self.seq_lens[reqs_start:],
            "input_positions": prefill_input_positions,
            "block_table": self.block_table[reqs_start:, ...],
            "max_query_len": max_query_len,
            "max_seq_lens": max_seq_lens,
            "query_start_loc": prefill_query_start_loc,
            "sin": sin,
            "cos": cos,
        }

    def build_decode_metadata(self, common_attn_metadata: Any) -> Any:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()

        actual_seq_lengths_q = query_start_loc_cpu[1 : self.num_decodes + 1].tolist()
        max_seq_lens = self.seq_lens[: self.num_decodes].max().item()
        self.seq_lens = self.seq_lens[: self.num_decodes]
        input_positions = input_positions[: self.num_decode_tokens]

        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_DECODE)
        self.block_table = self.block_table[:block_table_size]
        seq_lens_list = self.seq_lens.tolist()

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=True)

        return {
            "input_positions": input_positions,
            "block_table": self.block_table,
            "seq_lens": self.seq_lens,
            "seq_lens_list": seq_lens_list,
            "max_seq_lens": max_seq_lens,
            "attn_mask": None,
            "actual_seq_lengths_q": actual_seq_lengths_q,
            "sin": sin[: self.num_decode_tokens, ...],
            "cos": cos[: self.num_decode_tokens, ...],
        }


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendMLABackend310(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA_310" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["MLAAttentionImpl"]:
        return AscendMLAImpl310

    @staticmethod
    def get_builder_cls():
        return AscendMLAMetadataBuilder310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16]


class AscendMLAImpl310(MLAAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        del alibi_slopes, sliding_window, logits_soft_cap, attn_type, kv_sharing_target_layer_name
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = kwargs.get("q_lora_rank")
        self.kv_lora_rank = kwargs.get("kv_lora_rank")
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim")
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim")
        self.qk_head_dim = kwargs.get("qk_head_dim")
        self.v_head_dim = kwargs.get("v_head_dim", head_size)

        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.q_proj = kwargs.get("q_b_proj") if self.q_lora_rank is not None else kwargs.get("q_proj")
        self.kv_b_proj = kwargs.get("kv_b_proj")
        self.o_proj = kwargs.get("o_proj")
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")

        ascend_config = get_ascend_config()
        self.enable_kv_nz = ascend_config.enable_kv_nz

        self.W_UV = None
        self.W_UK_T = None

        backend_env = os.getenv("VLLM_ASCEND_310P_MLA_ATTN_BACKEND", "AUTO").upper()
        if backend_env not in {"AUTO", "FALLBACK", "NPU"}:
            logger.warning("Invalid VLLM_ASCEND_310P_MLA_ATTN_BACKEND=%s, fallback to AUTO.", backend_env)
            backend_env = "AUTO"
        self._mla_attn_backend = "FALLBACK" if backend_env == "AUTO" else backend_env
        logger.info("[310P MLA] attention backend requested=%s resolved=%s", backend_env, self._mla_attn_backend)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        del act_dtype
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if self.kv_b_proj is None:
            return
        if not isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod):
            logger.warning_once(
                "310P MLA currently only supports unquantized kv_b_proj. V up-projection may not work correctly.",
            )
            return

        kv_b_proj_weight = self.kv_b_proj.weight.data.T
        expected_shape = (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        if kv_b_proj_weight.shape != expected_shape:
            logger.warning_once("kv_b_proj weight shape mismatch: %s vs %s", kv_b_proj_weight.shape, expected_shape)
            return

        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UV = W_UV.transpose(0, 1).contiguous()
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        if self.W_UV is None:
            raise RuntimeError("W_UV not initialized. Call process_weights_after_loading first.")
        x = x.view(self.num_heads, -1, self.kv_lora_rank)
        x = torch.bmm(x.contiguous(), self.W_UV.contiguous()).permute(1, 0, 2)
        return x.reshape(-1, self.num_heads * self.v_head_dim)

    def _q_proj_and_k_up_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )
        q_nope = q_nope.transpose(0, 1)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        return ql_nope.transpose(0, 1), q_pe

    def get_num_actual_tokens(self, attn_metadata: AscendMLAMetadata310) -> int:
        return attn_metadata.num_actual_tokens

    def forward_mha(self, *args, **kwargs):
        raise NotImplementedError("forward_mha is not supported for MLA attention. Use forward() instead.")

    def forward_mqa(self, *args, **kwargs):
        raise NotImplementedError("forward_mqa is not supported for MLA attention. Use forward() instead.")

    def forward(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata310,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer_name, need_gather_q_kv
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        forward_context = get_forward_context()
        num_actual_tokens = self.get_num_actual_tokens(attn_metadata)

        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        output_padded = output
        o_proj_input_shape = (forward_context.num_tokens, self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        decode_preprocess_res = None
        prefill_preprocess_res = None
        if attn_metadata.num_decodes > 0:
            decode_preprocess_res = self._mla_preprocess_decode(
                hidden_states[:num_decode_tokens],
                kv_cache,
                attn_metadata,
            )

        if has_prefill:
            prefill_preprocess_res = self._mla_preprocess_prefill(
                hidden_states[num_decode_tokens:num_actual_tokens],
                kv_cache,
                attn_metadata,
                num_decode_tokens,
            )

        if decode_preprocess_res is not None:
            output_decode = self._forward_decode(
                decode_preprocess_res["ql_nope"],
                decode_preprocess_res["q_pe"],
                kv_cache,
                attn_metadata,
            )
            o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            output_prefill = self._forward_prefill(
                prefill_preprocess_res["q_nope"],
                prefill_preprocess_res["k_nope"],
                prefill_preprocess_res["v"],
                attn_metadata,
            )
            o_proj_input[num_decode_tokens:num_actual_tokens] = output_prefill

        output[...] = self.o_proj(o_proj_input)[0]
        del o_proj_input
        return output_padded

    def _mla_preprocess_decode(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata310,
    ) -> dict:
        bsz = hidden_states.shape[0]
        decode_meta = attn_metadata.decode
        cos = decode_meta["cos"].view(decode_meta["cos"].shape[0], decode_meta["cos"].shape[-1])
        sin = decode_meta["sin"].view(decode_meta["sin"].shape[0], decode_meta["sin"].shape[-1])

        if self.q_lora_rank is not None and self.q_a_layernorm is not None and self.fused_qkv_a_proj is not None:
            q_c = self.fused_qkv_a_proj(hidden_states)[0][:, : self.q_lora_rank]
            q_c = self.q_a_layernorm(q_c)
        else:
            q_c = hidden_states

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self._rope_single(q_pe, cos, sin)

        if self.kv_lora_rank is not None and self.kv_a_layernorm is not None and self.fused_qkv_a_proj is not None:
            kv_compressed = self.fused_qkv_a_proj(hidden_states)[0][:, self.q_lora_rank :]
        else:
            kv_compressed = self.kv_a_proj_with_mqa(hidden_states)[0]

        slots = attn_metadata.slot_mapping[:bsz]
        kv_c = kv_compressed[:, : self.kv_lora_rank]
        k_pe = kv_compressed[:, self.kv_lora_rank :]
        k_pe = k_pe.unsqueeze(1)
        if self.kv_a_layernorm is not None:
            kv_c = self.kv_a_layernorm(kv_c)

        k_pe = self._rope_single(k_pe, cos, sin)

        self._store_to_cache_310(kv_cache, kv_c, k_pe, slots)

        return {
            "ql_nope": ql_nope,
            "q_pe": q_pe,
        }

    def _mla_preprocess_prefill(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata310,
        num_decode_tokens: int,
    ) -> dict:
        num_prefill_tokens = hidden_states.shape[0]
        prefill_meta = attn_metadata.prefill
        cos = prefill_meta["cos"]
        sin = prefill_meta["sin"]

        if self.q_lora_rank is not None and self.q_a_layernorm is not None and self.fused_qkv_a_proj is not None:
            q_c = self.fused_qkv_a_proj(hidden_states)[0][:, : self.q_lora_rank]
            q_c = self.q_a_layernorm(q_c)
        else:
            q_c = hidden_states

        q = self.q_proj(q_c)[0].view(num_prefill_tokens, self.num_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]
        q_pe = self._rope_single(q_pe, cos, sin)

        if self.kv_lora_rank is not None and self.kv_a_layernorm is not None and self.fused_qkv_a_proj is not None:
            kv_compressed = self.fused_qkv_a_proj(hidden_states)[0][:, self.q_lora_rank :]
        else:
            kv_compressed = self.kv_a_proj_with_mqa(hidden_states)[0]

        kv_c = kv_compressed[:, : self.kv_lora_rank]
        k_pe = kv_compressed[:, self.kv_lora_rank :]
        k_pe = k_pe.unsqueeze(1)
        if self.kv_a_layernorm is not None:
            kv_c = self.kv_a_layernorm(kv_c)

        kv_nope = self.kv_b_proj(kv_c)[0].view(num_prefill_tokens, self.num_kv_heads, -1)
        k_nope = kv_nope[..., : self.qk_nope_head_dim]
        v = kv_nope[..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim]
        k_pe = self._rope_single(k_pe, cos, sin)

        slots = attn_metadata.slot_mapping[num_decode_tokens : num_decode_tokens + num_prefill_tokens]
        self._store_to_cache_310(kv_cache, kv_c, k_pe, slots)

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "k_nope": k_nope,
            "k_pe": k_pe,
            "v": v,
        }

    def _store_to_cache_310(
        self,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        slots: torch.Tensor,
    ):
        kv_c_cache = kv_cache[0]
        k_pe_cache = kv_cache[1]
        block_size = kv_c_cache.shape[1]

        kv_c_expanded = kv_c.unsqueeze(1).expand(-1, kv_c_cache.shape[2], -1)
        k_pe_expanded = k_pe.expand(-1, k_pe_cache.shape[2], -1)

        for i, slot in enumerate(slots):
            block_idx = slot.item() // block_size
            block_offset = slot.item() % block_size
            if block_idx < kv_c_cache.shape[0]:
                kv_c_cache[block_idx, block_offset, :, :] = kv_c_expanded[i].to(kv_c_cache.dtype)
                k_pe_cache[block_idx, block_offset, :, :] = k_pe_expanded[i].to(k_pe_cache.dtype)

    def _rope_single(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if cos.dim() == 4:
            cos = cos.squeeze(1).squeeze(1).view(B, 1, D)
            sin = sin.squeeze(1).squeeze(1).view(B, 1, D)
        elif cos.dim() == 2:
            cos = cos.view(B, 1, D)
            sin = sin.view(B, 1, D)
        else:
            cos = cos.view(B, 1, D)
            sin = sin.view(B, 1, D)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AscendMLAMetadata310,
    ) -> torch.Tensor:
        prefill_meta = attn_metadata.prefill
        num_tokens = q_nope.shape[0]

        # seq_lens in metadata includes both decode and prefill requests.
        # Slice the prefill portion to avoid mixing decode lengths into prefill.
        prefill_start = attn_metadata.num_decodes
        prefill_end = prefill_start + attn_metadata.num_prefills
        seq_len = prefill_meta["seq_lens"][prefill_start:prefill_end]
        real_tokens = int(seq_len.sum().item())
        aligned_tokens = int(q_nope.shape[0])
        delta = aligned_tokens - real_tokens
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = prefill_meta["attn_mask"]
        output = torch.empty(num_tokens, self.num_heads, self.v_head_dim, dtype=q_nope.dtype, device=q_nope.device)

        if self._mla_attn_backend == "NPU":
            torch_npu._npu_flash_attention(
                query=q_nope,
                key=k_nope,
                value=v,
                mask=mask,
                seq_len=seq_len,
                scale_value=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                out=output,
            )
        else:
            output.copy_(self._forward_prefill_fallback(q_nope, k_nope, v, seq_len, mask))

        return output.reshape(num_tokens, self.num_heads * self.v_head_dim)

    def _forward_decode(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AscendMLAMetadata310,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        num_tokens = ql_nope.shape[0]
        block_size = self.vllm_config.cache_config.block_size

        seq_lens = decode_meta["seq_lens"]
        if seq_lens.device != ql_nope.device:
            seq_lens = seq_lens.to(device=ql_nope.device, non_blocking=True)

        k_nope = kv_cache[0].view(-1, self.num_kv_heads, block_size, self.kv_lora_rank)
        k_pe = kv_cache[1].view(-1, self.num_kv_heads, block_size, self.qk_rope_head_dim)

        q_nope_input = ql_nope.view(num_tokens, self.num_heads, 1, -1).contiguous()
        q_pe_input = q_pe.view(num_tokens, self.num_heads, 1, -1)

        if self._mla_attn_backend == "NPU":
            # Only pass actual_seq_lengths_q when needed by special decode paths.
            actual_seq_lengths = None
            if attn_metadata.attn_state in {
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.ChunkedPrefill,
            }:
                actual_seq_lengths = decode_meta["actual_seq_lengths_q"]
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                q_nope_input,
                k_nope,
                k_nope,
                query_rope=q_pe_input,
                key_rope=k_pe,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD_NBSD",
                atten_mask=None,
                sparse_mode=0,
                scale=self.scale,
                block_table=decode_meta["block_table"],
                block_size=block_size,
                actual_seq_lengths_kv=decode_meta["seq_lens_list"],
                actual_seq_lengths=actual_seq_lengths,
            )
        else:
            attn_output = self._forward_decode_fallback(
                ql_nope=ql_nope,
                q_pe=q_pe,
                k_nope_cache=k_nope,
                k_pe_cache=k_pe,
                decode_meta=decode_meta,
                block_size=block_size,
            )

        return self._v_up_proj(attn_output)

    def _expand_kv_to_q_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.num_heads:
            return x
        if x.shape[1] != self.num_kv_heads or self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError(
                f"Invalid KV head configuration for expansion: x_heads={x.shape[1]}, "
                f"num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}",
            )
        group = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(group, dim=1)

    def _forward_prefill_fallback(
        self,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        v: torch.Tensor,
        seq_len: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del mask
        out = torch.zeros(
            q_nope.shape[0],
            self.num_heads,
            self.v_head_dim,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        k_nope_h = self._expand_kv_to_q_heads(k_nope)
        v_h = self._expand_kv_to_q_heads(v)

        offset = 0
        for seq_len_i in seq_len.tolist():
            cur_len = int(seq_len_i)
            if cur_len <= 0:
                continue
            q_i = q_nope[offset : offset + cur_len]
            k_i = k_nope_h[offset : offset + cur_len]
            v_i = v_h[offset : offset + cur_len]

            scores = torch.einsum("thd,shd->ths", q_i, k_i) * self.scale
            causal = torch.triu(
                torch.ones(cur_len, cur_len, dtype=torch.bool, device=q_nope.device),
                diagonal=1,
            )
            scores.masked_fill_(causal.unsqueeze(1), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[offset : offset + cur_len] = torch.einsum("ths,shd->thd", probs, v_i)
            offset += cur_len
        return out

    def _forward_decode_fallback(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        decode_meta: Any,
        block_size: int,
    ) -> torch.Tensor:
        block_table = decode_meta["block_table"]
        seq_lens_list = decode_meta["seq_lens_list"]
        num_tokens = ql_nope.shape[0]

        latent_output = torch.zeros(
            self.num_heads,
            num_tokens,
            self.kv_lora_rank,
            dtype=ql_nope.dtype,
            device=ql_nope.device,
        )

        for i in range(num_tokens):
            seq_len = int(seq_lens_list[i])
            if seq_len <= 0:
                continue

            needed_blocks = cdiv(seq_len, block_size)
            block_ids = block_table[i, :needed_blocks].tolist()

            k_lat_parts: list[torch.Tensor] = []
            k_pe_parts: list[torch.Tensor] = []
            remain = seq_len
            for block_id in block_ids:
                valid = min(block_size, remain)
                k_lat_parts.append(k_nope_cache[int(block_id), :, :valid, :].permute(1, 0, 2))
                k_pe_parts.append(k_pe_cache[int(block_id), :, :valid, :].permute(1, 0, 2))
                remain -= valid
                if remain <= 0:
                    break

            k_lat_seq = torch.cat(k_lat_parts, dim=0)
            k_pe_seq = torch.cat(k_pe_parts, dim=0)
            v_lat_seq = k_lat_seq

            k_lat_seq = self._expand_kv_to_q_heads(k_lat_seq)
            k_pe_seq = self._expand_kv_to_q_heads(k_pe_seq)
            v_lat_seq = self._expand_kv_to_q_heads(v_lat_seq)

            q_lat = ql_nope[i]
            q_pe_i = q_pe[i]

            score_lat = torch.einsum("hd,shd->hs", q_lat, k_lat_seq)
            score_rope = torch.einsum("hd,shd->hs", q_pe_i, k_pe_seq)
            probs = torch.softmax((score_lat + score_rope) * self.scale, dim=-1)
            ctx = torch.einsum("hs,shd->hd", probs, v_lat_seq)
            latent_output[:, i, :] = ctx

        return latent_output
