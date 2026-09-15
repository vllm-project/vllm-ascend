# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
#
import math
from itertools import islice

import torch
import vllm.envs as envs
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache as VllmDeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.attention.dsa_attn_kv_plan import get_dsv4_attn_kv_dtype
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.common.deepseek import (
    DeepseekForCausalLMMixin,
    get_spec_layer_idx_from_weight_name,
    init_attention_projections,
)
from vllm_ascend.models.common.deepseek import DeepseekMixtureOfExperts as DeepseekV2MixtureOfExperts
from vllm_ascend.models.common.deepseek import DeepseekMLP as DeepseekV2MLP
from vllm_ascend.models.common.deepseek import DeepseekMoE as DeepseekV4MoE
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.models.deepseek_v4.compressor import Compressor
from vllm_ascend.models.deepseek_v4.indexer import DeepseekV4Indexer
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.utils import (
    enable_custom_op,
    enable_dsa_cp,
    extract_dsv4_layer_index,
    get_dsv4_compress_ratio,
)
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_utils import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)

sequence_parallel_chunk = sp_shard


class AscendDeepseekV4SWACache(VllmDeepseekV4SWACache):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__(head_dim, window_size, torch.uint8, prefix, cache_config)
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.dtype = dtype

        self.block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        self.dtype = get_dsv4_attn_kv_dtype(vllm_config)
        if self.dtype == torch.float8_e4m3fn:
            vllm_config.cache_config.cache_dtype = "float8_e4m3fn"
        cached_head_size = self.head_dim + 128 if self.dtype == torch.float8_e4m3fn else self.head_dim
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=cached_head_size,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="deepseek_v4",
            alignment=None,
        )

    def forward(self): ...

    def get_attn_backend(self):
        from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend

        return AscendDSASWABackend


def precompute_freqs_cis_cpu(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis.to(x.device)).flatten(-2)
    y.copy_(x)
    return y


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(1 + torch.floor(positions / original_max_position_embeddings))
    # Broadcast over num_heads and head_dim
    return scaling[..., None, None]


class DeepseekV4Attention(nn.Module):
    swa_cache_cls = AscendDeepseekV4SWACache

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        max_position_embeddings: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
        need_gather_q_kv: bool = False,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        config_layer_idx = extract_dsv4_layer_index(config, prefix)
        init_attention_projections(self, config, quant_config, prefix, reduce_results)
        self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)

        if self.compress_ratio > 1:
            config.rope_parameters["rope_theta"] = config.compress_rope_theta
            rope_groups = ["default", f"c{self.compress_ratio}"]
        else:
            config.rope_parameters["rope_theta"] = config.rope_theta
            rope_groups = ["default"]
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=config.rope_parameters["rope_theta"],
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            rope_groups=rope_groups,
        )

        self.compressor: Compressor | None = None
        self.indexer: DeepseekV4Indexer | None = None

        use_index_cache = getattr(config, "use_index_cache", False)

        # IndexCache: decide whether this layer reuses topk from a previous
        # indexer-bearing layer. Refer: https://arxiv.org/abs/2603.12201
        # Only meaningful when this layer actually owns an Indexer (c4) and
        # IndexCache is enabled via hf-overrides. MTP layers are excluded
        # because spec_decode shares topk_indices_buffer at the model level
        # only, leaving impl-level references stale.
        skip_topk = False
        if self.compress_ratio == 4 and use_index_cache and ".mtp." not in prefix:
            compress_ratios = getattr(config, "compress_ratios", None) or []
            indexer_seq_idx = sum(1 for r in compress_ratios[:config_layer_idx] if r == 4)
            pattern = getattr(config, "index_topk_pattern", None)
            freq = getattr(config, "index_topk_freq", 1)
            if pattern is None:
                skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0
            else:
                assert pattern[0] == "F", "index_topk_pattern must start with 'F'"
                if 0 <= indexer_seq_idx < len(pattern):
                    skip_topk = pattern[indexer_seq_idx] == "S"

        if self.compress_ratio > 1:
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )  # Compressor(4, 128)

            if self.compress_ratio == 4:
                self.indexer = DeepseekV4Indexer(
                    vllm_config,
                    config,
                    self.compress_ratio,
                    skip_topk=skip_topk,
                    use_index_cache=use_index_cache,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    prefix=f"{prefix}.indexer",
                    topk_indices_buffer=topk_indices_buffer,
                )

        k_dtype = get_dsv4_attn_kv_dtype(vllm_config)
        swa_cache_layer = self.swa_cache_cls(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=k_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        dsa_modules = DSAModules(
            wq_a=self.wq_a,
            q_norm=self.q_norm,
            q_norm_without_weight=self.q_norm_without_weight,
            wq_b=self.wq_b,
            wkv=self.wkv,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer,
            compressor=self.compressor,
            swa_cache_layer=swa_cache_layer,
        )

        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            eps=self.eps,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            dsa_modules=dsa_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            # prefix=f'{prefix}.attn',
            prefix=f"{prefix}",
            need_gather_q_kv=need_gather_q_kv,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dsa_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV2DecoderLayer(nn.Module):
    attention_cls = DeepseekV4Attention

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = enable_dsa_cp()  # TODO: delete this when enable_dsa_cp is sunset.

        attn_cls = self.attention_cls

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
            need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
        )

        self.mlp = DeepseekV4MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def rms_norm_cast(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize once and provide the exact FP32 routing input."""
        if enable_custom_op():
            return torch.ops._C_ascend.npu_rms_norm_cast(
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, hidden_states.float()

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
            x, hc_fn, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps
        )
        return y, post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0), residual.unsqueeze(dim=0), post.unsqueeze(dim=0), comb.unsqueeze(dim=0)
        )
        return y.squeeze(dim=0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states.clone()
        full_num_tokens = positions.shape[0]
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        attn_kwargs = {"positions": positions, "hidden_states": hidden_states, "llama_4_scaling": llama_4_scaling}
        hidden_states = self.self_attn(**attn_kwargs)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        hidden_states, hidden_states_fp32 = self.rms_norm_cast(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
            already_sequence_parallel=(self.use_sequence_parallel_moe and self.enable_dsa_cp),
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        return hidden_states, residual


DeepseekV4DecoderLayer = DeepseekV2DecoderLayer


@support_torch_compile
class DeepseekV4Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False
    # vLLM #50514 validates and relays the model's existing PP aux payload.
    supports_aux_hidden_states_over_pp = True
    AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"
    decoder_layer_cls: type[DeepseekV2DecoderLayer] = DeepseekV2DecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        # Expose at model level so spec_decode/llm_base_proposer can share
        # this buffer with the MTP draft via attribute replacement.
        self.topk_indices_buffer = topk_indices_buffer

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer),
            prefix=f"{prefix}.layers",
        )
        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = make_pp_empty_intermediate_tensors(
            self,
            make_empty_intermediate_tensors,
        )

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_norm = RMSNorm(hc_dim, eps=config.rms_norm_eps, has_weight=False, dtype=torch.float32)

        # Pre-hc_head residual stream buffer for the speculative draft
        # (MTP / DSpark / DFlash). Only needed when the decoder consumes
        # target-model hidden states; allocating it unconditionally would
        # permanently cost max_num_batched_tokens * hc_dim per rank.
        # Aligned with upstream DeepSeekV4 (see vllm PR #50312).
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        self._mtp_hidden_buffer = (
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
            if get_pp_group().is_last_rank and needs_mtp_hidden_states
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        x_norm = self.hc_norm(x)
        mixes = torch.nn.functional.linear(x_norm, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None
        aux_hidden_states = get_pp_transport_tensors(
            intermediate_tensors,
            PPTransportDataType.AUX_HIDDEN_STATES,
        )

        if self.use_sequence_parallel_moe:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(forward_context.is_padding, hidden_states)
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)  # TODO: support PP with dsacp.

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = None
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        if pp_group.is_first_rank:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # (b, s, h) -> (b, s, c, h)
        moe_input_ids = input_ids
        if getattr(self, "needs_moe_input_ids", False):
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
                input_ids=moe_input_ids,
            )
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if self.use_sequence_parallel_moe:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)

        if not pp_group.is_last_rank:
            intermediate_tensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )
            return add_pp_transport_tensors(
                intermediate_tensors,
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        if self.use_sequence_parallel_moe:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)

        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekForCausalLMMixin, SupportsLoRA, SupportsEagle3):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = DeepseekV4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Set MoE hyperparameters
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states


__all__ = ["DeepseekV2MLP", "DeepseekV4MoE", "DeepseekV2MixtureOfExperts", "get_spec_layer_idx_from_weight_name"]
