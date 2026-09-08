# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-side wrapper for GLM-5 Indexer KPool MLA attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.indexer_kpool_mla_v1 import AscendIndexerKPoolMLABackend


@dataclass
class IndexerKPoolMLAModules:
    fused_qkv_a_proj: nn.Module
    q_a_layernorm: nn.Module
    q_b_proj: nn.Module
    kv_a_layernorm: nn.Module
    kv_b_proj: nn.Module
    rotary_emb: nn.Module | None
    indexer_rotary_emb: nn.Module | None
    o_proj: nn.Module
    indexer: nn.Module
    topk_indices_buffer: torch.Tensor | None


class IndexerKPoolMLACacheLayer(nn.Module, AttentionLayerBase):
    """Own the combined MLA KV/RoPE cache and the Ascend Indexer KPool MLA implementation."""

    # model_runner_v1 discovers GLM-5's three independent cache layers through
    # this marker.  Without it the main MLA cache spec is skipped, so only the
    # indexer_state/indexer metadata builders are created.
    cache_role = "kv"

    # vLLM 0.23's MLACommonMetadataBuilder unconditionally reads this field
    # from every layer represented by an MLA cache group.  The Ascend SFA
    # builder completely overrides build/_build and never consumes the
    # generic MLA prefill backend, so None is an intentional compatibility
    # sentinel rather than an executable backend.
    prefill_backend = None

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        modules: IndexerKPoolMLAModules,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_size = kv_lora_rank + qk_rope_head_dim
        if cache_config is None:
            raise ValueError("GLM-5 Indexer KPool MLA requires cache_config.")
        # GLM-5 Next's full MLA KV cache remains BF16 on every device. The A5
        # shared-KV operator consumes this cache directly without quantization.
        self.kv_cache_dtype = "bfloat16"
        self.block_size = cache_config.block_size
        self.attn_backend = AscendIndexerKPoolMLABackend
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads=num_heads,
            head_size=self.head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rotary_emb=modules.rotary_emb,
            indexer_rotary_emb=modules.indexer_rotary_emb,
            fused_qkv_a_proj=modules.fused_qkv_a_proj,
            q_b_proj=modules.q_b_proj,
            q_a_layernorm=modules.q_a_layernorm,
            q_proj=None,
            kv_a_proj_with_mqa=None,
            kv_a_layernorm=modules.kv_a_layernorm,
            kv_b_proj=modules.kv_b_proj,
            o_proj=modules.o_proj,
            indexer=modules.indexer,
            topk_indices_buffer=modules.topk_indices_buffer,
            layer_name=prefix,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype,
            vllm_config.model_config,
        )
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=cache_dtype,
            cache_dtype_str=None,
            compress_ratio=1,
            model_version="glm5_next",
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.impl.process_weights_after_loading(act_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class AscendIndexerKPoolMLAAttention(nn.Module):
    """GLM-5 Indexer KPool MLA wrapper that never constructs generic MLAAttention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        modules: IndexerKPoolMLAModules,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix = prefix
        self.mla_cache_layer = IndexerKPoolMLACacheLayer(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            modules=modules,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.cache_layers = (
            self.mla_cache_layer,
            modules.indexer.state_cache,
            modules.indexer.k_cache,
        )
        self.impl = self.mla_cache_layer.impl

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.mla_cache_layer.process_weights_after_loading(act_dtype)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        del positions  # RoPE positions are carried by Indexer KPool MLA metadata.
        output = torch.empty(
            (hidden_states.shape[0], self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.indexer_kpool_mla_forward(
            hidden_states,
            _EXTRA_CTX.flash_comm_v1_enabled,
            output,
            self.prefix,
        )
        return output


def _get_cache_tensor(
    layer: nn.Module,
    virtual_engine: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    cache = layer.kv_cache
    if isinstance(cache, (list, tuple)):
        cache = cache[virtual_engine]
    if isinstance(cache, (list, tuple)):
        if len(cache) == 1:
            cache = cache[0]
        elif all(isinstance(tensor, torch.Tensor) for tensor in cache):
            return tuple(cache)
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Indexer KPool MLA cache {type(layer).__name__} is not bound to a tensor.")
    return cache


def _get_cache_layer_name(layer: nn.Module) -> str:
    """返回 cache layer 在 ``attn_metadata`` 中使用的精确注册名。"""
    layer_name = getattr(layer, "layer_name", None)
    if isinstance(layer_name, str):
        return layer_name
    prefix = getattr(layer, "prefix", None)
    if isinstance(prefix, str):
        return prefix
    raise TypeError(
        "Indexer KPool MLA cache layer must expose a string "
        f"layer_name or prefix, got {type(layer).__name__}."
    )


def _collect_cache_metadata(wrapper: nn.Module, metadata: dict[str, object]) -> tuple[object, ...]:
    """按 cache tensor 的顺序精确取得三份 metadata，避免前缀匹配误收。"""
    cache_layer_names = tuple(_get_cache_layer_name(layer) for layer in wrapper.cache_layers)
    missing = tuple(name for name in cache_layer_names if name not in metadata)
    if missing:
        available = tuple(sorted(name for name in metadata if name.startswith(f"{wrapper.prefix}.")))
        raise KeyError(
            "Indexer KPool MLA metadata is missing cache layers "
            f"{missing}; metadata under {wrapper.prefix!r}: {available}."
        )
    return tuple(metadata[name] for name in cache_layer_names)


def indexer_kpool_mla_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    wrapper = forward_context.no_compile_layers[layer_name]
    metadata = forward_context.attn_metadata
    if metadata is None:
        wrapper.impl.forward(
            f"{wrapper.prefix}.attn",
            hidden_states,
            tuple(torch.tensor([]) for _ in wrapper.cache_layers),
            None,
            need_gather_q_kv,
            output=output,
        )
        return

    if not isinstance(metadata, dict):
        raise TypeError(
            "Indexer KPool MLA expects attention metadata keyed by layer name, "
            f"got {type(metadata).__name__}."
        )
    layer_metadata = _collect_cache_metadata(wrapper, metadata)
    virtual_engine = getattr(forward_context, "virtual_engine", 0) or 0
    caches = tuple(_get_cache_tensor(layer, virtual_engine) for layer in wrapper.cache_layers)
    wrapper.impl.forward(
        f"{wrapper.prefix}.attn",
        hidden_states,
        caches,
        layer_metadata,
        need_gather_q_kv,
        output=output,
    )


def indexer_kpool_mla_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="indexer_kpool_mla_forward",
    op_func=indexer_kpool_mla_forward,
    mutates_args=["output"],
    fake_impl=indexer_kpool_mla_forward_fake,
    dispatch_key="PrivateUse1",
)
