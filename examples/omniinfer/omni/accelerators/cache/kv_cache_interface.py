# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
import re
import torch

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.attention import AttentionType
from vllm.attention.layer import Attention
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
)
from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


logger = init_logger("vllm.v1.omni")

SINK = 1
RECENT = 3
BETA = 0.2
PATTERN = [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]

@dataclass
class OmniKVCacheConfig(KVCacheConfig):
    """
    The KV cache configuration of a model with different block numbers
    for different KV cache groups.
    """

    num_blocks_per_group: dict[type[KVCacheSpec], int]
    """The number of KV cache blocks per kv cache group"""


@dataclass
class OmniAttentionSpec(AttentionSpec):
    sink_blocks: int = SINK
    recent_blocks: int = RECENT

    def __post_init__(self):
        self.max_num_blocks = self.sink_blocks + self.recent_blocks
        self.sink = self.sink_blocks * self.block_size
        self.recent = self.recent_blocks * self.block_size
        self.max_compressed_len = self.max_num_blocks * self.block_size

    @property
    def type_id(self) -> str:
        return f"omni_attention_{self.sink}_{self.recent}_{self.block_size}_{self.page_size_bytes}"

    @property
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return self.max_num_blocks * self.page_size_bytes


class OmniMultiGroupBlockTable(MultiGroupBlockTable):
    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_batched_tokens: int, pin_memory: bool,
                 device: torch.device, *args, **kwargs) -> None:
        if len(args) > 0:
            raise RuntimeError("All arguments should be passed with keywords"
                               f", but {len(args)} positional args are given.")
        if 'block_size' in kwargs:
            block_size = kwargs['block_size']
        elif 'block_sizes' in kwargs:
            # when upgraded to vllm 0.9.2, this argument will change from block_size to block_sizes
            block_size = kwargs['block_sizes'][0]
        else:
            raise RuntimeError("Neither `block_size` nor `block_sizes` is given.")
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size should be a positive int, but is {block_size}.")
        self.block_tables = [
            BlockTable(max_num_reqs, cdiv(max_model_len, block_size),
                       max_num_batched_tokens, pin_memory, device),
            BlockTable(max_num_reqs, cdiv(max_model_len, block_size),
                       max_num_batched_tokens, pin_memory, device)
        ]


def get_kv_cache_config_omni_type(vllm_config: VllmConfig,
                                  kv_cache_spec: dict[str, KVCacheSpec],
                                  available_memory: int) -> OmniKVCacheConfig:
    """
    Generates the KV cache configuration for a model with two types of KV cache.
    It's assumed that the numbers of layers with these two types are approximately same.
    The ratio of memory allocated to them is now determined by a hyperparameter BETA.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) != 1:
        raise ValueError("page_sizes must have exactly one element.")
    page_size = page_sizes.pop()

    # the original number of blocks if layers were uniform
    num_blocks = int(available_memory // page_size // len(kv_cache_spec))
    num_blocks = max(num_blocks, 0)
    num_omni_layers = sum(PATTERN)
    num_full_layers = len(PATTERN) - num_omni_layers
    omni_num_blocks = int(num_blocks * BETA)
    # This computation is to ensure that the total number of blocks
    # of all layers does not exceed the original number.
    full_num_blocks = int(num_blocks * (1 + (1-BETA)*num_omni_layers/(num_full_layers+1)))

    # logging
    num_tokens = num_blocks * vllm_config.cache_config.block_size
    full_num_tokens = full_num_blocks * vllm_config.cache_config.block_size
    omni_num_tokens = omni_num_blocks * vllm_config.cache_config.block_size
    logger.info(f"GPU KV cache size: {num_tokens:,} tokens")
    logger.info(f"With Omni enabled, GPU KV Cache size: {full_num_tokens:,}"
                f" tokens in full layers and {omni_num_tokens:,} in swa layers.")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = num_tokens / vllm_config.model_config.max_model_len
    omni_max_concur = min(full_num_tokens / vllm_config.model_config.max_model_len,
                          omni_num_blocks / (SINK + RECENT))
    logger.info("Maximum concurrency for %s tokens per request changes from %.2fx to %.2fx",
                max_model_len_str, max_concurrency, omni_max_concur)

    # create a KVCacheConfig with exactly two groups
    # 1. divide the layers to full and omni groups
    # 2. compute size for each layer
    grouped_layer_names = [[], []]
    layer2size = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if isinstance(layer_spec, FullAttentionSpec):
            if layer_spec.sliding_window is not None:
                ValueError("Omni attention implements its own sliding window. \
                    Manually setting sliding window is not supported.")
            grouped_layer_names[0].append(layer_name)
            layer2size[layer_name] = KVCacheTensor(size=page_size*full_num_blocks)
        elif isinstance(layer_spec, OmniAttentionSpec):
            grouped_layer_names[1].append(layer_name)
            layer2size[layer_name] = KVCacheTensor(size=page_size*omni_num_blocks)
        else:
            raise RuntimeError(f"Unsupported KV Cache Spec type {type(layer_spec)}.")
    kv_cache_config = OmniKVCacheConfig(
        num_blocks=full_num_blocks,
        num_blocks_per_group={FullAttentionSpec: full_num_blocks, OmniAttentionSpec: omni_num_blocks},
        tensors=layer2size,
        kv_cache_groups=create_kv_cache_group_specs(kv_cache_spec,
                                                    grouped_layer_names)
    )
    return kv_cache_config


def get_omni_hybrid_kv_cache_spec(self: GPUModelRunner) -> dict[str, KVCacheSpec]:
    """Generates the KVCacheSpec by parsing the kv cache format from
    each attention module in the static forward context.
    Returns:
        A dictionary mapping layer names to their KV cache format.
    """
    layers = get_layers_from_vllm_config(self.vllm_config, Attention)
    block_size = self.vllm_config.cache_config.block_size
    use_mla = self.vllm_config.model_config.use_mla
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for layer_name, attn_module in layers.items():
        layer_idx = int(re.findall(r"model\.layers\.(\d+)", layer_name)[0])
        if attn_module.attn_type == AttentionType.DECODER:
            if PATTERN[layer_idx] == 1:
                kv_cache_spec[layer_name] = OmniAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla,
                    sink_blocks=SINK,
                    recent_blocks=RECENT)
            else:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla)
        else:
            raise NotImplementedError("Omni attention supports decoder-only models.")
    return kv_cache_spec
