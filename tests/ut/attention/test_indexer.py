# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadataBuilder,
)


def test_sfa_indexer_backend_contract():
    assert AscendSFAIndexerBackend.accept_output_buffer
    assert AscendSFAIndexerBackend.get_name() == "ASCEND_SFA_INDEXER"
    assert AscendSFAIndexerBackend.get_builder_cls() is AscendSFAIndexerMetadataBuilder
    assert AscendSFAIndexerBackend.get_kv_cache_shape(8, 128, 1, 160) == (
        8,
        128,
        1,
        160,
    )
    assert AscendSFAIndexerBackend.get_supported_kernel_block_sizes() == [128]


def test_sfa_indexer_metadata_builder_is_cache_only():
    kv_cache_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    device = torch.device("cpu")
    builder = AscendSFAIndexerMetadataBuilder(
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
    )

    assert builder.kv_cache_spec is kv_cache_spec
    assert builder.layer_names is layer_names
    assert builder.vllm_config is vllm_config
    assert builder.device == device
    assert builder.reorder_batch_threshold is None
    assert builder.get_cudagraph_support(vllm_config, kv_cache_spec) is AttentionCGSupport.UNIFORM_BATCH

    common_attn_metadata = MagicMock()
    assert builder.build(0, common_attn_metadata) is None
    assert builder.build_for_cudagraph_capture(common_attn_metadata) is None
