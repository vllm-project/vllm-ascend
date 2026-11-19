# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs


def make_vllm_config(num_layers=45, num_shared_buffers=3, **extra):
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": num_shared_buffers,
                **extra,
            },
        ),
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: num_layers, hf_text_config=SimpleNamespace(model_type="glm5_next_text")
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )


def make_cache_config(groups, *, num_blocks=3, offset=0):
    """Create native contiguous descriptors, including heterogeneous groups."""
    strides = [
        max(s.page_size_bytes for s in spec.kv_cache_specs.values())
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else spec.page_size_bytes
        for _, spec in groups
    ]
    size = offset + max(len(names) * num_blocks * stride for (names, _), stride in zip(groups, strides))
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec) for names, spec in groups],
        kv_cache_tensors=[
            KVCacheTensor(size=size, layers=names, layer_stride=num_blocks * stride, block_stride=stride, offset=offset)
            for (names, _), stride in zip(groups, strides)
        ],
    )
