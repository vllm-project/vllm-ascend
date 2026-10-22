# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator, MambaStateShapeCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.models.glm5next.cache_config import _align_glm5_next_cache_specs
from vllm_ascend.models.glm5next.cache_views import view_glm5_next_cache
from vllm_ascend.models.glm5next.kv_cache import get_kpool_tail_ring_capacity
from vllm_ascend.worker.v2.attn_utils import _reshape_mamba_kv_cache


@pytest.fixture
def glm_shared_physical_pages():
    blocks, offset, pool, mtp = 3, 16, 4, 5
    config = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=mtp))
    specs = {
        "main": AscendMLAAttentionSpec(
            block_size=40,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
        "indexer": AscendMLAAttentionSpec(
            block_size=40,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float8_e4m3fn,
            tokens_per_state=pool,
            scale_dim=1,
            scale_dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
        "tail": AscendIndexerKPoolTailSpec(
            block_size=get_kpool_tail_ring_capacity(config, pool),
            sliding_window=pool,
            compress_ratio=pool,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
        ),
        "mamba": MambaSpec(
            block_size=40,
            shapes=MambaStateShapeCalculator.kda_state_shape(1, 1, 2, conv_kernel_size=4, num_spec=mtp),
            dtypes=MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto"),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        ),
    }
    _align_glm5_next_cache_specs(specs)
    backings, raw_caches, descriptors = [], {}, []
    for names in (("main", "mamba"), ("indexer", "tail")):
        page_bytes = specs[names[0]].page_size_bytes
        backing = (torch.arange(blocks * page_bytes + 2 * offset) % 101).to(torch.uint8)
        raw = backing[offset:-offset]
        backings.append((backing, raw, page_bytes))
        raw_caches.update({name: raw for name in names})
        descriptors.append(SimpleNamespace(size=raw.numel(), layers=list(names), block_stride=0))

    backend = SimpleNamespace(get_kv_cache_shape=lambda n, block, heads, dim: (n, block, heads, dim))
    caches = {
        name: view_glm5_next_cache(
            name,
            spec,
            raw_caches[name],
            attn_backend=backend,
            kernel_block_size=8,
            num_blocks=blocks,
            get_kv_cache_dims=lambda *_: (8, 0),
        )
        for name, spec in specs.items()
        if name != "mamba"
    }
    caches["mamba"] = _reshape_mamba_kv_cache(raw_caches["mamba"], specs["mamba"])
    assert caches["main"][1].numel() == 0
    assert caches["indexer"][0].shape[0] == blocks * 5
    assert caches["tail"][0].shape[1] == pool + mtp
    assert specs["tail"].page_size_bytes > specs["tail"].unpadded_page_size_bytes
    kv_config = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_tensors=descriptors,
        kv_cache_groups=[SimpleNamespace(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()],
    )
    return SimpleNamespace(
        config=config, kv_config=kv_config, specs=specs, caches=caches, backings=backings, offset=offset
    )
