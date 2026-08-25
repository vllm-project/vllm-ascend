from types import SimpleNamespace

import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.models.dots3_note.model import Dots3NoteMLABackend
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    group_and_unify_kv_cache_specs,
)


def test_dots3_note_kv_plan_allocates_sliding_layers():
    specs = {
        "full": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        ),
        "indexer": AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        ),
        **{
            f"sliding_{index}": AscendSlidingWindowMLASpec(
                block_size=128,
                num_kv_heads=1,
                head_size=1088,
                dtype=torch.bfloat16,
                cache_dtype_str="auto",
                sliding_window=513,
            )
            for index in range(3)
        },
    }
    assert group_and_unify_kv_cache_specs(specs) is None
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = Dots3NoteMLABackend.supported_kv_cache_layouts()[0].name
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        speculative_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    groups = get_kv_cache_groups(vllm_config, specs)
    assert groups is not None
    config = get_kv_cache_config_from_groups(vllm_config, groups, available_memory=1 << 30)
    allocated_layers = {layer for tensor in config.kv_cache_tensors for layer in tensor.layers}
    assert allocated_layers == set(specs)
