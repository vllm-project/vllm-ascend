from types import SimpleNamespace

import torch

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_kv_cache_config_deepseek_v4,
    _get_kv_cache_groups_uniform_groups,
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
    grouped_specs = group_and_unify_kv_cache_specs(specs)
    assert grouped_specs is not None
    groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))

    _, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config,
        groups,
        available_memory=1 << 30,
    )

    allocated_layers = {layer_name for tensor in tensors for layer_name in tensor.shared_by}
    assert allocated_layers == set(specs)
