# SPDX-License-Identifier: Apache-2.0
"""Exercise the real runner grouping method with actual Attention layer types."""

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAImpl
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_actual_runner_grouping_keeps_gqa_heads_and_mla_rope_modes_independent(monkeypatch):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASH_MLA", "1")

    context = {}
    specs = {}
    gqa = AscendAttentionBackend.customize_spec(
        FullAttentionSpec(block_size=768, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    )
    mla = MLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    for name, heads, use_rope in [
        ("draft.gqa.0", 8, None),
        ("draft.gqa.1", 8, None),
        ("draft.gqa.2", 4, None),
        ("target.mla.0", 12, True),
        ("target.mla.1", 12, False),
        ("draft.mla.0", 8, True),
    ]:
        is_gqa = ".gqa." in name
        cls = Attention if is_gqa else MLAAttention
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.num_heads = heads
        impl_cls = AscendAttentionBackendImpl if is_gqa else AscendMLAImpl
        layer.impl = impl_cls.__new__(impl_cls)
        layer.impl.num_heads = heads
        if is_gqa:
            assert not hasattr(layer.impl, "use_mla_rope")
            backend = AscendAttentionBackend
        else:
            layer.impl.use_mla_rope = use_rope
            backend = AscendMLABackend
        layer.get_attn_backend = lambda backend=backend: backend
        context[name] = layer
        specs[name] = gqa if is_gqa else mla

    # Construction of NPU buffers is outside this grouping regression. Keep
    # the genuine runner method, layer filtering, keys and AttentionGroups.
    class RecordedBuilder:
        def __init__(self, spec, names, config, device):
            self.kv_cache_spec = spec
            self.layer_names = names
            self.num_heads = context[names[0]].num_heads

    monkeypatch.setattr(AscendAttentionBackend, "get_builder_cls", lambda: RecordedBuilder)
    monkeypatch.setattr(AscendMLABackend, "get_builder_cls", lambda: RecordedBuilder)
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    runner = SimpleNamespace(
        vllm_config=config,
        device=torch.device("cpu"),
        attn_groups=[],
        _check_and_update_cudagraph_mode=lambda *args: None,
        calculate_reorder_batch_threshold=lambda: None,
    )
    packed = UniformTypeKVCacheSpecs.from_specs(specs)
    assert packed is not None
    cache = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(context), packed)],
    )
    NPUModelRunner.initialize_attn_backend(runner, cache)
    groups = runner.attn_groups[0]
    assert sorted(sorted(group.layer_names) for group in groups) == sorted(
        [
            ["draft.gqa.0", "draft.gqa.1"],
            ["draft.gqa.2"],
            ["target.mla.0"],
            ["target.mla.1"],
            ["draft.mla.0"],
        ]
    )
    for group in groups:
        assert group.kv_cache_group_id == 0
        assert group.metadata_builders[0].num_heads == context[group.layer_names[0]].num_heads
    assert len({id(group.metadata_builders[0]) for group in groups}) == 5
