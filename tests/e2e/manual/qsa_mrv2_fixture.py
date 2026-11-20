# SPDX-License-Identifier: Apache-2.0
# Shared MRV2 fixture functions from test_qwen4_exp_cache_config.py.
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.models.qwen4_exp.cache_config import (
    build_qwen4_kv_cache_layout,
)
from vllm_ascend.worker.v2 import attn_utils as attn_utils_v2


def _groups() -> list[KVCacheGroupSpec]:
    # Match vLLM alignment: 256 tokens cover the full 33KiB GDN page.
    sources = [
        "model.layers.11.self_attn",
        "model.layers.47.self_attn",
        "model.mtp.layers.48.self_attn",
    ]
    main = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    gdn = MambaSpec(
        block_size=128,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    ple = MambaSpec(
        block_size=128,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
        tp_replicated=True,
    )
    hidden = HiddenStateCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    raw_sources = [sources[2], sources[0], sources[1]]
    compressed_sources = [sources[1], sources[2], sources[0]]
    return [
        KVCacheGroupSpec(
            [f"{source}.attn" for source in sources],
            main,
        ),
        KVCacheGroupSpec(
            [f"{source}.indexer.raw_key_cache" for source in raw_sources],
            raw,
        ),
        KVCacheGroupSpec(
            [f"{source}.indexer.compressed_key_cache" for source in compressed_sources],
            compressed,
        ),
        KVCacheGroupSpec(
            [
                "model.layers.0.linear_attn",
                "model.layers.1.linear_attn",
            ],
            gdn,
        ),
        KVCacheGroupSpec(["model.ple"], ple),
        KVCacheGroupSpec(["model.cache_only_layers.0"], hidden),
    ]


def _allocate_mrv2_kv_cache(
    config,
    monkeypatch,
    *,
    use_main: bool = False,
    kernel_block_sizes: list[int] | None = None,
    kv_transfer_config=None,
):
    """Allocate and materialize Qwen4Exp cache views through MRV2 only."""
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=kv_transfer_config,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils_v2, "enable_sfa", lambda _: False)
    backend = SimpleNamespace(full_cls_name=lambda: "qwen-cache-test")
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: backend, num_heads=1)
        for group in config.kv_cache_groups
        for name in group.layer_names
    }
    monkeypatch.setattr(
        attn_utils_v2,
        "get_layers_from_vllm_config",
        lambda *_: layers,
    )
    raw = attn_utils_v2._allocate_kv_cache(config, {}, torch.device("cpu"))
    block_sizes = kernel_block_sizes or [128] * len(config.kv_cache_groups)
    if use_main:
        monkeypatch.setattr(attn_utils_v2, "_allocate_kv_cache", lambda *_, **__: raw)
        caches = attn_utils_v2.allocate_kv_cache_main(
            config,
            torch.device("cpu"),
            None,
            block_sizes,
        )
    else:
        groups = [
            SimpleNamespace(
                kv_cache_group_id=i,
                kv_cache_spec=group.kv_cache_spec,
                layer_names=group.layer_names,
                backend=backend,
            )
            for i, group in enumerate(config.kv_cache_groups)
        ]
        caches = attn_utils_v2._reshape_kv_cache_v2(
            groups,
            raw,
            "auto",
            block_sizes,
            {},
            config,
        )
    layout = build_qwen4_kv_cache_layout(
        config.kv_cache_groups,
        config.num_blocks,
    )
    return raw, caches, layout
