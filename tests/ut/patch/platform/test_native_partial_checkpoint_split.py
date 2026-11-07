# SPDX-License-Identifier: Apache-2.0
"""Characterize the GLM sparse wrapper bypass using the real split functions.

This is a CPU counterexample, not a checkpoint-payload or model execution test.
The scheduler/request shells follow test_patch_mamba_block_aligned_split.py;
partial eligibility comes from real registered specs and the native resolver.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import resolve_cache_hit_alignment_tokens
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _skips_eagle_block_drop
from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_resolve_kv_cache_block_sizes
from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import (
    _mamba_block_aligned_split,
    _original_mamba_block_aligned_split,
)

# Resolved physical blocks of the measured deployment: P TP8 and D TP2.
_PRODUCER_BLOCK = 1152
_CONSUMER_BLOCK = 4352


def _resolved_glm_config(enable_prefix_caching, block_size=_PRODUCER_BLOCK, consumer=False):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    # Minimal resolved GLM group representatives; no weights or device tensors.
    # KPool raw-token block size equals the page block, with four tokens per
    # stored state.
    specs = [
        AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            tokens_per_state=4,
        ),
        MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=3,
        ),
        AscendIndexerKPoolTailSpec(
            block_size=4,
            sliding_window=4,
            compress_ratio=4,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float32,
        ),
    ]
    groups = [
        KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
        for name, spec in zip(("main", "indexer", "kda", "tail"), specs)
    ]
    kv_config = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=groups)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_cache_mode="align",
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=128,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=SimpleNamespace(is_kv_producer=not consumer, is_kv_consumer=consumer),
    )
    managers = [KVCacheSpecRegistry.get_manager_class(group.kv_cache_spec, group.role) for group in groups]
    assert managers[:2] == [FullAttentionManager, FullAttentionManager]
    assert issubclass(managers[2], MambaManager)
    assert managers[3] is KpoolTailManager
    assert all(manager.supports_fine_grained_hash_lookup for manager in managers[:3])
    assert not managers[3].supports_fine_grained_hash_lookup
    assert not specs[3].prefix_cacheable  # Private tail cannot veto partial hits.

    resolved_block_size, hash_size = _ascend_resolve_kv_cache_block_sizes(kv_config, config)
    assert (resolved_block_size, hash_size) == (block_size, 128)
    alignment = resolve_cache_hit_alignment_tokens(kv_config, config, resolved_block_size, hash_size)
    # Connector hashing remains 128 even with local prefix caching disabled.
    assert alignment == (128 if enable_prefix_caching else block_size)
    kv_config.cache_hit_alignment_tokens = alignment
    return config, kv_config, resolved_block_size, hash_size


def _scheduler(config, kv_config, block_size, hash_size, *, sparse, native):
    model_config = SimpleNamespace(index_topk=2048, index_kpool=4) if sparse else SimpleNamespace()
    # Apply the paired version's role policy to the direct native reference.
    keep_drop = not (native and _skips_eagle_block_drop(config.kv_transfer_config))
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=config.kv_transfer_config,
            model_config=SimpleNamespace(hf_text_config=model_config, hf_config=model_config),
        ),
        cache_config=config.cache_config,
        block_size=block_size,
        hash_block_size=hash_size,
        use_eagle=keep_drop,
        use_eagle_block_drop=keep_drop,
        num_hash_lookahead_tokens=0,  # Native KV connectors retain legacy hashes.
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=kv_config.cache_hit_alignment_tokens < block_size,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_shared_prefix_checkpoint=False,
    )


@pytest.mark.parametrize("enable_prefix_caching", [True, False], ids=["native-partial", "native-partial-off"])
@pytest.mark.parametrize("sparse", [True, False], ids=["sparse-kpool", "sparse-feature-off"])
def test_real_glm_alignment_exposes_sparse_partial_checkpoint_bypass(enable_prefix_caching, sparse):
    """Preserve the native lookahead tail stop when resolved groups permit it."""
    config, kv_config, block_size, hash_size = _resolved_glm_config(enable_prefix_caching)
    native = _scheduler(config, kv_config, block_size, hash_size, sparse=sparse, native=True)
    ascend = _scheduler(config, kv_config, block_size, hash_size, sparse=sparse, native=False)
    # Native lookahead hashes resolve the drop bit before either split call.
    for scheduler in (native, ascend):
        scheduler.num_hash_lookahead_tokens = 1
        scheduler.use_eagle_block_drop = False
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=4353,
        num_tokens=4353,
        shared_prefix_boundary=0,
    )

    # Both real functions first select the physical boundary, without a copy
    # of either split algorithm in the test.
    assert _original_mamba_block_aligned_split(native, request, 4353) == 3456
    assert _mamba_block_aligned_split(ascend, request, 4353) == 3456
    request.num_computed_tokens = 3456
    native_end = 3456 + _original_mamba_block_aligned_split(native, request, 897)
    ascend_end = 3456 + _mamba_block_aligned_split(ascend, request, 897)

    assert native_end == (4352 if enable_prefix_caching else 4353)
    assert ascend_end == native_end
    if sparse and enable_prefix_caching:
        assert (native_end, ascend_end) == (4352, 4352)
        request.num_computed_tokens = native_end
        assert _original_mamba_block_aligned_split(native, request, 1) == 1
    else:
        assert ascend_end == native_end
    assert ascend.use_eagle is True
    assert native.use_eagle_block_drop is False
    assert ascend.use_eagle_block_drop is False


def _consumer_schedulers():
    """Actual GLM sparse model on the D-side resolved block, native partial on."""
    config, kv_config, block_size, hash_size = _resolved_glm_config(True, block_size=_CONSUMER_BLOCK, consumer=True)
    assert not _skips_eagle_block_drop(config.kv_transfer_config)
    native = _scheduler(config, kv_config, block_size, hash_size, sparse=True, native=True)
    ascend = _scheduler(config, kv_config, block_size, hash_size, sparse=True, native=False)
    assert ascend.mamba_partial_cache_hit and native.mamba_partial_cache_hit
    # A consumer keeps the Eagle drop on both calls; nothing is cleared.
    assert native.use_eagle_block_drop is True and ascend.use_eagle_block_drop is True
    return kv_config, native, ascend


def test_consumer_continued_prefill_honors_native_checkpoint_stop():
    """A partial external hit followed by ordinary prefill must not bypass native.

    Previously any consumer with a computed prefix returned the whole chunk.
    """
    _, native, ascend = _consumer_schedulers()
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=9000,
        num_tokens=9000,
        shared_prefix_boundary=0,
    )
    external = _CONSUMER_BLOCK  # One loaded D block; the prompt remains in prefill.
    num_new_tokens = request.num_tokens - external

    native_tokens = _original_mamba_block_aligned_split(native, request, num_new_tokens, 0, external)
    ascend_tokens = _mamba_block_aligned_split(ascend, request, num_new_tokens, 0, external)

    assert ascend_tokens == native_tokens
    # Real native consumer stop: the hash-aligned prompt tail one hash page
    # earlier than a no-drop producer would use (Eagle drop kept), not the
    # chunk end that the old consumer bypass returned.
    assert native_tokens < num_new_tokens
    assert external + native_tokens == 8832
    assert ascend.use_eagle is True
    assert ascend.use_eagle_block_drop is True


def test_consumer_final_verifier_window_stays_intact():
    """The last prompt token padded to 1 + K must keep its full width.

    External loading leaves one prompt token mid-block; native alignment
    would stop at the next D block boundary and shorten the verifier window.
    """
    kv_config, native, ascend = _consumer_schedulers()
    num_speculative_tokens = kv_config.kv_cache_groups[2].kv_cache_spec.num_speculative_blocks
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=_CONSUMER_BLOCK,
        num_tokens=_CONSUMER_BLOCK,
        shared_prefix_boundary=0,
    )
    external = request.num_tokens - 1
    window = 1 + num_speculative_tokens

    # Characterize the real native split: it would truncate at the block end.
    assert _original_mamba_block_aligned_split(native, request, window, 0, external) == 1
    assert _mamba_block_aligned_split(ascend, request, window, 0, external) == window
    assert ascend.use_eagle is True
    assert ascend.use_eagle_block_drop is True
