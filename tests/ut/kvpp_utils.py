# SPDX-License-Identifier: Apache-2.0
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


def layer_name(index):
    return f"model.layers.{index}.self_attn.attn"


def indexer_name(index):
    return f"model.layers.{index}.self_attn.indexer.k_cache"


def make_kvpp_config(tp=3):
    return SimpleNamespace(
        additional_config={"enable_kvpp": True},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=17, num_nextn_predict_layers=1),
            enforce_eager=True,
            use_mla=True,
            is_hybrid=False,
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens_per_batch_size=None),
        kv_transfer_config=None,
        quant_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def make_kvpp_specs():
    # PP-local targets have uneven sizes; MTP is larger than every target.
    specs = {
        layer_name(index): AscendMLAAttentionSpec(
            block_size=2, num_kv_heads=1, head_size=size // 2, dtype=torch.int8, cache_sparse_sfa_c8=True
        )
        for index, size in zip(range(9, 18), (32, 48, 64, 32, 48, 64, 32, 48, 96))
    }
    specs[indexer_name(11)] = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    return specs


def make_cache_config(specs, num_blocks=3):
    kv_cache_tensors = []
    for name, spec in specs.items():
        size = num_blocks * spec.page_size_bytes
        if "shared_by" in KVCacheTensor.__dataclass_fields__:
            # vLLM #51718 (0.28.0 release lane): one descriptor per shared layer.
            kv_cache_tensors.append(
                KVCacheTensor(size=size, shared_by=[name], offset=0, block_stride=spec.page_size_bytes)
            )
        else:
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[name],
                    offset=0,
                    layer_stride=spec.page_size_bytes,
                    block_stride=spec.page_size_bytes,
                )
            )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=list(specs),
                kv_cache_spec=UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs),
            )
        ],
    )


class ManualExecutor:
    """Execute submitted work explicitly, without threads or fake Future.result."""

    def __init__(self, **_kwargs):
        self.pending: deque[tuple[Future[None], Callable[..., None], tuple[Any, ...]]] = deque()
        self.submitted = []

    def submit(self, fn, *args):
        future: Future[None] = Future()
        self.pending.append((future, fn, args))
        self.submitted.append((future, args))
        return future

    def run_next(self):
        future, fn, args = self.pending.popleft()
        future.set_result(fn(*args))

    def fail_next(self, error):
        future, _, _ = self.pending.popleft()
        future.set_exception(error)


def make_attention_cache_case(packed):
    from vllm.model_executor.layers.attention import MLAAttention

    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    from vllm_ascend.attention.mla_v1 import AscendMLABackend

    specs, layers = {}, {}
    for index, dims, packed_dim in ((9, (8, 4), 16), (10, (4, 4), 24), (11, (8, 4), 16)):
        name = layer_name(index)
        specs[name] = AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=packed_dim if packed else sum(dims),
            dtype=torch.int8 if packed else torch.bfloat16,
            cache_sparse_sfa_c8=packed,
        )
        layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(layer)
        layer.kv_lora_rank, layer.qk_rope_head_dim = dims
        layer.num_heads = 1
        layer.kv_sharing_target_layer_name = None
        layer.get_attn_backend = lambda: AscendMLABackend
        layers[name] = layer
    name = indexer_name(9)
    specs[name] = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    layers[name] = SimpleNamespace(
        num_heads=1, kv_sharing_target_layer_name=None, get_attn_backend=lambda: AscendSFAIndexerBackend
    )
    config = make_kvpp_config(2)
    config.speculative_config = None
    # The sparse-SFA-C8 reshape resolves the cache dtype from cache_config,
    # and the packed MLA caches are int8; "auto" would fall back to the model
    # dtype and fail on this fake config.
    config.cache_config.cache_dtype = "int8"
    config.model_config.hf_config.model_type = "deepseek_v2"
    return config, specs, layers


def assert_attention_cache_views(caches, raw, packed):
    expected_dims = {
        layer_name(9): (16,) if packed else (8, 4),
        layer_name(10): (24,) if packed else (4, 4),
        layer_name(11): (16,) if packed else (8, 4),
        indexer_name(9): (4, 1),
    }
    assert set(caches) == set(expected_dims)
    for name, dims in expected_dims.items():
        assert len(caches[name]) == len(dims)
        for component, (view, raw_part, dim) in enumerate(zip(caches[name], raw[name], dims)):
            expected_dtype = torch.int8 if packed else torch.bfloat16
            if name == indexer_name(9):
                expected_dtype = torch.int8 if component == 0 else torch.float16
            assert view.dtype == expected_dtype
            assert view.shape == (3, 2, 1, dim)
            assert view.untyped_storage().data_ptr() == raw_part.untyped_storage().data_ptr()
            assert view.storage_offset() * view.element_size() == raw_part.storage_offset()
            view.view(torch.int8).reshape(-1)[-1] = 23
            assert raw_part[-1].item() == 23
