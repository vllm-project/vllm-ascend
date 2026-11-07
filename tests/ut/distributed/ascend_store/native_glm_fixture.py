# SPDX-License-Identifier: Apache-2.0
"""Real GLM cache specs/views and CPU factory boundaries, without model weights."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.config import CacheConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import scheduler as native_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker as native_worker

from vllm_ascend.distributed.kv_transfer import _register_with_class_name_alias
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import native_worker as ascend_worker


def make_native_registration_inputs(*, use_recoverssm=False):
    """Return (config, cache_config, cache_tensors, layer) with real CPU views."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

    from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
    from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention

    # Execute the real loaded-layer shape/dtype methods without allocating model weights.
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.tp_size, layer.num_heads, layer.head_dim = 2, 4, 2
    layer.conv_size, layer.num_spec, layer.use_recoverssm = 4, 2, use_recoverssm
    layer._conv_state_dim_first = is_conv_state_dim_first()
    layer.model_config = SimpleNamespace(dtype=torch.float32)
    layer.cache_config = SimpleNamespace(mamba_cache_dtype="float32", mamba_ssm_cache_dtype="float32")
    shapes, dtypes = layer.get_state_shape(), layer.get_state_dtype()
    states = tuple(torch.empty((5, *shape), dtype=dtype) for shape, dtype in zip(shapes, dtypes, strict=True))
    layer.kv_cache = states
    state_spec = MambaSpec(block_size=8, shapes=shapes, dtypes=dtypes, mamba_type=MambaAttentionBackendEnum.GDN_ATTN)
    tail_spec = AscendIndexerKPoolTailSpec(
        block_size=8, num_kv_heads=1, head_size=8, dtype=torch.float32, sliding_window=2, compress_ratio=2
    )
    c8_spec = AscendMLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        num_heads=4,
        head_size=528,
        dtype=torch.int8,
        cache_sparse_sfa_c8=True,
        model_version="glm5_next",
    )
    groups = [
        KVCacheGroupSpec(["tail"], tail_spec),
        KVCacheGroupSpec(["c8"], c8_spec, host_resident=True),
        KVCacheGroupSpec(["kda"], state_spec),
    ]
    cache_config = KVCacheConfig(num_blocks=5, kv_cache_tensors=[], kv_cache_groups=groups, hisparse_host_num_blocks=3)
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="/models/glm-test", hf_text_config=SimpleNamespace(model_type="glm5_next")),
        speculative_config=None,
        num_hash_lookahead_tokens=0,
        compilation_config=SimpleNamespace(static_forward_context={"kda": layer}),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"store_tp_size": 4, "cache_prefix": "test"}),
    )
    c8 = torch.empty((3, 8, 1, 528), dtype=torch.int8)
    tail = torch.full((5, 8, 1, 16), 7.0)
    cache_tensors = {"tail": tail, "c8": (c8, torch.empty(0)), "kda": states}
    return config, cache_config, cache_tensors, layer


def make_glm_connectors(monkeypatch, tmp_path, *, use_recoverssm=False):
    config, cache_config, tensors, layer = make_native_registration_inputs(use_recoverssm=use_recoverssm)
    groups = list(cache_config.kv_cache_groups)
    groups[2] = replace(groups[2], kv_cache_spec=replace(groups[2].kv_cache_spec, mamba_cache_mode="align"))
    cache_config = replace(cache_config, kv_cache_groups=groups)
    config.parallel_config = SimpleNamespace(
        data_parallel_index=0,
        rank=0,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        tensor_parallel_size=2,
        world_size=2,
    )
    config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    config.cache_config = CacheConfig(block_size=8, prefix_match_unit=4, enable_prefix_caching=True)
    config.cache_config.kv_cache_layout = "LBNHC"
    config.kv_events_config = None
    config.model_config.get_num_layers = lambda parallel: 2
    config.model_config.get_total_num_kv_heads = lambda: 1
    config.model_config.use_mla = True
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={"store_tp_size": 4, "cache_prefix": "factory", "load_async": True},
    )
    store_config = tmp_path / "store.json"
    store_config.write_text(
        '{"master_server_address":"127.0.0.1:1","protocol":"ascend","global_segment_size":65536,"local_buffer_size":0}'
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(store_config))
    backend = Mock()
    backend._completion_unknown = False
    backend.config.prefer_alloc_in_same_node = True
    backend.config.preferred_segment = False
    monkeypatch.setattr(ascend_worker, "MooncakeBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(native_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(native_worker, "get_tensor_model_parallel_world_size", lambda: 2)
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(native_worker, "get_pcp_group", lambda: group)
    monkeypatch.setattr(native_worker, "get_dcp_group", lambda: group)
    monkeypatch.setattr(native_worker, "LookupKeyServer", Mock())
    monkeypatch.setattr(native_scheduler, "LookupKeyClient", Mock())
    for cls in (ascend_worker.AscendNativeStoreSendingThread, ascend_worker.AscendNativeStoreRecvingThread):
        monkeypatch.setattr(cls, "start", lambda self: self.ready_event.set())
    monkeypatch.setattr(KVConnectorFactory, "_registry", dict(KVConnectorFactory._registry))
    _register_with_class_name_alias(
        "MooncakeStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.native_connector",
        "AscendNativeStoreConnector",
        override=True,
    )
    worker_connector = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, cache_config)
    scheduler_connector = KVConnectorFactory.create_connector(config, KVConnectorRole.SCHEDULER, cache_config)
    return worker_connector, scheduler_connector, cache_config, tensors, backend, layer
