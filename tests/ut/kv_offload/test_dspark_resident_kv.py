# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    allocate_kv_offload_topk_buffer_pair,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _mla_spec(*, host=True, non_causal=False, block_size=128):
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        store_on_host=host,
        non_causal_multi_token_decode=non_causal,
    )


def _runner(*, method="dspark", sparse=True, last_rank=True, names=None):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
    runner.ascend_config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=sparse))
    runner.is_last_pp_rank = last_rank
    runner.speculator = SimpleNamespace(draft_attn_layer_names={"draft.layers.0.attn"} if names is None else names)
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.vllm_config.compilation_config = runner.compilation_config
    runner.shared_kv_cache_layers = {}
    return runner


def _prefill_runner(
    *,
    ids=None,
    producer=True,
    consumer=False,
    speculative=None,
    first_rank=True,
    pp=True,
    eager=True,
    cp=1,
    prefix=False,
    connector="SfaRemoteD2HConnector",
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    extra = {} if ids is None else {"dspark_aux_hidden_state_layer_ids": ids}
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector, kv_connector_extra_config=extra, is_kv_producer=producer, is_kv_consumer=consumer
        ),
        speculative_config=speculative,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=cp, decode_context_parallel_size=1),
    )
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=78), enforce_eager=eager, dtype=torch.bfloat16
    )
    runner.vllm_config.model_config = runner.model_config
    runner.model = SimpleNamespace(set_aux_hidden_state_layers=Mock(), make_empty_intermediate_tensors=Mock())
    runner.pp_handler = SimpleNamespace(configure_aux_hidden_state_relay=Mock())
    runner.use_pp = pp
    runner.is_first_pp_rank = first_rank
    runner.is_last_pp_rank = not pp or not first_rank
    runner.max_num_tokens = 16
    runner.device = torch.device("cpu")
    runner.use_aux_hidden_state_outputs = False
    runner.pd_dspark_aux_layer_ids = ()
    runner.intermediate_tensors = object()
    runner.speculator = SimpleNamespace() if speculative is not None and runner.is_last_pp_rank else None
    return runner


@pytest.mark.parametrize("method", [None, "mtp", "eagle3"])
def test_no_prefill_aux_option_preserves_original_loading(method):
    speculative = SimpleNamespace(method=method) if method is not None else None
    runner = _prefill_runner(speculative=speculative)
    original_buffer = runner.intermediate_tensors
    with patch.object(GPUModelRunner, "load_model") as parent:
        runner.load_model(True)
    parent.assert_called_once_with(True)
    assert runner.pd_dspark_aux_layer_ids == ()
    assert runner.intermediate_tensors is original_buffer
    assert not runner.use_aux_hidden_state_outputs
    runner.model.set_aux_hidden_state_layers.assert_not_called()


@pytest.mark.parametrize("first_rank,pp", [(True, False), (True, True), (False, True)])
def test_prefill_draft_kv_uses_parent_loaded_drafter_and_aux_states(first_rank, pp):
    layers = [2, 22, 38, 58, 74]
    speculative = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=layers)),
    )
    runner = _prefill_runner(speculative=speculative, first_rank=first_rank, pp=pp)
    original_buffer = runner.intermediate_tensors

    def parent_load(*args):
        assert runner.pd_dspark_aux_layer_ids == ()
        runner.use_aux_hidden_state_outputs = True

    with patch.object(GPUModelRunner, "load_model", side_effect=parent_load) as parent:
        runner.load_model()
    parent.assert_called_once_with(False)
    assert runner.pd_dspark_aux_layer_ids == tuple(layers)
    assert runner.use_aux_hidden_state_outputs
    assert (runner.speculator is not None) is runner.is_last_pp_rank
    assert runner.intermediate_tensors is original_buffer
    runner.model.set_aux_hidden_state_layers.assert_not_called()
    runner.model.make_empty_intermediate_tensors.assert_not_called()
    runner.pp_handler.configure_aux_hidden_state_relay.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_prefill_load_preserves_parent_validation_errors(error_type):
    speculative = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 22])),
    )
    runner = _prefill_runner(speculative=speculative, first_rank=False, pp=True)
    with (
        patch.object(GPUModelRunner, "load_model", side_effect=error_type("parent validation failed")) as parent,
        pytest.raises(error_type, match="parent validation failed"),
    ):
        runner.load_model()
    parent.assert_called_once_with(False)
    assert runner.pd_dspark_aux_layer_ids == ()


@pytest.mark.parametrize("ids", [[], [38, 22], [2, 2], [79], [-1], [True], "2,22,38"])
def test_invalid_prefill_aux_boundaries_fail_before_loading(ids):
    speculative = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=ids)),
    )
    runner = _prefill_runner(speculative=speculative)
    with patch.object(GPUModelRunner, "load_model") as parent, pytest.raises(ValueError, match="boundaries"):
        runner.load_model()
    parent.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [{"eager": False}, {"cp": 2}, {"prefix": True}],
)
def test_prefill_aux_capture_rejects_unsupported_topologies(options):
    speculative = SimpleNamespace(method="dspark")
    runner = _prefill_runner(speculative=speculative, **options)
    with patch.object(GPUModelRunner, "load_model") as parent, pytest.raises(ValueError):
        runner.load_model()
    parent.assert_not_called()


def test_no_aux_capture_preserves_local_prefix_caching():
    runner = _prefill_runner(prefix=True)
    with patch.object(GPUModelRunner, "load_model") as parent:
        runner.load_model()
    parent.assert_called_once_with(False)
    assert not runner.use_aux_hidden_state_outputs


def test_other_pd_backend_preserves_original_dspark_loading():
    runner = _prefill_runner(
        connector="MooncakeConnector", speculative=SimpleNamespace(method="dspark"), eager=False, cp=2, prefix=True
    )
    with patch.object(GPUModelRunner, "load_model") as parent:
        runner.load_model()
    parent.assert_called_once_with(False)
    assert runner.pd_dspark_aux_layer_ids == ()


@pytest.mark.parametrize("connector", [None, "MooncakeConnector", "SfaRemoteD2HConnector", "AscendStoreConnector"])
@pytest.mark.parametrize("method", ["dspark", "mtp", "eagle3"])
@pytest.mark.parametrize("layerwise", [False, True])
def test_worker_registers_draft_ownership_before_sfa_or_layerwise_connector_construction(connector, method, layerwise):
    from vllm_ascend.worker import worker as module

    worker = module.NPUWorker.__new__(module.NPUWorker)
    transfer = (
        SimpleNamespace(
            kv_connector=connector,
            kv_connector_extra_config={"backend": "memcache", "use_layerwise": layerwise},
            is_kv_producer=True,
            is_kv_consumer=False,
        )
        if connector is not None
        else None
    )
    worker.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        kv_transfer_config=transfer,
        model_config=SimpleNamespace(enable_sleep_mode=False),
    )
    worker.model_runner = SimpleNamespace(
        speculator=SimpleNamespace(draft_attn_layer_names={"draft.attn"}), initialize_kv_cache=Mock()
    )
    worker.use_v2_model_runner = True
    cache_config = SimpleNamespace(has_mamba_layers=False, needs_kv_cache_zeroing=False)
    enabled = method == "dspark" and (
        connector == "SfaRemoteD2HConnector" or connector == "AscendStoreConnector" and layerwise
    )

    def check_registration(config, cache):
        assert config is worker.vllm_config
        assert getattr(cache, "dspark_draft_layer_names", ()) == (("draft.attn",) if enabled else ())

    with patch.object(module, "ensure_kv_transfer_initialized", side_effect=check_registration) as initialize:
        worker.initialize_from_config(cache_config)
    initialize.assert_called_once_with(worker.vllm_config, cache_config)
    worker.model_runner.initialize_kv_cache.assert_called_once()
    assert hasattr(cache_config, "dspark_draft_layer_names") is enabled


@pytest.mark.parametrize("pd_connector", ["SfaRemoteD2HConnector", "MooncakeConnector"])
@pytest.mark.parametrize("method", ["dspark", "mtp", "eagle3"])
def test_worker_excludes_persistent_draft_from_layerwise_budget_for_any_dspark_connector(pd_connector, method):
    from vllm_ascend.worker import worker as module

    worker = module.NPUWorker.__new__(module.NPUWorker)
    worker.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={"connectors": [{"kv_connector": pd_connector}]},
            is_kv_producer=True,
            is_kv_consumer=False,
        ),
    )
    specs = {"target.attn": _mla_spec(), "draft.attn": _mla_spec()}
    worker.model_runner = SimpleNamespace(
        speculator=SimpleNamespace(draft_attn_layer_names={"draft.attn"}),
        get_kv_cache_spec=Mock(return_value=specs),
    )
    worker._get_layerwise_kv_cache_memory_info = Mock(return_value=(1, 1, 1.0))
    with (
        patch.object(module, "get_layerwise_reuse_config", return_value={}),
        patch.object(module.KVPPConfig, "from_vllm_config", return_value=SimpleNamespace(size=1)),
        patch.object(
            module,
            "get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        ),
    ):
        assert worker.get_kv_cache_spec() is specs
    excluded = {"draft.attn"} if method == "dspark" else set()
    worker._get_layerwise_kv_cache_memory_info.assert_called_once_with(specs, {}, excluded_layer_names=excluded)


def test_prefill_aux_config_delegates_to_backend_before_loading():
    from vllm_ascend.worker.v2 import model_runner as module

    runner = _prefill_runner()
    with (
        patch.object(
            module, "get_pd_dspark_aux_layer_ids", side_effect=ValueError("backend rejected config")
        ) as backend,
        patch.object(GPUModelRunner, "load_model") as parent,
        pytest.raises(ValueError, match="backend rejected config"),
    ):
        runner.load_model()
    backend.assert_called_once_with(runner.vllm_config)
    parent.assert_not_called()


@pytest.mark.parametrize("method", ["dspark", "mtp"])
@pytest.mark.parametrize("sparse", [False, True])
def test_draft_placement_changes_only_dspark_sparse(method, sparse):
    runner = _runner(method=method, sparse=sparse)
    target = _mla_spec()
    draft = _mla_spec(non_causal=True)
    original = {"model.layers.0.attn": target, "draft.layers.0.attn": draft}
    with patch.object(GPUModelRunner, "get_kv_cache_spec", return_value=original.copy()):
        result = runner.get_kv_cache_spec()
    assert result["model.layers.0.attn"] is target
    resident = method == "dspark" and sparse
    assert result["draft.layers.0.attn"].store_on_host is not resident
    assert result["draft.layers.0.attn"].non_causal_multi_token_decode
    assert draft.store_on_host  # No mutation of upstream specs.


@pytest.mark.parametrize("has_shared_layers", [False, True])
def test_dspark_ownership_is_exact_even_with_target_like_names(has_shared_layers):
    names = {"model.layers.3.attn", "head.cache"}
    runner = _runner(names=names)
    if not has_shared_layers:
        del runner.shared_kv_cache_layers
    specs = {name: _mla_spec() for name in [*sorted(names), "draft.layers.0.attn"]}
    with patch.object(GPUModelRunner, "get_kv_cache_spec", return_value=specs.copy()):
        result = runner.get_kv_cache_spec()
    assert all(not result[name].store_on_host for name in names)
    assert result["draft.layers.0.attn"].store_on_host


def test_dspark_non_last_pp_rank_has_no_resident_draft():
    runner = _runner(last_rank=False, names=set())
    runner.speculator = None
    specs = {"target.attn": _mla_spec()}
    with patch.object(GPUModelRunner, "get_kv_cache_spec", return_value=specs):
        assert runner.get_kv_cache_spec() is specs
    assert specs["target.attn"].store_on_host


@pytest.mark.parametrize("names", [None, set()])
def test_dspark_missing_loaded_ownership_fails_closed(names):
    runner = _runner()
    runner.speculator = SimpleNamespace(draft_attn_layer_names=names)
    with (
        patch.object(GPUModelRunner, "get_kv_cache_spec", return_value={}),
        pytest.raises(ValueError, match="loaded draft model"),
    ):
        runner.get_kv_cache_spec()


def test_dspark_rejects_gqa_checkpoint_cache():
    runner = _runner()
    spec = FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    with (
        patch.object(GPUModelRunner, "get_kv_cache_spec", return_value={"draft.layers.0.attn": spec}),
        pytest.raises(ValueError, match="MLA DSpark"),
    ):
        runner.get_kv_cache_spec()


@pytest.mark.parametrize("module_alias", [False, True])
def test_dspark_cannot_alias_target_kv(module_alias):
    runner = _runner()
    if module_alias:
        runner.compilation_config.static_forward_context = {
            "draft.layers.0.attn": SimpleNamespace(kv_sharing_target_layer_name="target.attn")
        }
    else:
        runner.shared_kv_cache_layers = {"draft.layers.0.attn": "target.attn"}
    with (
        patch.object(GPUModelRunner, "get_kv_cache_spec", return_value={}),
        pytest.raises(ValueError, match="share target-model"),
    ):
        runner.get_kv_cache_spec()


@pytest.mark.parametrize("producer,consumer", [(False, True), (False, False)])
def test_d_side_and_local_dspark_do_not_enable_p_prefill_generation(producer, consumer):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import get_pd_dspark_aux_layer_ids

    runner = _prefill_runner(
        speculative=SimpleNamespace(method="dspark"), producer=producer, consumer=consumer, eager=False
    )
    assert get_pd_dspark_aux_layer_ids(runner.vllm_config) == ()


def _cache_config(specs, num_blocks=3):
    groups = [KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


@pytest.mark.parametrize("method", ["dspark", "mtp"])
def test_offload_registry_follows_placement_not_name(method):
    specs = {
        "model.layers.0.attn": _mla_spec(),
        "model.layers.1.attn": _mla_spec(),
        "draft.layers.0.attn": _mla_spec(host=method == "mtp", non_causal=True),
        "some.indexer": AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.int8),
    }
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.kv_cache_config = _cache_config(specs)
    manager.num_target_layers = 2
    manager.tp_rank = 0
    manager._register_offload_layers(dict.fromkeys(specs, object()))
    assert manager.offload_layer_names == [
        name for name, spec in specs.items() if getattr(spec, "store_on_host", False)
    ]
    assert manager.mtp_layer_id == (2 if method == "mtp" else -1)
    assert manager._infer_group_block_sizes(manager.kv_cache_config) == 128


def test_mixed_target_draft_block_sizes_are_not_silently_accepted():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    config = _cache_config({"target": _mla_spec(), "draft": _mla_spec(host=False, block_size=16)})
    with pytest.raises(ValueError, match="shared block size"):
        manager._infer_group_block_sizes(config)


def test_resident_draft_is_counted_in_hbm_not_host_budget():
    target = _mla_spec()
    draft = _mla_spec(host=False, non_causal=True)
    indexer = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.int8)
    specs = {"target": target, "draft": draft, "indexer": indexer}
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    device_page_bytes = draft.page_size_bytes + indexer.page_size_bytes
    budget = plan_sparse_kv_offload_memory(specs, runtime, device_page_bytes * 2, 1 << 30, False)
    assert budget.limiting_factor == "npu"
    assert budget.final_num_blocks == 2
    assert budget.planned_device_bytes == 2 * device_page_bytes
    assert budget.planned_host_bytes == 2 * target.page_size_bytes
    config = _cache_config(specs)
    host_only = _cache_config({"target": target})
    assert get_sparse_kv_offload_cpu_pool_size_bytes(config) == get_sparse_kv_offload_cpu_pool_size_bytes(host_only)


@pytest.mark.parametrize("width", [1, 3, 7, 9])
@pytest.mark.parametrize("fused,keep_device", [(False, False), (True, False), (True, True)])
def test_hot_cache_rows_follow_request_slots_not_query_width(monkeypatch, width, fused, keep_device):
    runtime = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=width - 1),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64),
        cache_config=SimpleNamespace(block_size=128),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=4, qk_rope_head_dim=2)),
    )
    config = SimpleNamespace(topk_buffer_size=256, use_fused_copy_sfa=fused, keep_device_kv_cache=keep_device)
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        assert kwargs["device"] == "npu"
        return original_empty(*args, **{**kwargs, "device": "cpu"})

    monkeypatch.setattr(torch, "empty", cpu_empty)
    k, v = allocate_kv_offload_topk_buffer_pair(runtime, config)
    query_rows = min(64, 4 * width)
    request_rows = 2 * (4 + 2)
    expected_rows = (max(query_rows, request_rows) if keep_device else request_rows) if fused else query_rows
    stride = 512 if fused else 256
    assert k.shape == (expected_rows, stride, 1, 4)
    assert v.shape == (expected_rows, stride, 1, 2)
    assert v.data_ptr() == k.data_ptr() + k.numel() * k.element_size()
    if fused:
        assert not torch.count_nonzero(k) and not torch.count_nonzero(v)  # Private dummy rows start initialized.


def test_real_allocator_separates_host_target_and_persistent_mla_draft(monkeypatch):
    specs = {"target.attn": _mla_spec(), "draft.layers.0.attn": _mla_spec(host=False, non_causal=True)}
    config = _cache_config(specs)
    config.kv_cache_tensors = [
        KVCacheTensor(
            size=3 * spec.page_size_bytes,
            layers=[name],
            layer_stride=3 * spec.page_size_bytes,
            block_stride=spec.page_size_bytes,
            offset=0,
        )
        for name, spec in specs.items()
    ]
    runtime = SimpleNamespace(kv_transfer_config=None)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "is_deepseek_v41_cache", lambda _: False)
    monkeypatch.setattr(attn_utils, "get_layerwise_reuse_config", lambda _: None)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda _: False)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_: (512, 64))
    monkeypatch.setattr(attn_utils, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=False)),
    )
    host_calls = []
    host_marker = object()

    def allocate_host(k_bytes, v_bytes, alignment, rank, keep_device, allocate_device):
        host_calls.append((k_bytes, v_bytes, rank, keep_device))
        return host_marker

    monkeypatch.setattr(attn_utils, "allocate_kv_cache_tensors_for_sparse_kv_offload", allocate_host)
    raw = attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))
    assert raw["target.attn"] is host_marker
    draft_k, draft_v = raw["draft.layers.0.attn"]
    assert host_calls == [(3 * 128 * 512 * 2, 3 * 128 * 64 * 2, 0, False)]
    assert draft_k.numel() + draft_v.numel() == 3 * specs["draft.layers.0.attn"].page_size_bytes
    assert draft_k.data_ptr() != draft_v.data_ptr()
    assert draft_k.device.type == draft_v.device.type == "cpu"  # Requested device in this allocator UT.


def test_layerwise_reuse_retains_independent_draft_descriptors():
    target_names = [f"model.layers.{index}.attn" for index in range(5)]
    draft_names = [f"draft.layers.{index}.attn" for index in range(2)]
    names = target_names + draft_names
    spec = _mla_spec(host=False)
    layer_bytes = 3 * spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(names) * layer_bytes,
                layers=names,
                layer_stride=layer_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(get_num_layers=lambda _: 5),
        parallel_config=None,
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"backend": "memcache", "use_layerwise": True, "layerwise_num_shared_buffers": 1},
        ),
    )
    apply_layerwise_kv_cache_plan(config, runtime, excluded_layer_names=set(draft_names))
    assert len(config.kv_cache_tensors) == 4  # Independent target 0, shared targets 1..4, two draft layers.
    for name in draft_names:
        descriptor = next(tensor for tensor in config.kv_cache_tensors if name in tensor.layers)
        assert descriptor.layers == [name]
        assert descriptor.size == layer_bytes
    assert sorted(name for tensor in config.kv_cache_tensors for name in tensor.layers) == sorted(names)
