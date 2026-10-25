# SPDX-License-Identifier: Apache-2.0
"""DSpark P-side draft KV production and direct P-to-D page transfer."""

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextDescriptor,
    DSparkContextReceiver,
    DSparkContextSubmission,
    bind_dspark_context_receiver,
    configure_dspark_kv_transfer,
    find_dspark_context_connector,
    find_dspark_kv_connector,
    find_dspark_prefix_connector,
    resident_mla_context_group_ids,
    send_dspark_prefill_kv,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import (
    DraftKVCacheMetadata,
    build_draft_kv_metadata,
    build_draft_kv_read_batches,
)


def _descriptor(request_id="request", generation="allocation-1", prompt_tokens=6):
    return DSparkContextDescriptor(request_id, generation, prompt_tokens, (2, 22, 38, 58, 74), 4)


def _context_scheduler(draft_spec, *, method="dspark", wrapped=False):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import SFAPDRD2HScheduler

    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=None if method is None else SimpleNamespace(method=method)
    )
    scheduler.main_group_idx, scheduler.indexer_group_idx = 0, 1
    host_spec = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=True
    )
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=host_spec),
            SimpleNamespace(kv_cache_spec=SimpleNamespace()),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(kv_cache_specs={"draft": draft_spec}) if wrapped else draft_spec
            ),
        ]
    )
    return scheduler


@pytest.mark.parametrize("non_causal", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_scheduler_discovers_resident_mla_independent_of_attention_causality(non_causal, wrapped):
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        store_on_host=False,
        non_causal_multi_token_decode=non_causal,
    )
    assert _context_scheduler(spec, wrapped=wrapped)._find_dspark_context_groups() == (2,)
    assert spec.non_causal_multi_token_decode is non_causal


@pytest.mark.parametrize("method", [None, "mtp"])
def test_scheduler_does_not_require_dspark_groups_on_legacy_paths(method):
    assert _context_scheduler(SimpleNamespace(), method=method)._find_dspark_context_groups() == ()


def test_scheduler_conversion_retains_resident_draft_in_shared_block_group():
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    from vllm_ascend.patch.platform import patch_kv_cache_utils

    host_spec = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=True
    )
    draft_spec = replace(host_spec, store_on_host=False)
    uniform = UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"target": host_spec, "draft": draft_spec})
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["target", "draft"], kv_cache_spec=uniform)],
    )
    vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method="dspark"), kv_transfer_config=None)
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_config_from_groups", lambda *a, **k: config)
        monkeypatch.setattr(patch_kv_cache_utils, "is_deepseek_v41_cache", lambda *a, **k: False)
        monkeypatch.setattr(patch_kv_cache_utils, "_get_glm5_next_cache_layout", lambda *a, **k: None)
        monkeypatch.setattr(patch_kv_cache_utils, "_is_deepseek_v4_groups", lambda *a, **k: False)
        physical = patch_kv_cache_utils._ascend_get_kv_cache_config_from_groups(vllm_config, config.kv_cache_groups, 1)
    assert physical.dspark_context_group_ids == (0,)
    scheduler_config = generate_scheduler_kv_cache_config([physical])
    assert scheduler_config.kv_cache_groups[0].kv_cache_spec.store_on_host is True
    assert scheduler_config.dspark_context_group_ids == (0,)
    scheduler = _context_scheduler(draft_spec)
    scheduler.kv_cache_config = scheduler_config
    scheduler.main_group_idx = scheduler.indexer_group_idx = 0
    assert scheduler._find_dspark_context_groups() == (0,)
    assert physical.kv_cache_groups[0].kv_cache_spec is uniform
    assert resident_mla_context_group_ids(physical.kv_cache_groups) == (0,)


@pytest.mark.parametrize("group_ids", [(3,), (-1,), (True,), (0, 0)])
def test_scheduler_rejects_corrupt_resident_group_ownership(group_ids):
    scheduler = _context_scheduler(SimpleNamespace())
    scheduler.kv_cache_config.dspark_context_group_ids = group_ids
    with pytest.raises(RuntimeError, match="ownership is invalid"):
        scheduler._find_dspark_context_groups()


@pytest.mark.parametrize("nested", [False, True])
def test_multiconnector_routes_context_to_its_exact_child_metadata(nested):
    pd = SimpleNamespace(get_dspark_context_descriptor=Mock(), send_dspark_draft_kv=Mock())
    pd_meta = object()
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(), pd])
    metadata = SimpleNamespace(metadata=(object(), pd_meta))
    if nested:
        wrapper = SimpleNamespace(_connectors=[wrapper])
        metadata = SimpleNamespace(metadata=(metadata,))
    assert find_dspark_context_connector(wrapper, metadata) == (pd, pd_meta)


@pytest.mark.parametrize("nested", [False, True])
def test_prefix_store_metadata_route_is_independent_of_sfa_child_order(nested):
    store = SimpleNamespace(dspark_prefix_cache_enabled=True)
    store_metadata = object()
    wrapper = SimpleNamespace(_connectors=[store, SimpleNamespace()])
    metadata = SimpleNamespace(metadata=(store_metadata, object()))
    if nested:
        wrapper = SimpleNamespace(_connectors=[SimpleNamespace(), wrapper])
        metadata = SimpleNamespace(metadata=(object(), metadata))
    assert find_dspark_prefix_connector(wrapper, metadata) == (store, store_metadata)


def test_unconfigured_prefix_provider_has_no_metadata_route():
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(dspark_prefix_cache_enabled=False)])
    assert find_dspark_prefix_connector(wrapper, SimpleNamespace(metadata=(object(),))) is None


def test_kv_connector_lookup_finds_unique_sfa_child():
    pd = SimpleNamespace(configure_dspark_draft_layers=Mock(), send_dspark_draft_kv=Mock())
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(_connectors=[pd])])
    assert find_dspark_kv_connector(wrapper) is pd


def _transfer_config(connector="SfaRemoteD2HConnector", method="dspark", *, producer=True):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method) if method is not None else None,
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector,
            kv_connector_extra_config={},
            is_kv_producer=producer,
            is_kv_consumer=not producer,
        ),
    )


@pytest.mark.parametrize("producer", [False, True])
def test_backend_registers_loader_owned_draft_names_on_both_pd_roles(producer):
    config = _transfer_config(producer=producer)
    draft = SimpleNamespace(draft_attn_layer_names={"draft.z", "draft.a"})
    cache_config = SimpleNamespace()
    pd = SimpleNamespace(configure_dspark_draft_layers=Mock(), send_dspark_draft_kv=Mock())
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(), pd])
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context.get_kv_transfer_group",
        return_value=wrapper,
    ):
        configure_dspark_kv_transfer(config, draft, cache_config, is_last_pp_rank=True)
    assert cache_config.dspark_draft_layer_names == ("draft.a", "draft.z")
    pd.configure_dspark_draft_layers.assert_called_once_with(("draft.a", "draft.z"))


@pytest.mark.parametrize(
    "connector,method,last_rank",
    [
        ("MooncakeConnector", "dspark", True),
        ("AscendStoreConnector", "dspark", True),
        ("SfaRemoteD2HConnector", "mtp", True),
        ("SfaRemoteD2HConnector", None, True),
        ("SfaRemoteD2HConnector", "dspark", False),
        ("MultiConnector", "dspark", True),
    ],
)
def test_backend_opt_out_neither_inspects_draft_nor_mutates_cache(connector, method, last_rank):
    config = _transfer_config(connector, method)
    cache_config = SimpleNamespace()
    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context.get_kv_transfer_group") as group:
        configure_dspark_kv_transfer(config, None, cache_config, is_last_pp_rank=last_rank)
    group.assert_not_called()
    assert not hasattr(cache_config, "dspark_draft_layer_names")


@pytest.mark.parametrize("draft", [None, SimpleNamespace(draft_attn_layer_names=set())])
def test_selected_backend_still_rejects_missing_loaded_draft(draft):
    with pytest.raises(RuntimeError, match="draft"):
        configure_dspark_kv_transfer(_transfer_config(), draft, SimpleNamespace(), is_last_pp_rank=True)


@pytest.mark.parametrize("count", [0, 2])
def test_selected_backend_still_requires_exactly_one_runtime_connector(count):
    draft = SimpleNamespace(draft_attn_layer_names={"draft.attn"})
    children = [
        SimpleNamespace(configure_dspark_draft_layers=Mock(), send_dspark_draft_kv=Mock()) for _ in range(count)
    ]
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context.get_kv_transfer_group",
            return_value=SimpleNamespace(_connectors=children),
        ),
        pytest.raises(RuntimeError, match="exactly one SFA"),
    ):
        configure_dspark_kv_transfer(_transfer_config(), draft, SimpleNamespace(), is_last_pp_rank=True)
    for child in children:
        child.configure_dspark_draft_layers.assert_not_called()


@pytest.mark.parametrize(
    "connector,method,producer,sparse,last_rank",
    [
        ("SfaRemoteD2HConnector", "dspark", False, True, True),
        ("SfaRemoteD2HConnector", "dspark", True, True, True),
        ("SfaRemoteD2HConnector", "dspark", False, False, True),
        ("SfaRemoteD2HConnector", "dspark", False, True, False),
        ("SfaRemoteD2HConnector", "mtp", False, True, True),
        ("SfaRemoteD2HConnector", None, False, True, True),
        ("MooncakeConnector", "dspark", False, True, True),
        (None, "dspark", False, True, True),
    ],
)
def test_backend_receiver_binds_only_sparse_dspark_sfa_consumer(connector, method, producer, sparse, last_rank):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import dspark_context as module

    config = _transfer_config(connector, method, producer=producer)
    if connector is None:
        config.kv_transfer_config = None
    pd = SimpleNamespace(
        configure_dspark_draft_layers=Mock(), send_dspark_draft_kv=Mock(), bind_dspark_context_receiver=Mock()
    )
    receiver = object()
    with (
        patch.object(module, "get_kv_transfer_group", return_value=pd) as group,
        patch.object(module, "DSparkContextReceiver", return_value=receiver) as factory,
    ):
        bind_dspark_context_receiver(config, sparse_offload_enabled=sparse, is_last_pp_rank=last_rank, max_requests=4)
    enabled = connector == "SfaRemoteD2HConnector" and method == "dspark" and not producer and sparse and last_rank
    if enabled:
        group.assert_called_once_with()
        factory.assert_called_once_with(max_requests=4)
        pd.bind_dspark_context_receiver.assert_called_once_with(receiver)
    else:
        group.assert_not_called()
        factory.assert_not_called()
        pd.bind_dspark_context_receiver.assert_not_called()


def test_context_receiver_joins_target_and_draft_kv_and_keeps_lost_ack_tombstone():
    receiver = DSparkContextReceiver(max_requests=1)
    descriptor = _descriptor()
    receiver.register_request(descriptor)

    admission, reserved = receiver.begin_direct_transfer(descriptor)
    assert (admission, reserved) == (DSparkContextSubmission.ACCEPTED, True)
    assert receiver.begin_direct_transfer(descriptor) == (DSparkContextSubmission.BACKPRESSURE, False)
    receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    assert receiver.ready_requests() == set()

    receiver.finish_direct_transfer(descriptor, success=True)
    assert receiver.ready_requests() == {descriptor.request_id}
    receiver.retire_ready_request(descriptor.request_id, descriptor.generation)
    assert receiver.ready_requests() == set()
    assert receiver.begin_direct_transfer(descriptor) == (DSparkContextSubmission.ACCEPTED, False)

    receiver.discard_request_id(descriptor.request_id)
    next_descriptor = _descriptor(generation="allocation-2")
    receiver.register_request(next_descriptor)
    assert receiver.begin_direct_transfer(descriptor) == (DSparkContextSubmission.STALE, False)


def test_context_receiver_quarantines_failed_direct_copy():
    receiver = DSparkContextReceiver(max_requests=1)
    descriptor = _descriptor()
    receiver.register_request(descriptor)
    _, reserved = receiver.begin_direct_transfer(descriptor)
    assert reserved
    receiver.finish_direct_transfer(descriptor, success=False)
    receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    assert receiver.ready_requests() == set()
    with pytest.raises(RuntimeError, match="quarantined"):
        receiver.begin_direct_transfer(descriptor)


def _cache_metadata(group_id, base, *, block_size=4, num_blocks=8):
    return DraftKVCacheMetadata(
        group_id=group_id,
        block_size=block_size,
        num_blocks=num_blocks,
        base_addrs=(base,),
        block_strides=(16,),
        block_lens=(16,),
        block_scales=(1,),
        shapes=((4, 2),),
        dtypes=("torch.bfloat16",),
    )


def test_cache_metadata_uses_registered_draft_layer_names():
    config = SimpleNamespace(
        num_blocks=3,
        dspark_draft_layer_names=("draft.self_attn",),
        kv_cache_groups=[SimpleNamespace(layer_names=["draft.self_attn"], kv_cache_spec=SimpleNamespace(block_size=4))],
    )
    metadata = build_draft_kv_metadata(config, {"draft.self_attn": torch.empty((3, 4, 2), dtype=torch.bfloat16)})
    item = metadata["draft.self_attn"]
    assert (item.group_id, item.block_size, item.num_blocks) == (0, 4, 3)
    assert item.block_strides == item.block_lens == (16,)
    assert item.shapes == ((4, 2),)


def test_read_batches_map_different_p_and_d_groups_and_copy_only_prompt_rows():
    remote = {"draft.self_attn": _cache_metadata(7, 1000, num_blocks=8)}
    local = {"draft.self_attn": _cache_metadata(2, 4000, num_blocks=8)}
    batches = list(
        build_draft_kv_read_batches(
            remote,
            local,
            {"draft.self_attn": (1, 2)},
            {2: (4, 5)},
            prompt_tokens=6,
        )
    )
    assert len(batches) == 1
    peer_ptrs, local_ptrs, lengths = batches[0]
    assert peer_ptrs == [1016, 1032]
    assert local_ptrs == [4064, 4080]
    assert lengths == [16, 8]


def _wire_metadata(item):
    return {
        "draft.self_attn": {
            "group_id": item.group_id,
            "block_size": item.block_size,
            "num_blocks": item.num_blocks,
            "base_addrs": item.base_addrs,
            "block_strides": item.block_strides,
            "block_lens": item.block_lens,
            "block_scales": item.block_scales,
            "shapes": item.shapes,
            "dtypes": item.dtypes,
        }
    }


def test_read_thread_pulls_p_draft_pages_before_ack_with_independent_group_ids():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
        DSPARK_DRAFT_KV,
        DSPARK_DRAFT_KV_ACK,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import MembPullReadThread

    descriptor = _descriptor()
    receiver = DSparkContextReceiver(max_requests=1)
    receiver.register_request(descriptor)
    source, dest = _cache_metadata(7, 1000), _cache_metadata(2, 4000)
    reader = MembPullReadThread.__new__(MembPullReadThread)
    reader._state = SimpleNamespace(
        dspark_context_receiver=receiver,
        dspark_draft_kv_metadata={"draft.self_attn": dest},
        dspark_draft_blocks_by_req={"request": {2: (4, 5)}},
    )
    reader._p_sessions = {b"p": "p-session"}
    reader._p_pp_topology = {b"p": (1, 2)}
    reader._failed_requests = set()
    reader._lock = threading.Lock()
    reader.engine = SimpleNamespace(batch_transfer_sync_read=Mock(return_value=0))
    sock = Mock()
    message = (
        DSPARK_DRAFT_KV,
        descriptor.request_id,
        descriptor.generation,
        descriptor.prompt_tokens,
        descriptor.aux_layer_ids,
        descriptor.hidden_size,
        _wire_metadata(source),
        {7: (1, 2)},
    )

    reader._handle_dspark_draft_kv(b"p", message, sock, msgspec.msgpack.Encoder())

    reader.engine.batch_transfer_sync_read.assert_called_once_with("p-session", [4064, 4080], [1016, 1032], [16, 8])
    reply = msgspec.msgpack.decode(sock.send_multipart.call_args.args[0][2])
    assert reply == [DSPARK_DRAFT_KV_ACK, "request", "allocation-1", b"accepted"]
    assert receiver.begin_direct_transfer(descriptor) == (DSparkContextSubmission.ACCEPTED, False)


@pytest.mark.parametrize("split_prefill", [False, True])
def test_runner_writes_draft_kv_on_p_then_sends_only_page_metadata(monkeypatch, split_prefill):
    from vllm_ascend.worker.v2 import model_runner as module

    descriptor = _descriptor(prompt_tokens=71)
    pd = SimpleNamespace(
        get_dspark_context_descriptor=Mock(return_value=descriptor),
        send_dspark_draft_kv=Mock(),
    )
    request_meta = SimpleNamespace(
        dspark_context_generation=descriptor.generation,
        local_block_ids=[[index for index in range(20)]],
    )
    pd_meta = SimpleNamespace(requests={"request": request_meta})
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(), pd])
    metadata = SimpleNamespace(metadata=(SimpleNamespace(), pd_meta))
    monkeypatch.setattr(module, "get_kv_transfer_group", lambda: wrapper)

    runner = module.NPUModelRunner.__new__(module.NPUModelRunner)
    runner.pd_dspark_aux_layer_ids = descriptor.aux_layer_ids
    runner.is_last_pp_rank = True
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: descriptor.hidden_size)
    runner.speculator = SimpleNamespace(
        get_draft_context_group_layout=Mock(return_value=((0,), (0,), {0: 4})),
        initialize_local_context=Mock(),
    )
    runner._dspark_prefill_progress = {}
    aux = [torch.full((75, 4), index, dtype=torch.bfloat16) for index in range(5)]
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request"],
        prefill_len_np=[71],
        num_computed_tokens_np=[0],
        num_scheduled_tokens=[73],
        query_start_loc_np=[2, 75],
    )
    runner.execute_model_state = SimpleNamespace(aux_hidden_states=aux, input_batch=batch)

    if split_prefill:
        batch.num_scheduled_tokens = [64]
        request_meta.local_block_ids = [list(range(16))]
        runner._maybe_send_draft_kv(SimpleNamespace(kv_connector_metadata=metadata, finished_req_ids=set()))
        pd.send_dspark_draft_kv.assert_not_called()
        assert runner._dspark_prefill_progress == {"request": (descriptor.generation, 64)}
        batch.num_computed_tokens_np = [64]
        batch.num_scheduled_tokens = [9]
        batch.query_start_loc_np = [66, 75]
        request_meta.local_block_ids = [list(range(20))]

    runner._maybe_send_draft_kv(SimpleNamespace(kv_connector_metadata=metadata, finished_req_ids=set()))

    calls = runner.speculator.initialize_local_context.call_args_list
    assert [(call.args[0].token_offset, call.args[0].num_tokens) for call in calls] == [(0, 64), (64, 7)]
    features = torch.cat(aux, dim=-1)
    torch.testing.assert_close(calls[0].args[1], features[2:66])
    torch.testing.assert_close(calls[1].args[1], features[66:73])
    pd.send_dspark_draft_kv.assert_called_once_with("request", descriptor, {0: tuple(range(18))})
    assert runner._dspark_prefill_progress == {}


def _prefix_prefill_case(offset=4, prompt_tokens=12):
    descriptor = _descriptor(prompt_tokens=prompt_tokens)
    suffix_tokens = prompt_tokens - offset
    calls = []
    speculator = SimpleNamespace(
        get_draft_context_group_layout=Mock(return_value=((0,), (0,), {0: 4})),
        initialize_local_context=Mock(side_effect=lambda *args: calls.append(("project", args))),
    )
    connector = SimpleNamespace(
        get_dspark_context_descriptor=Mock(return_value=descriptor),
        send_dspark_draft_kv=Mock(side_effect=lambda *args: calls.append(("send", args))),
    )

    def restore_prefix(*args):
        calls.append(("restore", args))
        return offset

    provider = SimpleNamespace(
        restore_dspark_prefix=Mock(side_effect=restore_prefix),
        save_dspark_prefix=Mock(side_effect=lambda *args: calls.append(("save", args))),
    )
    aux = [torch.full((suffix_tokens, 4), index, dtype=torch.bfloat16) for index in range(5)]
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=[descriptor.request_id],
        prefill_len_np=[prompt_tokens],
        num_computed_tokens_np=[offset],
        num_scheduled_tokens=[suffix_tokens],
        query_start_loc_np=[0, suffix_tokens],
    )
    requests = {descriptor.request_id: SimpleNamespace(local_block_ids=[[3, 4, 5]])}
    return descriptor, speculator, connector, provider, aux, batch, requests, calls


@pytest.mark.parametrize("offset", [4, 8])
def test_prefix_restore_precedes_suffix_projection_and_complete_draft_transfer(offset):
    descriptor, speculator, connector, provider, aux, batch, requests, calls = _prefix_prefill_case(offset)
    metadata = object()
    progress: dict[str, tuple[str, int]] = {}

    send_dspark_prefill_kv(
        speculator,
        batch,
        aux,
        requests,
        connector,
        progress,
        (),
        prefix_connector=provider,
        prefix_metadata=metadata,
    )

    assert [name for name, _ in calls] == ["restore", "project", "save", "send"]
    blocks = {0: (3, 4, 5)}
    provider.restore_dspark_prefix.assert_called_once_with(metadata, descriptor.request_id, offset, blocks)
    chunk, features, source_blocks = speculator.initialize_local_context.call_args.args
    assert (chunk.token_offset, chunk.num_tokens) == (offset, descriptor.prompt_tokens - offset)
    assert source_blocks == blocks
    torch.testing.assert_close(features, torch.cat(aux, dim=-1))
    provider.save_dspark_prefix.assert_called_once_with(metadata, descriptor.request_id, 12, blocks)
    connector.send_dspark_draft_kv.assert_called_once_with(descriptor.request_id, descriptor, blocks)
    assert progress == {}


def test_prefix_is_restored_once_across_chunked_suffix_prefill():
    descriptor, speculator, connector, provider, aux, batch, requests, calls = _prefix_prefill_case()
    metadata = object()
    progress: dict[str, tuple[str, int]] = {}
    batch.num_scheduled_tokens = [4]
    batch.query_start_loc_np = [0, 4]
    send_dspark_prefill_kv(
        speculator,
        batch,
        [features[:4] for features in aux],
        requests,
        connector,
        progress,
        (),
        prefix_connector=provider,
        prefix_metadata=metadata,
    )
    assert progress == {descriptor.request_id: (descriptor.generation, 8)}
    connector.send_dspark_draft_kv.assert_not_called()

    batch.num_computed_tokens_np = [8]
    send_dspark_prefill_kv(
        speculator,
        batch,
        [features[4:] for features in aux],
        requests,
        connector,
        progress,
        (),
        prefix_connector=provider,
        prefix_metadata=metadata,
    )
    assert [name for name, _ in calls] == ["restore", "project", "save", "project", "save", "send"]
    provider.restore_dspark_prefix.assert_called_once()
    assert [
        (call.args[0].token_offset, call.args[0].num_tokens)
        for call in speculator.initialize_local_context.call_args_list
    ] == [
        (4, 4),
        (8, 4),
    ]
    assert [call.args[2] for call in provider.save_dspark_prefix.call_args_list] == [8, 12]
    assert progress == {}


@pytest.mark.parametrize("restored_tokens", [0, 3])
def test_incomplete_draft_prefix_proof_stops_before_projection(restored_tokens):
    _, speculator, connector, provider, aux, batch, requests, _ = _prefix_prefill_case()
    provider.restore_dspark_prefix.side_effect = None
    provider.restore_dspark_prefix.return_value = restored_tokens
    with pytest.raises(RuntimeError, match="chunks are not contiguous"):
        send_dspark_prefill_kv(
            speculator,
            batch,
            aux,
            requests,
            connector,
            {},
            (),
            prefix_connector=provider,
            prefix_metadata=object(),
        )
    speculator.initialize_local_context.assert_not_called()
    provider.save_dspark_prefix.assert_not_called()
    connector.send_dspark_draft_kv.assert_not_called()


def test_target_only_prefix_cannot_skip_missing_draft_context():
    _, speculator, connector, _, aux, batch, requests, _ = _prefix_prefill_case()
    with pytest.raises(RuntimeError, match="prefix|contiguous"):
        send_dspark_prefill_kv(speculator, batch, aux, requests, connector, {}, ())
    speculator.initialize_local_context.assert_not_called()
    connector.send_dspark_draft_kv.assert_not_called()


def test_failed_prefix_save_does_not_publish_pd_draft_transfer():
    _, speculator, connector, provider, aux, batch, requests, _ = _prefix_prefill_case()
    provider.save_dspark_prefix.side_effect = RuntimeError("draft prefix save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        send_dspark_prefill_kv(
            speculator,
            batch,
            aux,
            requests,
            connector,
            {},
            (),
            prefix_connector=provider,
            prefix_metadata=object(),
        )
    speculator.initialize_local_context.assert_called_once()
    connector.send_dspark_draft_kv.assert_not_called()


def test_context_slots_allow_incremental_p_prefill_allocation():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
        DSparkContextChunk,
        build_draft_context_slot_mappings,
    )

    chunk = DSparkContextChunk(_descriptor(prompt_tokens=71), 0, 64)
    kwargs = dict(
        draft_group_ids=(0,),
        block_sizes_by_group={0: 4},
        layer_group_ids=(0,),
        device=torch.device("cpu"),
    )
    slots = build_draft_context_slot_mappings(chunk, draft_block_ids_by_group={0: tuple(range(16))}, **kwargs)
    torch.testing.assert_close(slots[0], torch.arange(64, dtype=torch.int32))
    with pytest.raises(ValueError, match="prefill chunk"):
        build_draft_context_slot_mappings(chunk, draft_block_ids_by_group={0: tuple(range(15))}, **kwargs)


def test_p_runner_excludes_persistent_draft_from_target_scratch_plan(monkeypatch):
    from vllm_ascend.worker.v2 import model_runner as module

    runner = module.NPUModelRunner.__new__(module.NPUModelRunner)
    runner.vllm_config = _transfer_config()
    runner.ascend_config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    runner.is_last_pp_rank = True
    runner.speculator = SimpleNamespace(draft_attn_layer_names={"draft.layers.0.attn"})
    original = SimpleNamespace(dspark_draft_layer_names=("draft.layers.0.attn",))
    register = Mock()
    monkeypatch.setattr(module, "configure_dspark_kv_transfer", register)

    class PlanReached(Exception):
        pass

    def check_plan(config, runtime, *, excluded_layer_names):
        assert config is not original and runtime is runner.vllm_config
        register.assert_called_once_with(runtime, runner.speculator, config, is_last_pp_rank=True)
        assert excluded_layer_names == {"draft.layers.0.attn"}
        raise PlanReached

    monkeypatch.setattr(module, "apply_layerwise_kv_cache_plan", check_plan)
    with pytest.raises(PlanReached):
        runner.initialize_kv_cache(original)
