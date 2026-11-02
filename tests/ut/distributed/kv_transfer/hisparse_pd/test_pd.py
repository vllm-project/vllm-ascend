# SPDX-License-Identifier: Apache-2.0

import ctypes
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def layouts(m, buffers, *, source):
    result = []
    # Two main layers share a manager but have separate payloads. The indexer
    # has an expanded HMA view, a nonzero layer offset and a different stride.
    for index, (name, role, rows, width) in enumerate(
        (
            ("layer0", "hisparse_source", 4, 6),
            ("layer1", "hisparse_source", 4, 6),
            ("indexer", "hisparse_indexer", 4, 2),
        )
    ):
        stride = rows * width * (3 if source else 5)
        raw = torch.full((10 * stride + 17,), -19, dtype=torch.bfloat16)
        offset = index + 1
        view = torch.as_strided(raw, (10, rows, width), (stride, width, 1), offset)
        if source:
            view.copy_(torch.arange(view.numel()).reshape(view.shape) + 1000 * index)
        buffers[name] = raw, view
        result.append(
            m.protocol.LayerLayout(
                name,
                0 if index < 2 else 3,
                role,
                rows,
                width * 2,
                view.data_ptr(),
                stride * 2,
                10,
            )
        )
    return result


@pytest.mark.parametrize("start,end", [(0, 11), (3, 10), (4, 12), (9, 10)])
def test_real_strided_payload_partial_pages(modules, start, end):
    src, dst = {}, {}
    a, b = layouts(modules, src, source=True), layouts(modules, dst, source=False)
    source_ids, destination_ids = {0: [7, 1, 4], 3: [5, 2, 8]}, {0: [2, 8, 3], 3: [6, 1, 7]}
    expected = {name: raw.clone() for name, (raw, _) in dst.items()}
    for layout in b:
        view = dst[layout.name][1]
        target = torch.as_strided(expected[layout.name], view.shape, view.stride(), view.storage_offset())
        for token in range(start, end):
            page, offset = divmod(token, 4)
            target[destination_ids[layout.group_id][page], offset] = src[layout.name][1][
                source_ids[layout.group_id][page], offset
            ]
    for destination, source, count in modules.protocol.plan_reads(a, b, source_ids, destination_ids, start, end):
        ctypes.memmove(destination, source, count)
    for name, (raw, _) in dst.items():
        assert torch.equal(raw, expected[name]), name  # Includes all padding/other pages.


def test_64bit_addresses_and_reject_missing_geometry(modules):
    source = layouts(modules, {}, source=True)
    destination = layouts(modules, {}, source=False)
    source = [replace(layer, base=2**45 + i * 2**36, block_stride=2**33) for i, layer in enumerate(source)]
    reads = modules.protocol.plan_reads(source, destination, {0: [9], 3: [8]}, {0: [1], 3: [2]}, 0, 4)
    assert reads[0][1] == 2**45 + 9 * 2**33
    for invalid in (source[:-1], source + [source[0]], [replace(source[0], row_bytes=8), *source[1:]]):
        with pytest.raises(ValueError):
            modules.protocol.plan_reads(invalid, destination, {0: [9], 3: [8]}, {0: [1], 3: [2]}, 0, 4)
    with pytest.raises(ValueError, match="Missing block"):
        modules.protocol.plan_reads(source, destination, {0: [], 3: [8]}, {0: [1], 3: [2]}, 0, 4)


def test_lease_does_not_free_on_rank_mismatch_timeout_or_unfinished_read(modules, monkeypatch):
    clock = [0]
    monkeypatch.setattr(modules.protocol.time, "monotonic", lambda: clock[0])
    registry = modules.protocol.LeaseRegistry(unclaimed_timeout=5)
    manifest = {"identity": "model", "rank": 0, "world_size": 2}
    registry.publish("key", "request", manifest)
    with pytest.raises(ValueError, match="rank"):
        registry.acquire("key", "reader", "model", 1, 2)
    assert registry.take_finished() == set()
    registry.acquire("key", "reader", "model", 0, 2)
    clock[0] = 1000
    assert registry.take_finished() == set()  # Claimed leases never expire.
    with pytest.raises(ValueError):
        registry.release("key", "reader", drained=False)
    with pytest.raises(ValueError):
        registry.release("key", "other", drained=True)
    registry.release("key", "reader", drained=True)
    registry.release("key", "reader", drained=True)
    assert registry.take_finished() == {"request"}
    assert registry.take_finished() == set()
    registry.publish("unclaimed", "unused", manifest)
    clock[0] += 6
    assert registry.take_finished() == {"unused"}
    assert registry.acquire("unclaimed", "reader", "model", 0, 2) == {"status": "pending"}


def test_actual_upstream_aggregator_waits_all_ranks_including_failure(modules):
    aggregator = modules.aggregator(2)

    def output(done=(), failed=()):
        return NS(
            kv_connector_output=NS(
                finished_sending=None,
                finished_recving=set(done),
                failed_recving=set(failed),
                expected_finished_count=0,
                kv_connector_stats=None,
                kv_connector_worker_meta=None,
                kv_cache_events=None,
                invalid_block_ids=set(),
            )
        )

    partial = aggregator.aggregate([output(["r"], ["r"]), output()])
    assert not partial.kv_connector_output.finished_recving
    assert not partial.kv_connector_output.failed_recving
    complete = aggregator.aggregate([output(), output(["r"])])
    assert complete.kv_connector_output.finished_recving == {"r"}
    assert complete.kv_connector_output.failed_recving == {"r"}


def test_actual_multi_preserves_hisparse_completion_and_child_metadata(modules):
    seen = []
    children = [
        NS(
            bind_connector_metadata=lambda meta, index=i: seen.append(("bind", index, meta)),
            update_connector_output=lambda output, index=i: seen.append(
                (
                    "update",
                    index,
                    output.finished_recving,
                    output.failed_recving,
                    output.kv_connector_worker_meta,
                )
            ),
        )
        for i in range(2)
    ]
    multi = object.__new__(modules.multi.MultiConnector)
    multi._connectors = children
    multi._extra_async_saves = {}
    metadata = modules.multi.MultiKVConnectorMetadata(metadata=("hisparse", "pd"))
    multi.bind_connector_metadata(metadata)
    worker_meta = modules.multi.MultiKVConnectorWorkerMetadata(metadata=("host-mirror", None))
    output = NS(finished_recving={"r"}, failed_recving=set(), kv_connector_worker_meta=worker_meta)
    multi.update_connector_output(output)
    assert seen == [
        ("bind", 0, "hisparse"),
        ("bind", 1, "pd"),
        ("update", 0, {"r"}, set(), "host-mirror"),
        ("update", 1, {"r"}, set(), None),
    ]
    assert output.kv_connector_worker_meta is worker_meta


@pytest.mark.parametrize("cancel,failure", [(False, False), (True, False), (False, True)])
def test_real_zmq_worker_read_then_ack_or_retain_on_dma_error(modules, monkeypatch, cancel, failure):
    monkeypatch.setattr(torch, "npu", NS(set_device=lambda _: None, synchronize=lambda _: None), raising=False)
    src, dst = {}, {}
    source, destination = layouts(modules, src, source=True), layouts(modules, dst, source=False)
    registry = modules.protocol.LeaseRegistry()
    ticket = {"lease_id": "lease", "num_tokens": 11}
    registry.publish(
        "lease",
        "source-request",
        {
            "version": 1,
            "identity": "model",
            "rank": 0,
            "world_size": 1,
            "ticket": ticket,
            "session": "session",
            "blocks": {0: [7, 1, 4], 3: [5, 2, 8]},
            "layers": modules.protocol.encode_layouts(source),
        },
    )
    endpoint_port = port()
    server = modules.transport.LeaseServer(f"tcp://127.0.0.1:{endpoint_port}", registry)
    event = threading.Event()

    def read(session, descriptors):
        assert registry.active() == 1
        assert registry.take_finished() == set()
        if failure:
            raise RuntimeError("DMA uncertain")
        for destination_ptr, source_ptr, size in descriptors:
            ctypes.memmove(destination_ptr, source_ptr, size)
        if cancel:
            event.set()

    worker = object.__new__(modules.worker.HiSparsePDWorker)
    worker.__dict__.update(
        device=0,
        rank=0,
        world_size=1,
        identity="model",
        peer_host="127.0.0.1",
        peer_port=endpoint_port,
        timeout=5,
        layouts=destination,
        max_descriptors=2,
        reader=NS(read=read),
        kv_cache_config=NS(hisparse_shared_host_pool=False),
    )
    job = {"request_id": "consumer", "ticket": ticket, "start_token": 0, "blocks": {0: [2, 8, 3], 3: [6, 1, 7]}}
    try:
        if failure:
            with pytest.raises(RuntimeError, match="uncertain"):
                worker._load(job, event)
            assert registry.active() == 1
            assert registry.take_finished() == set()
        else:
            assert worker._load(job, event) is (not cancel)
            assert registry.take_finished() == {"source-request"}
            if not cancel:
                for layout in destination:
                    a = [src[layout.name][1][i] for i in {0: [7, 1, 4], 3: [5, 2, 8]}[layout.group_id]]
                    b = [dst[layout.name][1][i] for i in job["blocks"][layout.group_id]]
                    assert torch.equal(torch.cat(a)[:11], torch.cat(b)[:11])
    finally:
        server.close()


def make_config(producer=True):
    extra = {
        "host": "127.0.0.1",
        "producer_host": "127.0.0.1",
        "control_port": 29500,
        "producer_control_port": 29600,
        "model_identity": "fixture-weights-sha",
        "memfabric_transfer_protocol": "device_urma",
    }
    config = NS(
        kv_transfer_config=NS(
            kv_connector_extra_config=extra,
            is_kv_producer=producer,
            is_kv_consumer=not producer,
        ),
        model_config=NS(enforce_eager=True, hf_config=NS(to_dict=lambda: {"architectures": ["DSA"]})),
        speculative_config=None,
        attention_config=NS(hisparse_config=object()),
        parallel_config=NS(tensor_parallel_size=2),
    )
    groups = [NS(role="hisparse_source", is_eagle_group=False), NS(role="hisparse_indexer", is_eagle_group=False)]
    return config, NS(hisparse_host_num_blocks=10, transfer_group_ids=(0, 1), kv_cache_groups=groups)


def test_scheduler_ticket_all_group_blocks_and_exact_prompt(modules):
    config, cache = make_config()
    cls = modules.connector.HiSparsePDConnector
    producer = cls(config, modules.base.KVConnectorRole.SCHEDULER, cache)
    assert not cls.supports_sleep_mode(config.kv_transfer_config)
    request = NS(
        request_id="P",
        prompt_token_ids=list(range(11)),
        num_computed_tokens=11,
        num_prompt_tokens=11,
        status=NS(name="FINISHED_LENGTH_CAPPED"),
        kv_transfer_params={"do_remote_decode": True},
    )
    with pytest.raises(ValueError, match="MultiConnector"):
        producer.request_finished_all_groups(request, ([5, 2, 9], [4, 7, 1]))
    producer.bind_hisparse_sibling(type("HiSparseConnector", (), {})())
    retained, parameters = producer.request_finished_all_groups(request, ([5, 2, 9], [4, 7, 1]))
    assert retained and producer.has_pending_push_work()
    config, cache = make_config(False)
    consumer = cls(config, modules.base.KVConnectorRole.SCHEDULER, cache)
    consumer.bind_hisparse_sibling(type("HiSparseConnector", (), {})())
    request.request_id = "D"
    request.kv_transfer_params = parameters
    assert consumer.get_num_new_matched_tokens(request, 3) == (8, True)
    blocks = NS(get_block_ids=lambda: ([1, 2, 3], [8, 9, 4]))
    consumer.update_state_after_alloc(request, blocks, 8)
    assert consumer.get_loaded_kv_cache_group_ids(request) == (0, 1)
    metadata = consumer.build_connector_meta(None)
    assert metadata.imports[0]["start_token"] == 3
    assert metadata.imports[0]["blocks"] == {0: [1, 2, 3], 1: [8, 9, 4]}
    assert consumer.build_connector_meta(None).imports == []
    request.prompt_token_ids[4] += 1
    with pytest.raises(ValueError, match="token prefix"):
        consumer.get_num_new_matched_tokens(request, 0)
    producer.update_connector_output(NS(finished_sending={"P"}))
    producer.build_connector_meta(None)
    assert not producer.has_pending_block_frees()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config, _: setattr(config.model_config, "enforce_eager", False),
        lambda config, _: setattr(config, "speculative_config", object()),
        lambda config, _: setattr(config.parallel_config, "pipeline_parallel_size", 2),
        lambda config, _: setattr(config, "lora_config", object()),
        lambda _, cache: setattr(cache, "hisparse_host_num_blocks", None),
        lambda config, _: config.kv_transfer_config.kv_connector_extra_config.pop("model_identity"),
    ],
)
def test_reject_unsupported_configuration(modules, mutation):
    config, cache = make_config()
    mutation(config, cache)
    with pytest.raises(ValueError):
        modules.connector.HiSparsePDConnector(config, modules.base.KVConnectorRole.SCHEDULER, cache)


def test_real_tensor_binding_and_storage_registration_dedup(modules):
    host = torch.zeros(2 * 8 * 4 * 6, dtype=torch.bfloat16)
    mapping = NS(
        device_ptr=host.data_ptr(),
        nbytes=host.numel() * 2,
        address_of=lambda tensor: tensor.data_ptr(),
    )
    raw = torch.zeros(8 * 12 * 4 * 2, dtype=torch.bfloat16)
    caches, context = {}, {}
    groups, tensors = [], []
    for index in range(2):
        name = f"layer{index}"
        caches[name] = host[index * 8 * 4 * 6 : (index + 1) * 8 * 4 * 6].view(8, 4, 6)
        context[name] = NS(hisparse_cache=NS(runtime=NS(cache_ops=NS(host_mapping=mapping))))
    groups.append(
        NS(
            role="hisparse_source",
            enable_kv_transfer=True,
            is_eagle_group=False,
            layer_names=["layer0", "layer1"],
            kv_cache_spec=NS(block_size=4),
        )
    )
    for index in range(2):
        name = f"indexer{index}"
        # Last expanded page bound matches the real Ascend binding contract.
        caches[name] = (torch.as_strided(raw, (7 * 12 + 1, 4, 1, 2), (8, 2, 2, 1), index * 8),)
        context[name] = NS(hisparse_block_stride=12)
        tensors.append(NS(layers=[name], block_stride=12 * 4 * 2 * 2))
        groups.append(
            NS(
                role="hisparse_indexer",
                enable_kv_transfer=True,
                is_eagle_group=False,
                layer_names=[name],
                kv_cache_spec=NS(block_size=4),
            )
        )
    config = NS(kv_cache_groups=groups, kv_cache_tensors=tensors, num_blocks=8, hisparse_host_num_blocks=8)
    with pytest.raises(ValueError, match="Ascend device"):
        modules.layout.build_layouts(caches, config, context, producer=True)
    actual, regions, owners = modules.layout.build_layouts(
        caches, config, context, producer=True, allow_cpu_indexer=True
    )
    assert len(actual) == 4 and len(regions) == 2 and owners
    indexer = actual[-1]
    assert indexer.base == raw.data_ptr() + 8 * 2
    assert indexer.address(7) == raw.data_ptr() + 8 * 2 + 7 * 12 * 8 * 2
    context["indexer1"].hisparse_block_stride = 1
    with pytest.raises(ValueError, match="manager block stride"):
        modules.layout.build_layouts(caches, config, context, producer=True, allow_cpu_indexer=True)


def test_shutdown_refusal_keeps_ack_path_and_tombstones_are_bounded(modules, monkeypatch):
    clock = [0]
    monkeypatch.setattr(modules.protocol.time, "monotonic", lambda: clock[0])
    registry = modules.protocol.LeaseRegistry(ack_retry_window=10, max_tombstones=2)
    manifest = {"identity": "model", "rank": 0, "world_size": 1}
    for index in range(4):
        registry.publish(str(index), f"r{index}", manifest)
        registry.acquire(str(index), "consumer", "model", 0, 1)
        with pytest.raises(RuntimeError, match="source leases"):
            registry.begin_shutdown()
        # A failed close did not disable the producer's release handler.
        registry.release(str(index), "consumer", drained=True)
    assert len(registry._released) == 2
    registry.release("3", "consumer", drained=True)
    clock[0] += 11
    with pytest.raises(ValueError):
        registry.release("3", "consumer", drained=True)
    assert not registry._released
    registry.begin_shutdown()
    with pytest.raises(ValueError, match="closing"):
        registry.acquire("new", "consumer", "model", 0, 1)


@pytest.mark.parametrize("failure_kind", ["cancelled", "invalid_manifest"])
def test_safe_failures_before_dma_do_not_crash_worker(modules, monkeypatch, failure_kind):
    monkeypatch.setattr(torch, "npu", NS(set_device=lambda _: None), raising=False)
    worker = object.__new__(modules.worker.HiSparsePDWorker)
    worker.__dict__.update(
        device=0, rank=0, world_size=1, identity="model", peer_host="127.0.0.1", peer_port=1, timeout=5
    )
    cancelled = threading.Event()
    released = []
    if failure_kind == "cancelled":
        cancelled.set()
    else:
        monkeypatch.setattr(modules.worker, "acquire", lambda *args: {"version": -1})
    monkeypatch.setattr(modules.worker, "rpc", lambda *args: released.append(args) or {"status": "released"})
    assert worker._load({"request_id": "r", "ticket": {"lease_id": "lease"}}, cancelled) is False
    assert len(released) == (failure_kind == "invalid_manifest")


def test_drained_read_with_lost_ack_is_failed_request(modules, monkeypatch):
    monkeypatch.setattr(torch, "npu", NS(set_device=lambda _: None, synchronize=lambda _: None), raising=False)
    worker = object.__new__(modules.worker.HiSparsePDWorker)
    reads = []
    worker.__dict__.update(
        device=0,
        rank=0,
        world_size=1,
        identity="model",
        peer_host="127.0.0.1",
        peer_port=1,
        timeout=5,
        layouts=[],
        max_descriptors=1,
        kv_cache_config=NS(hisparse_shared_host_pool=False),
        reader=NS(read=lambda *args: reads.append(args)),
    )
    ticket = {"lease_id": "lease", "num_tokens": 1}
    monkeypatch.setattr(
        modules.worker,
        "acquire",
        lambda *args: {
            "version": 1,
            "ticket": ticket,
            "layers": [],
            "blocks": {},
            "session": "test",
        },
    )
    monkeypatch.setattr(modules.worker, "plan_reads", lambda *args, **kwargs: [(16, 32, 8)])

    def lost_ack(*args):
        raise TimeoutError("reply lost")

    monkeypatch.setattr(modules.worker, "rpc", lost_ack)
    assert not worker._load({"request_id": "r", "ticket": ticket, "blocks": {}, "start_token": 0}, threading.Event())
    assert len(reads) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_upstream_import_publishes_only_after_rank_completion(modules, failed):
    calls = []
    state = NS(valid_pages=set(), ready_prefix_pages=0)
    coordinator = modules.import_contract()
    coordinator._resident_requests = {"r"}
    coordinator._pending_imports = {}
    coordinator.resident_managers = [NS(block_size=128)]
    coordinator._get_request_state = lambda _: state
    coordinator._plan_page_transfer = lambda *args, **kwargs: calls.append(("restore", args, kwargs))
    coordinator._publish_host_blocks_if_ready = lambda request: calls.append(("publish", request))
    coordinator.record_pending_host_import("r", 257)
    scheduler = object.__new__(modules.hisparse_scheduler)
    scheduler.coordinator = coordinator
    multi = object.__new__(modules.multi.MultiConnector)
    multi._connectors = [scheduler, NS(update_connector_output=lambda _: None)]
    multi._extra_async_saves = {}
    aggregator = modules.aggregator(2)

    def output(done, failure=False):
        return NS(
            kv_connector_output=NS(
                finished_sending=None,
                finished_recving={"r"} if done else set(),
                failed_recving={"r"} if failure else set(),
                expected_finished_count=0,
                kv_connector_stats=None,
                kv_connector_worker_meta=None,
                kv_cache_events=None,
                invalid_block_ids=set(),
            )
        )

    partial = aggregator.aggregate([output(True, failed), output(False)]).kv_connector_output
    multi.update_connector_output(partial)
    assert coordinator._pending_imports == {"r": 257} and not calls and not state.valid_pages
    complete = aggregator.aggregate([output(False), output(True)]).kv_connector_output
    multi.update_connector_output(complete)
    assert not coordinator._pending_imports
    if failed:
        assert not calls and not state.valid_pages
    else:
        assert state.valid_pages == {0, 1} and state.ready_prefix_pages == 2
        assert calls == [("restore", ("r", 2), {"after_forward": False, "restore": True}), ("publish", "r")]


@pytest.mark.parametrize(
    "names",
    [
        ["HiSparsePDConnector"],
        ["HiSparsePDConnector", "HiSparseConnector"],
        ["HiSparseConnector", "HiSparsePDConnector", "Other"],
    ],
)
def test_composition_rejects_missing_reordered_or_extra_children(modules, names):
    config = NS(
        kv_transfer_config=NS(
            kv_connector_extra_config={
                "connectors": [{"kv_connector": name} for name in names],
            }
        )
    )
    with pytest.raises(ValueError, match="children"):
        modules.ascend_multi(config, None, None)


@pytest.mark.parametrize("refuse", [False, True])
def test_pd_pre_release_fences_upstream_cache_owner(modules, refuse):
    seen = []
    owner = type("HiSparseConnector", (), {})()
    owner.shutdown = lambda: seen.append("host-unmap")

    def drain():
        seen.append("pd-drain")
        if refuse:
            raise RuntimeError("live remote reader")

    pd = NS(
        bind_hisparse_sibling=lambda child: seen.append(("sibling", child)),
        prepare_cache_release=drain,
        shutdown=lambda: seen.append("pd-shutdown"),
    )
    config = NS(
        kv_transfer_config=NS(
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "HiSparseConnector"},
                    {"kv_connector": "HiSparsePDConnector"},
                ]
            }
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        test_children=[owner, pd],
    )
    multi = modules.ascend_multi(config, None, None)
    assert seen == [("sibling", owner)]
    seen.clear()
    if refuse:
        with pytest.raises(RuntimeError, match="live remote"):
            multi.shutdown()
        assert seen == ["pd-drain"]
    else:
        multi.shutdown()
        assert seen == ["pd-drain", "host-unmap", "pd-shutdown"]
