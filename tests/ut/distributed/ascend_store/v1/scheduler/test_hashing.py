"""v1 request hashes preserve upstream identity without changing engine hashes."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils
from vllm.v1.request import Request

from tests.ut.distributed.ascend_store.v1.helpers import begin_step, make_topology
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlocks,
    FakeLookupClient,
    make_config,
    make_output,
    new_request_data,
)
from tests.ut.distributed.ascend_store.v1.test_factory import factory_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import factory
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import resolve
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousBulkScheduler,
    LayerwiseScheduler,
    SynchronousBulkScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.hashing import RequestHashes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.rows import lookup_chunk_ranges


@pytest.fixture(params=("sha256", "sha256_cbor"))
def hash_function(request, monkeypatch):
    function = get_hash_fn_by_name(request.param)
    # EngineCore initializes this shared seed before request hashing. Keep each
    # algorithm's test isolated from other tests that consume the upstream seed.
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", getattr(kv_cache_utils, "NONE_HASH", None), raising=False)
    monkeypatch.setattr(kv_cache_utils, "_NONE_HASH_SEED", kv_cache_utils._NONE_HASH_SEED)
    kv_cache_utils.init_none_hash(function)
    return function


def make_hashed_request(hash_function, block_size=64, *, request_id="request", token_count=128, **kwargs):
    return Request(
        request_id,
        list(range(token_count)),
        SamplingParams(max_tokens=64),
        None,
        block_hasher=kv_cache_utils.get_request_block_hasher(block_size, hash_function),
        **kwargs,
    )


def test_matching_grain_reuses_engine_hashes_without_retaining_request(hash_function):
    request = make_hashed_request(hash_function)
    hashes = RequestHashes()
    assert hashes.get(request) == tuple(request.block_hashes)
    request.append_output_token_ids(list(range(128, 192)))
    assert hashes.get(request) == tuple(request.block_hashes)
    assert hashes._requests == {}


@pytest.mark.parametrize("identity", ("plain", "salt", "lora", "multimodal", "embeddings"))
def test_fine_hashes_match_upstream_and_leave_request_unchanged(hash_function, identity):
    kwargs = {
        "plain": {},
        "salt": {"cache_salt": "tenant-a"},
        "lora": {"lora_request": SimpleNamespace(lora_name="adapter-a")},
        "multimodal": {
            "mm_features": [SimpleNamespace(identifier="image-a", mm_position=SimpleNamespace(offset=10, length=26))]
        },
        "embeddings": {"prompt_embeds": torch.arange(256, dtype=torch.float32).reshape(128, 2)},
    }[identity]
    request = make_hashed_request(hash_function, **kwargs)
    expected = make_hashed_request(hash_function, 16, **kwargs)
    engine_hashes = list(request.block_hashes)
    engine_embedding_cache = dict(request._prompt_embeds_per_block_hashes)
    hashes = RequestHashes(kv_cache_utils.get_request_block_hasher(16, hash_function))

    assert hashes.get(request) == tuple(expected.block_hashes)
    assert len(hashes.get(request)) == 8
    assert request.block_hashes == engine_hashes
    assert request._prompt_embeds_per_block_hashes == engine_embedding_cache
    if identity != "plain":
        plain = make_hashed_request(hash_function, 16)
        assert hashes.get(request) != tuple(plain.block_hashes)


def test_decode_hashes_only_new_full_blocks_and_reuses_partial_tail(hash_function, monkeypatch):
    request = make_hashed_request(hash_function, token_count=31)
    hashes = RequestHashes(kv_cache_utils.get_request_block_hasher(16, hash_function))
    native_hash = Mock(wraps=kv_cache_utils.hash_block_tokens)
    monkeypatch.setattr(kv_cache_utils, "hash_block_tokens", native_hash)
    first = hashes.get(request)
    assert len(first) == native_hash.call_count == 1
    assert hashes.get(request) == first
    assert native_hash.call_count == 1

    request.append_output_token_ids(31)
    second = hashes.get(request)
    assert len(second) == native_hash.call_count == 2
    assert second[:1] == first
    request.append_output_token_ids(list(range(32, 48)))
    third = hashes.get(request)
    assert len(third) == native_hash.call_count == 3
    assert third[:2] == second
    assert request.block_hashes == []


def test_request_id_reuse_and_discard_do_not_reuse_another_prefix(hash_function):
    hashes = RequestHashes(kv_cache_utils.get_request_block_hasher(16, hash_function))
    first = make_hashed_request(hash_function, cache_salt="first")
    second = make_hashed_request(hash_function, cache_salt="second")
    first_hashes = hashes.get(first)
    second_hashes = hashes.get(second)
    assert first_hashes != second_hashes
    hashes.discard(second.request_id)
    assert hashes._requests == {}
    assert hashes.get(first) == first_hashes
    hashes.clear()
    assert hashes._requests == {}


@pytest.mark.parametrize("scheduler_type", (SynchronousBulkScheduler, AsynchronousBulkScheduler, LayerwiseScheduler))
@pytest.mark.parametrize("release", ("finished", "preempted"))
def test_scheduler_uses_fine_hashes_for_lookup_load_and_request_cleanup(hash_function, scheduler_type, release):
    lookup = FakeLookupClient(128)
    config = replace(make_config(granularity=64, groups=(0, 1)), hash_block_size=16)
    scheduler = scheduler_type(config, lookup, block_hasher=kv_cache_utils.get_request_block_hasher(16, hash_function))
    request = make_hashed_request(hash_function)
    expected = tuple(make_hashed_request(hash_function, 16).block_hashes)
    assert scheduler.get_num_new_matched_tokens(request, 0)[0] == 127
    assert lookup.queries[0][2] == expected
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4], [5, 6]), 127)
    output = make_output(
        new=(new_request_data(request.request_id, ([1, 2, 3, 4], [5, 6]), 127),),
        scheduled_tokens={request.request_id: 1},
    )
    command = scheduler.build_step(output).load_commands[0]
    assert command.block_hashes == expected
    assert len(request.block_hashes) == 2
    for block_size, indices in ((32, (1, 3, 5, 7)), (64, (3, 7))):
        rows = lookup_chunk_ranges(128, command.block_hashes, block_size=block_size, hash_block_size=16)
        assert rows[2] == tuple(expected[index] for index in indices)
    scheduler.build_step(make_output(**{release: [request.request_id]}))
    assert scheduler._request_hashes._requests == {}
    scheduler.close()


def test_factory_restores_prefix_off_multigroup_dcp_store_and_decode(configuration_case, hash_function, monkeypatch):
    config, cache = configuration_case
    config.cache_config.enable_prefix_caching = False
    config.cache_config.prefix_caching_hash_algo = hash_function.__name__
    config.kv_transfer_config.kv_connector_extra_config["save_decode_cache"] = True
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = 2
    config.parallel_config.world_size = 2
    spec = cache.kv_cache_groups[0].kv_cache_spec
    cache.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=replace(spec, block_size=size), layer_names=[f"model.layers.{index}"])
        for index, size in enumerate((16, 32))
    ]
    cache.transfer_group_ids = (0, 1)
    engine_resolver = Mock(return_value=(64, 64))
    monkeypatch.setattr(resolve.kv_cache_utils, "resolve_kv_cache_block_sizes", engine_resolver)
    lookup = FakeLookupClient()
    monkeypatch.setattr(factory, "LookupClient", lambda *_args, **_kwargs: lookup)
    scheduler = factory.create_kv_pool_scheduler(config, cache, "unused")
    request = make_hashed_request(hash_function)
    expected = make_hashed_request(hash_function, 16)
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        assert lookup.queries[0][2] == tuple(expected.block_hashes)
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4], [5, 6]), 0)
        step = scheduler.build_step(
            make_output(
                new=(new_request_data("request", ([1, 2, 3, 4], [5, 6]), 0),), scheduled_tokens={"request": 128}
            )
        )
        assert step.store.commands[0].block_hashes == tuple(expected.block_hashes)
        request.append_output_token_ids(list(range(128, 192)))
        expected.append_output_token_ids(list(range(128, 192)))
        step = scheduler.build_step(
            make_output(cached=(("request", ([7, 8], [9]), 128),), scheduled_tokens={"request": 64})
        )
        command = step.store.commands[0]
        assert (command.start_token, command.end_token) == (128, 192)
        assert command.block_hashes == tuple(expected.block_hashes)
        assert len(request.block_hashes) == 3
        assert not config.cache_config.enable_prefix_caching
        assert resolve.kv_cache_utils.resolve_kv_cache_block_sizes is engine_resolver
    finally:
        scheduler.close()
    assert scheduler._request_hashes._requests == {}


def test_prefix_off_bulk_objects_load_with_prefix_on_hashes(hash_function, monkeypatch):
    request = make_hashed_request(hash_function, token_count=64)
    hashes = RequestHashes(kv_cache_utils.get_request_block_hasher(16, hash_function)).get(request)
    objects = {}
    ranks = []
    source_ids = ((1, 2), (1,))
    for dcp_rank in (0, 1):
        topology = make_topology(
            group_ids=(0, 1), physical_layers=(0,), tp_rank=dcp_rank, tp_size=2, dcp_rank=dcp_rank, dcp_size=2
        )
        groups = tuple(
            replace(group, kv_cache_spec=replace(group.kv_cache_spec, block_size=size))
            for group, size in zip(topology.groups, (32, 64), strict=True)
        )
        topology = replace(topology, groups=groups, cache_transfer_granularity=64, hash_block_size=16)
        writer, source, write_sdk = factory_worker(monkeypatch, "mooncake", False, topology, objects)
        expected = {
            group.layer_names[0]: tuple(tensor[list(ids)].clone() for tensor in source[group.layer_names[0]])
            for group, ids in zip(groups, source_ids, strict=True)
        }
        try:
            begin_step(writer, store=(RangeStoreCommand("request", 0, 64, source_ids, hashes, 64, 17),))
            writer.finish_step()
            assert all(completion.evidence.succeeded for completion in writer.fence_previous_store())
            assert writer.take_released_store_job_ids() == {17}
            writer.end_step()
        finally:
            writer.close()
        assert write_sdk.closed
        ranks.append((topology, expected))
    assert len(objects) == 6

    # A prefix-on engine produces these fine hashes directly. Its request ID
    # differs, but the same tokens and identities must address both DCP shards.
    fine_request = make_hashed_request(hash_function, 16, request_id="reader", token_count=64)
    reader_hashes = tuple(fine_request.block_hashes)
    target_ids = ((6, 7), (7,))
    for topology, expected in ranks:
        reader, destination, read_sdk = factory_worker(monkeypatch, "mooncake", False, topology, objects)
        try:
            for tensors in destination.values():
                for tensor in tensors:
                    tensor.fill_(-1)
            assert reader.lookup(LookupRequest(0, 64, (0, 1), reader_hashes)).available_end_token == 64
            begin_step(reader, load=(LoadCommand("reader", 0, 64, target_ids, reader_hashes),))
            reader.start_load()
            assert not reader.collect_load_result().failed_locations
            for group, ids in zip(topology.groups, target_ids, strict=True):
                name = group.layer_names[0]
                for actual, wanted in zip(destination[name], expected[name], strict=True):
                    torch.testing.assert_close(actual[list(ids)], wanted)
                    assert torch.all(actual[0] == -1)
            reader.end_step()
        finally:
            reader.close()
        assert read_sdk.closed
    assert len(request.block_hashes) == 1
