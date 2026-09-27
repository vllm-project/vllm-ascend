# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.request import RequestStatus

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_mooncake_layerwise import (
    TestMooncakeWorkerSessionPreparation as _WorkerFixture,
)
from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta, RequestTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pd_transfer import (
    PoolPDTransfer,
    prepare_pool_pd_forward,
    requires_layerwise_eager_load,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler


def make_scheduler(role="kv_producer", use_v2=False, backend="mooncake"):
    config = make_config(role, {"backend": backend, "pool_pd": True})
    config.speculative_config = None
    config.model_config.dtype = "bfloat16"
    config.model_config.revision = None
    config.cache_config.cache_dtype = "auto"
    config.use_v2_model_runner = use_v2
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib"):
        scheduler = KVPoolScheduler(config, use_layerwise=True)
    scheduler.store_scheduler.batch_is_committed.side_effect = lambda keys: [True] * len(keys)
    return scheduler


def test_v2_runner_is_rejected_without_the_pre_forward_adapter():
    with pytest.raises(ValueError, match="V1 runner"):
        make_scheduler(use_v2=True)


def make_request(length=20, req_id="prefill"):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(range(length)),
        block_hashes=[bytes([i + 1]) * 32 for i in range(length // 16)],
        lora_request=None,
        mm_features=[],
        kv_transfer_params=None,
        num_tokens=length,
        num_computed_tokens=length,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )


@pytest.mark.parametrize(
    "hits,expected", [([True, True, False], 32), ([True, False, True], 16), ([False] * 3, 0), ([True] * 3, 47)]
)
def test_producer_prefix_reuse_only_claims_contiguous_committed_blocks(hits, expected):
    scheduler = make_scheduler(backend="memcache")
    scheduler.vllm_config.kv_transfer_config.kv_connector_extra_config["pool_pd_prefix_reuse"] = True
    scheduler.store_scheduler.batch_is_committed.side_effect = lambda keys: hits[: len(keys)]
    request = make_request(48)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (expected, False)
    if expected:
        assert scheduler.load_specs[request.request_id].kvpool_store_skip_tokens == (48 if expected == 47 else expected)


@pytest.mark.parametrize("layerwise,sync,expected", [(True, True, True), (True, False, False), (False, True, False)])
def test_only_actual_layerwise_loads_bypass_graph_replay(layerwise, sync, expected):
    config = SimpleNamespace(kv_connector_extra_config={"use_layerwise": layerwise})
    assert requires_layerwise_eager_load(SimpleNamespace(has_sync_kv_loads=sync), config) is expected
    assert not requires_layerwise_eager_load(SimpleNamespace(has_sync_kv_loads=sync), None)


@pytest.mark.parametrize("enabled", [False, True])
def test_producer_prepares_layerwise_tasks_before_forward_without_remote_hits(enabled):
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
    from vllm.v1.worker import kv_connector_model_runner_mixin as runner

    events = []
    connector = MagicMock(spec=KVConnectorBase)
    connector.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    connector.start_load_kv.side_effect = lambda _: events.append("prepare")
    connector.wait_for_save.side_effect = lambda: events.append("save_done")
    output = SimpleNamespace(has_sync_kv_loads=False, kv_connector_metadata=object(), finished_req_ids=set())
    config = SimpleNamespace(kv_connector_extra_config={"pool_pd": enabled})
    local_output = prepare_pool_pd_forward(output, config)
    with (
        patch.object(runner, "get_kv_transfer_group", return_value=connector),
        patch.object(runner, "get_forward_context"),
        runner.KVConnectorModelRunnerMixin._get_kv_connector_output(local_output),
    ):
        events.append("forward")
    assert events == (
        ["bind", "prepare", "forward", "save_done"] if enabled else ["bind", "forward", "prepare", "save_done"]
    )
    assert output.has_sync_kv_loads is False


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 300])
@pytest.mark.parametrize("backend", ["mooncake", "memcache"])
def test_handoff_covers_exact_prompt_and_replays_only_one_token(length, backend):
    producer = make_scheduler(backend=backend)
    request = make_request(length)
    assert producer.get_num_new_matched_tokens(request, 0) == (0, False)
    _, params = producer.request_finished(request, [])
    assert params["pool_pd"]["status"] == "ready"
    consumer = make_scheduler("kv_consumer", backend=backend)
    request.request_id = "different-decode-request"
    request.kv_transfer_params = params
    assert consumer.get_num_new_matched_tokens(request, 0) == (length - 1, False)
    spec = consumer.load_specs[request.request_id]
    assert spec.kvpool_store_skip_tokens == length
    assert params["pool_pd"]["tail_tokens"] == length % 16


def test_pd_never_publishes_a_missing_tail_as_ready():
    scheduler = make_scheduler()
    request = make_request()
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.store_scheduler.batch_is_committed.side_effect = lambda keys: ["pd-tail" not in key for key in keys]
    _, params = scheduler.request_finished(request, [])
    assert params["pool_pd"]["status"] == "failed"


def test_prefill_snapshot_remains_valid_if_scheduler_has_advanced_beyond_prompt():
    scheduler = make_scheduler()
    request = make_request()
    scheduler.get_num_new_matched_tokens(request, 0)
    request.num_computed_tokens += 1
    _, params = scheduler.request_finished(request, [])
    assert params["pool_pd"]["status"] == "ready"
    assert params["pool_pd"]["num_tokens"] == 20


def test_decoder_rejects_expired_pool_objects_before_claiming_a_hit():
    producer = make_scheduler()
    request = make_request()
    producer.get_num_new_matched_tokens(request, 0)
    _, request.kv_transfer_params = producer.request_finished(request, [])
    consumer = make_scheduler("kv_consumer")
    consumer.store_scheduler.batch_is_committed.side_effect = lambda keys: [False] * len(keys)
    with pytest.raises(ValueError, match="no longer readable"):
        consumer.get_num_new_matched_tokens(request, 0)
    assert request.request_id not in consumer.load_specs


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_tokens", 21),
        ("block_size", 32),
        ("layout_id", "wrong"),
        ("prompt_digest", "wrong"),
        ("version", 2),
        ("block_hashes", []),
    ],
)
def test_decoder_rejects_wrong_snapshot(field, value):
    transfer = PoolPDTransfer.create(list(range(20)), 16, "layout", ["aabb"])
    wire = transfer.to_wire() | {"status": "ready", field: value}
    with pytest.raises(ValueError):
        PoolPDTransfer.from_wire(wire, list(range(20)), 16, "layout")


def test_final_partial_chunk_is_saved_even_without_a_new_complete_block():
    scheduler = make_scheduler()
    request = make_request()
    scheduler.get_num_new_matched_tokens(request, 0)
    tracker = RequestTracker(
        req_id=request.request_id, token_len=16, allocated_block_ids=[1, 2], num_saved_tokens=0, num_prompt_tokens=20
    )
    first = scheduler._build_req_meta(tracker, request.block_hashes, None, request.prompt_token_ids, False)
    assert not first.is_last_chunk
    assert first.partial_block_index is None
    tracker.token_len = 20
    last = scheduler._build_req_meta(tracker, request.block_hashes, None, request.prompt_token_ids, False)
    assert last.can_save
    assert last.is_last_chunk
    assert last.partial_block_index == 1
    assert last.pd_transfer["tail_tokens"] == 4


def test_tail_key_is_independent_of_decoder_request_id_and_local_hashes():
    worker = _WorkerFixture._make_worker()
    transfer = PoolPDTransfer.create(list(range(20)), 16, "layout", ["aabb"])
    request = ReqMeta(
        "decode-id",
        block_ids=[10, 11],
        block_hashes=[b"different"],
        load_spec=LoadSpec(0, 19, can_load=True, kvpool_store_skip_tokens=20),
        pd_transfer=transfer.to_wire(),
    )
    slots = worker._prepare_layerwise_get_session(request)
    assert slots == [("model@aabb@0", 10, 0), (f"model@{transfer.tail_id}@0", 11, 1)]


def test_tail_only_prompt_uses_a_real_put_session():
    worker = _WorkerFixture._make_worker()
    worker.m_store.exists.return_value = [0]
    worker.m_store.batch_put_start.return_value = [0]
    transfer = PoolPDTransfer.create([42], 16, "layout", [])
    request = ReqMeta(
        "p", block_ids=[10], can_save=True, partial_block_index=0, target_token_len=1, pd_transfer=transfer.to_wire()
    )
    worker._prepare_layerwise_put_session(request)
    worker.m_store.batch_put_start.assert_called_once_with([f"model@{transfer.tail_id}@0"], [60])


def test_repeated_prompt_does_not_overwrite_shared_complete_blocks():
    worker = _WorkerFixture._make_worker()
    worker.m_store.exists.side_effect = lambda keys: [0 if "pd-tail" in key else 1 for key in keys]
    worker.m_store.batch_put_start.return_value = [0]
    transfer = PoolPDTransfer.create(list(range(20)), 16, "layout", ["6830"])
    request = ReqMeta(
        "p",
        block_ids=[10, 11],
        block_hashes=[b"h0"],
        save_end_token=16,
        target_token_len=20,
        can_save=True,
        partial_block_index=1,
        pd_transfer=transfer.to_wire(),
    )
    worker._prepare_layerwise_put_session(request)
    worker.m_store.batch_put_start.assert_called_once_with([f"model@{transfer.tail_id}@0"], [60])
    assert request.save_block_keys == [None]
