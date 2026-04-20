import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.put_calls = []
        self.get_calls = []

    def set_device(self):
        pass

    def register_buffer(self, ptrs, lengths):
        pass

    def exists(self, keys):
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((keys, addrs, sizes))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((keys, addrs, sizes))


class FakeKey:
    def __init__(self, value):
        self._value = value

    def to_string(self):
        return self._value

    def split_layers(self, num_layers):
        return [FakeLayerKey(self._value, i) for i in range(num_layers)]


class FakeLayerKey:
    def __init__(self, value, layer_id):
        self._value = value
        self.layer_id = layer_id

    def to_string(self):
        return f"{self._value}@{self.layer_id}"


def _make_pool_worker(
    kv_role="kv_producer",
    block_size=8,
    num_layers=2,
    use_layerwise=False,
    num_kv_head=4,
    tp_size=1,
    pp_size=1,
    use_mla=False,
    backend_name="mooncake",
    load_async=False,
    consumer_is_to_put=False,
    exists_result=None,
    enable_kv_events=False,
):
    """Create a KVPoolWorker with mocked dependencies."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
        KVPoolWorker,
    )

    worker = object.__new__(KVPoolWorker)
    worker.dp_rank = 0
    worker.use_mla = use_mla
    worker.use_sparse = False
    worker.use_layerwise = use_layerwise
    worker.tp_rank = 0
    worker.tp_size = tp_size
    worker.pp_size = pp_size
    worker.pp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.dcp_size = 1
    worker.dcp_rank = 0
    worker.kv_role = kv_role
    worker.load_async = load_async
    worker.consumer_is_to_put = consumer_is_to_put
    worker.backend = backend_name
    worker.original_block_size = block_size
    worker.block_size = block_size
    worker.current_layer = 0
    worker.num_layers = num_layers
    worker.num_kv_head = num_kv_head
    worker.put_step = 1
    worker.head_or_tp_rank = 0
    worker.enable_kv_events = enable_kv_events

    worker.metadata = KeyMetadata("model", 0, 0, 0, 0)
    worker.token_database = ChunkedTokenDatabase(worker.metadata, block_size, None)
    worker.m_store = FakeStore(exists_result or [])
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.finished_store_req = set()

    return worker


class TestKVPoolWorkerCheckAllLayersExists(unittest.TestCase):
    def test_all_layers_exist(self):
        worker = _make_pool_worker(num_layers=3)
        # 2 chunks * 3 layers = 6 results, all 1
        res = [1, 1, 1, 1, 1, 1]
        result = worker.check_all_layers_exists(res, 3)
        self.assertEqual(result, [1, 1])

    def test_some_layers_missing(self):
        worker = _make_pool_worker(num_layers=3)
        # chunk 0: all exist, chunk 1: layer 1 missing
        res = [1, 1, 1, 1, 0, 1]
        result = worker.check_all_layers_exists(res, 3)
        self.assertEqual(result, [1, 0])

    def test_empty(self):
        worker = _make_pool_worker(num_layers=2)
        result = worker.check_all_layers_exists([], 2)
        self.assertEqual(result, [])

    def test_single_chunk(self):
        worker = _make_pool_worker(num_layers=2)
        result = worker.check_all_layers_exists([1, 1], 2)
        self.assertEqual(result, [1])

    def test_single_chunk_missing(self):
        worker = _make_pool_worker(num_layers=2)
        result = worker.check_all_layers_exists([1, 0], 2)
        self.assertEqual(result, [0])


class TestKVPoolWorkerFindMinFirstNonOneIndex(unittest.TestCase):
    def test_basic(self):
        worker = _make_pool_worker()
        arr = [[1, 1, 0], [1, 0, 1]]
        self.assertEqual(worker.find_min_first_non_one_index(arr), 1)

    def test_all_ones(self):
        worker = _make_pool_worker()
        arr = [[1, 1, 1], [1, 1, 1]]
        self.assertEqual(worker.find_min_first_non_one_index(arr), -1)

    def test_first_is_non_one(self):
        worker = _make_pool_worker()
        arr = [[0, 1], [1, 1]]
        self.assertEqual(worker.find_min_first_non_one_index(arr), 0)

    def test_empty(self):
        worker = _make_pool_worker()
        arr = [[]]
        self.assertEqual(worker.find_min_first_non_one_index(arr), -1)

    def test_single_row(self):
        worker = _make_pool_worker()
        arr = [[1, 0, 1]]
        self.assertEqual(worker.find_min_first_non_one_index(arr), 1)


class TestKVPoolWorkerLookup(unittest.TestCase):
    def test_lookup_all_found(self):
        worker = _make_pool_worker(block_size=8, exists_result=[1, 1])
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup(16, ["h0", "h1"], use_layerwise=False)
        # All exist -> return end of last block
        self.assertEqual(result, 16)

    def test_lookup_partial(self):
        worker = _make_pool_worker(block_size=8, exists_result=[1, 0])
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup(16, ["h0", "h1"], use_layerwise=False)
        # Second block missing, returns starts[1] = 8
        self.assertEqual(result, 8)

    def test_lookup_none_found(self):
        worker = _make_pool_worker(block_size=8, exists_result=[0, 0])
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_exception(self):
        worker = _make_pool_worker(block_size=8)
        worker.m_store.exists = MagicMock(side_effect=Exception("conn"))
        result = worker.lookup(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_layerwise(self):
        worker = _make_pool_worker(block_size=8, num_layers=2, exists_result=[1, 1, 1, 1])
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup(16, ["h0", "h1"], use_layerwise=True)
        self.assertEqual(result, 16)


class TestKVPoolWorkerGetAndClearFinished(unittest.TestCase):
    def test_basic(self):
        worker = _make_pool_worker()
        mock_send_thread = MagicMock()
        mock_send_thread.stored_requests = {"r1": 0, "r2": 1}
        mock_send_thread.stored_requests.copy.return_value = {"r1": 0, "r2": 1}
        worker.kv_send_thread = mock_send_thread

        meta = AscendConnectorMetadata(set(), set())
        worker.finished_store_req.add("r1")

        finished = worker.get_and_clear_finished_requests({"r2"}, meta)
        self.assertIn("r1", finished)

    def test_preempted_deleted(self):
        worker = _make_pool_worker()
        mock_send_thread = MagicMock()
        mock_send_thread.stored_requests = {}
        mock_send_thread.stored_requests.copy.return_value = {}
        worker.kv_send_thread = mock_send_thread

        meta = AscendConnectorMetadata(set(), {"r_pre"})
        worker.get_and_clear_finished_requests(set(), meta)
        mock_send_thread.delete_finished_stored_request.assert_called_with("r_pre")

    def test_finished_req_with_zero_remain(self):
        worker = _make_pool_worker()
        mock_send_thread = MagicMock()
        mock_send_thread.stored_requests = {"r3": 0}
        mock_send_thread.stored_requests.copy.return_value = {"r3": 0}
        mock_send_thread.stored_requests.get.return_value = 0
        worker.kv_send_thread = mock_send_thread

        meta = AscendConnectorMetadata(set(), set())
        finished = worker.get_and_clear_finished_requests({"r3"}, meta)
        self.assertIn("r3", finished)

    def test_finished_req_with_nonzero_remain(self):
        worker = _make_pool_worker()
        mock_send_thread = MagicMock()
        mock_send_thread.stored_requests = {"r4": 2}
        mock_send_thread.stored_requests.copy.return_value = {"r4": 2}
        mock_send_thread.stored_requests.get.return_value = 2
        worker.kv_send_thread = mock_send_thread

        meta = AscendConnectorMetadata(set(), set())
        finished = worker.get_and_clear_finished_requests({"r4"}, meta)
        self.assertNotIn("r4", finished)
        self.assertIn("r4", worker.finished_store_req)


class TestKVPoolWorkerGetKvEvents(unittest.TestCase):
    def test_no_events(self):
        worker = _make_pool_worker(enable_kv_events=False)
        result = worker.get_kv_events()
        self.assertEqual(result, [])

    def test_with_events(self):
        worker = _make_pool_worker(enable_kv_events=True)
        mock_thread = MagicMock()
        mock_event = MagicMock()
        mock_thread.get_kv_events.return_value = [mock_event]
        worker.kv_send_thread = mock_thread
        result = worker.get_kv_events()
        self.assertEqual(result, [mock_event])

    def test_enabled_but_no_thread(self):
        worker = _make_pool_worker(enable_kv_events=True)
        worker.kv_send_thread = None
        result = worker.get_kv_events()
        self.assertEqual(result, [])


class TestKVPoolWorkerGetFinished(unittest.TestCase):
    def test_producer_role(self):
        worker = _make_pool_worker(kv_role="kv_producer", load_async=False)
        mock_send_thread = MagicMock()
        mock_send_thread.stored_requests = {}
        mock_send_thread.stored_requests.copy.return_value = {}
        worker.kv_send_thread = mock_send_thread

        meta = AscendConnectorMetadata(set(), set())
        done_send, done_recv = worker.get_finished(set(), meta)
        self.assertIsInstance(done_send, set)
        self.assertEqual(done_recv, set())

    def test_with_async_recv(self):
        worker = _make_pool_worker(kv_role="kv_producer", load_async=True)
        mock_send_thread = MagicMock()
        mock_send_thread.stored_requests = {}
        mock_send_thread.stored_requests.copy.return_value = {}
        worker.kv_send_thread = mock_send_thread

        mock_recv_thread = MagicMock()
        mock_recv_thread.get_and_clear_finished_requests.return_value = {"r1"}
        worker.kv_recv_thread = mock_recv_thread

        meta = AscendConnectorMetadata(set(), set())
        done_send, done_recv = worker.get_finished(set(), meta)
        self.assertEqual(done_recv, {"r1"})


class TestKVPoolWorkerWaitForSave(unittest.TestCase):
    def test_wait_for_save(self):
        worker = _make_pool_worker()
        mock_send_thread = MagicMock()
        worker.kv_send_thread = mock_send_thread

        mock_event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0, 1],
            block_hashes=[],
            can_save=True,
            current_event=None,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)

        with patch("torch.npu.Event", return_value=mock_event):
            worker.wait_for_save(meta)

        mock_send_thread.add_stored_request.assert_called_once_with("r1")
        mock_send_thread.add_request.assert_called_once()

    def test_wait_for_save_skip_no_save(self):
        worker = _make_pool_worker()
        mock_send_thread = MagicMock()
        worker.kv_send_thread = mock_send_thread

        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[],
            can_save=False,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)

        worker.wait_for_save(meta)
        mock_send_thread.add_stored_request.assert_not_called()


class TestKVPoolWorkerStartLoadKv(unittest.TestCase):
    def test_start_load_kv_no_load(self):
        worker = _make_pool_worker()
        meta = AscendConnectorMetadata(set(), set())
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0, 1],
            block_hashes=[],
            load_spec=None,
        )
        meta.add_request(req)
        worker.start_load_kv(meta)
        # No error, no load

    def test_start_load_kv_sync(self):
        worker = _make_pool_worker(block_size=8, load_async=False)
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])

        load_spec = LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        self.assertEqual(len(worker.m_store.get_calls), 1)

    def test_start_load_kv_async(self):
        worker = _make_pool_worker(block_size=8, load_async=True)
        mock_recv = MagicMock()
        worker.kv_recv_thread = mock_recv

        load_spec = LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0, 1],
            block_hashes=["h0", "h1"],
            load_spec=load_spec,
        )
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(req)
        worker.start_load_kv(meta)
        mock_recv.add_request.assert_called_once()


class TestKVPoolWorkerLookupScheduler(unittest.TestCase):
    def test_lookup_scheduler_all_found(self):
        worker = _make_pool_worker(block_size=8, tp_size=1, pp_size=1, num_kv_head=4)
        worker.m_store.exists_result = [1, 1]
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup_scheduler(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_scheduler_partial(self):
        worker = _make_pool_worker(block_size=8, tp_size=1, pp_size=1, num_kv_head=4)
        worker.m_store.exists_result = [1, 0]
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup_scheduler(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 8)

    def test_lookup_scheduler_exception(self):
        worker = _make_pool_worker(block_size=8)
        worker.m_store.exists = MagicMock(side_effect=Exception("error"))
        result = worker.lookup_scheduler(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 0)

    def test_lookup_scheduler_multi_tp(self):
        worker = _make_pool_worker(
            block_size=8, tp_size=2, pp_size=1, num_kv_head=4
        )
        # 2 hashes * 2 tp = 4 keys
        worker.m_store.exists_result = [1, 1, 1, 1]
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup_scheduler(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_scheduler_multi_pp(self):
        worker = _make_pool_worker(
            block_size=8, tp_size=1, pp_size=2, num_kv_head=4
        )
        # 2 hashes * 1 tp * 2 pp = 4 keys
        worker.m_store.exists_result = [1, 1, 1, 1]
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup_scheduler(16, ["h0", "h1"], use_layerwise=False)
        self.assertEqual(result, 16)

    def test_lookup_scheduler_layerwise(self):
        worker = _make_pool_worker(
            block_size=8, tp_size=1, pp_size=1, num_kv_head=4, num_layers=2
        )
        # 2 hashes * 2 layers = 4 keys
        worker.m_store.exists_result = [1, 1, 1, 1]
        worker.token_database.set_kv_caches_base_addr([1000])
        worker.token_database.set_block_len([64])
        result = worker.lookup_scheduler(16, ["h0", "h1"], use_layerwise=True)
        self.assertEqual(result, 16)


if __name__ == "__main__":
    unittest.main()
