# SPDX-License-Identifier: Apache-2.0

from copy import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole, KVConnectorTransferResults
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakeBaseConnector,
    MooncakeConnector,
    MooncakePullConnector,
    MooncakePushConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import DONE_RECVING_MSG

from .helpers import make_full_spec
from .test_pull_scheduler import make_pull_scheduler, make_sending_thread
from .test_pull_worker import make_connector_worker


def make_facade() -> MooncakeBaseConnector:
    connector = MooncakeBaseConnector.__new__(MooncakeBaseConnector)
    connector.connector_scheduler = MagicMock()
    connector.connector_worker = MagicMock()
    connector._connector_metadata = MooncakeConnectorMetadata()
    return connector


def test_mooncake_connector_aliases_pull_connector() -> None:
    assert MooncakeConnector is MooncakePullConnector


def test_loaded_group_ids_hook_is_version_gated() -> None:
    assert "get_loaded_kv_cache_group_ids" in MooncakeBaseConnector.__dict__


def test_loaded_group_ids_returns_transfer_groups() -> None:
    # A P/D transfer restores the request's own blocks, so the scheduler must
    # skip zeroing (and SWA replay) for every transfer group, not only the
    # prefix-cacheable ones.
    connector = make_facade()
    connector._kv_cache_config = SimpleNamespace(transfer_group_ids=(0, 2))  # type: ignore[attr-defined]
    assert connector.get_loaded_kv_cache_group_ids(MagicMock()) == (0, 2)


def test_required_layout_is_hnd_for_regular_attention_and_none_for_mla() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) == "HND"  # type: ignore[arg-type]

    config.model_config.use_mla = True
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) is None  # type: ignore[arg-type]

    config.model_config = None
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) is None  # type: ignore[arg-type]


def test_facade_delegates_scheduler_methods() -> None:
    connector = make_facade()
    request = MagicMock()
    blocks = MagicMock()
    output = MagicMock()
    connector.connector_scheduler.get_num_new_matched_tokens.return_value = (8, True)
    connector.connector_scheduler.request_finished.return_value = (True, {"remote": True})

    assert connector.get_num_new_matched_tokens(request, 4) == (8, True)
    connector.update_state_after_alloc(request, blocks, 8)
    connector.update_connector_output(output)
    assert connector.request_finished(request, [1, 2]) == (True, {"remote": True})

    connector.connector_scheduler.update_state_after_alloc.assert_called_once_with(request, blocks, 8)
    connector.connector_scheduler.request_finished.assert_called_once_with(request, ([1, 2],))


def test_facade_delegates_worker_methods_and_metadata() -> None:
    connector = make_facade()
    connector.connector_worker.get_finished.return_value = ({"sent"}, {"received"})
    results = KVConnectorTransferResults(
        finished_sending={"sent"}, finished_recving={"received", "failed"}, failed_recving={"failed"}
    )
    connector.connector_worker.get_transfer_results.return_value = results
    connector.connector_worker.get_block_ids_with_load_errors.return_value = {10}
    connector.connector_worker.xfer_handshake_metadata = MagicMock()
    connector._connector_metadata.requests["request"] = MagicMock()

    connector.register_kv_caches({})
    connector.start_load_kv(MagicMock())

    assert connector.get_finished(set()) == ({"sent"}, {"received"})
    assert connector.get_transfer_results(set()) is results
    connector.connector_worker.get_transfer_results.assert_called_once_with()
    assert connector.get_block_ids_with_load_errors() == {10}
    assert connector.get_handshake_metadata() is connector.connector_worker.get_handshake_metadata.return_value
    connector.connector_worker.get_handshake_metadata.assert_called_once_with()
    connector.connector_worker.start_load_kv.assert_called_once_with(connector._connector_metadata)


@pytest.mark.parametrize(
    "connector_cls, prefix", [(MooncakePullConnector, "MooncakePull"), (MooncakePushConnector, "MooncakePush")]
)
def test_connector_selects_implementation_by_role(connector_cls, prefix) -> None:
    config = MagicMock()
    config.kv_transfer_config.engine_id = "engine"
    config.kv_transfer_config.kv_role = "kv_producer"
    kv_cache_config = MagicMock()

    def initialize_base(self, *_args, **_kwargs) -> None:
        self.engine_id = "engine"
        self.connector_scheduler = None
        self.connector_worker = None

    with (
        patch.object(MooncakeBaseConnector, "__init__", initialize_base),
        patch.object(
            connector_module,
            f"{prefix}ConnectorScheduler",
            return_value=MagicMock(),
        ) as scheduler_cls,
        patch.object(
            connector_module,
            f"{prefix}ConnectorWorker",
            return_value=MagicMock(),
        ) as worker_cls,
    ):
        scheduler_connector = connector_cls(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        worker_connector = connector_cls(config, KVConnectorRole.WORKER, kv_cache_config)

    assert scheduler_connector.connector_scheduler is scheduler_cls.return_value
    assert worker_connector.connector_worker is worker_cls.return_value

    assert scheduler_connector.connector_worker is None
    assert worker_connector.connector_scheduler is None
    scheduler_cls.assert_called_once_with(config, "engine", kv_cache_config)
    worker_cls.assert_called_once_with(config, "engine", kv_cache_config)
    scheduler_connector.shutdown()
    scheduler_cls.return_value.shutdown.assert_called_once_with()


def test_idle_producer_ack_releases_blocks_through_native_worker_output(monkeypatch: pytest.MonkeyPatch) -> None:
    connector_scheduler = make_pull_scheduler()
    connector_scheduler._sending_thread = make_sending_thread()
    scheduler_connector = make_facade()
    scheduler_connector.connector_scheduler = connector_scheduler
    scheduler_connector.connector_worker = None
    block_size = connector_scheduler.group_block_size[0]
    cache_manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=make_full_spec())],
        ),
        max_model_len=block_size * 4,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=False,
    )
    request = Request(
        request_id="request-p",
        prompt_token_ids=list(range(block_size * 2)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    request.kv_transfer_params = {"do_remote_decode": True}
    free_before = cache_manager.block_pool.get_num_free_blocks()
    blocks = cache_manager.allocate_slots(request, num_new_tokens=request.num_prompt_tokens)
    assert blocks is not None
    retained_blocks = list(blocks.blocks[0])
    request.append_output_token_ids(123)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delay_free, _ = scheduler_connector.request_finished_all_groups(request, blocks.get_block_ids())
    assert delay_free
    assert all(block.ref_cnt == 1 for block in retained_blocks)

    # Deliver the D scheduler's global completion at the network boundary.
    socket = MagicMock()
    socket.recv_multipart.side_effect = [
        [b"consumer", b"", msgspec.msgpack.encode((DONE_RECVING_MSG, request.request_id))],
        KeyboardInterrupt(),
    ]
    with pytest.raises(KeyboardInterrupt):
        connector_scheduler._sending_thread._run_busy_loop(socket)

    workers = []
    for _ in range(2):
        worker = make_connector_worker(monkeypatch)
        facade = make_facade()
        facade.connector_scheduler = None
        facade.connector_worker = worker
        # Skip cache registration; exercise the actual V2 no-forward lifecycle.
        native = ActiveKVConnector.__new__(ActiveKVConnector)
        native.vllm_config = worker.vllm_config
        native.kv_connector = facade
        native._pending_load_kwargs = None
        native._disabled = False
        workers.append(native)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = scheduler_connector
    scheduler.kv_cache_manager = cache_manager
    scheduler.requests = {request.request_id: request}
    scheduler.defer_block_free = False
    aggregator = KVOutputAggregator(expected_finished_count=len(workers))

    for first_completion in (True, False):
        step = SchedulerOutput.make_empty()
        step.kv_connector_metadata = scheduler_connector.build_connector_meta(step)
        if first_completion:
            # Building metadata must not release the scheduler's retained blocks.
            assert all(block.ref_cnt == 1 for block in retained_blocks)
        worker_outputs = [native.no_forward(step) for native in workers]
        if first_completion:
            assert all(output.kv_connector_output is not None for output in worker_outputs), (
                "The completion ACK must survive V2 empty-output compaction"
            )
        # The aggregator mutates its selected output; do not mutate the shared
        # EMPTY_MODEL_RUNNER_OUTPUT returned on the following idle step.
        output = aggregator.aggregate([copy(worker_output) for worker_output in worker_outputs])
        assert output is not None
        assert output.kv_connector_output is not None
        assert output.kv_connector_output.finished_sending == ({request.request_id} if first_completion else None)
        scheduler._update_from_kv_xfer_finished(output.kv_connector_output)
        assert all(block.ref_cnt == 0 for block in retained_blocks)
        assert cache_manager.block_pool.get_num_free_blocks() == free_before
        assert request.request_id not in scheduler.requests
