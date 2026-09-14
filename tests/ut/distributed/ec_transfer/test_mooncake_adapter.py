# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ECConnectorOutput

from vllm_ascend.distributed.ec_transfer import register_connector
from vllm_ascend.distributed.ec_transfer.mooncake import (
    AscendECMooncakeConnector,
    AscendMooncakeTransfer,
    _AscendECMooncakeWorker,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_registers_concrete_ascend_connector() -> None:
    with patch.dict(ECConnectorFactory._registry):
        register_connector()
        connector = ECConnectorFactory._registry["ECMooncakeConnector"]()

    assert connector is AscendECMooncakeConnector
    assert not inspect.isabstract(connector)


def test_rejects_non_ascend_transport() -> None:
    with pytest.raises(ValueError, match="mooncake_protocol='ascend'"):
        AscendMooncakeTransfer("127.0.0.1", "rdma")


def test_transfer_uses_shared_engine_for_registration_and_write() -> None:
    engine = Mock()
    engine.get_rpc_port.return_value = 1234
    engine.batch_register_memory.return_value = 0
    engine.batch_transfer_sync_write.return_value = 0
    tensor = Mock(nbytes=4096)
    tensor.data_ptr.return_value = 2097152
    with patch(
        "vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine",
        return_value=engine,
    ) as get_engine:
        transfer = AscendMooncakeTransfer("127.0.0.1", "ascend")
        get_engine.assert_not_called()
        transfer.ensure_ready()
        assert transfer.local_session() == "127.0.0.1:1234"
        assert transfer.register_memory(tensor) == 0
        transfer.write("peer:1234", [2097152], [4194304], [4096])
        get_engine.assert_called_once_with("127.0.0.1", device_name=None)
    engine.batch_register_memory.assert_called_once_with([2097152], [4096])
    engine.batch_transfer_sync_write.assert_called_once_with("peer:1234", [2097152], [4194304], [4096])
    engine.batch_transfer_sync_write.return_value = -1
    with pytest.raises(RuntimeError, match="failed with status -1"):
        transfer.write("peer:1234", [2097152], [4194304], [4096])


@pytest.mark.parametrize("close_status", [0, -1])
def test_transfer_retains_failed_unregistration_until_close(close_status: int) -> None:
    engine = Mock()
    engine.unregister_memory.return_value = -1
    engine.batch_unregister_memory.return_value = close_status
    tensor = Mock(nbytes=4096)
    tensor.data_ptr.return_value = 2097152
    with patch(
        "vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine",
        return_value=engine,
    ):
        transfer = AscendMooncakeTransfer("127.0.0.1", "ascend")
        transfer.register_memory(tensor)
        assert not transfer.unregister_memory(tensor)
        assert transfer._pending_unregister[2097152] is tensor
        transfer.close()
        transfer.close()
    engine.batch_unregister_memory.assert_called_once_with([2097152])
    assert transfer._pending_unregister == ({2097152: tensor} if close_status else {})
    engine.close.assert_not_called()


def test_transfer_rejects_direct_source_registration() -> None:
    transfer = AscendMooncakeTransfer("127.0.0.1", "ascend")
    with patch("vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine") as get_engine:
        with pytest.raises(RuntimeError, match="aligned staging pool"):
            transfer.acquire_sources([Mock()])
        assert transfer.release_sources([])
        transfer.close()
        get_engine.assert_not_called()


@pytest.mark.parametrize("is_producer, is_consumer", [(True, False), (False, True), (True, True)])
def test_worker_reuses_upstream_lifecycle_with_lazy_ascend_data_plane(is_producer: bool, is_consumer: bool) -> None:
    config = SimpleNamespace(
        is_producer=is_producer,
        is_consumer=is_consumer,
        control_host="127.0.0.1",
        control_port=14579,
        control_timeout_ms=10,
        buffer_device="cuda",
        pool_size=4096,
        protocol="ascend",
    )
    vllm_config = SimpleNamespace(ec_transfer_config=SimpleNamespace(ec_connector_extra_config={}))
    with (
        patch(
            "vllm_ascend.distributed.ec_transfer.mooncake.MooncakeECConfig.from_vllm_config",
            return_value=config,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.mooncake.transfer.MooncakeTransfer._ensure_engine",
            side_effect=AssertionError("upstream engine must stay lazy"),
        ),
        patch(
            "vllm_ascend.distributed.ec_transfer.mooncake.global_te.get_transfer_engine",
            side_effect=AssertionError("Ascend engine must stay lazy"),
        ),
    ):
        worker = _AscendECMooncakeWorker(vllm_config)

        try:
            assert worker._buffer_device == "npu"
            assert isinstance(worker._transfer, AscendMooncakeTransfer)
            assert worker._consumer_memory._transfer is worker._transfer
            assert worker._producer_memory._transfer is worker._transfer
            assert worker._reservations._memory is worker._consumer_memory
            assert worker._consumer_memory.tensor is None
            assert worker._producer_memory.tensor is None
            assert worker._control_server is None
            if is_producer:
                dispatched = threading.Event()

                def submit_batches(*args, **kwargs):
                    dispatched.set()
                    return False

                with patch.object(worker._producer_pushes, "submit_batches", side_effect=submit_batches):
                    worker._push_ready.set()
                    assert dispatched.wait(timeout=5), "upstream dispatcher did not process work"
            else:
                assert worker._dispatcher is None
        finally:
            worker.close()
        assert worker._shutdown
        if is_producer:
            assert not worker._dispatcher.is_alive()


def test_v1_no_forward_preserves_ec_output() -> None:
    runner = object.__new__(NPUModelRunner)
    runner.encoder_cache = {}
    runner.vllm_config = object()
    ec_output = ECConnectorOutput(finished_sending={"image"})

    @contextmanager
    def output_context(*args, **kwargs):
        yield ec_output

    runner.maybe_get_ec_connector_output = output_context
    with (
        patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=True),
        patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False),
    ):
        output = runner._no_forward_output(SimpleNamespace(ec_connector_metadata=object()))

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.ec_connector_output is ec_output


def test_v1_no_forward_without_metadata_is_noop() -> None:
    runner = object.__new__(NPUModelRunner)
    runner.encoder_cache = {}
    runner.vllm_config = object()
    with (
        patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=True),
        patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False),
    ):
        output = runner._no_forward_output(SimpleNamespace(ec_connector_metadata=None))

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
