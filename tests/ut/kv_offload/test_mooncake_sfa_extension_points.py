# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import (
    MooncakeSFAConnectorWorker,
    MooncakeSFAPullRecvingThread,
)

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.helpers import make_pp_metadata


def test_sfa_facade_forwards_sparse_preparation_to_its_worker() -> None:
    connector = MooncakePullConnector.__new__(MooncakePullConnector)
    worker = MagicMock(spec=MooncakeSFAConnectorWorker)
    connector.connector_worker = worker
    metadata = MooncakeSFAConnectorMetadata()
    prepare = getattr(connector, "prepare_sparse_kv_offload", None)
    assert prepare is not None
    prepare(metadata)
    worker.prepare_sparse_kv_offload.assert_called_once_with(metadata)


def test_sfa_receiver_rejects_missing_reader_before_starting() -> None:
    with pytest.raises(ValueError, match="requires a staging reader"):
        MooncakeSFAPullRecvingThread(staging_reader=None)


@pytest.mark.parametrize("result", [0, -1])
def test_execute_bucket_waits_for_staged_landing_and_propagates_failure(result):
    reader = MagicMock()
    reader.batch_transfer_sync_read.return_value = result
    thread = make_thread(engine=MagicMock(), staging_reader=reader)
    remote = make_pp_metadata()
    entries = {0: {(0, 0): [("request", [1], [2])]}}
    if result < 0:
        with pytest.raises(RuntimeError, match="transfer failed"):
            thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, remote, entries)
    else:
        thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, remote, entries)
    reader.batch_transfer_sync_read.assert_called_once_with("10.0.0.1:9000", [1128], [5256], [128], "request")
    thread.engine.batch_transfer_sync_read.assert_not_called()
