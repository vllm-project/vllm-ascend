# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_dsa_metadata as dsa,
)


def _endpoint(rank: int = 0) -> dsa.RemoteEndpoint:
    return dsa.RemoteEndpoint(
        remote_host=f"10.0.0.{rank + 1}",
        remote_port=9000 + rank,
        remote_engine_id=f"prefill-{rank}",
    )


def _source() -> dsa.RemoteSource:
    return dsa.RemoteSource(
        remote_request_id="prefill-request",
        endpoints_by_prefill_rank=(_endpoint(0), _endpoint(1)),
        indexer_block_ids=(2, 3),
        main_block_ids=(4, 5),
    )


def _request(request_id: str = "decode-request") -> dsa.DsaStepRequest:
    return dsa.DsaStepRequest(
        request_id=request_id,
        source=_source(),
        main_host_block_ids=(10, 11),
        indexer_hbm_block_ids=(20, 21),
    )


def test_remote_endpoint_validates_port() -> None:
    with pytest.raises(ValueError, match="1..65535"):
        dsa.RemoteEndpoint("127.0.0.1", 0, "engine")


def test_remote_source_normalizes_block_ids() -> None:
    source = dsa.RemoteSource(
        "prefill-request",
        [_endpoint()],
        [1, 2],
        [3, 4],
    )
    assert source.endpoints_by_prefill_rank == (_endpoint(),)
    assert source.indexer_block_ids == (1, 2)
    assert source.main_block_ids == (3, 4)


@pytest.mark.parametrize(
    "field",
    ("indexer_block_ids", "main_block_ids"),
)
def test_remote_source_rejects_empty_block_ids(field: str) -> None:
    kwargs = {
        "remote_request_id": "prefill-request",
        "endpoints_by_prefill_rank": (_endpoint(),),
        "indexer_block_ids": (1,),
        "main_block_ids": (2,),
    }
    kwargs[field] = ()
    with pytest.raises(ValueError, match="must not be empty"):
        dsa.RemoteSource(**kwargs)


def test_step_request_owns_only_one_shot_destinations() -> None:
    request = _request()
    assert request.main_host_block_ids == (10, 11)
    assert request.indexer_hbm_block_ids == (20, 21)
    assert not hasattr(request, "lifecycle")
    assert not hasattr(request, "main_reservation_id")


def test_connector_metadata_sorts_requests() -> None:
    metadata = dsa.DsaConnectorMetadata(
        (_request("request-b"), _request("request-a"))
    )
    assert tuple(item.request_id for item in metadata.requests) == (
        "request-a",
        "request-b",
    )


def test_connector_metadata_rejects_conflicting_request() -> None:
    first = _request()
    second = dsa.DsaStepRequest(
        request_id=first.request_id,
        source=first.source,
        main_host_block_ids=(12,),
        indexer_hbm_block_ids=(22,),
    )
    with pytest.raises(ValueError, match="conflicting"):
        dsa.DsaConnectorMetadata((first, second))


def test_transfer_failure_requires_phase() -> None:
    with pytest.raises(ValueError, match="requires a failure phase"):
        dsa.DsaLocalResult(
            request_id="request",
            tp_rank=0,
            kind=dsa.DsaLocalResultKind.TRANSFER_FAILED,
        )


def test_worker_metadata_aggregates_tp_results() -> None:
    left = dsa.DsaWorkerResultMetadata(
        (
            dsa.DsaLocalResult(
                "request",
                0,
                dsa.DsaLocalResultKind.RECEIVE_COMPLETE,
            ),
        )
    )
    right = dsa.DsaWorkerResultMetadata(
        (
            dsa.DsaLocalResult(
                "request",
                1,
                dsa.DsaLocalResultKind.RECEIVE_COMPLETE,
            ),
        )
    )
    merged = left.aggregate(right)
    assert tuple(result.tp_rank for result in merged.results) == (0, 1)


def test_worker_metadata_rejects_conflicting_result() -> None:
    success = dsa.DsaWorkerResultMetadata(
        (
            dsa.DsaLocalResult(
                "request",
                0,
                dsa.DsaLocalResultKind.RECEIVE_COMPLETE,
            ),
        )
    )
    failure = dsa.DsaWorkerResultMetadata(
        (
            dsa.DsaLocalResult(
                "request",
                0,
                dsa.DsaLocalResultKind.TRANSFER_FAILED,
                dsa.DsaTransferPhase.INDEXER_D2D,
            ),
        )
    )
    with pytest.raises(ValueError, match="conflicting"):
        success.aggregate(failure)
