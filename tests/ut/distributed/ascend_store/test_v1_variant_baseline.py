"""Observable baselines shared by the remaining AscendStore v1 slices."""

from __future__ import annotations

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)

from .v1.helpers import (
    FakeBackend,
    build_admitted_store_batch,
    make_topology,
    make_worker,
    worker_backend_io,
)


class _NamedBulkBackend(FakeBackend):
    def __init__(self, native_result) -> None:
        super().__init__()
        self.native_result = native_result

    def store(self, keys, addresses, sizes):
        self.calls.append(("store", keys, addresses, sizes))
        return self.native_result


@pytest.mark.parametrize(
    ("backend_name", "native_result"),
    [
        ("mooncake", [0]),
        ("memcache", [0]),
    ],
)
def test_bulk_store_uses_port_without_backend_name_branch(
    backend_name: str,
    native_result,
) -> None:
    backend = _NamedBulkBackend(native_result)
    worker, resources, _ = make_worker(
        backend,
        physical_layers=(0,),
        backend_name=backend_name,
    )
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"hash",), 4, 17)

    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        worker.finish_step()
        (completion,) = worker.fence_previous_store()

        native_call = next(call for call in backend.calls if call[0] == "store")
        assert native_call == (
            "store",
            [completion.evidence.transfer_evidence[0].source.key],
            [[1064]],
            [[32]],
        )
        assert completion.evidence.succeeded
        assert completion.evidence.source_release_confirmed
        assert completion.store_job_id == 17
        worker.end_step()
    finally:
        worker.close()
        assert resources.closed


def test_sparse_transfer_groups_keep_original_identity_in_bulk_trace_and_evidence() -> None:
    topology = make_topology(group_ids=(1, 3), physical_layers=(0,))
    backend = FakeBackend()
    backend.put_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=topology)
    command = RangeStoreCommand(
        "request",
        TokenRange(0, 3),
        ((), (1,), (), (2,)),
        (b"hash",),
        3,
        23,
    )
    batch = build_admitted_store_batch(worker, (command,))
    assert batch is not None

    try:
        arguments = worker._materialize_concrete_bulk_arguments(batch, store=True)
        (completion,) = worker_backend_io(worker).store_materialized(batch, arguments)

        put_call = next(call for call in backend.calls if call[0] == "put")
        assert ["@group:1@" in key for key in put_call[1]] == [True, False]
        assert ["@group:3@" in key for key in put_call[1]] == [False, True]
        assert put_call[2] == ((11064,), (31128,))
        assert put_call[3] == ((24,), (24,))
        assert [item.source.group_id for item in completion.evidence.transfer_evidence] == [1, 3]
        assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [1, 2]
        assert [item.source_release_confirmed for item in completion.evidence.transfer_evidence] == [True, True]
        assert [item.result_code for item in completion.evidence.transfer_evidence] == [0, -1]
        assert not completion.evidence.succeeded
        assert completion.evidence.source_release_confirmed
    finally:
        worker.close()
        assert resources.closed
