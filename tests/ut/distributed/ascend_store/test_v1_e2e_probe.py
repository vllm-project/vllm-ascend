# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the E2E probe rejects failed or silently skipped Backend copies on CPU."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.engine import EngineCoreOutputs, UtilityOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

from tests.e2e.common.kv_pool.ascendstore_v1_probe import (
    _snapshot_buffers,
    clear_worker_local_kv,
    collect_worker_io_probe,
    install_worker_io_probe,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.lowering import (
    bind_transfer_rows,
    enumerate_transfer_work,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.evidence import (
    StoreEvidence,
    TransferEvidence,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.representation import (
    KVBlockAssignment,
    KVBlockAssignmentBatch,
    KVChunk,
    PhysicalCoordinate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.selection import (
    TransferWork,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.transfer import (
    BoundGroupPlan,
    ContiguousLayoutPlan,
    SubmissionPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend.arguments import (
    materialize_ranges,
)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cache = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    raw_bytes = cache.view(torch.uint8).flatten()
    objects: dict[str, torch.Tensor] = {}
    backend = SimpleNamespace(mode="copy")
    layout = ContiguousLayoutPlan(
        (0,),
        0,
        64,
        0,
        np.asarray([cache.data_ptr()], dtype=np.uint64),
        np.asarray([64], dtype=np.uint64),
        np.asarray([16], dtype=np.uint64),
        np.asarray([0], dtype=np.uint64),
        4,
    )
    plan = BoundGroupPlan(
        0,
        (PhysicalCoordinate(),),
        ("key-",),
        (layout,),
        (SubmissionPlan(None, (0,)),),
        2,
    )
    chunks = tuple(KVChunk(0, index, TokenRange(index * 4, (index + 1) * 4), str(index)) for index in range(2))
    rows = bind_transfer_rows(
        plan,
        KVBlockAssignmentBatch(
            0,
            tuple(KVBlockAssignment(chunk, index, 4) for index, chunk in enumerate(chunks)),
        ),
    )
    work = enumerate_transfer_work((rows,))[0]

    def store(selected_work):
        arguments = materialize_ranges(selected_work)
        for key, addresses, sizes in zip(arguments.keys, arguments.addresses, arguments.sizes, strict=True):
            offset = addresses[0] - cache.data_ptr()
            objects[key] = raw_bytes[offset : offset + sizes[0]].clone()
        evidence = tuple(TransferEvidence(source, 0, True) for source in selected_work.sources)
        return StoreEvidence(evidence, True, True)

    def get(keys, addresses, sizes):
        if backend.mode == "fail":
            return [-1] * len(keys)
        if backend.mode == "short":
            return [0]
        if backend.mode == "missing":
            return None
        if backend.mode in ("copy", "partial"):
            for index, (key, buffers, lengths) in enumerate(zip(keys, addresses, sizes, strict=True)):
                if index and backend.mode == "partial":
                    break
                offset = buffers[0] - cache.data_ptr()
                raw_bytes[offset : offset + lengths[0]].copy_(objects[key])
        return [0] * len(keys)

    def load(selected_work):
        if selected_work.empty:
            return ()
        arguments = materialize_ranges(selected_work)
        return tuple(
            TransferEvidence(source, code)
            for source, code in zip(
                selected_work.sources,
                backend.get(arguments.keys, arguments.addresses, arguments.sizes),
                strict=True,
            )
        )

    backend.get = get
    runtime = SimpleNamespace(
        _resources=SimpleNamespace(kv_caches={"layer": cache}, backend=backend),
        _backend_io=SimpleNamespace(store=store, load=load),
        _program=SimpleNamespace(topology=SimpleNamespace(cache_transfer_granularity=4)),
        fence_previous_store=Mock(),
    )
    connector_type = type(
        "AscendStoreV1Connector",
        (),
        {"__module__": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"},
    )
    connector = connector_type()
    connector.runtime = runtime
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.kv_transfer", SimpleNamespace(get_kv_transfer_group=lambda: connector)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    return SimpleNamespace(runtime=runtime, work=work, cache=cache, connector=connector)


def test_probe_verifies_real_copy_after_erasing_source(worker: SimpleNamespace) -> None:
    assert install_worker_io_probe(worker) == 4
    expected = worker.cache.clone()
    worker.runtime._backend_io.store(worker.work)
    cold_evidence = clear_worker_local_kv(worker)
    worker.runtime.fence_previous_store.assert_called_once()
    assert torch.count_nonzero(worker.cache) == 0
    assert cold_evidence["get_calls"] == 0
    assert cold_evidence["local_kv_cleared"]

    worker.runtime._backend_io.load(worker.work)
    assert torch.equal(worker.cache, expected)
    warm_evidence = collect_worker_io_probe(worker)
    assert warm_evidence["get_calls"] == 1
    assert warm_evidence["loaded_keys"] == cold_evidence["stored_keys"]
    assert warm_evidence["loaded_ranges"] == [[0, 4], [4, 8]]
    assert warm_evidence["loaded_bytes"] == cold_evidence["stored_bytes"] == 128


def test_probe_evidence_survives_collective_rpc_codec(worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    install_worker_io_probe(worker)
    worker.runtime._backend_io.store(worker.work)
    clear_worker_local_kv(worker)
    worker.runtime._backend_io.load(worker.work)
    evidence = collect_worker_io_probe(worker)
    outputs = EngineCoreOutputs(utility_output=UtilityOutput(call_id=1, result=UtilityResult([evidence])))

    # Exercise the utility response codec used by LLM.collective_rpc, not a direct callback or pickle round trip.
    frames = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(frames).utility_output
    assert decoded is not None and decoded.result is not None
    assert decoded.result.result == [evidence]


@pytest.mark.parametrize("mode", ["noop", "partial", "fail", "short", "missing"])
def test_probe_rejects_false_get_success(worker: SimpleNamespace, mode: str) -> None:
    install_worker_io_probe(worker)
    worker.runtime._backend_io.store(worker.work)
    clear_worker_local_kv(worker)
    worker.runtime._resources.backend.mode = mode
    with pytest.raises(AssertionError):
        worker.runtime._backend_io.load(worker.work)
    assert collect_worker_io_probe(worker)["get_calls"] == 0


def test_probe_does_not_count_an_empty_load_as_get(worker: SimpleNamespace) -> None:
    install_worker_io_probe(worker)
    assert worker.runtime._backend_io.load(TransferWork(None, ())) == ()
    assert collect_worker_io_probe(worker)["get_calls"] == 0


def test_probe_rejects_a_different_connector(worker: SimpleNamespace) -> None:
    worker.connector.__class__.__module__ = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
    with pytest.raises(AssertionError):
        install_worker_io_probe(worker)


def test_snapshot_uses_storage_offsets_and_rejects_out_of_bounds() -> None:
    storage = torch.arange(16, dtype=torch.int32)
    cache = storage[3::2]
    snapshot = _snapshot_buffers((cache,), (storage.data_ptr() + 4,), (8,))[0]
    assert torch.equal(snapshot, storage.view(torch.uint8)[4:12])
    storage.zero_()
    assert torch.count_nonzero(snapshot) > 0
    with pytest.raises(AssertionError, match="outside registered KV storage"):
        _snapshot_buffers((cache,), (storage.data_ptr() + 60,), (8,))
