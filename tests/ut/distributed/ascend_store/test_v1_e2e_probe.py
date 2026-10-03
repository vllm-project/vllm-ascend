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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    LoadCommand,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules.memory import (
    KVMemoryRule,
    bulk_arguments,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend.io import (
    BackendIO,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.batch import (
    KVGroupBatch,
    KVTransferBatch,
)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cache = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    raw_bytes = cache.view(torch.uint8).flatten()
    objects: dict[str, torch.Tensor] = {}
    backend = SimpleNamespace(mode="copy")
    memory = KVMemoryRule(
        group_ids=(0,),
        block_sizes={0: 4},
        align_state_group_ids=frozenset(),
        physical_layers={0: (0,)},
        base_addresses={0: (cache.data_ptr(),)},
        block_lengths={0: (64,)},
        block_strides={0: (64,)},
        layer_entry_offsets={0: (0, 1)},
        strided_slice_count=1,
        consumer_pipeline_partitions=None,
        store_pipeline_ranks=None,
        data_plane="bulk",
        requires_global_offsets=False,
        object_sizes=None,
        object_offsets=None,
    )
    rules = SimpleNamespace(
        memory=memory,
        format_ranges=lambda key_axes, ranges, *, object_bases=None, selected_objects=None: bulk_arguments(
            key_axes,
            ranges,
            selected_objects,
        ),
    )
    batch = KVTransferBatch(
        ("request",),
        (
            KVGroupBatch(
                0,
                np.asarray([0, 1], dtype=np.uint64),
                np.asarray([4, 4], dtype=np.uint64),
                (("key-0", "key-1"),),
                np.asarray([0, 2], dtype=np.intp),
                (0,),
                64,
            ),
        ),
    )

    def put(keys, addresses, sizes):
        for key, key_addresses, key_sizes in zip(keys, addresses, sizes, strict=True):
            offset = key_addresses[0] - cache.data_ptr()
            objects[key] = raw_bytes[offset : offset + key_sizes[0]].clone()
        return [0] * len(keys)

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

    backend.put = put
    backend.get = get
    backend_io = BackendIO(backend, SimpleNamespace(name="fake"))
    backend_io.bind_rules(rules)
    load_command = LoadCommand("request", TokenRange(0, 8), ((0, 1),), ("0", "1"))
    runtime = SimpleNamespace(
        _resources=SimpleNamespace(kv_caches={"layer": cache}, backend=backend),
        _backend_io=backend_io,
        _bound_rules=rules,
        _spec=SimpleNamespace(topology=SimpleNamespace(cache_transfer_granularity=4)),
        _build_load_batch=lambda _commands: batch,
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
    return SimpleNamespace(
        runtime=runtime,
        batch=batch,
        load_command=load_command,
        cache=cache,
        connector=connector,
    )


def test_probe_verifies_real_copy_after_erasing_source(worker: SimpleNamespace) -> None:
    assert install_worker_io_probe(worker) == 4
    expected = worker.cache.clone()
    worker.runtime._backend_io.store_batch(worker.batch)
    cold_evidence = clear_worker_local_kv(worker)
    worker.runtime.fence_previous_store.assert_called_once()
    assert torch.count_nonzero(worker.cache) == 0
    assert cold_evidence["get_calls"] == 0
    assert cold_evidence["local_kv_cleared"]

    load_batch = worker.runtime._build_load_batch((worker.load_command,))
    worker.runtime._backend_io.load_batch(load_batch)
    assert torch.equal(worker.cache, expected)
    warm_evidence = collect_worker_io_probe(worker)
    assert warm_evidence["get_calls"] == 1
    assert warm_evidence["loaded_keys"] == cold_evidence["stored_keys"]
    assert warm_evidence["loaded_ranges"] == [[0, 4], [4, 8]]
    assert warm_evidence["loaded_bytes"] == cold_evidence["stored_bytes"] == 128


def test_probe_evidence_survives_collective_rpc_codec(worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    install_worker_io_probe(worker)
    worker.runtime._backend_io.store_batch(worker.batch)
    clear_worker_local_kv(worker)
    load_batch = worker.runtime._build_load_batch((worker.load_command,))
    worker.runtime._backend_io.load_batch(load_batch)
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
    worker.runtime._backend_io.store_batch(worker.batch)
    clear_worker_local_kv(worker)
    worker.runtime._resources.backend.mode = mode
    load_batch = worker.runtime._build_load_batch((worker.load_command,))
    with pytest.raises(AssertionError):
        worker.runtime._backend_io.load_batch(load_batch)
    assert collect_worker_io_probe(worker)["get_calls"] == 0


def test_probe_does_not_count_an_empty_load_as_get(worker: SimpleNamespace) -> None:
    install_worker_io_probe(worker)
    assert worker.runtime._backend_io.load_batch(KVTransferBatch((), ())) == ()
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
