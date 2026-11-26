# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise inherited worker synchronization without replacing Core methods."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm_ascend.hisparse.backend import AscendHiSparseBackend  # noqa: E402


@pytest.mark.parametrize("extra_position", [None, "before", "between", "after"])
def test_pd_host_mirror_defers_only_shared_consumer_writer(monkeypatch, extra_position):
    from vllm.distributed import parallel_state
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
    from vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd import _HiSparseMooncakePullConnector, _HiSparsePDWorker
    from vllm_ascend.hisparse import host_mirror
    from vllm_ascend.hisparse.connector import HiSparseConnector
    from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker

    calls = []
    device = NS(type="npu")
    output = NS(device=device, stream_id=1, device_index=0, device_type=20)
    output.wait_event = lambda event: calls.append("output wait")
    compute = NS(device=device, wait_event=lambda event: calls.append("compute wait"))
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: compute)
    monkeypatch.setattr(torch, "Stream", lambda **kwargs: output)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(host_mirror._DirectMirrorSources, "bind", Mock(return_value=NS(can_defer=lambda worker: True)))

    worker = AscendHiSparseConnectorWorker.__new__(AscendHiSparseConnectorWorker)
    worker._initialized = worker.is_host_writer = True
    worker.backend = AscendHiSparseBackend()
    worker.shared_host_region = object()
    worker.cache_handles = [NS(runtime=NS(device=device))]
    worker._forward_ready_event = worker.host_write_event = Mock()
    worker._pending_dma_descriptors = []
    worker._dma_submitted = False
    worker._in_step = True
    worker._release_completed_dma_descriptors = Mock()

    def submit_mirror(event):
        calls.append("target DMA")
        worker._dma_submitted = True
        worker._pending_dma_descriptors.append(object())

    worker._finish_mirror_phase = submit_mirror
    hisparse = HiSparseConnector.__new__(HiSparseConnector)
    hisparse.connector_worker = worker
    hisparse.finish_forward = lambda: calls.append("regular HiSparse")
    mooncake = _HiSparseMooncakePullConnector.__new__(_HiSparseMooncakePullConnector)
    mooncake.connector_worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    mooncake.connector_worker.kv_transfer_config = NS(is_kv_consumer=True)
    mooncake.finish_forward = lambda: calls.append("Mooncake")
    extra = NS(finish_forward=lambda: calls.append("extra"))
    order = ["hisparse", "Mooncake"]
    if extra_position is not None:
        order.insert({"before": 0, "between": 1, "after": 2}[extra_position], "extra")
    children = {"hisparse": hisparse, "Mooncake": mooncake, "extra": extra}
    multi = AscendMultiConnector.__new__(AscendMultiConnector)
    multi._connectors = [children[name] for name in order]
    inner = ActiveKVConnector.__new__(ActiveKVConnector)
    inner.kv_connector, inner._disabled = multi, False
    inner.pre_forward = Mock()
    inner.post_forward = Mock()
    inner.finalize_step = Mock(side_effect=lambda: calls.append("finalize"))
    runner = NS(kv_connector=inner, output_copy_stream=output, speculator=object())

    proxy = host_mirror.overlap_target_host_mirror(runner)
    assert isinstance(proxy, host_mirror.HostMirrorConnector)
    scheduler_output = NS(has_sync_kv_loads=True, skip_drafts=False)
    batch = NS(has_prefill=False, num_tokens=1, num_tokens_after_padding=1)
    proxy.begin_execution(True)
    proxy.pre_forward(scheduler_output, batch, attn_metadata={})
    proxy.finish_forward()
    assert not worker._in_step
    proxy.post_forward(set(), defer_finalize=True)
    expected = [step for name in order for step in (["target DMA", "output wait"] if name == "hisparse" else [name])]
    assert calls == expected
    proxy.finalize_step()
    assert calls == [*expected, "compute wait", "finalize"]

    calls.clear()
    scheduler_output.skip_drafts = True
    proxy.begin_execution(True)
    proxy.pre_forward(scheduler_output, batch, attn_metadata={})
    proxy.finish_forward()
    assert calls == ["regular HiSparse" if name == "hisparse" else name for name in order]
    worker.is_host_writer = False
    assert host_mirror.overlap_target_host_mirror(runner) is inner
    worker.is_host_writer = True
    mooncake.connector_worker.kv_transfer_config.is_kv_consumer = False
    assert host_mirror.overlap_target_host_mirror(runner) is inner


@pytest.mark.parametrize("shared", [False, True])
def test_finish_previous_step_orders_local_wait_before_shared_broadcast(monkeypatch, shared):
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker

    from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker

    calls: list[object] = []
    stream = NS(wait_event=lambda event: calls.append(("wait", event)))
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: stream)
    group = NS(broadcast=lambda tensor, src: calls.append(("broadcast", tensor, src)))
    monkeypatch.setattr(worker, "get_tp_group", lambda: group)
    monkeypatch.setattr(worker.HiSparseConnectorWorker, "_finish_previous_step", lambda self: calls.append("finish"))

    state = AscendHiSparseConnectorWorker.__new__(AscendHiSparseConnectorWorker)
    state.host_write_event = object()
    state.shared_host_region = object() if shared else None
    state._host_write_sync = object() if shared else None
    state._finish_previous_step()

    expected: list[object] = ["finish"]
    if shared:
        expected += [("wait", state.host_write_event), ("broadcast", state._host_write_sync, 0)]
    assert calls == expected


def test_worker_extends_core_dma_setup(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker as core_worker

    from vllm_ascend.hisparse import worker

    calls = []
    monkeypatch.setattr(worker, "initialize_worker_dma", lambda state: calls.append("ascend"))
    monkeypatch.setattr(core_worker.HiSparseConnectorWorker, "_init_dma", lambda state: calls.append("core"))
    state = worker.AscendHiSparseConnectorWorker.__new__(worker.AscendHiSparseConnectorWorker)
    state._init_dma()
    assert calls == ["core", "ascend"]


def test_worker_inherits_core_step_dma_and_finish():
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker

    for name in ("start_step", "_submit_dma_descriptors", "finish_forward"):
        assert getattr(AscendHiSparseConnectorWorker, name) is getattr(HiSparseConnectorWorker, name)


def test_worker_uses_h2d_for_restore_and_resets_direction_on_failure(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker

    submit = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", submit, raising=False)
    descriptors = torch.empty(1, dtype=torch.int64)

    def restore(worker, transfers):
        worker.backend.swap_blocks_batch(descriptors, descriptors, descriptors)
        if not transfers:
            raise RuntimeError("restore failed")

    monkeypatch.setattr(HiSparseConnectorWorker, "_restore_pages", restore)
    worker = AscendHiSparseConnectorWorker.__new__(AscendHiSparseConnectorWorker)
    worker.backend = AscendHiSparseBackend()
    worker.shared_host_region = None

    worker._restore_pages([object()])
    with pytest.raises(RuntimeError, match="restore failed"):
        worker._restore_pages([])
    worker.backend.swap_blocks_batch(descriptors, descriptors, descriptors)

    assert [call.args[3] for call in submit.call_args_list] == [0, 0, 1]


@pytest.mark.parametrize(
    ("device_base", "device_type", "batch_size"),
    [
        (4096, "A5", 2),
        (8192, "A5", 2),
        (8192, "A5", 64),
        (8192, "A5", 65),
        (8192, "A3", 48),
        (8192, "A3", 49),
    ],
)
def test_follower_restores_from_local_device_mapping(monkeypatch, device_base, device_type, batch_size):
    import sys

    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker

    submit = Mock()
    sparse_copy = Mock(return_value=0)
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", submit, raising=False)
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", NS(offload=NS(sparse_copy=sparse_copy)))

    page_bytes = (1152 if device_type == "A3" else 656) * 128
    offsets = torch.arange(batch_size, dtype=torch.int64) * page_bytes + 16
    source = 4096 + offsets
    destination = 16384 + offsets
    sizes = torch.full_like(source, page_bytes)

    def restore(worker, transfers):
        worker.backend.swap_blocks_batch(source, destination, sizes)

    monkeypatch.setattr(HiSparseConnectorWorker, "_restore_pages", restore)
    worker = AscendHiSparseConnectorWorker.__new__(AscendHiSparseConnectorWorker)
    worker.backend = AscendHiSparseBackend()
    worker.is_host_writer = False
    mapping = NS(host_address=4096, device_ptr=device_base, device=torch.device("cpu"))
    worker.shared_host_region = NS(base_tensor=NS(_hisparse_registration=mapping))

    worker._restore_pages([object()])

    submit.assert_not_called()
    sparse_copy.assert_called_once()
    src, dst, lengths, count, device = sparse_copy.call_args.args
    assert src.tolist() == (device_base + offsets).tolist()
    assert dst.tolist() == destination.tolist()
    assert lengths.dtype == count.dtype == torch.int32
    assert lengths.tolist() == [page_bytes] * batch_size
    assert count.tolist() == [batch_size]
    assert device == mapping.device
    assert source.tolist() == (4096 + offsets).tolist()

    worker.backend.swap_blocks_batch(source, destination, sizes)
    assert submit.call_args.args[3] == 1


def test_worker_reuses_core_local_events(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.hisparse import worker

    events = (object(), object())
    manager = NS(is_host_writer=False, shared_region=None)
    monkeypatch.setattr(worker, "uses_memfabric_pd", lambda transfer: True)
    monkeypatch.setattr(worker.AscendHiSparseOffloadManager, "from_host_pool", lambda pool: manager)

    def initialize(state, *args, **kwargs):
        assert kwargs["shared_host_region"] is None
        assert kwargs["is_host_writer"] is False
        state.host_write_events = events
        state.host_write_event = events[1]

    monkeypatch.setattr(HiSparseConnectorWorker, "initialize", initialize)
    state = worker.AscendHiSparseConnectorWorker.__new__(worker.AscendHiSparseConnectorWorker)
    state.kv_cache_config = object()
    state.initialize([NS(runtime=NS(host_pool=NS(vllm_config=NS(kv_transfer_config=None))))], [], 1, 1, "cpu", [])
    assert state.host_write_events is events
    assert state.host_write_event is events[1]
