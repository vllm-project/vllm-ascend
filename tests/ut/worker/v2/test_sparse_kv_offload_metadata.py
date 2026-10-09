# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for reusable draft-offload metadata buffers."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from zlib import adler32

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1 import utils as v1_utils

from vllm_ascend.worker.v2.spec_decode.autoregressive import sparse_kv_offload
from vllm_ascend.worker.v2.spec_decode.autoregressive.sparse_kv_offload import SparseKVOffloadMetadata


@pytest.fixture(autouse=True)
def _cpu_buffers(monkeypatch):
    monkeypatch.setattr(v1_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(sparse_kv_offload, "is_pin_memory_available", lambda: False)


def _metadata(*, enabled=True, num_steps=3):
    return SparseKVOffloadMetadata(
        enabled=enabled,
        num_steps=num_steps,
        max_num_reqs=4,
        max_num_tokens=8,
        device=torch.device("cpu"),
    )


def _batch(*, dummy=False):
    return SimpleNamespace(req_ids=["live-a", "live-b", "not-scheduled"], is_dummy=dummy)


def _state(*, pool=True):
    if not pool:
        return SimpleNamespace(_offload_pool_slots=None, _offload_pool_active=None)
    slots = v1_utils.CpuGpuBuffer(4, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    active = v1_utils.CpuGpuBuffer(4, dtype=torch.bool, device=torch.device("cpu"), pin_memory=False)
    slots.cpu.copy_(torch.tensor([7, 8, 9, 10]))
    active.cpu.copy_(torch.tensor([True, False, True, False]))
    return SimpleNamespace(_offload_pool_slots=slots, _offload_pool_active=active)


def _descriptor(mode=CUDAGraphMode.FULL, *, num_reqs=4):
    return SimpleNamespace(cg_mode=mode, num_reqs=num_reqs, num_tokens=8)


def test_disabled_does_not_allocate_or_read_batch(monkeypatch):
    allocate = MagicMock(side_effect=AssertionError("disabled offload allocated buffers"))
    pin_memory = MagicMock(side_effect=AssertionError("disabled offload queried pin memory"))
    update = MagicMock(side_effect=AssertionError("disabled offload updated metadata"))
    monkeypatch.setattr(sparse_kv_offload, "CpuGpuBuffer", allocate)
    monkeypatch.setattr(sparse_kv_offload, "is_pin_memory_available", pin_memory)
    monkeypatch.setattr(sparse_kv_offload, "update_sparse_kv_offload_metadata", update)
    metadata = _metadata(enabled=False)

    assert metadata.req_ids_buffers == metadata.token_to_req_buffers == []
    assert metadata.build_kwargs(object(), object(), 2, object(), object(), 1) is None
    allocate.assert_not_called()
    pin_memory.assert_not_called()
    update.assert_not_called()


def test_enabled_allocates_one_pair_per_step_once(monkeypatch):
    allocate = MagicMock(wraps=v1_utils.CpuGpuBuffer)
    pin_memory = MagicMock(return_value=False)
    monkeypatch.setattr(sparse_kv_offload, "CpuGpuBuffer", allocate)
    monkeypatch.setattr(sparse_kv_offload, "is_pin_memory_available", pin_memory)

    metadata = _metadata()

    pin_memory.assert_called_once_with()
    assert allocate.call_count == 6
    for step in range(3):
        req_call, token_call = allocate.call_args_list[2 * step : 2 * step + 2]
        assert req_call.args == (4,)
        assert token_call.args == (8,)
        assert req_call.kwargs == {"dtype": torch.int64, "device": torch.device("cpu"), "pin_memory": False}
        assert token_call.kwargs == {"dtype": torch.int32, "device": torch.device("cpu"), "pin_memory": False}
    assert len({buffer.gpu.data_ptr() for buffer in metadata.req_ids_buffers}) == 3
    assert len({buffer.gpu.data_ptr() for buffer in metadata.token_to_req_buffers}) == 3


def test_full_graph_metadata_preserves_live_ids_pool_cpu_views_and_step(monkeypatch):
    metadata = _metadata()
    state = _state()
    batch = _batch()
    descriptor = _descriptor()
    query_start = np.array([0, 2, 5], dtype=np.int32)
    update = MagicMock(wraps=sparse_kv_offload.update_sparse_kv_offload_metadata)
    monkeypatch.setattr(sparse_kv_offload, "update_sparse_kv_offload_metadata", update)

    kwargs = metadata.build_kwargs(batch, state, 2, descriptor, query_start, 1)

    assert kwargs is not None
    update.assert_called_once()
    args = update.call_args.args
    assert args[:5] == (5, 2, 8, 4, ["live-a", "live-b"])
    assert args[5] is query_start
    assert args[6] is metadata.req_ids_buffers[1]
    assert args[7] is metadata.token_to_req_buffers[1]
    assert set(kwargs) == {
        "req_ids_tensor",
        "token_to_req",
        "req_topk_buffer_slots",
        "req_topk_buffer_active",
        "copy_sfa_draft_index",
        "offload_dummy",
    }
    assert kwargs["req_ids_tensor"].tolist() == [adler32(b"live-a"), adler32(b"live-b"), 0, 0]
    assert kwargs["token_to_req"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert kwargs["req_topk_buffer_slots"].data_ptr() == state._offload_pool_slots.cpu.data_ptr()
    assert kwargs["req_topk_buffer_slots"].tolist() == [7, 8, 9, 10]
    assert kwargs["req_topk_buffer_active"].data_ptr() == state._offload_pool_active.cpu.data_ptr()
    assert kwargs["copy_sfa_draft_index"] == 1
    assert kwargs["offload_dummy"] is False


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_eager_and_piecewise_use_actual_tokens_and_unpadded_requests(mode):
    metadata = _metadata()

    kwargs = metadata.build_kwargs(
        _batch(), _state(pool=False), 2, _descriptor(mode, num_reqs=0), np.array([0, 2, 5]), 0
    )

    assert kwargs is not None
    assert kwargs["req_ids_tensor"].shape == (2,)
    assert kwargs["token_to_req"].tolist() == [0, 0, 1, 1, 1]
    assert kwargs["req_topk_buffer_slots"] is None
    assert kwargs["req_topk_buffer_active"] is None
    assert kwargs["copy_sfa_draft_index"] == 0


def test_repeated_step_reuses_addresses_clears_padding_and_keeps_other_steps(monkeypatch):
    metadata = _metadata()
    allocate = MagicMock(side_effect=AssertionError("step execution allocated a new metadata buffer"))
    monkeypatch.setattr(sparse_kv_offload, "CpuGpuBuffer", allocate)
    query_start = np.array([0, 2, 5], dtype=np.int32)
    step0 = metadata.build_kwargs(_batch(), _state(), 2, _descriptor(), query_start, 0)
    step1 = metadata.build_kwargs(_batch(), _state(), 2, _descriptor(), query_start, 1)
    assert step0 is not None
    assert step1 is not None
    saved_step0 = {key: step0[key].clone() for key in ("req_ids_tensor", "token_to_req")}
    previous_addresses = {key: step1[key].data_ptr() for key in ("req_ids_tensor", "token_to_req")}

    repeated = metadata.build_kwargs(
        SimpleNamespace(req_ids=["replacement"]), _state(), 1, _descriptor(), np.array([0, 1]), 1
    )

    assert repeated is not None
    for key in previous_addresses:
        assert repeated[key].data_ptr() == previous_addresses[key]
        assert repeated[key].data_ptr() != step0[key].data_ptr()
        assert torch.equal(step0[key], saved_step0[key])
    assert repeated["req_ids_tensor"].tolist() == [adler32(b"replacement"), 0, 0, 0]
    assert repeated["token_to_req"].tolist() == [0] * 8
    assert repeated["offload_dummy"] is False
    allocate.assert_not_called()


def test_dummy_uses_distinct_nonzero_ids_when_live_request_list_is_empty(monkeypatch):
    metadata = _metadata()
    update = MagicMock(wraps=sparse_kv_offload.update_sparse_kv_offload_metadata)
    monkeypatch.setattr(sparse_kv_offload, "update_sparse_kv_offload_metadata", update)

    kwargs = metadata.build_kwargs(
        SimpleNamespace(req_ids=[], is_dummy=True), _state(pool=False), 2, _descriptor(), np.array([0, 2, 5]), 2
    )

    assert kwargs is not None
    assert update.call_args.args[4] == ["offload-dummy-0", "offload-dummy-1"]
    assert kwargs["req_ids_tensor"].tolist() == [adler32(b"offload-dummy-0"), adler32(b"offload-dummy-1"), 0, 0]
    assert kwargs["offload_dummy"] is True
    assert kwargs["copy_sfa_draft_index"] == 2


def test_batch_with_no_live_ids_does_not_reuse_previous_request_identity():
    metadata = _metadata()
    metadata.build_kwargs(_batch(), _state(), 2, _descriptor(), np.array([0, 2, 5]), 0)

    kwargs = metadata.build_kwargs(
        SimpleNamespace(req_ids=[], is_dummy=False), _state(pool=False), 2, _descriptor(), np.array([0, 2, 5]), 0
    )

    assert kwargs is not None
    assert kwargs["req_ids_tensor"].tolist() == [0, 0, 0, 0]
    assert kwargs["offload_dummy"] is False
