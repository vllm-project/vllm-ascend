# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend._310p.worker.v2 import pp


@pytest.fixture
def make_handler(monkeypatch):
    def create(*, is_last_rank):
        group = SimpleNamespace(
            is_last_rank=is_last_rank,
            last_rank=3,
            ranks=[1, 3],
            world_size=2,
            group_ranks=[[0, 2], [1, 3]],
        )
        monkeypatch.setattr(pp, "get_pp_group", lambda: group)
        monkeypatch.setattr(pp.dist, "new_subgroups_by_enumeration", lambda *args, **kwargs: (object(), []))
        return pp.PPHandler310(max_num_reqs=4)

    return create


def make_batch():
    return SimpleNamespace(
        num_reqs=3,
        idx_mapping_np=np.array([0, 1, 2]),
        num_computed_tokens_np=np.array([0, 4, 6]),
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        prefill_len_np=np.array([4, 4, 4]),
        max_seq_len_np=None,
    )


def test_pipeline_send_uses_int64_wire_format_sync_fallback(monkeypatch, make_handler):
    handler = make_handler(is_last_rank=True)
    try:
        send = Mock(return_value=Mock())
        monkeypatch.setattr(pp.dist, "isend", send)
        tokens = torch.tensor([[11], [12], [13]], dtype=torch.int32)
        counts = torch.tensor([0, 1, 1], dtype=torch.int32)
        handler.begin_sample_step()
        handler.broadcast(tokens, counts, torch.zeros_like(counts), make_batch())
        send.assert_not_called()
        handler.submit_sample_output(None)

        payload = send.call_args.args[0]
        torch.testing.assert_close(payload, torch.tensor([[11, 12, 13], [0, 1, 1]], dtype=torch.int64))
        assert send.call_args.kwargs["dst"] == 1
    finally:
        handler.close()


def test_pipeline_send_reuses_async_output_event(monkeypatch, make_handler):
    handler = make_handler(is_last_rank=True)
    try:
        work = Mock()
        send = Mock(return_value=work)
        monkeypatch.setattr(pp.dist, "isend", send)
        event = Mock()
        async_output = SimpleNamespace(
            copy_event=event,
            sampled_token_ids=np.array([[11], [12], [13]], dtype=np.int32),
            num_sampled_tokens_np=np.array([0, 1, 1], dtype=np.int32),
        )
        tokens = torch.tensor([[11], [12], [13]], dtype=torch.int32)
        counts = torch.tensor([0, 1, 1], dtype=torch.int32)
        handler.begin_sample_step()
        handler.broadcast(tokens, counts, torch.zeros_like(counts), make_batch())
        handler.submit_sample_output(SimpleNamespace(_inner=async_output))
        handler._pending_relays[-1].result(timeout=5)

        event.synchronize.assert_called_once()
        payload = send.call_args.args[0]
        torch.testing.assert_close(payload, torch.tensor([[11, 12, 13], [0, 1, 1]], dtype=torch.int64))
        assert send.call_args.kwargs["dst"] == 1
        work.wait.assert_called_once()
    finally:
        handler.close()


def test_pipeline_receive_filters_cancelled_and_prefill_rows(monkeypatch, make_handler):
    handler = make_handler(is_last_rank=False)
    work = Mock()

    def receive(payload, *, src, group):
        assert src == 3
        assert group is handler.broadcast_group
        payload.copy_(torch.tensor([[11, 12, 13], [0, 1, 1]], dtype=torch.int64))
        return work

    monkeypatch.setattr(pp.dist, "irecv", receive)
    batch = make_batch()
    assert handler.get_prev_sampled_outputs() is None
    assert not handler.receive(batch, recv_drafts=False)
    batch.idx_mapping_np[:] = 3
    handler.on_req_idx_freed(1)
    assert handler.get_prev_sampled_outputs() is None
    pending = handler.get_prev_sampled_outputs()

    work.wait.assert_called_once()
    assert pending is not None
    assert pending["idx_mapping"].tolist() == [-1, -1, 2]
    assert pending["sampled_tokens"][2, 0].item() == 13


def test_310p_pp_activation_transport_uses_batched_p2p_without_metadata(monkeypatch):
    from vllm.sequence import IntermediateTensors

    from vllm_ascend._310p import worker_310p

    device_group = object()
    group = SimpleNamespace(rank_in_group=0, world_size=2, ranks=[0, 2], device_group=device_group)
    monkeypatch.setattr(worker_310p, "get_pp_group", lambda: group)
    recorded_ops: list[list[object]] = []

    def fake_batch_isend_irecv(ops):
        recorded_ops.append(list(ops))
        return [Mock() for _ in ops]

    monkeypatch.setattr(worker_310p.dist, "batch_isend_irecv", fake_batch_isend_irecv)
    monkeypatch.setattr(worker_310p.dist, "P2POp", lambda op, tensor, peer, grp: (op, tensor.shape, peer, grp))

    worker = object.__new__(worker_310p.NPUWorker310)
    worker.device = torch.device("cpu")
    worker._pp_send_buffers = ()
    handles = worker._pp_send_tensor_dict(
        {
            "residual": torch.zeros((4, 8), dtype=torch.float16),
            "hidden_states": torch.ones((4, 8), dtype=torch.float16),
        }
    )
    assert len(handles) == 2
    assert recorded_ops[0] == [
        (worker_310p.dist.isend, torch.Size([4, 8]), 2, device_group),
        (worker_310p.dist.isend, torch.Size([4, 8]), 2, device_group),
    ]

    group.rank_in_group = 1
    worker.model_runner = SimpleNamespace(
        intermediate_tensors=IntermediateTensors(
            {
                "residual": torch.empty((16, 8), dtype=torch.float16),
                "hidden_states": torch.empty((16, 8), dtype=torch.float16),
            }
        )
    )
    recv_tensors, recv_handles, postprocess = worker._pp_recv_tensor_dict(num_tokens=4)
    assert list(recv_tensors.keys()) == ["hidden_states", "residual"]
    assert all(t.shape == (4, 8) for t in recv_tensors.values())
    assert len(recv_handles) == 2
    assert postprocess == []
    assert recorded_ops[1] == [
        (worker_310p.dist.irecv, torch.Size([4, 8]), 0, device_group),
        (worker_310p.dist.irecv, torch.Size([4, 8]), 0, device_group),
    ]


def test_310p_pp_default_cudagraph_cap_tracks_microbatch(monkeypatch):
    from vllm_ascend import platform as ascend_platform
    from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=None,
            cudagraph_capture_sizes=None,
            prefill_cudagraph_capture_token_sizes=None,
            max_prefill_cudagraph_capture_tokens=None,
            decode_cudagraph_capture_request_sizes=None,
            max_decode_cudagraph_capture_requests=None,
            max_decode_cudagraph_capture_tokens=None,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        speculative_config=None,
    )
    monkeypatch.setattr(
        ascend_platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.COMPATIBILITY),
    )
    assert ascend_platform._get_reduced_cg_cap(vllm_config) == 4

    vllm_config.parallel_config.pipeline_parallel_size = 1
    assert ascend_platform._get_reduced_cg_cap(vllm_config) == 8

    vllm_config.parallel_config.pipeline_parallel_size = 2
    monkeypatch.setattr(
        ascend_platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.STANDARD),
    )
    assert ascend_platform._get_reduced_cg_cap(vllm_config) == 8
