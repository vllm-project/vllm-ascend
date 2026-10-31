# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.v1.worker.pp.boundary import PPBoundary
from vllm.v1.worker.pp.buffer import PPSendBuffer
from vllm.v1.worker.pp.feedback import PPDeviceFeedback


def _hccl_buffer_worker(rank, rendezvous, first_two_posted):
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch

    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    with set_current_vllm_config(VllmConfig()):
        group = GroupCoordinatorPatch([[0, 1]], rank, "hccl", False, group_name="pp")
    feedback = PPDeviceFeedback(torch.npu, group.cpu_group.get_group_store(), rank, 30)
    try:
        # Initialize the HCCL connection before delaying receives.
        dist.all_reduce(torch.ones(1, device="npu"), group=group.device_group)
        torch.npu.synchronize()
        # P2P initializes a separate HCCL communicator on its first transfer.
        if rank == 0:
            group.send_tensor_dict({"hidden_states": torch.zeros(128, 4, 4096, device="npu")})
        else:
            group.recv_tensor_dict()
        torch.npu.synchronize()
        group.barrier()
        if rank == 0:
            buffer = PPSendBuffer(2, torch.npu, metadata_handles=1)
            output = torch.empty(128, 4, 4096, dtype=torch.bfloat16, device="npu")
            buffer.reserve({"hidden_states": output})
            for batch_id in range(12):
                print(f"producer acquiring batch {batch_id}", flush=True)
                buffer.acquire()
                feedback.begin(PPBoundary(batch_id, 20 + batch_id % 3, batch_id), 64 if batch_id % 2 else 128, 1)
                feedback.forward_start()
                value = output[: 64 if batch_id % 2 else 128]
                value.fill_(batch_id)
                buffer.send(
                    {"hidden_states": value, "dynamic_pp_header": (batch_id, 20 + batch_id % 3, batch_id)},
                    group.isend_tensor_dict,
                    on_snapshot=feedback.finish,
                )
                print(f"producer posted batch {batch_id}", flush=True)
                output.fill_(-100)
                if batch_id == 1:
                    first_two_posted.set()
            buffer.drain()
            assert buffer.num_sends == 12
        else:
            assert first_two_posted.wait(45), "producer could not post two batches before receive"
            for batch_id in range(12):
                tensors, handles, postprocess = group.irecv_tensor_dict()
                for handle in handles:
                    handle.wait()
                for callback in postprocess:
                    callback()
                feedback.begin(PPBoundary(batch_id, 20 + batch_id % 3, batch_id), 64 if batch_id % 2 else 128, 1)
                feedback.forward_start()
                tensors["hidden_states"].square()
                feedback.finish()
                assert tensors["dynamic_pp_header"] == (batch_id, 20 + batch_id % 3, batch_id)
                value = tensors["hidden_states"].cpu()
                assert value.shape == (64 if batch_id % 2 else 128, 4, 4096)
                torch.testing.assert_close(value, torch.full_like(value, batch_id), rtol=0, atol=0)
        torch.npu.synchronize()
        feedback.poll()
        group.barrier()
        feedback.poll()
        if rank == 0:
            samples = feedback.take_samples()
            assert [sample.batch_id for sample in samples] == list(range(12))
            assert all(sample.pp0_ms > 0 and sample.pp1_ms > 0 for sample in samples)
            assert feedback.queue_depth == 0 and feedback.consumed_batch_id == 11
            print("Device feedback: all 12 batch timings paired, final ready queue=0", flush=True)
    finally:
        group.destroy()
        dist.destroy_process_group()


def test_hccl_delayed_receiver_preserves_pending_dsv4_outputs(tmp_path, monkeypatch):
    """Exercise the production dictionary transport, stream ordering and reuse."""
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    context = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        _hccl_buffer_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", context.Event()),
        nprocs=2,
        join=True,
    )
