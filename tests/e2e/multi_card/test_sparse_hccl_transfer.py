# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real trainer and two TP-like workers, including repeated sparse chunks.

Run with three free NPUs:
    ASCEND_RT_VISIBLE_DEVICES=8,9,10 torchrun --standalone --nproc-per-node=3 \
        tests/e2e/multi_card/test_sparse_hccl_transfer.py
"""

import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerInitInfo,
    SparseHCCLTrainerWeightTransferEngine,
    SparseHCCLWeightTransferEngine,
    SparseHCCLWeightTransferInitInfo,
    SparseHCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)


class ShardedModel(torch.nn.Module):
    def __init__(self, shard, device):
        super().__init__()
        self.shard = shard
        self.weight = torch.nn.Parameter(
            torch.arange(8, device=device, dtype=torch.bfloat16).reshape(4, 2)[shard].clone()
        )

    def load_weights(self, weights):
        for name, source in weights:
            assert name == "weight"
            self.weight.copy_(source[self.shard])
        return {"weight"}


class Client:
    def __init__(self):
        self.expected = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)

    def command(self, operation, payload=None):
        dist.broadcast_object_list([(operation, payload)], src=0)

    def init_weight_transfer_engine(self, info):
        self.command("init", info)
        dist.barrier()

    def start_weight_update(self):
        self.command("start")

    def update_weights(self, info):
        self.command("update", (info, self.expected.tolist()))
        dist.barrier()

    def finish_weight_update(self, weight_version=None):
        self.command("finish")


def run():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"npu:{torch.npu.current_device()}")
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 3
    if rank == 0:
        client = Client()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        engine = SparseHCCLTrainerWeightTransferEngine.trainer_init(
            SparseHCCLTrainerInitInfo("127.0.0.1", port, 3, rank=0), client=client
        )
        client.start_weight_update()
        for indices, values in [([0, 7], [10, 70]), ([0, 3], [20, 30]), ([2, 5], [-2, -5])]:
            cpu_indices = torch.tensor(indices, dtype=torch.long)
            cpu_values = torch.tensor(values, dtype=torch.bfloat16)
            client.expected.view(-1).index_copy_(0, cpu_indices, cpu_values)
            # Split a repeated checkpoint name across separate patches in one chunk.
            patches = [
                SparseWeightPatch(
                    "weight",
                    cpu_indices[i : i + 1].to(device, dtype=torch.int32),
                    cpu_values[i : i + 1].to(device),
                    (4, 2),
                )
                for i in range(2)
            ]
            engine.send_weight_chunk(patches)
        client.finish_weight_update()
        client.start_weight_update()
        for indices, values in [([1, 6], [11, 66]), ([1, 4], [21, 44]), ([0, 7], [-10, -70])]:
            cpu_indices = torch.tensor(indices, dtype=torch.long)
            cpu_values = torch.tensor(values, dtype=torch.bfloat16)
            client.expected.view(-1).index_copy_(0, cpu_indices, cpu_values)
            info = SparseHCCLWeightTransferUpdateInfo(
                names=["weight", "weight"],
                dtype_names=["bfloat16", "bfloat16"],
                shapes=[[4, 2], [4, 2]],
                num_updates_list=[1, 1],
                packed=True,
                verify=True,
            )
            engine.send_packed_chunk(info, cpu_indices.to(device, dtype=torch.int32), cpu_values.to(device))
        client.finish_weight_update()
        client.command("stop")
        engine.shutdown()
        print(
            "SPARSE-HCCL-TRANSFER PASS: 3 ranks, 6 chunks, standard + packed, repeated names, BF16 equality", flush=True
        )
    else:
        shard = slice((rank - 1) * 2, rank * 2)
        model = ShardedModel(shard, device)
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_index=0, world_size=2, rank=rank - 1),
            model_config=SimpleNamespace(),
        )
        engine = SparseHCCLWeightTransferEngine(SimpleNamespace(), config, device, model)
        while True:
            command = [None]
            dist.broadcast_object_list(command, src=0)
            operation, payload = command[0]
            if operation == "init":
                engine.init_transfer_engine(SparseHCCLWeightTransferInitInfo(**payload))
                dist.barrier()
            elif operation == "start":
                engine.start_weight_update()
            elif operation == "update":
                info, expected = payload
                engine.update_weights(info)
                expected = torch.tensor(expected, device=device, dtype=torch.bfloat16)[shard]
                assert torch.equal(model.weight.view(torch.int16), expected.view(torch.int16))
                dist.barrier()
            elif operation == "finish":
                engine.finish_weight_update()
            elif operation == "stop":
                engine.shutdown()
                break
            else:
                raise AssertionError(operation)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
