# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two logical pipeline stages update an existing HCCL rollout server.

Start a TP2 server on devices 0,1, then run this example on devices 2,3:

    ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-0.6B \
        --tensor-parallel-size 2 --load-format dummy \
        --weight-transfer-config '{"backend":"hccl"}' \
        --additional-config '{"rl_config":{"enabled":true}}'
    torchrun --standalone --nproc-per-node=2 \
        examples/rl/rlhf_hccl_multi_source.py --trainer-device-offset 2

Requires vLLM with ownership planning (#63256, #62557). Each demo rank loads
the entire HF model to make identical global metadata easy to inspect; a real
PP trainer should export global metadata and materialize only its own stage.
The CPU Gloo group coordinates pause/resume in this example, independently of
the HCCL transfer group. A failed collective requires restarting the job.
"""

import argparse
from functools import cached_property

import requests
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerInitInfo,
    HCCLTrainerWeightTransferEngine,
)


class StageSource(WeightSource):
    """Same ownership declaration on both ranks, stage-local iteration."""

    def __init__(self, model, rank):
        self.model = model
        self.rank = rank

    def _owner(self, name):
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(int(parts[index + 1]) >= self.model.config.num_hidden_layers // 2)
        return 0 if "embed_tokens" in name else 1

    @cached_property
    def metadata(self):
        return [
            ParamMeta(n, p.dtype, tuple(p.shape), owners=(self._owner(n),)) for n, p in self.model.named_parameters()
        ]

    def iter_full(self):
        for name, param in self.model.named_parameters():
            if self._owner(name) == self.rank:
                yield name, param

    def __iter__(self):
        yield from self.iter_full()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trainer-device-offset", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--packed", action="store_true", help="opt in to packed transfers; requires packed NPU validation"
    )
    parser.add_argument("--buffer-size-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if not hasattr(NCCLTrainerWeightTransferEngine, "_broadcast_plan"):
        raise RuntimeError("multi-source HCCL requires vLLM ownership planning (#63256 and #62557)")
    if args.rounds < 1 or args.trainer_device_offset < 0 or args.buffer_size_bytes < 1:
        parser.error("rounds/buffer size must be positive and trainer device offset non-negative")
    dist.init_process_group("gloo")
    engine = None
    try:
        if dist.get_world_size() != 2:
            raise ValueError("launch this two-stage example with --nproc-per-node=2")
        rank = dist.get_rank()
        device = torch.device("npu", args.trainer_device_offset + rank)
        torch.npu.set_device(device)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
        endpoint = [None]
        if rank == 0:
            response = requests.get(f"{args.base_url}/get_world_size", timeout=30)
            response.raise_for_status()
            endpoint[0] = (get_ip(), get_open_port(), response.json()["world_size"] + 2)
        dist.broadcast_object_list(endpoint, src=0)
        address, port, world_size = endpoint[0]
        engine = HCCLTrainerWeightTransferEngine.trainer_init(
            HCCLTrainerInitInfo(
                rank=rank,
                master_address=address,
                master_port=port,
                world_size=world_size,
                packed=args.packed,
                packed_buffer_size_bytes=args.buffer_size_bytes,
            ),
            client=HTTPVLLMWeightSyncClient(args.base_url, timeout=300),
            source=StageSource(model, rank),
        )
        for _ in range(args.rounds):
            # A training integration performs its policy update before this step.
            if rank == 0:
                response = requests.post(f"{args.base_url}/pause", timeout=60)
                response.raise_for_status()
            dist.barrier()
            engine.send_weights()
            dist.barrier()
            if rank == 0:
                response = requests.post(f"{args.base_url}/resume", timeout=60)
                response.raise_for_status()
            dist.barrier()
    finally:
        # Failed updates deliberately leave generation paused.
        if engine is not None:
            engine.shutdown()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
