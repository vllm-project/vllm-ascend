# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run tiny PLE GVA correctness, isolation and replay checks on one NPU.

Usage: python tests/manual/check_ple_gva.py --device 0
Requires an isolated MemFabric install exposing offload.Offload().
"""

import argparse
import importlib.util
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401


def load_backend():
    path = Path(__file__).parents[2] / "vllm_ascend/distributed/ple_gva.py"
    spec = importlib.util.spec_from_file_location("ple_gva", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_rank(rank, devices, port, store_port):
    torch.npu.set_device(devices[rank])
    device = torch.device("npu", devices[rank])
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    group = SimpleNamespace(rank_in_group=rank, world_size=2, ranks=[0, 1], cpu_group=dist.group.WORLD)
    table = load_backend().PLEGvaTable((8, 128), device, group=group, store_url=f"tcp://127.0.0.1:{store_port}")
    try:
        expected = torch.arange(8 * 128).reshape(8, 128).bfloat16()
        if rank == 0:
            table.tensor.copy_(expected)
        else:
            assert table.tensor.is_meta and table.pointer == 0
        dist.barrier()
        ids = torch.tensor([10, 17, 9, 11, 18], device=device)
        actual = table.lookup(ids, 10, 18).cpu()
        reference = torch.stack((expected[0], expected[7], torch.zeros(128), expected[1], torch.zeros(128))).bfloat16()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        print(f"shared owner/consumer rank {rank}: PASS", flush=True)
    finally:
        table.close()
        dist.destroy_process_group()


def check_private(device_index, check_adapter=False):
    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)
    module = load_backend()
    first = module.PLEGvaTable((8, 128), device)
    second = module.PLEGvaTable((8, 128), device)
    try:
        weight = torch.arange(8 * 128).reshape(8, 128).bfloat16()
        first.tensor.copy_(weight)
        second.tensor.fill_(7)

        def reference(ids):
            valid = (ids >= 10) & (ids < 18)
            return torch.where(valid[:, None], weight[(ids - 10).clamp(0, 7)], 0)

        for count in (0, 1, 3, 48, 49, 64, 65, 129):
            ids_cpu = torch.tensor(([9, 10, 17, 18, -1, 2**40, 10] * ((count + 6) // 7))[:count], dtype=torch.int64)
            ids = ids_cpu.to(device)
            actual = first.lookup(ids, 10, 18)
            torch.testing.assert_close(actual.cpu(), reference(ids_cpu), rtol=0, atol=0)
        print("eager lookup: PASS", flush=True)
        ids = torch.tensor([10, 17, 18], device=device)
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            for _ in range(3):
                first.lookup(ids, 10, 18)
        torch.npu.current_stream(device).wait_stream(stream)
        torch.npu.synchronize(device)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = first.lookup(ids, 10, 18)
        changed = torch.tensor([17, -1, 10])
        ids.copy_(changed.to(device))
        graph.replay()
        torch.npu.synchronize(device)
        torch.testing.assert_close(output.cpu(), reference(changed), rtol=0, atol=0)
        graph.reset()
        print("ACLGraph changed-ID replay: PASS", flush=True)
        if check_adapter:
            # Import lazily so the raw-backend diagnostic can run without a
            # matching vLLM installation.
            from vllm_ascend.models.qwen4_exp import ple_embedding

            adapter = object.__new__(ple_embedding.AscendPLEHostEmbedding)
            torch.nn.Module.__init__(adapter)
            adapter._host_table = first
            adapter._context_key = "test.ple_gva"
            adapter.embedding_dim = 128
            adapter.etp_data_parallel_size = 1
            adapter.tp_size = 1
            adapter.reduce_results = False
            adapter.shard_indices = SimpleNamespace(org_vocab_start_index=10, org_vocab_end_index=18)
            context = SimpleNamespace(no_compile_layers={adapter._context_key: adapter})
            with patch.object(ple_embedding, "get_forward_context", return_value=context):
                compiled = torch.compile(adapter, backend="eager", fullgraph=True)
                torch.testing.assert_close(compiled(ids).cpu(), reference(changed), rtol=0, atol=0)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    output = compiled(ids)
                changed = torch.tensor([10, 11, 12])
                ids.copy_(changed.to(device))
                graph.replay()
                torch.npu.synchronize(device)
                torch.testing.assert_close(output.cpu(), reference(changed), rtol=0, atol=0)
                graph.reset()
            print("adapter fullgraph + ACLGraph: PASS", flush=True)
        first.close()
        actual = second.lookup(torch.tensor([0], device=device), 0, 8)
        torch.testing.assert_close(actual.cpu(), torch.full((1, 128), 7, dtype=torch.bfloat16))
        print("independent pool lifetime: PASS", flush=True)
    finally:
        first.close()
        second.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--shared-devices", type=int, nargs=2)
    parser.add_argument("--port", type=int, default=31591)
    parser.add_argument("--store-port", type=int, default=31592)
    args = parser.parse_args()
    if args.shared_devices:
        mp.spawn(shared_rank, args=(args.shared_devices, args.port, args.store_port), nprocs=2)
    else:
        check_private(args.device, args.adapter)


if __name__ == "__main__":
    main()
