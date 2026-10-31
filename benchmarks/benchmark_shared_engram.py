# SPDX-License-Identifier: Apache-2.0
"""Real shared-pool API and unchanged native lookup, no mapping mocks."""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.npu import (
    gather_dequantize_engram_int8,
    gather_dequantize_host_uva,
)

ROWS, WIDTH = 16384, 256


def source_tables(layer):
    codes = (torch.arange(ROWS * WIDTH).remainder(255) - 127).to(torch.int8).view(ROWS, WIDTH)
    scales = (torch.arange(ROWS * 8).remainder(8).float() + layer).view(ROWS, 8) / 16
    return codes, scales


def latency(fn):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(8):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 50)
    return statistics.median(samples), output.cpu()


def main(args):
    root = Path(args.root)
    from vllm_ascend.models.deepseek_v41.engram import shared_pool as pool

    if args.role == "owner":
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    generation = root / "pool"
    if args.role == "owner":
        owner = pool.SharedPoolOwner(
            generation, "probe-model", device=f"npu:{args.device}", host=args.host, world_size=65
        )
        try:
            for layer in (1, 14):
                for key, values in zip(("codes", "scales"), source_tables(layer)):
                    buffer = owner.add(f"{layer}.{key}", ROWS, values.shape[1], values.dtype)
                    buffer.publish(0, values)
            owner.ready()
            print("POOL_READY", flush=True)
            deadline = time.monotonic() + 1800
            while not (root / "stop").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("owner test deadline")
                time.sleep(0.2)
        finally:
            owner.close(timeout=30)
        print("OWNER_CLOSED_PASS", flush=True)
        return
    clients, views = [], {}
    try:
        for layer in (1, 14):
            client = pool.SharedPoolClient(generation, "probe-model", device=f"npu:{args.device}")
            clients.append(client)
            for key, width, dtype in (("codes", WIDTH, torch.int8), ("scales", 8, torch.float32)):
                views[layer, key] = client.view(f"{layer}.{key}", 0, ROWS, rows=ROWS, width=width, dtype=dtype)
        runtime = clients[0].runtime
        assert runtime is not None and runtime is clients[1].runtime, "layers must share the BM singleton"
        assert runtime.references == 2
        for client in clients:
            for buffer in client.buffers.values():
                assert buffer.handle.local_mem_size(buffer.runtime.bm.BmMemType.HOST) == 0
                assert buffer.handle.local_mem_size(buffer.runtime.bm.BmMemType.DEVICE) == 0
        result = dict(
            tag=args.tag,
            consumer_pool_hbm_bytes=0,
            consumer_pool_dram_bytes=0,
            hbm_reference_allocated_separately=True,
            cases=[],
        )
        for tokens in (32, 128, 512):
            ids_cpu = torch.randint(0, ROWS, (tokens, 24), generator=torch.Generator().manual_seed(tokens))
            ids = ids_cpu.npu()
            codes, scales = views[1, "codes"], views[1, "scales"]
            data, factors = source_tables(1)
            expected = (
                data[ids_cpu.flatten()].float() * factors[ids_cpu.flatten()].repeat_interleave(32, 1)
            ).bfloat16()
            hbm_codes, hbm_scales = data.npu(), factors.npu()
            host_fn = partial(gather_dequantize_host_uva, codes, scales, ids, local_heads=24)
            hbm_fn = partial(gather_dequantize_engram_int8, hbm_codes, hbm_scales, ids, WIDTH, local_heads=24)
            torch.testing.assert_close(host_fn().cpu(), expected, rtol=0, atol=0)
            hbm_us, hbm_output = latency(hbm_fn)
            gva_us, gva_output = latency(host_fn)
            torch.testing.assert_close(gva_output, expected, rtol=0, atol=0)
            torch.testing.assert_close(hbm_output, expected, rtol=0, atol=0)
            result["cases"].append(dict(tokens=tokens, heads=24, hbm_us=hbm_us, gva_us=gva_us))
            print(json.dumps(result["cases"][-1]), flush=True)
            del hbm_codes, hbm_scales, ids
        # One layer can detach while the other still reads the owner's table.
        clients[0].close()
        assert clients[1].runtime is runtime and runtime.references == 1
        ids_cpu = torch.tensor([[0], [ROWS - 1], [17]], dtype=torch.int64)
        ids = ids_cpu.npu()
        actual = gather_dequantize_host_uva(views[14, "codes"], views[14, "scales"], ids)
        data, factors = source_tables(14)
        expected = (data[ids_cpu.flatten()].float() * factors[ids_cpu.flatten()].repeat_interleave(32, 1)).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        clients[1].close()
        (root / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("SHARED_POOL_DIRECT_LOOKUP_PASS", flush=True)
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("owner", "consumer"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tag", default="consumer")
    main(parser.parse_args())
