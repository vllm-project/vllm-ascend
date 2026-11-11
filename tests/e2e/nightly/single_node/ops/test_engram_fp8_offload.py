# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 950 multi-rank FP8 Engram correctness and lookup latency benchmark.

Run with torchrun --standalone --nproc-per-node=2 <this-file>.
The optional --rows-per-rank/--fetch-rows/--iterations control the benchmark.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

WIDTH = 256
BLOCK_SIZE = 32
WEIGHT_KEY = "model.layers.1.engram.embed.weight"
SCALE_KEY = "model.layers.1.engram.embed.scale"


@torch.inference_mode()
def verify_full_graph_fetch(table, reference, device, fetch_rows):
    """Capture the real CANN fetch/wait/dequant and replay changed IDs.

    A Python callback that only completes capture-time data cannot pass this:
    all later outputs must follow device input updates without a host wait.
    """
    main, side = torch.npu.Stream(), torch.npu.Stream()
    ready, inputs_ready = torch.npu.Event(), torch.npu.Event()
    static_ids = torch.zeros(fetch_rows, dtype=torch.int64, device=device)
    main.wait_stream(torch.npu.current_stream())
    graphs, snapshots = {}, []

    def forward(gear):
        inputs_ready.record(main)
        with torch.npu.stream(side):
            side.wait_event(inputs_ready)
            result = table(static_ids[:gear])
            ready.record(side)
        main.wait_event(ready)
        result.record_stream(main)
        return result.clone()

    with torch.npu.stream(main):
        for gear in (fetch_rows, fetch_rows // 2, fetch_rows):
            forward(gear)
            main.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=main):
                output = forward(gear)
            graphs[gear] = (graph, output)
        for step, gear in enumerate((fetch_rows, fetch_rows // 2) * 4):
            offset = step * (reference.shape[0] // 3)
            static_ids.copy_((torch.arange(fetch_rows, device=device) + offset) % reference.shape[0])
            graph, output = graphs[gear]
            graph.replay()
            expected_ids = (torch.arange(gear) + offset) % reference.shape[0]
            snapshots.append((output.clone(), reference[expected_ids]))
    # One final verification wait; none in the input-update / replay loop.
    main.synchronize()
    for actual, expected in snapshots:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def run_smoke(rows_per_rank=512, fetch_rows=128, iterations=20):
    # Importing torch_npu initializes the NPU backend only in the hardware run.
    import torch_npu  # noqa: F401
    from vllm.logger import logger

    from vllm_ascend.models.deepseek_v41.engram.elastic import (
        ElasticEngramEmbedding,
        create_engram_process_group,
    )

    if rows_per_rank < 2 or fetch_rows < 6 or iterations < 1:
        raise ValueError("rows_per_rank >= 2, fetch_rows >= 6 and iterations >= 1 are required")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        world = dist.get_world_size()
        if world < 2:
            raise ValueError("Run on at least two Ascend 950 ranks")
        device = torch.device("npu", torch.npu.current_device())
        rows = rows_per_rank * world
        generator = torch.Generator().manual_seed(42)
        weights = torch.randn((rows - 1, WIDTH), generator=generator).to(torch.float8_e4m3fn)
        scale_bytes = torch.randint(120, 134, (rows - 1, WIDTH // BLOCK_SIZE), generator=generator).to(torch.uint8)
        reference = torch.zeros((rows, WIDTH), dtype=torch.bfloat16)
        reference[:-1] = (
            (
                weights.float().reshape(rows - 1, -1, BLOCK_SIZE)
                * torch.ldexp(torch.ones_like(scale_bytes, dtype=torch.float32), scale_bytes.int() - 127).unsqueeze(-1)
            )
            .reshape(rows - 1, WIDTH)
            .to(torch.bfloat16)
        )
        ids_cpu = torch.randint(rows, (fetch_rows,), generator=generator)
        ids_cpu[:6] = torch.tensor([0, rows_per_rank - 1, rows_per_rank, rows - 2, rows - 1, 0])
        ids = ids_cpu.to(device)
        with tempfile.TemporaryDirectory(prefix="engram-fp8-smoke-") as root:
            save_file({WEIGHT_KEY: weights, SCALE_KEY: scale_bytes}, Path(root) / "model.safetensors")
            group, owns_group = create_engram_process_group(world, layer_id=1)
            table = ElasticEngramEmbedding(
                rows,
                WIDTH,
                world,
                group=group,
                owns_group=owns_group,
                layer_id=1,
                storage_format="mxfp8",
                minimum_rows=rows - 1,
                device=device,
            )
            try:
                table.load_checkpoint(root, WEIGHT_KEY, chunk_rows=127)
                table.offload_weights()
                actual = table(ids).cpu()
                assert torch.equal(actual.view(torch.int16), reference[ids_cpu].view(torch.int16))
                assert table(ids[:0]).shape == (0, WIDTH)
                verify_full_graph_fetch(table, reference, device, fetch_rows)
                # Use separate buffers/process groups, as in the model. Both
                # launches must survive together until their staged wait/Wkv.
                if __package__:
                    from .test_engram_elastic_multistream import run_smoke as verify_multistream
                else:
                    from test_engram_elastic_multistream import run_smoke as verify_multistream

                late_group, owns_late_group = create_engram_process_group(world, layer_id=14)
                late_table = ElasticEngramEmbedding(
                    rows,
                    WIDTH,
                    world,
                    group=late_group,
                    owns_group=owns_late_group,
                    layer_id=14,
                    storage_format="mxfp8",
                    minimum_rows=rows - 1,
                    device=device,
                )
                try:
                    late_table.load_checkpoint(root, WEIGHT_KEY, chunk_rows=127)
                    late_table.offload_weights()
                    verify_multistream({1: table, 14: late_table}, {1: reference, 14: reference})
                finally:
                    late_table.destroy()
                for _ in range(3):
                    table(ids)
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    table(ids)
                torch.npu.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
                logger.info(
                    "FOR-ENGRAM smoke passed: rank=%d storage=%s rows_per_rank=%d fetch_rows=%d "
                    "iterations=%d lookup_ms=%.3f bf16_bits=exact scale_hbm_bytes=%d",
                    dist.get_rank(),
                    table.storage_format,
                    rows_per_rank,
                    fetch_rows,
                    iterations,
                    elapsed_ms,
                    table._scale.numel() if table._scale is not None else 0,
                )
            finally:
                table.destroy()
    finally:
        dist.destroy_process_group()


def test_engram_fp8_offload():
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

    if not get_current_hardware_profile().supports(HardwareCapability.ENGRAM_MXFP8):
        pytest.skip("Engram FP8 ElasticBuffer requires Ascend 950")
    if "LOCAL_RANK" in os.environ:
        run_smoke()
        return

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
    ]
    completed = subprocess.run(
        command,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    print(completed.stdout)
    assert completed.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-rank", type=int, default=512)
    parser.add_argument("--fetch-rows", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run_smoke(args.rows_per_rank, args.fetch_rows, args.iterations)
