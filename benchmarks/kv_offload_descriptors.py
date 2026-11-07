# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU byte-copy scenarios and host-only timing for simple offload descriptors.

Run from a source checkout. The DMA boundary is replaced with a CPU copy or
recorder; these checks do not exercise CANN, NPU streams, or device completion.
"""

import argparse
import ctypes
import importlib.util
import itertools
import json
import platform
import random
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

_SOURCE = Path(__file__).resolve().parents[1] / (
    "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/npu_mem_ops.py"
)


def verify(module):
    records = []

    def record(src, dst, sizes, direction):
        if any(t.dtype != torch.int64 or not t.is_contiguous() or t.device.type != "cpu" for t in (src, dst, sizes)):
            raise RuntimeError("DMA descriptor storage contract changed")
        records.append((src.tolist(), dst.tolist(), sizes.tolist(), direction))

    def check_mapping(src_ids, dst_ids, layers, block_bytes=32, direction=0, descriptor_count=None):
        records.clear()
        bases = np.arange(layers, dtype=np.int64) * 2**48
        params = module.BatchMemcpyParams(
            bases + 2**44,
            bases + 2**45,
            np.broadcast_to(np.asarray(block_bytes, dtype=np.int64), (layers,)),
            layers,
            direction,
        )
        module.copy_blocks(src_ids, dst_ids, params)
        expected = [(layer, src, dst) for layer in range(layers) for src, dst in zip(src_ids, dst_ids)]
        actual = []
        for srcs, dsts, sizes, observed_direction in records:
            if observed_direction != direction:
                raise RuntimeError("copy direction changed")
            for src, dst, size in zip(srcs, dsts, sizes):
                layer = int(np.searchsorted(params.src_bases, src, side="right")) - 1
                width = int(params.bpb[layer])
                source_block = (src - int(params.src_bases[layer])) // width
                target_block = (dst - int(params.dst_bases[layer])) // width
                if size <= 0 or size % width:
                    raise RuntimeError("partial or empty descriptor")
                actual.extend((layer, source_block + offset, target_block + offset) for offset in range(size // width))
        if descriptor_count is not None and sum(len(row[0]) for row in records) != descriptor_count:
            raise RuntimeError(f"descriptor count differs from {descriptor_count=}")
        if actual != expected:
            raise RuntimeError(f"byte mapping differs for {layers=}, blocks={len(src_ids)}")

    cases = 0
    with patch.object(torch.ops._C_ascend, "swap_blocks_batch", record, create=True):
        for n in range(5):
            for src in itertools.product(range(4), repeat=n):
                for dst in itertools.permutations(range(4), n):
                    check_mapping(list(src), list(dst), 3)
                    cases += 1
        for n in (511, 512, 513, 4096):
            for run in (1, 2, 16, n):
                for direction in (0, 1):
                    check_mapping(
                        [i + 3 * (i // run) for i in range(n)],
                        [i + 5 * (i // run) for i in range(n)],
                        64,
                        direction=direction,
                    )
                    cases += 1
        check_mapping(list(range(512)), list(range(512)), 64, block_bytes=2**23)
        cases += 1
        for n, expected_count in ((511, 511 * 64), (512, 64), (513, 64)):
            check_mapping(list(range(n)), list(range(n)), 64, descriptor_count=expected_count)
            cases += 1
        check_mapping(list(range(512)), list(range(512)), 64, block_bytes=[32 * (i + 1) for i in range(64)])
        check_mapping(list(range(2**32, 2**32 + 512)), list(range(2**33, 2**33 + 512)), 64, descriptor_count=64)
        cases += 2
        records.clear()
        module.copy_blocks([], [1], module.BatchMemcpyParams(np.array([]), np.array([]), np.array([]), 0, 0))
        if records:
            raise RuntimeError("empty source submitted a transfer")

    def copy_on_cpu(src, dst, sizes, direction):
        for source, target, length in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(target, source, length)

    for dtype in (torch.int8, torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32):
        source = {str(i): torch.arange(1024 * 8).remainder(113).reshape(1024, 8).to(dtype) for i in range(64)}
        target = {name: torch.full_like(tensor, -7) for name, tensor in source.items()}
        ids = list(range(512))
        destinations = list(range(511, 1023))
        params = module.build_params(source, target, module.DIRECTION_H2D)
        with patch.object(torch.ops._C_ascend, "swap_blocks_batch", copy_on_cpu, create=True):
            for _ in range(2):
                for tensor in source.values():
                    tensor.add_(1)
                expected = {name: tensor.clone() for name, tensor in target.items()}
                for name in expected:
                    expected[name][destinations] = source[name][ids]
                module.copy_blocks(ids, destinations, params)
                for name, tensor in target.items():
                    if not torch.equal(tensor.view(torch.uint8), expected[name].view(torch.uint8)):
                        raise RuntimeError(f"copied bytes differ for {dtype=}, tensor={name}")
                cases += 1

    def fail(*args):
        raise RuntimeError("DMA submission failed")

    with patch.object(torch.ops._C_ascend, "swap_blocks_batch", fail, create=True):
        try:
            module.copy_blocks(ids, destinations, params)
        except RuntimeError as exc:
            if str(exc) != "DMA submission failed":
                raise
        else:
            raise RuntimeError("DMA submission failure was swallowed")
    try:
        module.copy_blocks([0], [0, 1], params)
    except AssertionError as exc:
        if str(exc) != "src/dst block counts must match":
            raise
    else:
        raise RuntimeError("mismatched block counts were accepted")
    return cases


def benchmark(module, baseline):
    rng = random.Random(1407)
    counts = []

    def record(src, dst, sizes, direction):
        counts[:] = [src.numel(), src.nbytes + dst.nbytes + sizes.nbytes]

    rows = []
    with patch.object(torch.ops._C_ascend, "swap_blocks_batch", record, create=True):
        for layers in (1, 8, 64, 96):
            params = module.BatchMemcpyParams(
                np.arange(layers, dtype=np.int64) * 10**9 + 10**10,
                np.arange(layers, dtype=np.int64) * 10**9 + 10**12,
                np.arange(1, layers + 1, dtype=np.int64) * 32,
                layers,
                module.DIRECTION_H2D,
            )
            for n in (16, 128, 511, 512, 513, 4096):
                for run in (1, 16, n):
                    src = [i + 3 * (i // run) for i in range(n)]
                    dst = [i + 5 * (i // run) for i in range(n)]
                    samples = {"baseline": [], "candidate": []}
                    measurements = {}
                    for repetition in range(65):
                        implementations = [("baseline", baseline), ("candidate", module)]
                        rng.shuffle(implementations)
                        for name, implementation in implementations:
                            started = time.perf_counter_ns()
                            implementation.copy_blocks(src, dst, params)
                            elapsed = time.perf_counter_ns() - started
                            if repetition >= 5:
                                samples[name].append(elapsed / 1000)
                            measurements[name] = {"descriptors": counts[0], "descriptor_bytes": counts[1]}
                    for name in samples:
                        measurements[name]["median_us"] = statistics.median(samples[name])
                    rows.append({"layers": layers, "blocks": n, "run_length": run, "measurements": measurements})
    return rows


def load_source(path):
    spec = importlib.util.spec_from_file_location("offload_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Compare host construction time with this source file")
    args = parser.parse_args()
    module = load_source(_SOURCE)
    result = {
        "source": str(_SOURCE),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if args.baseline:
        result["baseline"] = str(args.baseline)
        result["measurements"] = benchmark(module, load_source(args.baseline))
    else:
        result["scenarios"] = verify(module)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
