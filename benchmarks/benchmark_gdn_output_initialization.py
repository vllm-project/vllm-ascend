# SPDX-License-Identifier: Apache-2.0
"""Isolate GDN output initialization and complete write-back; no GDN math kernels.

This measures the indicated local operation chain, not model throughput.
Run in a matched vllm-ascend NPU environment. No speedup threshold is enforced.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu


def measure(fn, iterations):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    begin = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations, (time.perf_counter() - begin) * 1e6 / iterations


def cases(args):
    for rows in args.rows:
        source = torch.randn(rows, args.width, device="npu", dtype=torch.bfloat16)
        for padding in (0, 7):
            shape = (rows + padding, args.width)

            def baseline(source=source, shape=shape, rows=rows):
                output = torch.zeros(shape, device=source.device, dtype=source.dtype)
                output[:rows].copy_(source)
                return output

            def optimized(source=source, shape=shape, rows=rows, padding=padding):
                output = torch.empty(shape, device=source.device, dtype=source.dtype)
                if padding:
                    output[rows:].zero_()
                output[:rows].copy_(source)
                return output

            yield dict(rows=rows, padding=padding, avoided_zero_bytes=source.nbytes), baseline, optimized


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 7, 128, 4868])
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if min(*args.rows, args.width, args.rounds, args.iterations) <= 0:
        parser.error("all sizes and iteration counts must be positive")
    torch.manual_seed(17)
    print(
        json.dumps(
            dict(
                device=torch.npu.get_device_name(),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                dtype="bfloat16",
                width=args.width,
            )
        )
    )
    for label, baseline, optimized in cases(args):
        expected = baseline()
        expected = expected.clone() if isinstance(expected, torch.Tensor) else tuple(x.clone() for x in expected)
        torch.testing.assert_close(expected, optimized(), rtol=0, atol=0)
        for _ in range(10):
            baseline()
            optimized()
        times = {"baseline": [], "optimized": []}
        for index in range(args.rounds):
            order = [("baseline", baseline), ("optimized", optimized)]
            if index % 2:
                order.reverse()
            for name, fn in order:
                times[name].append(measure(fn, args.iterations))
        print(
            json.dumps(
                dict(
                    case=label,
                    event_median_us={k: statistics.median(t[0] for t in v) for k, v in times.items()},
                    wall_median_us={k: statistics.median(t[1] for t in v) for k, v in times.items()},
                    rounds_event_wall_us=times,
                )
            )
        )


if __name__ == "__main__":
    main()
