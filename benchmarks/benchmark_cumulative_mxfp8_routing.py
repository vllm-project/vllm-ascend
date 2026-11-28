# SPDX-License-Identifier: Apache-2.0
"""Compare MXFP8 routing plus cumsum with directly cumulative routing on A5.

This measures the indicated local operation chain, not model throughput.
Run in a matched vllm-ascend NPU environment. No speedup threshold is enforced.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu

from vllm_ascend.device.device_op import DeviceOperator


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
    if args.width % 32:
        raise ValueError("MXFP8 width must be divisible by 32")
    experts, top_k = 256, 8
    for rows in args.rows:
        x = torch.randn(rows, args.width, device="npu", dtype=torch.bfloat16)
        ids = torch.rand(rows, experts, device="npu").topk(top_k, dim=-1).indices.to(torch.int32)
        kwargs = dict(
            active_num=ids.numel(),
            expert_num=experts,
            active_expert_range=[0, experts],
            quant_mode=3,
            act_quant_type=torch.float8_e4m3fn,
        )

        def baseline(x=x, ids=ids, kwargs=kwargs):
            result = DeviceOperator.npu_moe_init_routing(x, ids, expert_tokens_num_type=1, **kwargs)
            return result[2].to(torch.int64).cumsum(0)

        def optimized(x=x, ids=ids, kwargs=kwargs):
            result = DeviceOperator.npu_moe_init_routing(x, ids, expert_tokens_num_type=0, **kwargs)
            return result[2].to(torch.int64)

        yield dict(rows=rows, experts=experts, top_k=top_k), baseline, optimized


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 7, 128, 4868])
    parser.add_argument("--width", type=int, default=2048)
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
