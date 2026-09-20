# SPDX-License-Identifier: Apache-2.0
"""CPU capture microbenchmark, not an NPU serving performance measurement."""

import argparse
import dataclasses
import json
import platform
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from vllm_ascend.dfx.config import DfxConfig
from vllm_ascend.dfx.recorder import FlightRecorder


@dataclasses.dataclass
class Schedule:
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_new_reqs: list = dataclasses.field(default_factory=list)
    scheduled_spec_decode_tokens: dict = dataclasses.field(default_factory=dict)
    finished_req_ids: set = dataclasses.field(default_factory=set)


class HostView:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return HostView(self.values[key])

    def numpy(self):
        return self.values


def run(num_reqs, tokens_per_req, blocks_per_req, iterations):
    req_ids = [f"request-{i}" for i in range(num_reqs)]
    counts = np.full(num_reqs, tokens_per_req, dtype=np.int32)
    total = num_reqs * tokens_per_req
    positions = np.arange(total, dtype=np.int64)
    schedule = Schedule(dict(zip(req_ids, [tokens_per_req] * num_reqs)), total)
    model = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=req_ids, num_reqs=num_reqs, num_computed_tokens_cpu=np.zeros(num_reqs, dtype=np.int32)
        ),
        input_ids=SimpleNamespace(cpu=HostView(np.arange(total, dtype=np.int32))),
        requests={key: SimpleNamespace(block_ids=(list(range(blocks_per_req)),)) for key in req_ids},
        use_async_scheduling=True,
        speculative_config=None,
        use_dcp=False,
        dfx_recorder=None,
    )

    def step():
        if model.dfx_recorder is not None:
            model.dfx_recorder.record_schedule(schedule)
        if model.dfx_recorder is not None and model.dfx_recorder.config.track_block_ownership:
            model.dfx_recorder.record_lifecycle(model, schedule)
        if model.dfx_recorder is not None:
            model.dfx_recorder.record_batch(model, schedule, positions, counts)
        frame = None
        if model.dfx_recorder is not None:
            frame = model.dfx_recorder.begin_device(model, None, None, None)
        if model.dfx_recorder is not None:
            model.dfx_recorder.end_device(frame)
        if model.dfx_recorder is not None:
            recorder = model.dfx_recorder
            result = SimpleNamespace(
                req_ids=req_ids,
                req_id_to_index={req: row for row, req in enumerate(req_ids)},
                sampled_token_ids=[[recorder.execution_id % 30000] for _ in req_ids],
                num_nans_in_logits=None,
                logprobs=None,
            )
            recorder.record_output(result, recorder.output_context(req_ids, 32000))

    def sample():
        for _ in range(100):
            step()
        elapsed = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            step()
            elapsed.append((time.perf_counter_ns() - start) / 1000)
        return {"p50_us": float(np.percentile(elapsed, 50)), "p99_us": float(np.percentile(elapsed, 99))}

    off = sample()
    with tempfile.TemporaryDirectory(prefix="ascend-dfx-bench-") as directory:
        model.dfx_recorder = FlightRecorder(DfxConfig(output_dir=directory), {"benchmark": True})
        try:
            on = sample()
            stats = model.dfx_recorder.stats()
        finally:
            model.dfx_recorder.close()
    return {
        "requests": num_reqs,
        "tokens_per_request": tokens_per_req,
        "blocks_per_request": blocks_per_req,
        "off_synthetic_runner_checks": off,
        "on_schedule_batch_output_detection": on,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    print(
        json.dumps(
            {
                "scope": "Synthetic CPU-only; excludes NPU execution, new requests, triggers and JSON writer load",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "iterations": args.iterations,
                "results": [run(*case, args.iterations) for case in [(1, 1, 16), (32, 1, 16), (32, 128, 128)]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
