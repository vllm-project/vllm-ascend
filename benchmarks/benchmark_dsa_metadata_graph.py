# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure host submission time for PA_ND SAS/QLI metadata on one NPU.

Run from the repository root with vLLM Ascend, torch-npu and pytest installed:
    python -m benchmarks.benchmark_dsa_metadata_graph --iterations 100
"""

import argparse
from functools import partial
from time import perf_counter

import pytest
import torch

from tests.e2e.pull_request.one_card.aclgraph.test_dsa_metadata_graph import _capture_metadata_graph
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
        parser.error("requires the custom PA_ND SAS/QLI metadata backend")

    print("kind,requests,eager_submission_us,graph_submission_us")
    for kind in (0, 4, 128, "qli"):
        for num_reqs in (1, 4, 16, 32):
            with (
                pytest.MonkeyPatch.context() as monkeypatch,
                _capture_metadata_graph(kind, num_reqs, monkeypatch) as case,
            ):
                _, build, _, _, _, _, graph, stream = case
                times = []
                for submit in (partial(build, False, 1, 128), graph.replay):
                    submit()
                    stream.synchronize()
                    started = perf_counter()
                    for _ in range(args.iterations):
                        submit()
                    elapsed = perf_counter() - started
                    stream.synchronize()
                    times.append(elapsed * 1e6 / args.iterations)
                print(f"{kind},{num_reqs},{times[0]:.3f},{times[1]:.3f}")


if __name__ == "__main__":
    main()
