# SPDX-License-Identifier: Apache-2.0
"""Bound the startup overhead of reporting shared DeepSeek-V4.1 Engram registrations."""

from contextlib import ExitStack
from datetime import timedelta
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.config.logging import LoggingConfig
from vllm.logger import configure_logging, logger

import vllm_ascend.ops  # noqa: F401 - initialize the operators before the device adaptor
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.engram.npu import SharedUvaBuffer

WORLD_SIZE = 4
PAIRS = 6
# A CI runner only needs ~16.5 MiB of shared memory for these four fresh mappings.
LAYER_ROWS = ((1, 32768), (14, 32768))
TABLES = (("_codes_uva", 256, torch.float8_e4m3fn), ("_scales_uva", 8, torch.uint8))
# Allow scheduling jitter on small tables and scale the limit with registration time.
ABSOLUTE_BUDGET_S = 0.005
RELATIVE_BUDGET = 0.10


def _registration_seconds(group, layer_rows, *, diagnostics):
    device = torch.device("npu", torch.npu.current_device())
    modules = {layer: torch.nn.Module() for layer, _ in layer_rows}
    with ExitStack() as buffers:
        dist.barrier(group=group.cpu_group)
        start = perf_counter()
        for layer, rows in layer_rows:
            for name, width, dtype in TABLES:
                kwargs = {"checkpoint_layer_id": layer} if diagnostics else {}
                buffer = SharedUvaBuffer((rows, width), dtype, device, group, name, modules[layer], **kwargs)
                buffers.callback(buffer.close)
        torch.npu.synchronize(device)
        elapsed = perf_counter() - start
    # Time the real CANN call, any INFO output, and the constructor's group synchronization.
    # Unregistration and the reduction below are outside the measured interval.
    seconds = torch.tensor(elapsed, dtype=torch.float64, device="cpu")
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX, group=group.cpu_group)
    return seconds.item()


def _check_overhead(rank, rendezvous):
    torch.npu.set_device(rank)
    configure_logging(LoggingConfig(configure_logging=True, log_level="INFO", pylogging_config_file=None))
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=WORLD_SIZE, timeout=timedelta(minutes=2)
    )
    try:
        group = SimpleNamespace(cpu_group=dist.group.WORLD, rank_in_group=rank, world_size=WORLD_SIZE)
        for enabled in (False, True):
            _registration_seconds(group, LAYER_ROWS, diagnostics=enabled)

        controls, deltas = [], []
        for pair in range(PAIRS):
            # Reverse order on alternate pairs to balance cache and scheduling effects.
            first = bool(pair % 2)
            times = {}
            for enabled in (first, not first):
                times[enabled] = _registration_seconds(group, LAYER_ROWS, diagnostics=enabled)
            controls.append(times[False])
            deltas.append(times[True] - times[False])
        baseline, overhead = median(controls), median(deltas)
        budget = max(ABSOLUTE_BUDGET_S, RELATIVE_BUDGET * baseline)
        if rank == 0:
            logger.info(
                "Shared Engram benchmark control=%.6fs paired_overhead=%.6fs budget=%.6fs", baseline, overhead, budget
            )
        assert overhead <= budget, f"paired overhead {overhead:.6f}s exceeds {budget:.6f}s: {deltas=}"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5, reason="requires Ascend A5 shared host registration"
)
def test_shared_engram_registration_logging_overhead(tmp_path, capfd):
    if torch.npu.device_count() < WORLD_SIZE:
        pytest.skip("requires four NPU devices")
    mp.spawn(_check_overhead, args=((tmp_path / "rendezvous").as_uri(),), nprocs=WORLD_SIZE, join=True)
    captured = capfd.readouterr()
    output = captured.out + captured.err
    # A disabled handler would silently benchmark registrations without writing the diagnostics.
    expected = (1 + PAIRS) * len(LAYER_ROWS) * len(TABLES)
    for rank in range(WORLD_SIZE):
        for event in ("start", "end"):
            assert output.count(f"Shared Engram registration {event} rank={rank} ") == expected
