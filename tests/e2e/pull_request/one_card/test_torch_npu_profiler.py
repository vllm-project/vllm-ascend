# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import torch
from vllm.config import ProfilerConfig, VllmConfig

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper


def test_exports_opt_in_operator_call_stack(tmp_path: Path):
    torch.npu.set_device(0)
    inputs = torch.full((256,), -1.0, device="npu")
    inputs.relu()
    torch.npu.synchronize()

    clear_ascend_config()
    try:
        init_ascend_config(VllmConfig(additional_config={"torch_npu_profiler_with_stack": True}))
        profiler = TorchNPUProfilerWrapper(
            ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=str(tmp_path),
                torch_profiler_with_stack=True,
            ),
            "operator-stack-test",
        )
        profiler.start()
        try:
            inputs.relu()
            torch.npu.synchronize()
        finally:
            profiler.stop()
    finally:
        clear_ascend_config()

    # Check that a new NPU operation can finish after profiling stops.
    assert inputs.relu().sum().item() == 0

    (operator_details,) = tmp_path.rglob("operator_details.csv")
    with operator_details.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        stack_column = next(column for column in reader.fieldnames or [] if column.casefold() == "call stack")
        assert any(row["Name"] == "aten::relu" and row[stack_column].strip() for row in reader), (
            "Expected a recorded call stack for aten::relu"
        )
