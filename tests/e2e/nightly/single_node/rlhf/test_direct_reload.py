# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-model Direct transport matrix with runtime and original-graph probes.

Each cell cold-loads generated checkpoint A and compares A -> A -> B -> A against
independent cold references. No sleep is implicit in these BF16 cells.
"""

from dataclasses import replace

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import DIRECT_QWEN_CASES, WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.direct import run_direct_reload_transaction


@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("backend", ["hccl", "npu_ipc"])
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.id, marks=pytest.mark.e2e_model(case.model)) for case in DIRECT_QWEN_CASES],
)
def test_direct_reload_transport_matrix(case: WeightUpdateModelCase, backend: str, packed: bool, execution_mode: str):
    run_direct_reload_transaction(case, backend, packed, execution_mode)


@pytest.mark.e2e_model(DIRECT_QWEN_CASES[0].model)
@pytest.mark.parametrize("sleep_level", [1, 2], ids=["sleep1", "sleep2"])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_direct_reload_dense_hccl_sleep(packed: bool, execution_mode: str, sleep_level: int):
    run_direct_reload_transaction(DIRECT_QWEN_CASES[0], "hccl", packed, execution_mode, sleep_level=sleep_level)


@pytest.mark.e2e_model(DIRECT_QWEN_CASES[0].model)
@pytest.mark.parametrize("sleep_level", [1, 2], ids=["sleep1", "sleep2"])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_direct_reload_dense_ipc_sleep(packed: bool, execution_mode: str, sleep_level: int):
    case = replace(DIRECT_QWEN_CASES[0], tensor_parallel_size=2)
    run_direct_reload_transaction(case, "npu_ipc", packed, execution_mode, sleep_level=sleep_level)


@pytest.mark.e2e_model(DIRECT_QWEN_CASES[1].model)
@pytest.mark.parametrize("sleep_level", [1, 2], ids=["sleep1", "sleep2"])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("backend", ["hccl", "npu_ipc"])
def test_direct_reload_moe_sleep(backend: str, packed: bool, execution_mode: str, sleep_level: int):
    run_direct_reload_transaction(DIRECT_QWEN_CASES[1], backend, packed, execution_mode, sleep_level=sleep_level)
