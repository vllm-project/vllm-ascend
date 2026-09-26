# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``slot_mapping_triton_warmup``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.model_executor.warmup.helpers import make_mock_worker
from vllm_ascend.model_executor.warmup import slot_mapping_triton_warmup as sw


@patch.object(sw, "HAS_TRITON", True)
def test_calls_block_table_prewarm():
    worker = make_mock_worker()
    prewarm = MagicMock()
    worker.model_runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(prewarm_fused_slot_mapping_kernels=prewarm)
    )
    sw.slot_mapping_triton_warmup(worker)
    prewarm.assert_called_once_with()


@patch.object(sw, "HAS_TRITON", True)
def test_skips_runner_without_multi_group_block_table():
    worker = make_mock_worker()
    worker.model_runner = SimpleNamespace()
    sw.slot_mapping_triton_warmup(worker)
    worker.model_runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=object()))
    sw.slot_mapping_triton_warmup(worker)


@patch.object(sw, "HAS_TRITON", False)
def test_skips_without_triton():
    worker = make_mock_worker()
    prewarm = MagicMock()
    worker.model_runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(prewarm_fused_slot_mapping_kernels=prewarm)
    )
    sw.slot_mapping_triton_warmup(worker)
    prewarm.assert_not_called()
