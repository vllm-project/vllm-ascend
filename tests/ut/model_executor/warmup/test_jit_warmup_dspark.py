# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the DSpark SWA indices warmup wrapper."""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops.triton import dspark_swa_indices as ds

# The registration gates and key spaces under test are only meaningful
# with a real Triton kernel behind them; the without-device CI leg has no
# triton installed.
pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")


def test_single_key_pinned_by_deployment_config():
    keys = ds._DSPARK_SWA_INDICES_KERNEL.get_warmup_keys(
        window_size=4096,
        block_size=128,
        index_width=4608,
        block_table_width=1024,
        num_speculative_tokens=3,
    )
    (key,) = keys
    assert key.window_size == 4096
    assert key.block_size == 128
    assert key.index_w == 4608
    assert key.block_w == 1024
    assert key.row_pow2 == 1024
    assert key.q_pow2 == 4
    # 1024 is divisible by 16: representative of the stride's div-16 class.
    assert key.bt_stride == 16

    inputs = ds._DSPARK_SWA_INDICES_KERNEL.warmup_inputs(key)
    assert inputs["out_slots"].shape == (1, 1, 4608)
    assert inputs["out_lens"].dtype == torch.int64


def test_warmup_inputs_reproduce_pointer_specialization():
    keys = ds._DSPARK_SWA_INDICES_KERNEL.get_warmup_keys(
        window_size=4096,
        block_size=128,
        index_width=4608,
        block_table_width=1024,
        num_speculative_tokens=3,
    )
    inputs = ds._DSPARK_SWA_INDICES_KERNEL.warmup_inputs(keys[0])
    assert inputs["block_table"].strides == (16, 1)
    assert inputs["out_slots"].strides == (4608, 4608, 1)
    # Capacity scalars are do_not_specialize; fixed values suffice.
    assert inputs["num_slots"] == 1


def test_register_respects_supported_gate_and_window():
    registry = make_registry(None)

    # Unsupported geometry is refused even inside a window.
    with registry.activate():
        assert (
            ds.register_dspark_swa_indices_warmup(
                window_size=4096,
                block_size=100,
                index_width=4608,
                block_table_width=1024,
                num_speculative_tokens=3,
            )
            is False
        )
    assert len(registry) == 0

    with registry.activate():
        assert (
            ds.register_dspark_swa_indices_warmup(
                window_size=4096,
                block_size=128,
                index_width=4608,
                block_table_width=1024,
                num_speculative_tokens=3,
            )
            is True
        )
    assert len(registry) == 1

    # Outside a window: reports success but registers nothing.
    assert (
        ds.register_dspark_swa_indices_warmup(
            window_size=4096,
            block_size=128,
            index_width=4608,
            block_table_width=1024,
            num_speculative_tokens=3,
        )
        is True
    )
    assert len(registry) == 1
