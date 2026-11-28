# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Engram replayed-slots Triton JIT warmup wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops.triton import engram_replayed_slots as rs
from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")


def test_wrapper_uses_ascend_base_class():
    assert isinstance(rs._PAD_V2_REPLAYED_SLOTS_KERNEL, AscendTritonJitKernel)


@pytest.mark.parametrize(
    ("group_stride", "expected_stride_rep", "replay_window", "expected_window_rep"),
    [
        (8192, 16, 4096, 16),
        (100, 2, 129, 2),
        (1, 1, 1, 1),
    ],
)
def test_compile_key_normalizes_scalar_specializations(
    group_stride, expected_stride_rep, replay_window, expected_window_rep
):
    keys = rs._PAD_V2_REPLAYED_SLOTS_KERNEL.get_warmup_keys(
        slot_mappings_dtype=torch.int32,
        group_stride=group_stride,
        positions_dtype=torch.int64,
        replay_window=replay_window,
        num_groups=2,
    )
    (key,) = keys
    assert key.slot_mappings_dtype == torch.int32
    assert key.group_stride == expected_stride_rep
    assert key.positions_dtype == torch.int64
    assert key.replay_window == expected_window_rep
    assert key.num_groups == 2
    assert key.block == 1024

    inputs = rs._PAD_V2_REPLAYED_SLOTS_KERNEL.warmup_inputs(key)
    assert inputs["slot_mappings"].dtype == torch.int32
    assert inputs["slot_mappings"].strides == (expected_stride_rep, 1)
    assert inputs["cacheable_groups"].dtype == torch.int32
    assert inputs["cacheable_groups"].shape == (2,)
    assert inputs["query_start_loc"].dtype == torch.int32
    assert inputs["positions"].dtype == torch.int64
    assert inputs["replay_start"].dtype == torch.int32
    assert inputs["replay_window"] == expected_window_rep
    assert inputs["pad_slot_id"] == PAD_SLOT_ID
    assert inputs["NUM_GROUPS"] == 2
    assert inputs["BLOCK"] == 1024
    assert inputs["num_reqs"] == 1


def test_register_respects_gates_and_active_window():
    registry = make_registry(None)

    with registry.activate():
        assert (
            rs.register_pad_v2_replayed_slots_warmup(
                slot_mappings_dtype=torch.int32,
                group_stride=8192,
                positions_dtype=torch.int64,
                replay_window=0,
                num_groups=2,
            )
            is False
        )
        assert (
            rs.register_pad_v2_replayed_slots_warmup(
                slot_mappings_dtype=torch.int32,
                group_stride=8192,
                positions_dtype=torch.int64,
                replay_window=4096,
                num_groups=0,
            )
            is False
        )
        assert (
            rs.register_pad_v2_replayed_slots_warmup(
                slot_mappings_dtype=torch.int32,
                group_stride=0,
                positions_dtype=torch.int64,
                replay_window=4096,
                num_groups=2,
            )
            is False
        )
    assert len(registry) == 0

    with registry.activate():
        assert (
            rs.register_pad_v2_replayed_slots_warmup(
                slot_mappings_dtype=torch.int32,
                group_stride=8192,
                positions_dtype=torch.int64,
                replay_window=4096,
                num_groups=2,
            )
            is True
        )
    assert len(registry) == 1

    # Outside a window: reports eligibility without mutating the registry.
    assert (
        rs.register_pad_v2_replayed_slots_warmup(
            slot_mappings_dtype=torch.int32,
            group_stride=100,
            positions_dtype=torch.int64,
            replay_window=4096,
            num_groups=2,
        )
        is True
    )
    assert len(registry) == 1


def test_engram_model_state_registers_local_and_pcp_layouts():
    from vllm_ascend.models.deepseek_v41.engram import model_state as engram_state

    registry = make_registry(None)
    state = object.__new__(engram_state.EngramModelState)
    state._replay_groups = None
    state.pcp_manager = SimpleNamespace(_global_batch_slot_mappings=torch.zeros((3, 100), dtype=torch.int32))

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_replay_tokens=4096, cacheable=True)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_replay_tokens=0, cacheable=False)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(sliding_window=2048, cacheable=True)),
        ]
    )
    local_slot_mappings = torch.zeros((3, 8192), dtype=torch.int32)
    positions = torch.zeros(8192, dtype=torch.int64)

    with (
        patch("vllm_ascend.core.kv_cache_interface.is_prefix_cacheable", side_effect=lambda s: s.cacheable),
        registry.activate(),
    ):
        assert (
            state.register_replayed_slots_warmup(
                kv_cache_config,
                slot_mappings=local_slot_mappings,
                positions=positions,
            )
            is True
        )

    # Registration must not allocate or mutate self._replay_groups.
    assert state._replay_groups is None
    registrations = registry._registrations[rs._PAD_V2_REPLAYED_SLOTS_KERNEL]
    keys = {
        key
        for args, kwargs in registrations
        for key in rs._PAD_V2_REPLAYED_SLOTS_KERNEL.get_warmup_keys(*args, **kwargs)
    }
    # Distinct stride classes: local 8192 (div16 -> 16) and PCP 100 (generic -> 2).
    assert keys == {
        rs.PadV2ReplayedSlotsKernel.CompileKey(
            slot_mappings_dtype=torch.int32,
            group_stride=16,
            positions_dtype=torch.int64,
            replay_window=16,
            num_groups=2,
            block=1024,
        ),
        rs.PadV2ReplayedSlotsKernel.CompileKey(
            slot_mappings_dtype=torch.int32,
            group_stride=2,
            positions_dtype=torch.int64,
            replay_window=16,
            num_groups=2,
            block=1024,
        ),
    }


def test_engram_model_state_skips_when_no_replay_or_no_slot_mappings():
    from vllm_ascend.models.deepseek_v41.engram import model_state as engram_state

    registry = make_registry(None)
    state = object.__new__(engram_state.EngramModelState)
    state._replay_groups = (0, torch.tensor([0], dtype=torch.int32))
    state.pcp_manager = None

    with registry.activate():
        assert (
            state.register_replayed_slots_warmup(
                None,
                slot_mappings=torch.zeros((1, 64), dtype=torch.int32),
                positions=torch.zeros(64, dtype=torch.int64),
            )
            is False
        )
        state._replay_groups = (4096, torch.tensor([0], dtype=torch.int32))
        assert state.register_replayed_slots_warmup(None, slot_mappings=None, positions=None) is False
    assert len(registry) == 0


def test_wrapper_runtime_and_warmup_bind_exact_kernel_args():
    slot_mappings = torch.zeros((2, 64), dtype=torch.int32)
    cacheable_groups = torch.tensor([0, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    positions = torch.arange(8, dtype=torch.int64)
    replay_start = torch.tensor([2, 0], dtype=torch.int32)

    mock_kernel = MagicMock()
    with patch.object(rs, "_pad_v2_replayed_slots_kernel", mock_kernel):
        rs._PAD_V2_REPLAYED_SLOTS_KERNEL(
            slot_mappings,
            cacheable_groups,
            query_start_loc,
            positions,
            replay_start,
            replay_window=4096,
            pad_slot_id=PAD_SLOT_ID,
            NUM_GROUPS=2,
            BLOCK=1024,
            num_reqs=2,
        )

    mock_kernel.__getitem__.assert_called_once_with((2,))
    _, launch_kwargs = mock_kernel.__getitem__.return_value.call_args
    assert set(launch_kwargs) == set(rs._pad_v2_replayed_slots_kernel.arg_names)
    assert launch_kwargs["slot_mappings_ptr"] is slot_mappings
    assert launch_kwargs["group_stride"] == 64
    assert launch_kwargs["cacheable_groups_ptr"] is cacheable_groups
    assert launch_kwargs["query_start_loc_ptr"] is query_start_loc
    assert launch_kwargs["positions_ptr"] is positions
    assert launch_kwargs["replay_start_ptr"] is replay_start
    assert launch_kwargs["replay_window"] == 4096
    assert launch_kwargs["pad_slot_id"] == PAD_SLOT_ID
    assert launch_kwargs["NUM_GROUPS"] == 2
    assert launch_kwargs["BLOCK"] == 1024

    # Verify the compile() warmup path invokes kernel.warmup(grid=(1,), **kwargs)
    # with the exact same parameter names.
    (key,) = rs._PAD_V2_REPLAYED_SLOTS_KERNEL.get_warmup_keys(
        slot_mappings_dtype=torch.int32,
        group_stride=64,
        positions_dtype=torch.int64,
        replay_window=4096,
        num_groups=2,
    )
    warmup_kernel = MagicMock()
    with patch.object(rs, "_pad_v2_replayed_slots_kernel", warmup_kernel):
        rs._PAD_V2_REPLAYED_SLOTS_KERNEL.compile(key)
    warmup_kernel.warmup.assert_called_once()
    _, warmup_kwargs = warmup_kernel.warmup.call_args
    assert warmup_kwargs["grid"] == (1,)
    assert set(warmup_kwargs) - {"grid"} == set(rs._pad_v2_replayed_slots_kernel.arg_names)
    assert warmup_kwargs["group_stride"] == 16
    assert warmup_kwargs["replay_window"] == 16
    assert warmup_kwargs["pad_slot_id"] == PAD_SLOT_ID
    assert warmup_kwargs["NUM_GROUPS"] == 2
    assert warmup_kwargs["BLOCK"] == 1024


def test_prepare_attn_launches_wrapper_on_non_pcp_replay():
    import numpy as np
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.utils import CpuGpuBuffer

    from vllm_ascend.models.deepseek_v41.engram import model_state as engram_state
    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    state = object.__new__(engram_state.EngramModelState)
    state.pcp_manager = None
    state.decoder_replay_state = None
    state._replay_groups = (4096, torch.tensor([0, 2], dtype=torch.int32))
    state._replay_start_np = np.array([0, 128, 0, 0], dtype=np.int32)
    state._replay_start = CpuGpuBuffer(4, dtype=torch.int32, device="cpu", pin_memory=False)
    state._global_replay_start = CpuGpuBuffer(4, dtype=torch.int32, device="cpu", pin_memory=False)

    slot_mappings = torch.zeros((3, 64), dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        is_dummy=False,
        is_prefilling_np=np.array([True, True], dtype=np.bool_),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int64),
    )

    mock_kernel = MagicMock()
    with (
        patch.object(engram_state, "_pad_v2_replayed_slots_kernel", mock_kernel),
        patch.object(AscendModelState, "prepare_attn", return_value={}),
    ):
        state.prepare_attn(input_batch, CUDAGraphMode.NONE, (), slot_mappings, [], None)

    mock_kernel.__getitem__.assert_called_once_with((2,))
    _, launch_kwargs = mock_kernel.__getitem__.return_value.call_args
    assert launch_kwargs["slot_mappings_ptr"] is slot_mappings
    assert launch_kwargs["group_stride"] == 64
    assert launch_kwargs["replay_window"] == 4096
    assert launch_kwargs["pad_slot_id"] == PAD_SLOT_ID
    assert launch_kwargs["NUM_GROUPS"] == 2
    assert launch_kwargs["BLOCK"] == 1024
