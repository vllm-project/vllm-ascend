# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from vllm_ascend.patch.worker.patch_mamba_utils import (
    _do_mamba_copy_block_npu,
    _stage_mamba_copy_metadata,
    preprocess_mamba,
)


def _copy_buffer():
    gpu = MagicMock()
    gpu_view = MagicMock()
    gpu.__getitem__.return_value = gpu_view
    return SimpleNamespace(copy_to_gpu=MagicMock(), gpu=gpu), gpu_view


def test_mamba_copy_metadata_is_staged_asynchronously_during_preprocess():
    src_ptrs, _ = _copy_buffer()
    dst_ptrs, _ = _copy_buffer()
    sizes, _ = _copy_buffer()
    copy_bufs = SimpleNamespace(
        offset=2,
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        sizes=sizes,
    )

    _stage_mamba_copy_metadata(copy_bufs)

    for buffer in (src_ptrs, dst_ptrs, sizes):
        buffer.copy_to_gpu.assert_called_once_with(2)


def test_mamba_state_copy_uses_previously_staged_metadata():
    src_ptrs, src_view = _copy_buffer()
    dst_ptrs, dst_view = _copy_buffer()
    sizes, sizes_view = _copy_buffer()
    copy_bufs = SimpleNamespace(
        offset=2,
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        sizes=sizes,
    )

    with patch("vllm_ascend.patch.worker.patch_mamba_utils._batch_memcpy_triton") as batch_memcpy:
        _do_mamba_copy_block_npu(copy_bufs)

    for buffer in (src_ptrs, dst_ptrs, sizes):
        buffer.copy_to_gpu.assert_not_called()
        assert buffer.gpu.__getitem__.call_args_list == [call(slice(None, 2))]
    batch_memcpy.assert_called_once_with(src_view, dst_view, sizes_view)


def test_preprocess_stages_metadata_but_defers_state_copy():
    copy_bufs = SimpleNamespace(
        offset=0,
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(num_speculative_blocks=1, block_size=7),
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=[],
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=[]),
        num_scheduled_tokens={"req": 7},
    )
    input_batch = SimpleNamespace(
        req_ids=["req"],
        num_accepted_tokens_cpu=np.array([2], dtype=np.int32),
    )
    requests = {"req": SimpleNamespace(num_computed_tokens=7)}
    mamba_state_idx = {"req": 0}

    def collect_metadata(copy_buffers, *_args):
        copy_buffers.offset = 1

    with (
        patch(
            "vllm_ascend.patch.worker.patch_mamba_utils.mamba_utils.collect_mamba_copy_meta",
            side_effect=collect_metadata,
        ) as collect,
        patch("vllm_ascend.patch.worker.patch_mamba_utils._can_launch_triton_batch_memcpy", return_value=True),
        patch("vllm_ascend.patch.worker.patch_mamba_utils._stage_mamba_copy_metadata") as stage,
        patch("vllm_ascend.patch.worker.patch_mamba_utils._do_mamba_copy_block_npu") as state_copy,
    ):
        preprocess_mamba(
            scheduler_output,
            SimpleNamespace(),
            SimpleNamespace(),
            mamba_state_idx,
            input_batch,
            requests,
            {},
            (),
            copy_bufs,
        )

    collect.assert_called_once()
    stage.assert_called_once_with(copy_bufs)
    state_copy.assert_not_called()
    assert input_batch.num_accepted_tokens_cpu.tolist() == [1]


def test_load_only_step_does_not_hide_remote_state_copy_on_next_forward():
    copy_bufs = SimpleNamespace(
        offset=0,
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(num_speculative_blocks=7, block_size=128),
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=[],
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=[]),
        num_scheduled_tokens={"req": 0},
    )
    input_batch = SimpleNamespace(
        req_ids=["req"],
        num_accepted_tokens_cpu=np.array([1], dtype=np.int32),
    )
    requests = {"req": SimpleNamespace(num_computed_tokens=0)}
    mamba_state_idx: dict[str, int] = {}

    with (
        patch("vllm_ascend.patch.worker.patch_mamba_utils.mamba_utils.collect_mamba_copy_meta") as collect,
        patch(
            "vllm_ascend.patch.worker.patch_mamba_utils._can_launch_triton_batch_memcpy",
            return_value=True,
        ),
        patch("vllm_ascend.patch.worker.patch_mamba_utils._stage_mamba_copy_metadata") as stage,
    ):
        preprocess_mamba(
            scheduler_output,
            SimpleNamespace(),
            SimpleNamespace(),
            mamba_state_idx,
            input_batch,
            requests,
            {},
            (),
            copy_bufs,
        )

        assert "req" not in mamba_state_idx
        collect.assert_not_called()
        stage.assert_called_once_with(copy_bufs)

        scheduler_output.num_scheduled_tokens["req"] = 8
        requests["req"].num_computed_tokens = 8191
        stage.reset_mock()

        def collect_metadata(copy_buffers, *_args):
            copy_buffers.offset = 1

        collect.side_effect = collect_metadata
        preprocess_mamba(
            scheduler_output,
            SimpleNamespace(),
            SimpleNamespace(),
            mamba_state_idx,
            input_batch,
            requests,
            {},
            (),
            copy_bufs,
        )

    collect.assert_called_once()
    assert collect.call_args.args[4:7] == (63, 64, 0)
    stage.assert_called_once_with(copy_bufs)
    assert mamba_state_idx["req"] == 64
