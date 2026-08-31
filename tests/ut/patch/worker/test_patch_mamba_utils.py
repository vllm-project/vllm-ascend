# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.patch.worker.patch_mamba_utils import (
    _do_mamba_copy_block_npu,
    postprocess_mamba_align_gpu_npu,
    preprocess_mamba,
)


def test_preprocess_stages_metadata_but_defers_state_copy():
    # Separate CPU-backed buffers let us check staging without an NPU.
    buffers = [
        CpuGpuBuffer(2, dtype=dtype, device=torch.device("cpu"), pin_memory=False)
        for dtype in (torch.int64, torch.int64, torch.int32)
    ]
    copy_bufs = SimpleNamespace(
        offset=0,
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(num_speculative_blocks=1, block_size=7),
        src_ptrs=buffers[0],
        dst_ptrs=buffers[1],
        sizes=buffers[2],
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
        for buffer, value in zip(buffers, (100, 200, 32)):
            buffer.np[0] = value
        copy_buffers.offset = 1

    with (
        patch(
            "vllm_ascend.patch.worker.patch_mamba_utils.mamba_utils.collect_mamba_copy_meta",
            side_effect=collect_metadata,
        ),
        patch("vllm_ascend.patch.worker.patch_mamba_utils._can_launch_triton_batch_memcpy", return_value=True),
        patch("vllm_ascend.patch.worker.patch_mamba_utils._batch_memcpy_triton") as state_copy,
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

        state_copy.assert_not_called()
        for buffer, value in zip(buffers, (100, 200, 32)):
            torch.testing.assert_close(buffer.gpu, torch.tensor([value, 0], dtype=buffer.gpu.dtype))
            # Later host reuse must not overwrite metadata staged for this step.
            buffer.cpu.fill_(-1)

        _do_mamba_copy_block_npu(copy_bufs)

    state_copy.assert_called_once()
    for actual, value in zip(state_copy.call_args.args, (100, 200, 32)):
        torch.testing.assert_close(actual, torch.tensor([value], dtype=actual.dtype))
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


def test_fused_mamba_path_keeps_accepted_tokens_on_device():
    state_idx = CpuGpuBuffer(2, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    src_col = CpuGpuBuffer(2, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    token_bias = CpuGpuBuffer(2, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    align_ctx = SimpleNamespace(
        is_initialized=True,
        mamba_state_idx_buf=state_idx,
        precopy_src_col_buf=src_col,
        precopy_token_bias_buf=token_bias,
        run_fused_precopy=MagicMock(),
    )
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
        num_accepted_tokens_cpu=np.array([99], dtype=np.int32),
    )
    requests = {"req": SimpleNamespace(num_computed_tokens=7)}
    accepted_tokens = torch.tensor([3, 1], dtype=torch.int32)

    with patch(
        "vllm_ascend.patch.worker.patch_mamba_utils._can_launch_triton_batch_memcpy",
        return_value=True,
    ):
        preprocess_mamba(
            scheduler_output,
            SimpleNamespace(),
            SimpleNamespace(),
            {},
            input_batch,
            requests,
            {},
            (),
            copy_bufs,
            align_ctx=align_ctx,
            num_accepted_tokens_gpu=accepted_tokens,
        )

    call_kwargs = align_ctx.run_fused_precopy.call_args.kwargs
    torch.testing.assert_close(call_kwargs["state_idx_gpu"][:1], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(call_kwargs["src_col_gpu"][:1], torch.tensor([0], dtype=torch.int32))
    torch.testing.assert_close(call_kwargs["token_bias_gpu"][:1], torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(accepted_tokens, torch.tensor([1, 1], dtype=torch.int32))
    assert input_batch.num_accepted_tokens_cpu.tolist() == [99]


def test_fused_mamba_postprocess_does_not_copy_accepted_tokens_to_cpu():
    ctx = SimpleNamespace(
        is_initialized=True,
        mamba_state_idx_buf=SimpleNamespace(gpu=torch.tensor([0], dtype=torch.int32)),
        num_scheduled_tokens_buf=SimpleNamespace(gpu=torch.tensor([8], dtype=torch.int32)),
        num_computed_tokens_buf=SimpleNamespace(gpu=torch.tensor([128], dtype=torch.int32)),
        num_draft_tokens_buf=SimpleNamespace(gpu=torch.tensor([7], dtype=torch.int32)),
        run_fused_postprocess=MagicMock(),
    )
    cpu_counts = torch.tensor([-1], dtype=torch.int32)

    postprocess_mamba_align_gpu_npu(
        bufs=SimpleNamespace(postprocess_align=ctx),
        num_reqs=1,
        num_accepted_tokens_gpu=torch.tensor([4], dtype=torch.int32),
        num_accepted_tokens_cpu_tensor=cpu_counts,
        input_batch=SimpleNamespace(),
        kv_cache_config=SimpleNamespace(),
        forward_context={},
        mamba_state_copy_funcs=(),
    )

    ctx.run_fused_postprocess.assert_called_once()
    torch.testing.assert_close(cpu_counts, torch.tensor([-1], dtype=torch.int32))
