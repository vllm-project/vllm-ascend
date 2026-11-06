# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.patch.worker.patch_v1.patch_mamba_utils import (
    _do_mamba_copy_block_npu,
    postprocess_mamba_align,
    preprocess_mamba,
    stage_mamba_postprocess_inputs,
)


@pytest.mark.parametrize("new_api", [False, True])
def test_postprocess_staging_retains_state_indices_and_active_rows(new_api):
    buf = CpuGpuBuffer(3, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    buf.cpu.fill_(-1)
    buf.gpu.fill_(-1)
    ctx = SimpleNamespace(mamba_state_idx_buf=buf)
    scheduler, requests = object(), object()
    req_ids = ["second", "first", "inactive"]
    indices = {"first": 7, "second": 3}
    seen: list[str] = []

    def new_stage(actual_ctx, actual_scheduler, ids, count, actual_requests, *, num_decodes=None):
        assert actual_ctx is ctx and actual_scheduler is scheduler and actual_requests is requests
        assert ids is req_ids and count == 2
        # The copied Ascend preprocess owns the index upload, including reorder.
        torch.testing.assert_close(buf.gpu, torch.tensor([3, 7, -1], dtype=torch.int32))
        seen.append("new")

    def old_stage(actual_ctx, actual_scheduler, ids, count, actual_requests, state_indices):
        assert state_indices is indices
        for i, req_id in enumerate(ids[:count]):
            actual_ctx.mamba_state_idx_buf.np[i] = state_indices[req_id]
        actual_ctx.mamba_state_idx_buf.copy_to_gpu(count)
        new_stage(actual_ctx, actual_scheduler, ids, count, actual_requests)

    native = SimpleNamespace(stage_postprocess_inputs_to_gpu=new_stage if new_api else old_stage)
    if new_api:
        native.postprocess_mamba_gpu = object()
    with patch("vllm_ascend.patch.worker.patch_v1.patch_mamba_utils.mamba_utils", native):
        stage_mamba_postprocess_inputs(ctx, scheduler, req_ids, 2, requests, indices)
    assert seen == ["new"]


@pytest.mark.parametrize("legacy_fallback", [False, True])
def test_postprocess_preserves_prefix_migration_and_310p_fallback(legacy_fallback):
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    seen: list[str] = []

    def new_postprocess(*, num_accepted_tokens, run_prefix_state_migration):
        assert num_accepted_tokens is accepted
        assert run_prefix_state_migration is True
        seen.append("new")

    def fallback(*, num_accepted_tokens):
        assert num_accepted_tokens is accepted
        seen.append("fallback")

    native = SimpleNamespace(postprocess_mamba_gpu=new_postprocess)
    if legacy_fallback:
        native.postprocess_mamba_align_gpu = fallback
    with patch("vllm_ascend.patch.worker.patch_v1.patch_mamba_utils.mamba_utils", native):
        postprocess_mamba_align(num_accepted_tokens=accepted)
    assert seen == ["fallback" if legacy_fallback else "new"]


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
            "vllm_ascend.patch.worker.patch_v1.patch_mamba_utils.mamba_utils.collect_mamba_copy_meta",
            side_effect=collect_metadata,
        ),
        patch("vllm_ascend.patch.worker.patch_v1.patch_mamba_utils._can_launch_triton_batch_memcpy", return_value=True),
        patch("vllm_ascend.patch.worker.patch_v1.patch_mamba_utils._batch_memcpy_triton") as state_copy,
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
        patch("vllm_ascend.patch.worker.patch_v1.patch_mamba_utils.mamba_utils.collect_mamba_copy_meta") as collect,
        patch(
            "vllm_ascend.patch.worker.patch_v1.patch_mamba_utils._can_launch_triton_batch_memcpy",
            return_value=True,
        ),
        patch("vllm_ascend.patch.worker.patch_v1.patch_mamba_utils._stage_mamba_copy_metadata") as stage,
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


def test_layerwise_mamba_copy_is_grouped_by_layer():
    """Per-layer scheduling: prepare stages all layers once; each layer's
    do_mamba_copy_block_for_layer consumes only its own slice; finish validates
    all layers executed."""
    from vllm_ascend.patch.worker.patch_v1 import patch_mamba_utils as pm

    bufs = SimpleNamespace(
        src_ptrs=CpuGpuBuffer(8, dtype=torch.int64, device=torch.device("cpu"), pin_memory=False),
        dst_ptrs=CpuGpuBuffer(8, dtype=torch.int64, device=torch.device("cpu"), pin_memory=False),
        sizes=CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False),
        offset=0,
        _layer_copy_metadata={
            "layers.0.linear_attn": ([11, 12], [21, 22], [31, 32]),
            "layers.1.linear_attn": ([13, 14], [23, 24], [33, 34]),
        },
        _layer_copy_slices={},
        _layer_copy_staged=False,
        _layer_tensor_copy_pairs={},
        _tensor_copy_pairs=[],
    )

    copy_calls = []
    orig = pm._batch_memcpy_triton
    pm._batch_memcpy_triton = lambda s, d, z: copy_calls.append((list(s), list(d), list(z)))
    try:
        pm.prepare_mamba_copy_by_layer(bufs)
        assert bufs._layer_copy_staged is True
        assert bufs.src_ptrs.np[:4].tolist() == [11, 12, 13, 14]

        pm.do_mamba_copy_block_for_layer(bufs, "layers.0.linear_attn")
        pm.do_mamba_copy_block_for_layer(bufs, "layers.1.linear_attn")

        assert copy_calls == [
            ([11, 12], [21, 22], [31, 32]),
            ([13, 14], [23, 24], [33, 34]),
        ]

        pm.finish_mamba_copy_by_layer(bufs)
        assert bufs.offset == 0
    finally:
        pm._batch_memcpy_triton = orig

    # finish must raise if a layer was never executed
    bufs2 = SimpleNamespace(
        src_ptrs=CpuGpuBuffer(8, dtype=torch.int64, device=torch.device("cpu"), pin_memory=False),
        dst_ptrs=CpuGpuBuffer(8, dtype=torch.int64, device=torch.device("cpu"), pin_memory=False),
        sizes=CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False),
        offset=0,
        _layer_copy_metadata={"layers.2.linear_attn": ([15], [25], [35])},
        _layer_copy_slices={},
        _layer_copy_staged=False,
        _layer_tensor_copy_pairs={},
        _tensor_copy_pairs=[],
    )
    try:
        pm.finish_mamba_copy_by_layer(bufs2)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "finish must raise when a loaded layer never executed its copy"
