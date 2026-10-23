# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from vllm.v1.utils import CpuGpuBuffer

from tests.ut.models.qwen4_exp.test_cache_config import _allocate, _config, _specs
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.patch.worker.patch_mamba_utils import (
    _collect_mamba_copy_meta_torch,
    _do_mamba_copy_block_npu,
    _do_mamba_copy_block_torch,
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


def test_layerwise_mamba_copy_is_grouped_by_layer():
    """Per-layer scheduling: prepare stages all layers once; each layer's
    do_mamba_copy_block_for_layer consumes only its own slice; finish validates
    all layers executed."""
    from vllm_ascend.patch.worker import patch_mamba_utils as pm

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


def test_qwen_merged_gdn_ple_copies_each_state_with_its_own_stride():
    model_config, specs = _config(cycles=1), _specs(1)
    groups = get_qwen4_exp_kv_cache_groups(model_config, specs)
    config = get_qwen4_exp_kv_cache_config(model_config, groups, 3 * get_qwen4_exp_pool_bytes_per_block(groups))
    _, _, caches = _allocate(config)
    state_names = groups[1].layer_names
    for marker, name in enumerate(state_names, 1):
        for state in caches[name]:
            state[0].fill_(marker)

    def copy_state(state, block_ids, source_index, token_bias):
        source = state[block_ids[source_index]]
        return SimpleNamespace(start_addr=source.data_ptr(), num_elements=source.numel())

    copy_funcs = {specs[name].mamba_type: (copy_state,) * len(specs[name].shapes) for name in state_names}
    num_states = sum(len(caches[name]) for name in state_names)
    buffers = SimpleNamespace(offset=0, sizes=SimpleNamespace(np=np.zeros(num_states, dtype=np.int32)))
    request = SimpleNamespace(block_ids=([0, 1, 2], [0, 1, 2], [0, 1, 2]))
    context = {name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()}
    _collect_mamba_copy_meta_torch(buffers, config, copy_funcs, [1], 0, 2, 0, request, context)
    assert buffers.offset == 7  # Three Conv/SSM pairs and one PLE state.
    assert buffers.sizes.np.tolist() == [2560, 32768] * 3 + [20480]
    _do_mamba_copy_block_torch(buffers)
    for marker, name in enumerate(state_names, 1):
        for state in caches[name]:
            assert torch.all(state[2] == marker)
            assert not torch.count_nonzero(state[1])
    assert not torch.count_nonzero(caches["model.layers.3.self_attn.indexer.compressed_key_cache"])
