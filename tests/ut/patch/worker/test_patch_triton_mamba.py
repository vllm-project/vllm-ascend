#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch
import vllm.model_executor.layers.mamba.ops.mamba_ssm as mamba_ssm
import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan as ssd_chunk_scan
import vllm.model_executor.layers.mamba.ops.ssd_combined as ssd_combined

import vllm_ascend.ops.triton.mamba.selective_state_update as ascend_ssu
import vllm_ascend.ops.triton.mamba.ssd_chunk_scan as ascend_ssd_chunk_scan
from vllm_ascend.patch.worker import patch_triton


@pytest.fixture(autouse=True)
def clear_ssu_config_cache():
    ascend_ssu._get_optimal_ssm_config_npu_cached.cache_clear()
    yield
    ascend_ssu._get_optimal_ssm_config_npu_cached.cache_clear()


def _contiguous_strides(shape):
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


class _FakeNPUTensor:
    class _Device:
        type = "npu"
        index = 0

    def __init__(self, shape):
        self.shape = shape
        self.device = self._Device()
        self.dtype = torch.float16
        self._strides = _contiguous_strides(shape)

    def stride(self, dim):
        return self._strides[dim]

    def dim(self):
        return len(self.shape)

    def new_empty(self, shape):
        return _FakeNPUTensor(shape)


def test_chunk_scan_dummy_optional_kernel_args_are_non_null():
    reference = torch.empty(2, 3)

    actual = ascend_ssd_chunk_scan._chunk_scan_optional_kernel_args(reference, None, (1, 1))

    assert actual.shape == (1, 1)
    assert actual.device == reference.device
    assert actual.dtype == reference.dtype


def test_chunk_scan_initial_states_dummy_reuses_states_pointer():
    states = torch.empty(2, 3, 4, 5)

    actual = ascend_ssd_chunk_scan._chunk_scan_initial_states_kernel_arg(states, None)

    assert actual is states
    assert actual.data_ptr() == states.data_ptr()


def test_chunk_scan_initial_states_arg_uses_real_tensor():
    states = torch.empty(2, 3, 4, 5)
    initial_states = torch.empty(6, 3, 4, 5)

    actual = ascend_ssd_chunk_scan._chunk_scan_initial_states_kernel_arg(states, initial_states)

    assert actual is initial_states


def test_chunk_scan_initial_states_strides():
    initial_states = torch.empty(6, 3, 4, 5)

    actual = ascend_ssd_chunk_scan._chunk_scan_initial_states_strides(initial_states)

    assert actual == initial_states.stride()


def test_chunk_scan_default_context_fits_in_one_launch():
    meta = {
        "BLOCK_SIZE_M": ascend_ssd_chunk_scan._CHUNK_SCAN_BLOCK_SIZE_M,
        "BLOCK_SIZE_N": ascend_ssd_chunk_scan._CHUNK_SCAN_BLOCK_SIZE_N,
    }
    launch_ranges = list(
        ascend_ssd_chunk_scan._chunk_scan_launch_ranges(
            chunk_size=128,
            headdim=64,
            nchunks=2048,
            nheads=16,
        )
    )
    grid = ascend_ssd_chunk_scan._chunk_scan_grid(
        chunk_size=128,
        headdim=64,
        nchunks=launch_ranges[0][1],
        nheads=16,
    )(meta)

    assert meta["BLOCK_SIZE_M"] == 128
    assert launch_ranges == [(0, 2048)]
    assert grid == (1, 2048, 16)
    assert grid[0] * grid[1] * grid[2] <= ascend_ssd_chunk_scan._ASCEND_MAX_CORE_DIM


def test_chunk_scan_one_million_context_is_split_below_npu_coredim():
    meta = {
        "BLOCK_SIZE_M": ascend_ssd_chunk_scan._CHUNK_SCAN_BLOCK_SIZE_M,
        "BLOCK_SIZE_N": ascend_ssd_chunk_scan._CHUNK_SCAN_BLOCK_SIZE_N,
    }
    launch_ranges = list(
        ascend_ssd_chunk_scan._chunk_scan_launch_ranges(
            chunk_size=128,
            headdim=64,
            nchunks=8192,
            nheads=16,
        )
    )

    assert launch_ranges == [
        (0, 2048),
        (2048, 2048),
        (4096, 2048),
        (6144, 2048),
    ]
    assert sum(nchunks for _, nchunks in launch_ranges) == 8192
    for _, nchunks in launch_ranges:
        grid = ascend_ssd_chunk_scan._chunk_scan_grid(
            chunk_size=128,
            headdim=64,
            nchunks=nchunks,
            nheads=16,
        )(meta)
        assert grid[0] * grid[1] * grid[2] <= ascend_ssd_chunk_scan._CHUNK_SCAN_MAX_PROGRAMS_PER_LAUNCH
        assert grid[0] * grid[1] * grid[2] <= ascend_ssd_chunk_scan._ASCEND_MAX_CORE_DIM


def test_chunk_scan_npu_without_initial_states_uses_ascend_kernel(monkeypatch):
    captured = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid
            return self

        def __call__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ascend_ssd_chunk_scan, "_chunk_scan_fwd_kernel_npu", FakeKernel())

    seqlen = 16
    nheads = 2
    headdim = 4
    ngroups = 1
    dstate = 8
    nchunks = 2
    chunk_size = 8

    cb = _FakeNPUTensor((nchunks, ngroups, chunk_size, chunk_size))
    x = _FakeNPUTensor((seqlen, nheads, headdim))
    dt = _FakeNPUTensor((nheads, nchunks, chunk_size))
    dA_cumsum = _FakeNPUTensor((nheads, nchunks, chunk_size))
    C = _FakeNPUTensor((seqlen, ngroups, dstate))
    states = _FakeNPUTensor((nchunks, nheads, headdim, dstate))
    cu_chunk_seqlens = _FakeNPUTensor((nchunks + 1,))
    out = _FakeNPUTensor((seqlen, nheads, headdim))
    seq_idx = _FakeNPUTensor((nchunks,))

    ascend_ssd_chunk_scan._chunk_scan_fwd_npu(
        cb,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,
        seq_idx,
    )

    assert captured["HAS_INITSTATES"] is False
    assert captured["chunk_start"] == 0
    assert captured["initstates_ptr"] is states
    assert captured["stride_init_states_batch"] == 0
    assert captured["stride_init_states_head"] == 0
    assert captured["stride_init_states_hdim"] == 0
    assert captured["stride_init_states_dstate"] == 0


def test_chunk_scan_npu_initial_states_uses_ascend_kernel(monkeypatch):
    captured = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid
            return self

        def __call__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ascend_ssd_chunk_scan, "_chunk_scan_fwd_kernel_npu", FakeKernel())

    seqlen = 16
    nheads = 2
    headdim = 4
    ngroups = 1
    dstate = 8
    nchunks = 2
    chunk_size = 8
    batch = 2

    cb = _FakeNPUTensor((nchunks, ngroups, chunk_size, chunk_size))
    x = _FakeNPUTensor((seqlen, nheads, headdim))
    dt = _FakeNPUTensor((nheads, nchunks, chunk_size))
    dA_cumsum = _FakeNPUTensor((nheads, nchunks, chunk_size))
    C = _FakeNPUTensor((seqlen, ngroups, dstate))
    states = _FakeNPUTensor((nchunks, nheads, headdim, dstate))
    cu_chunk_seqlens = _FakeNPUTensor((nchunks + 1,))
    out = _FakeNPUTensor((seqlen, nheads, headdim))
    seq_idx = _FakeNPUTensor((nchunks,))
    initial_states = _FakeNPUTensor((batch, nheads, headdim, dstate))

    ascend_ssd_chunk_scan._chunk_scan_fwd_npu(
        cb,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,
        seq_idx,
        initial_states=initial_states,
    )

    assert captured["HAS_INITSTATES"] is True
    assert captured["chunk_start"] == 0
    assert captured["initstates_ptr"] is initial_states
    assert captured["stride_init_states_batch"] == initial_states.stride(0)
    assert captured["stride_init_states_head"] == initial_states.stride(1)
    assert captured["stride_init_states_hdim"] == initial_states.stride(2)
    assert captured["stride_init_states_dstate"] == initial_states.stride(3)


def test_chunk_scan_patch_updates_combined_import_binding():
    assert ssd_chunk_scan._chunk_scan_fwd is patch_triton._chunk_scan_fwd_npu
    assert ssd_combined._chunk_scan_fwd is patch_triton._chunk_scan_fwd_npu


def test_ssu_patch_uses_ascend_910b3_tuned_configs(monkeypatch):
    monkeypatch.setattr(ascend_ssu.mamba_ssm, "get_ssm_device_name", lambda: "Ascend910B3")
    ascend_ssu._get_optimal_ssm_config_npu_cached.cache_clear()

    assert ascend_ssu.try_get_optimal_ssm_config_npu(64, 128, 1, 16, "float32", False) == (64, 1)
    assert ascend_ssu.try_get_optimal_ssm_config_npu(64, 128, 2, 16, "float32", False) == (64, 4)
    assert ascend_ssu.try_get_optimal_ssm_config_npu(64, 128, 4, 16, "float32", False) == (64, 1)


def test_ssu_config_falls_back_for_other_shapes(monkeypatch):
    sentinel = (8, 2)
    monkeypatch.setattr(ascend_ssu, "_ORIGINAL_TRY_GET_OPTIMAL_SSM_CONFIG", lambda *args: sentinel)
    monkeypatch.setattr(ascend_ssu.mamba_ssm, "get_ssm_device_name", lambda: "Ascend910B3")
    ascend_ssu._get_optimal_ssm_config_npu_cached.cache_clear()

    assert ascend_ssu.try_get_optimal_ssm_config_npu(128, 128, 1, 16, "float32", False) == sentinel
    assert ascend_ssu.try_get_optimal_ssm_config_npu(64, 128, 1, 16, "float16", False) == sentinel


def test_ssu_config_honors_upstream_benchmark_override_after_cache(monkeypatch):
    monkeypatch.setattr(ascend_ssu.mamba_ssm, "get_ssm_device_name", lambda: "Ascend910B3")
    ascend_ssu._get_optimal_ssm_config_npu_cached.cache_clear()

    args = (64, 128, 1, 16, "float32", False)
    assert ascend_ssu.try_get_optimal_ssm_config_npu(*args) == (64, 1)

    monkeypatch.setattr(ascend_ssu.mamba_ssm, "_ssm_config_override", (8, 2))
    monkeypatch.setattr(
        ascend_ssu,
        "_ORIGINAL_TRY_GET_OPTIMAL_SSM_CONFIG",
        lambda *unused: ascend_ssu.mamba_ssm._ssm_config_override,
    )

    assert ascend_ssu.try_get_optimal_ssm_config_npu(*args) == (8, 2)


def test_ssu_patch_updates_vllm_binding():
    assert mamba_ssm.try_get_optimal_ssm_config is patch_triton.try_get_optimal_ssm_config_npu
