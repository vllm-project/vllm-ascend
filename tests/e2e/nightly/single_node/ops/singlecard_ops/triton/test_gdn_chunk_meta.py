# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import gc

import pytest
import torch

from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)
from vllm_ascend.ops.triton.gdn_chunk_meta import build_chunk_meta_device
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize(
    ("cu_seqlens_data", "chunk_size", "input_dtype", "output_dtype"),
    [
        ([0, 4, 7], 4, torch.int32, torch.int32),
        ([0, 4, 4, 12], 4, torch.int32, torch.int32),
        ([0, 128, 1024, 1024, 4096], 64, torch.int64, torch.int32),
    ],
)
@torch.inference_mode()
def test_build_chunk_meta_device_correctness(
    cu_seqlens_data: list[int],
    chunk_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    init_device_properties_triton()
    device = "npu"

    cu_seqlens_cpu = torch.tensor(cu_seqlens_data, dtype=input_dtype)
    cu_seqlens = cu_seqlens_cpu.to(device)

    expected_chunk_indices = prepare_chunk_indices(cu_seqlens_cpu, chunk_size).to(output_dtype)
    expected_chunk_offsets = prepare_chunk_offsets(cu_seqlens_cpu, chunk_size).to(output_dtype)
    expected_update_chunk_offsets = prepare_update_chunk_offsets(cu_seqlens_cpu, chunk_size).to(output_dtype)
    expected_final_chunk_indices = prepare_final_chunk_indices(cu_seqlens_cpu, chunk_size).to(output_dtype)

    out_chunk_indices = torch.empty(
        tuple(expected_chunk_indices.shape),
        dtype=output_dtype,
        device=device,
    )
    out_chunk_offsets = torch.empty(
        tuple(expected_chunk_offsets.shape),
        dtype=output_dtype,
        device=device,
    )
    out_update_chunk_offsets = torch.empty(
        tuple(expected_update_chunk_offsets.shape),
        dtype=output_dtype,
        device=device,
    )
    out_final_chunk_indices = torch.empty(
        tuple(expected_final_chunk_indices.shape),
        dtype=output_dtype,
        device=device,
    )

    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        out_chunk_indices=out_chunk_indices,
        out_chunk_offsets=out_chunk_offsets,
        out_update_chunk_offsets=out_update_chunk_offsets,
        out_final_chunk_indices=out_final_chunk_indices,
    )

    torch.testing.assert_close(out_chunk_indices.cpu(), expected_chunk_indices)
    torch.testing.assert_close(out_chunk_offsets.cpu(), expected_chunk_offsets)
    torch.testing.assert_close(out_update_chunk_offsets.cpu(), expected_update_chunk_offsets)
    torch.testing.assert_close(out_final_chunk_indices.cpu(), expected_final_chunk_indices)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
