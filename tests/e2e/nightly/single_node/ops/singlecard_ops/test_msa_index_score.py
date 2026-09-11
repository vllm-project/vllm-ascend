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

from __future__ import annotations

import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

BLOCK_SIZE = 128
SCORE_ALIGNMENT = 16
FILL_THRESHOLD = -1.0e30


def _load_golden_module():
    repo_root = Path(__file__).resolve().parents[6]
    golden_path = (
        repo_root
        / "csrc"
        / "attention"
        / "msa_index_score"
        / "tests"
        / "golden"
        / "msa_index_score_golden.py"
    )
    spec = spec_from_file_location("msa_index_score_golden", golden_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


golden = _load_golden_module()


def _prefix_sum(lengths: tuple[int, ...]) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32)


def _is_ascend_950() -> bool:
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


def _discrete_random(
    shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    # Binary fractions keep independent CPU and NPU reductions deterministic.
    return (
        torch.randint(-8, 9, shape, generator=generator, dtype=torch.int32).float()
        / 8.0
    )


def _run_case(
    dtype: torch.dtype,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    table_width: int,
    page_axis_gap: int,
) -> None:
    if dtype == torch.float8_e4m3fn and not _is_ascend_950():
        pytest.skip("FLOAT8_E4M3FN MsaIndexScore requires Ascend 950")

    generator = torch.Generator().manual_seed(2026)
    total_q = sum(q_lens)
    num_heads = 8
    head_dim = 128
    required_blocks = max(math.ceil(length / BLOCK_SIZE) for length in kv_lens)
    assert required_blocks <= table_width
    num_pages = max(required_blocks + 3, 4)

    query_values = _discrete_random((total_q, num_heads, head_dim), generator)
    logical_key_values = _discrete_random(
        (num_pages, BLOCK_SIZE, 1, head_dim), generator
    )

    block_table = torch.zeros((len(q_lens), table_width), dtype=torch.int32)
    for batch_id, kv_len in enumerate(kv_lens):
        num_blocks = math.ceil(kv_len / BLOCK_SIZE)
        if num_blocks:
            block_table[batch_id, :num_blocks] = torch.randperm(
                num_pages, generator=generator
            )[:num_blocks].int()

    base_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    query = query_values.to(base_dtype).npu()
    key_storage = torch.full(
        (num_pages * page_axis_gap, BLOCK_SIZE, 1, head_dim),
        7.0,
        dtype=base_dtype,
    )
    key_storage[::page_axis_gap].copy_(logical_key_values.to(base_dtype))
    key_storage = key_storage.npu()
    if dtype == torch.float8_e4m3fn:
        query = query.to(dtype)
        key_storage = key_storage.to(dtype)
    key = key_storage[::page_axis_gap]

    if page_axis_gap > 1:
        compact_page_stride = BLOCK_SIZE * head_dim
        assert not key.is_contiguous()
        assert key.stride(0) == page_axis_gap * compact_page_stride

    actual_seq_qlen = _prefix_sum(q_lens)
    actual_seq_klen = torch.tensor(kv_lens, dtype=torch.int32)
    start_loc = torch.tensor(
        [max(0, math.ceil(length / BLOCK_SIZE) - 1) for length in kv_lens],
        dtype=torch.int32,
    )
    atten_mask = torch.zeros((2048, 2048), dtype=torch.int8, device="npu")

    expected = golden.msa_index_score_golden(
        golden.MsaIndexScoreGoldenInputs(
            query=query.float().cpu().numpy(),
            key=key.float().cpu().numpy(),
            block_table=block_table.numpy(),
            actual_seq_qlen=actual_seq_qlen.numpy(),
            actual_seq_klen=actual_seq_klen.numpy(),
            start_loc=start_loc.numpy(),
            sparse_mode=golden.SPARSE_MODE_RIGHT_DOWN,
            block_size=BLOCK_SIZE,
            local_blocks=0,
        )
    )

    actual = torch.ops._C_ascend.npu_msa_index_score(
        query,
        key,
        block_table.npu(),
        start_loc.npu(),
        atten_mask=atten_mask,
        actual_seq_qlen=actual_seq_qlen.npu(),
        actual_seq_klen=actual_seq_klen.npu(),
        layout_key="BBND",
        sparse_mode=3,
        init_blocks=0,
        local_blocks=0,
    )
    torch.npu.synchronize()

    actual_cpu = actual.float().cpu()
    expected_cpu = torch.from_numpy(expected)
    expected_width = math.ceil(table_width / SCORE_ALIGNMENT) * SCORE_ALIGNMENT
    assert actual_cpu.shape == (num_heads, total_q, expected_width)

    expected_fill = expected_cpu <= FILL_THRESHOLD
    actual_fill = actual_cpu <= FILL_THRESHOLD
    torch.testing.assert_close(actual_fill, expected_fill, rtol=0, atol=0)

    valid = ~expected_fill
    tolerance = 2.0e-2 if dtype == torch.float8_e4m3fn else 1.0e-3
    torch.testing.assert_close(
        actual_cpu[valid],
        expected_cpu[valid],
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    ("dtype", "q_lens", "kv_lens", "table_width", "page_axis_gap"),
    [
        pytest.param(torch.float16, (32, 17), (300, 130), 8, 1, id="fp16-prefill"),
        pytest.param(torch.bfloat16, (32, 17), (300, 130), 8, 1, id="bf16-prefill"),
        pytest.param(
            torch.bfloat16, (32, 17), (300, 130), 8, 2, id="bf16-strided-page-axis"
        ),
        pytest.param(torch.float8_e4m3fn, (32, 17), (300, 130), 8, 1, id="fp8-prefill"),
        pytest.param(
            torch.float8_e4m3fn, (32, 17), (300, 130), 8, 2, id="fp8-strided-page-axis"
        ),
        pytest.param(
            torch.float8_e4m3fn, (1, 1, 1, 1), (900, 512, 129, 1), 8, 1, id="fp8-decode"
        ),
        pytest.param(
            torch.float8_e4m3fn, (2,), (5,), 257, 1, id="fp8-wide-block-table"
        ),
    ],
)
@torch.inference_mode()
def test_msa_index_score_matches_cpu_golden(
    dtype: torch.dtype,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    table_width: int,
    page_axis_gap: int,
) -> None:
    _run_case(dtype, q_lens, kv_lens, table_width, page_axis_gap)
