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

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_golden_module():
    repo_root = Path(__file__).resolve().parents[3]
    golden_path = (
        repo_root / "csrc" / "attention" / "msa_index_score" / "tests" / "golden" / "msa_index_score_golden.py"
    )
    spec = spec_from_file_location("msa_index_score_golden", golden_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


golden = _load_golden_module()


def _run_golden(
    query: np.ndarray,
    key: np.ndarray,
    block_table: np.ndarray,
    *,
    actual_seq_klen: int,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    return golden.msa_index_score_golden(
        golden.MsaIndexScoreGoldenInputs(
            query=query,
            key=key,
            block_table=block_table,
            actual_seq_qlen=np.array([0, query.shape[0]], dtype=np.int32),
            actual_seq_klen=np.array([actual_seq_klen], dtype=np.int32),
            start_loc=np.array([0], dtype=np.int32),
            sparse_mode=golden.SPARSE_MODE_DEFAULT,
            block_size=key.shape[1],
            scale=scale,
            local_blocks=0,
        )
    )


def test_msa_index_score_golden_handles_paged_non_quant_input() -> None:
    query = np.array(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [2, -1]],
        ],
        dtype=np.float16,
    )
    key = np.array(
        [
            [[[5, 6]], [[7, 8]]],
            [[[1, 2]], [[3, 4]]],
        ],
        dtype=np.float16,
    )

    actual = _run_golden(
        query,
        key,
        np.array([[1, 0]], dtype=np.int32),
        actual_seq_klen=3,
    )

    expected = np.array(
        [
            [[3, 5], [7, 11]],
            [[4, 6], [2, 4]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual[:, :, :2], expected)
    assert np.all(actual[:, :, 2:] == golden.NEG_INF)


def test_msa_index_score_golden_dequantizes_int8_key() -> None:
    query = np.array([[[2, 1]]], dtype=np.float16)
    key = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.int8)
    scale = np.array([[[0.5, 2.0]]], dtype=np.float16)

    actual = _run_golden(
        query,
        key,
        np.array([[0]], dtype=np.int32),
        actual_seq_klen=2,
        scale=scale,
    )

    np.testing.assert_allclose(actual[0, 0, 0], 20.0)
    assert np.all(actual[:, :, 1:] == golden.NEG_INF)


def test_msa_index_score_golden_accepts_noncontiguous_page_axis() -> None:
    query = np.arange(8, dtype=np.float16).reshape(2, 2, 2)
    key_storage = np.arange(24, dtype=np.float16).reshape(6, 2, 1, 2)
    key = key_storage[::2]
    block_table = np.array([[2, 0]], dtype=np.int32)

    assert not key.flags.c_contiguous
    actual = _run_golden(query, key, block_table, actual_seq_klen=3)
    contiguous = _run_golden(
        query,
        np.ascontiguousarray(key),
        block_table,
        actual_seq_klen=3,
    )

    np.testing.assert_array_equal(actual, contiguous)
