#
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
#
"""The KV cache compression helpers must behave the same on both spec layouts.

vLLM renamed ``compress_ratio`` to ``tokens_per_state``, redefined
``storage_block_size`` from a row count to an optional token width, and dropped
``indexes_kv_by_block_stride``. Only one layout is importable at a time, so
these tests assert the invariants that must hold on either one rather than
restating which layout was detected.
"""

from dataclasses import replace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.base import TestBase
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    block_stride_indexing_kwargs,
    compression_kwargs,
    get_storage_block_size,
    optional_spec_compress_ratio,
    serialized_spec_compress_ratio,
    spec_compress_ratio,
)

BLOCK_SIZE = 128
COMPRESS_RATIO = 4


def _mla_spec(compress_ratio: int) -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        **compression_kwargs(compress_ratio),
    )


class TestCompressionKwargs(TestBase):
    def test_kwargs_are_accepted_by_the_spec_constructor(self):
        spec = _mla_spec(COMPRESS_RATIO)

        self.assertEqual(spec_compress_ratio(spec), COMPRESS_RATIO)

    def test_uncompressed_spec_reports_ratio_one(self):
        self.assertEqual(spec_compress_ratio(_mla_spec(1)), 1)


class TestStorageBlockSize(TestBase):
    def test_compression_folds_out_of_the_scheduler_block(self):
        # Ascend kernels index physical rows, so one scheduler block covers
        # block_size // compress_ratio of them regardless of spec layout.
        self.assertEqual(get_storage_block_size(_mla_spec(COMPRESS_RATIO)), BLOCK_SIZE // COMPRESS_RATIO)

    def test_uncompressed_spec_keeps_the_whole_block(self):
        self.assertEqual(get_storage_block_size(_mla_spec(1)), BLOCK_SIZE)

    def test_non_mla_spec_keeps_the_whole_block(self):
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )

        self.assertEqual(get_storage_block_size(spec), BLOCK_SIZE)

    def test_real_page_size_bytes_uses_physical_rows(self):
        compressed = _mla_spec(COMPRESS_RATIO)
        uncompressed = _mla_spec(1)

        self.assertEqual(compressed.real_page_size_bytes * COMPRESS_RATIO, uncompressed.real_page_size_bytes)


class TestBlockStrideIndexing(TestBase):
    def test_kwargs_are_accepted_by_replace(self):
        # Either the field exists and is set, or it is gone and the helper
        # contributes nothing; both must leave replace() working.
        spec = replace(_mla_spec(1), **block_stride_indexing_kwargs(True))

        self.assertEqual(get_storage_block_size(spec), BLOCK_SIZE)

    def test_disabled_matches_the_default(self):
        self.assertEqual(
            replace(_mla_spec(1), **block_stride_indexing_kwargs(False)),
            _mla_spec(1),
        )


class TestOptionalCompressRatio(TestBase):
    def test_mla_spec_reports_its_ratio(self):
        self.assertEqual(optional_spec_compress_ratio(_mla_spec(COMPRESS_RATIO)), COMPRESS_RATIO)

    def test_non_mla_spec_declares_no_compression(self):
        # None keeps callers falling back to the model config; a ratio of 1
        # would instead pin them to the "c1" cache family.
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )

        self.assertIsNone(optional_spec_compress_ratio(spec))


class TestSerializedSpecCompressRatio(TestBase):
    def test_reads_either_peer_field_name(self):
        self.assertEqual(serialized_spec_compress_ratio({"tokens_per_state": COMPRESS_RATIO}), COMPRESS_RATIO)
        self.assertEqual(serialized_spec_compress_ratio({"compress_ratio": COMPRESS_RATIO}), COMPRESS_RATIO)

    def test_missing_field_declares_no_compression(self):
        self.assertIsNone(serialized_spec_compress_ratio({"block_size": BLOCK_SIZE}))

    def test_non_positive_ratio_is_clamped(self):
        # Some models tag "uncompressed" with a zero ratio.
        self.assertEqual(serialized_spec_compress_ratio({"compress_ratio": 0}), 1)
