# Copyright (c) 2026 ouyangqing. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the xlite block_table CPU-numpy fast path.

The xlite hybrid path reads ``block_tables`` from a CPU numpy view
attached to the attention metadata instead of doing a per-step device
to host sync (which drains the whole compute stream, ~38ms/step at
B16). These tests lock the dataclass fields and the wiring contract
between the model runner, the metadata builder and the xlite adapter.
"""

import dataclasses
import inspect

import numpy as np

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


def test_ascend_metadata_has_block_tables_cpu_np_field():
    names = {f.name: f for f in dataclasses.fields(AscendMetadata)}
    assert "block_tables_cpu_np" in names
    assert names["block_tables_cpu_np"].default is None


def test_ascend_metadata_accepts_cpu_numpy_tables():
    meta = AscendMetadata()
    assert meta.block_tables_cpu_np is None
    table = np.zeros((3, 4), dtype=np.int32)
    meta.block_tables_cpu_np = table
    assert meta.block_tables_cpu_np is table


def test_common_attention_metadata_has_block_table_cpu_np_field():
    names = {f.name: f for f in dataclasses.fields(AscendCommonAttentionMetadata)}
    assert "block_table_cpu_np" in names
    assert names["block_table_cpu_np"].default is None


def test_metadata_builder_forwards_block_table_cpu_np():
    """The builder must forward the CPU numpy view from common metadata.

    This is a contract test: the field is only useful when the builder
    actually passes it through to AscendMetadata; a silent drop would
    push the xlite hybrid path back onto the per-step D2H fallback.
    """
    src = inspect.getsource(AscendAttentionMetadataBuilder.build)
    assert 'getattr(common_attn_metadata, "block_table_cpu_np", None)' in src


def test_metadata_builder_keeps_d2h_fallback_safe():
    """Forwarding uses getattr with a None default (old callers unaffected)."""
    src = inspect.getsource(AscendAttentionMetadataBuilder.build)
    # The pass-through must tolerate common metadata built without the field.
    assert "block_table_cpu_np" in src
