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
"""Unit tests for the xlite V2 attention-metadata bridge.

All tests run on CPU. The xlite runtime extension (``xlite._C``) is not part
of the CI dependencies, so a minimal stub is installed when it is missing;
only pure-Python metadata logic is exercised.
"""

import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

# The real xlite._C probes the NPU runtime at import time and *aborts the
# process* (not an ImportError) on CPU-only runners, so it must only be
# imported when a real NPU is present. Otherwise install the stub below.
_HAVE_XLITE = False
try:
    import subprocess as _sp

    _sp.run(["npu-smi", "info"], capture_output=True, check=True)
    import xlite._C  # noqa: F401

    _HAVE_XLITE = True
except Exception:  # pragma: no cover - CPU-only runners take the stub path
    _HAVE_XLITE = False
    import enum

    class _AttnKind(enum.IntEnum):
        MHA = 0
        MLA = 1
        HYBRID = 2
        DSA = 3

    xlite_mod = types.ModuleType("xlite")
    xlite_c = types.ModuleType("xlite._C")
    # setattr keeps the dynamic ModuleType attributes invisible to mypy.
    for _name, _value in {
        "AttnMHA": _AttnKind.MHA,
        "AttnMLA": _AttnKind.MLA,
        "AttnHybrid": _AttnKind.HYBRID,
        "AttnDSA": _AttnKind.DSA,
        "AttnMeta": type("AttnMeta", (), {}),
        "AttnMetaV2": type("AttnMetaV2", (), {}),
        # required by vllm_ascend/xlite/utils.py
        "Model": type("Model", (), {}),
        "ModelConfig": type("ModelConfig", (), {}),
        "Runtime": MagicMock,
        "ScoringFuncSoftmax": 0,
        "ScoringFuncSigmoid": 1,
    }.items():
        setattr(xlite_c, _name, _value)
    for _name in (
        "graph_capture_begin",
        "graph_capture_end",
        "graph_destroy",
        "graph_exec",
        "set_gdn_chunk_callback",
    ):
        setattr(xlite_c, _name, MagicMock())
    sys.modules.setdefault("xlite", xlite_mod)
    sys.modules.setdefault("xlite._C", xlite_c)

from vllm_ascend.xlite.metadata import build_hybrid_step_metadata  # noqa: E402
from vllm_ascend.xlite.xlite import XliteWrapper  # noqa: E402


def _attn_metadata(
    *,
    num_decodes=2,
    num_prefills=0,
    seq_lens_list=(10, 20),
    actual_seq_lengths_q=(1, 2),
    num_actual_tokens=2,
    with_device=True,
    block_tables_cpu_np=None,
):
    """AscendMetadata-shaped namespace: 2 decode requests, contexts 9/19.

    ``actual_seq_lengths_q`` follows the vLLM convention: *cumulative* query
    lengths, so the default ``(1, 2)`` means two 1-token decode requests.
    """
    ns = SimpleNamespace(
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        seq_lens_list=list(seq_lens_list),
        actual_seq_lengths_q=list(actual_seq_lengths_q),
        num_actual_tokens=num_actual_tokens,
        block_tables_cpu_np=block_tables_cpu_np,
    )
    if with_device:
        n = len(seq_lens_list)
        ns.query_start_loc_gpu = torch.tensor([0] + list(actual_seq_lengths_q), dtype=torch.int32)
        ns.seq_lens_gpu = torch.tensor(seq_lens_list[:n], dtype=torch.int32)
        ns.slot_mapping = torch.arange(num_actual_tokens + 2, dtype=torch.int64) + 100
        ns.block_tables = torch.tensor([[5, 0], [7, 0], [9, 9]][:n], dtype=torch.int32)
    return ns


def test_build_hybrid_step_metadata_zero_copy_aliases():
    meta_in = _attn_metadata()
    positions = torch.tensor([9, 19], dtype=torch.int64)

    step = build_hybrid_step_metadata(meta_in, positions)

    assert step is not None
    assert step.num_reqs == 2
    assert step.num_actual_tokens == 2
    assert step.query_lens == [1, 1]
    assert step.cached_lens == [9, 19]
    meta = step.attn_meta
    # Derived device vectors (int32).
    assert meta.lens.dtype == torch.int32
    assert meta.lens.tolist() == [1, 1]
    assert meta.cached_lens.tolist() == [9, 19]
    # xlite's queryStartLoc is the exclusive prefix sum: vllm's 0-prefixed
    # cumulative tensor sliced to [num_reqs].
    assert meta.query_start_loc.tolist() == [0, 1]
    # vLLM keeps slot_mapping in int64; the xlite kernels read it as int32.
    assert meta.slot_mapping[0].dtype == torch.int32
    assert meta.slot_mapping[0].tolist() == [100, 101]
    # Zero-copy views of the vLLM-Ascend buffers.
    assert meta.block_tables[0].data_ptr() == meta_in.block_tables.data_ptr()
    assert meta.query_start_loc.data_ptr() == meta_in.query_start_loc_gpu.data_ptr()
    assert meta.positions.tolist() == [9, 19]
    assert meta.positions.is_contiguous()
    # Host lists for the C++ tile-size selection.
    assert meta.lens_cpu == [1, 1]
    assert meta.cached_lens_cpu == [9, 19]


def test_query_start_loc_is_exclusive_prefix_sum():
    meta_in = _attn_metadata(
        seq_lens_list=(13, 28),
        actual_seq_lengths_q=(3, 8),
        num_actual_tokens=8,
    )
    positions = torch.arange(8, dtype=torch.int64)

    step = build_hybrid_step_metadata(meta_in, positions)

    assert step is not None
    assert step.query_lens == [3, 5]
    assert step.cached_lens == [10, 23]
    assert step.attn_meta.query_start_loc.tolist() == [0, 3]
    assert step.attn_meta.lens.tolist() == [3, 5]


def test_build_rejects_query_lens_sum_mismatch():
    meta_in = _attn_metadata(num_actual_tokens=3)  # query lens sum is 2
    positions = torch.arange(3, dtype=torch.int64)

    assert build_hybrid_step_metadata(meta_in, positions) is None


def test_build_rejects_missing_device_tensors():
    meta_in = _attn_metadata(with_device=False)
    positions = torch.arange(2, dtype=torch.int64)

    assert build_hybrid_step_metadata(meta_in, positions) is None


def test_build_rejects_dummy_block_table_when_host_copy_present():
    # First request has context but an all-zero (dummy/capture) first block.
    block_tables_cpu_np = [[0, 0], [7, 0]]
    meta_in = _attn_metadata(block_tables_cpu_np=block_tables_cpu_np)
    positions = torch.arange(2, dtype=torch.int64)

    assert build_hybrid_step_metadata(meta_in, positions) is None


def test_build_accepts_fresh_prefill_with_zero_block_table():
    # cached_len == 0 (fresh prefill): the first block id may legitimately be
    # unwritten; the dummy-table check only guards requests WITH context.
    block_tables_cpu_np = [[0, 0], [7, 0]]
    meta_in = _attn_metadata(
        num_decodes=0,
        num_prefills=2,
        seq_lens_list=(10, 20),
        actual_seq_lengths_q=(10, 20),
        num_actual_tokens=20,
        block_tables_cpu_np=block_tables_cpu_np,
    )
    positions = torch.arange(20, dtype=torch.int64)

    step = build_hybrid_step_metadata(meta_in, positions)

    assert step is not None
    assert step.cached_lens == [0, 10]


def test_num_reqs_hint_used_when_counts_missing():
    meta_in = _attn_metadata(num_decodes=0, num_prefills=0)
    positions = torch.arange(2, dtype=torch.int64)

    step = build_hybrid_step_metadata(meta_in, positions, num_reqs_hint=2)

    assert step is not None
    assert step.num_reqs == 2


def test_build_rejects_zero_requests():
    meta_in = _attn_metadata(num_decodes=0, num_prefills=0)
    positions = torch.arange(2, dtype=torch.int64)

    assert build_hybrid_step_metadata(meta_in, positions) is None


def test_positions_2d_is_flattened():
    meta_in = _attn_metadata()
    positions = torch.tensor([[9, 19]], dtype=torch.int64)

    step = build_hybrid_step_metadata(meta_in, positions)

    assert step is not None
    assert step.attn_meta.positions.tolist() == [9, 19]


def test_host_lens_fallback_via_router_lists():
    # No seq_lens_list / actual_seq_lengths_q: derived from the raw tensors.
    meta_in = _attn_metadata()
    meta_in.seq_lens_list = None
    meta_in.actual_seq_lengths_q = None
    meta_in.query_start_loc = torch.tensor([0, 3, 8], dtype=torch.int64)
    meta_in.seq_lens_cpu = torch.tensor([13, 28], dtype=torch.int64)
    meta_in.num_actual_tokens = 8
    meta_in.query_start_loc_gpu = torch.tensor([0, 3, 8], dtype=torch.int32)
    meta_in.slot_mapping = torch.arange(10, dtype=torch.int64) + 100
    positions = torch.arange(8, dtype=torch.int64)

    step = build_hybrid_step_metadata(meta_in, positions)

    assert step is not None
    assert step.query_lens == [3, 5]
    assert step.cached_lens == [10, 23]


def test_hybrid_branch_forwards_exactly_once():
    """Regression: the hybrid branch must run exactly ONE xlite forward per
    step between gather and scatter (a duplicated forward would advance the
    recurrent GDN state twice per token)."""
    src = inspect.getsource(XliteWrapper.__call__)
    hybrid = src.split("step = build_hybrid_step_metadata")[1].split("attn_metadata_router")[0]
    assert hybrid.count("self.xlite_model.forward_v2(") == 1
    assert hybrid.count("self.xlite_model.forward_with_inputs_embeds_v2(") == 1
    assert hybrid.count("self._hybrid_gather(") == 1
    assert hybrid.count("self._hybrid_scatter(") == 1
    gather_at = hybrid.index("self._hybrid_gather(")
    scatter_at = hybrid.index("self._hybrid_scatter(")
    assert gather_at < hybrid.index("self.xlite_model.forward_v2(") < scatter_at
    assert gather_at < hybrid.index("self.xlite_model.forward_with_inputs_embeds_v2(") < scatter_at
