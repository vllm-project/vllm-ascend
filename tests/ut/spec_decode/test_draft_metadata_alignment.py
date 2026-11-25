# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FULL-graph draft updates preserve row alignment after builder padding."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackend
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionBackend,
    AscendFlashAttentionMetadata,
)
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

# These CPU tests inspect the real FA3 backend class, without invoking its
# optional device kernels. Keep the extension mock scoped to this import so
# collection works without flash-attention-npu and does not leak the mock.
with patch.dict(sys.modules, {"flash_attn_npu_3": Mock()}):
    from vllm_ascend.attention.flash_attention_v3 import (
        AscendFlashAttentionBackend as NativeFA3Backend,
    )


def _speculator(kind, query_count):
    cls = AscendDFlashSpeculator if kind == "dflash" else AscendDSparkSpeculator
    spec = cls.__new__(cls)
    spec.num_query_per_req = query_count
    spec.attn_architecture = "MLA" if kind == "dspark_mla" else "GQA"
    backend = AscendMLABackend if kind == "dspark_mla" else AscendAttentionBackend
    spec.attn_backends = {name: backend for name in ("draft.2", "draft.0", "draft", "fia", "external")}
    return spec


def _metadata(kind, num_reqs, query_count, extra_row, offset=0):
    # Three live requests occupy four graph slots. The builder can append a
    # fifth synthetic request to cover the Q tail beyond the zero-length slot.
    lengths = [101 + offset, 202 + offset, 303 + offset] + [0] * (num_reqs - 3)
    query_ends = [query_count * (i + 1) for i in range(3)] + [3 * query_count] * (num_reqs - 3)
    if extra_row:
        lengths.append(0)
        query_ends.append(num_reqs * query_count)
    rows = len(lengths)
    table = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2) + offset
    fields = dict(
        actual_seq_lengths_q=query_ends,
        seq_lens_list=lengths,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
    )
    if kind == "dspark_mla":
        fields["block_table"] = table
        decode = SimpleNamespace(**fields)
        return SimpleNamespace(decode=decode, untouched=object()), decode, table
    fields.update(seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32), block_tables=table)
    metadata = SimpleNamespace(**fields)
    return metadata, metadata, table


@pytest.mark.parametrize("kind", ["dspark_gqa", "dflash"])
@pytest.mark.parametrize(
    "num_reqs,query_count,extra_row",
    [(4, 5, True), (4, 6, True), (54, 5, True), (4, 5, False)],
)
def test_fullgraph_update_aligns_rows_without_changing_request_data(kind, num_reqs, query_count, extra_row):
    spec = _speculator(kind, query_count)
    layers = {}
    originals = {}
    for name, offset in [("draft.2", 0), ("draft.0", 1000)]:
        outer, metadata, table = _metadata(kind, num_reqs, query_count, extra_row, offset)
        layers[name] = outer
        originals[name] = (
            metadata,
            metadata.seq_lens.clone(),
            table.clone(),
            metadata.seq_lens,
            table,
        )
    original_order = list(layers)

    result = spec._update_draft_attn_metadata(layers, num_reqs)
    if kind != "dflash":
        assert result is layers
    assert list(layers) == original_order
    for name, outer in layers.items():
        metadata, lengths, table, original_lengths, original_table = originals[name]
        assert metadata.actual_seq_lengths_q == [query_count * (i + 1) for i in range(num_reqs)]
        assert metadata.actual_seq_lengths_q[-1] == num_reqs * query_count
        assert metadata.seq_lens_list == lengths[:num_reqs].tolist()
        torch.testing.assert_close(metadata.seq_lens, lengths[:num_reqs])
        torch.testing.assert_close(metadata.seq_lens_cpu, lengths[:num_reqs])
        actual_table = metadata.block_tables
        torch.testing.assert_close(actual_table, table[:num_reqs])
        # Slicing metadata must not modify the builder's backing buffers.
        torch.testing.assert_close(original_lengths, lengths)
        torch.testing.assert_close(original_table, table)

    # Reusing already-aligned metadata must not remove any more rows.
    spec._update_draft_attn_metadata(layers, num_reqs)
    for metadata, lengths, table, _, _ in originals.values():
        assert metadata.seq_lens_list == lengths[:num_reqs].tolist()
        torch.testing.assert_close(metadata.seq_lens, lengths[:num_reqs])
        actual_table = metadata.block_tables
        torch.testing.assert_close(actual_table, table[:num_reqs])


@pytest.mark.parametrize("kind", ["dspark_gqa", "dflash"])
def test_fullgraph_update_preserves_absent_block_tables(kind):
    spec = _speculator(kind, 5)
    outer, metadata, _ = _metadata(kind, 4, 5, True)
    metadata.block_tables = None
    spec._update_draft_attn_metadata({"draft": outer}, 4)
    assert metadata.actual_seq_lengths_q == [5, 10, 15, 20]
    assert metadata.seq_lens_list == [101, 202, 303, 0]
    torch.testing.assert_close(metadata.seq_lens, torch.tensor([101, 202, 303, 0], dtype=torch.int32))
    torch.testing.assert_close(metadata.seq_lens_cpu, metadata.seq_lens)
    assert metadata.block_tables is None


def test_dspark_update_leaves_external_flashmla_metadata_untouched():
    spec = _speculator("dspark_mla", 5)
    # No FIA-specific fields: the external backend must be skipped entirely.
    external = SimpleNamespace(external_flashmla=object(), decode=object())
    outer, metadata, _ = _metadata("dspark_mla", 4, 5, True)
    before = vars(external).copy()
    layers = {"external": external, "fia": outer}
    assert spec._update_draft_attn_metadata(layers, 4) is layers
    assert vars(external) == before
    assert metadata.actual_seq_lengths_q == [5, 10, 15, 20]
    assert len(metadata.seq_lens_list) == metadata.seq_lens.shape[0] == metadata.block_table.shape[0] == 5


class OtherAttentionBackend(AscendAttentionBackend):
    """A subclass must not inherit FIA-specific trimming implicitly."""


@pytest.mark.parametrize("kind", ["dspark_gqa", "dflash"])
@pytest.mark.parametrize(
    "backend",
    [
        AscendFlashAttentionBackend,
        NativeFA3Backend,
        AscendC8MXFPAttentionBackend,
        OtherAttentionBackend,
    ],
)
@pytest.mark.parametrize("cpu_lengths", ["none", "absent", "populated"])
def test_fullgraph_update_preserves_non_fia_kv_metadata(kind, backend, cpu_lengths):
    spec = _speculator(kind, 5)
    spec.attn_backends["other"] = backend
    fia, _, _ = _metadata(kind, 4, 5, True)
    other, _, _ = _metadata(kind, 4, 5, True)
    if cpu_lengths == "none":
        other = AscendFlashAttentionMetadata(seq_lens=other.seq_lens, block_tables=other.block_tables)
    elif cpu_lengths == "absent":
        del other.seq_lens_list
        del other.seq_lens_cpu
    before = vars(other).copy()

    # A mixed layer dictionary must trim only the registered FIA layer,
    # even if another backend supplies the same populated CPU fields.
    spec._update_draft_attn_metadata({"draft.0": fia, "other": other}, 4)

    assert fia.seq_lens_list == [101, 202, 303, 0]
    assert fia.block_tables.shape[0] == 4
    assert other.actual_seq_lengths_q == [5, 10, 15, 20]
    assert set(vars(other)) == set(before) | {"actual_seq_lengths_q"}
    for name, value in before.items():
        if name != "actual_seq_lengths_q":
            assert getattr(other, name) is value


def test_dspark_mla_keeps_original_q_rewrite_without_trimming_kv():
    spec = _speculator("dspark_mla", 5)
    outer, metadata, _ = _metadata("dspark_mla", 4, 5, True)
    before = vars(metadata).copy()

    spec._update_draft_attn_metadata({"draft.0": outer}, 4)

    assert metadata.actual_seq_lengths_q == [5, 10, 15, 20]
    for name, value in before.items():
        if name != "actual_seq_lengths_q":
            assert getattr(metadata, name) is value
