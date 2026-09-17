# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def make_speculator(architecture):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.attn_architecture = architecture
    spec.num_query_per_req = 5
    return spec


@pytest.mark.parametrize("num_reqs_padded", [1, 4])
def test_direct_mla_builder_updates_speculative_metadata(monkeypatch, num_reqs_padded):
    spec = make_speculator("MLA")
    metadata = {
        "draft": SimpleNamespace(
            attn_state=AscendAttentionState.PrefillCacheHit,
            decode=SimpleNamespace(actual_seq_lengths_q=[5] * num_reqs_padded),
        )
    }
    builder = MagicMock(return_value=metadata)
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", builder)

    result = spec._build_draft_attn_metadata(
        num_reqs=1, num_reqs_padded=num_reqs_padded, num_tokens_padded=num_reqs_padded * 5, step=5
    )

    builder.assert_called_once_with(
        num_reqs=1, num_reqs_padded=num_reqs_padded, num_tokens_padded=num_reqs_padded * 5, step=5
    )
    assert result is metadata
    assert result["draft"].attn_state == AscendAttentionState.ChunkedPrefill
    assert result["draft"].decode.actual_seq_lengths_q == [5 * (i + 1) for i in range(num_reqs_padded)]
    assert not hasattr(result["draft"], "actual_seq_lengths_q")


@pytest.mark.parametrize("metadata_cls", [SimpleNamespace, AscendDSAMetadata, AscendSFAMetadata])
def test_direct_non_dense_mla_builder_preserves_upstream_metadata(monkeypatch, metadata_cls):
    spec = make_speculator(None)
    metadata = metadata_cls.__new__(metadata_cls)
    query_lengths = [5, 5]
    metadata.actual_seq_lengths_q = query_lengths
    initial_attn_state = getattr(metadata, "attn_state", None)
    layers = {"draft": metadata}
    builder = MagicMock(return_value=layers)
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", builder)

    assert spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=2) is layers
    builder.assert_called_once_with(num_reqs=1, num_reqs_padded=2)
    assert metadata.actual_seq_lengths_q is query_lengths
    assert not hasattr(metadata, "decode")
    assert getattr(metadata, "attn_state", None) is initial_attn_state


@pytest.mark.parametrize("architecture", [None, "MLA"])
def test_direct_builder_preserves_empty_metadata(monkeypatch, architecture):
    spec = make_speculator(architecture)
    metadata = {}
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", MagicMock(return_value=metadata))
    assert spec._build_draft_attn_metadata(num_reqs=0, num_reqs_padded=1, num_tokens_padded=5) is metadata


@pytest.mark.parametrize("local_reqs,padded_reqs", [(1, 2), (2, 2), (1, 4)])
def test_mla_eager_dp_metadata_covers_padded_tokens(monkeypatch, local_reqs, padded_reqs):
    spec = make_speculator("MLA")
    spec.num_query_per_req = 3
    metadata = {"draft": SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[3] * local_reqs))}
    builder = MagicMock(return_value=metadata)
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", builder)

    result = spec._build_draft_attn_metadata(
        num_reqs=local_reqs, num_reqs_padded=local_reqs, num_tokens_padded=padded_reqs * 3, step=3
    )

    builder.assert_called_once_with(
        num_reqs=local_reqs, num_reqs_padded=padded_reqs, num_tokens_padded=padded_reqs * 3, step=3
    )
    assert result["draft"].decode.actual_seq_lengths_q == [3 * (i + 1) for i in range(padded_reqs)]
