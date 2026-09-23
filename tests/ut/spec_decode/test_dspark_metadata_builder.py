# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import AttentionConfig
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def make_speculator(architecture):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.attn_architecture = architecture
    spec.use_dcp = False
    spec.requires_non_causal = True
    spec.vllm_config = SimpleNamespace(
        attention_config=AttentionConfig(), parallel_config=SimpleNamespace(decode_context_parallel_size=1)
    )
    spec.num_query_per_req = 5
    spec.input_buffers = SimpleNamespace(positions=torch.arange(32))
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
    assert result["draft"].attn_state == AscendAttentionState.PrefillCacheHit
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

    assert spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=2, num_tokens_padded=10, step=5) is layers
    builder.assert_called_once_with(num_reqs=1, num_reqs_padded=2, num_tokens_padded=10, step=5)
    assert metadata.actual_seq_lengths_q is query_lengths
    assert not hasattr(metadata, "decode")
    assert getattr(metadata, "attn_state", None) is initial_attn_state


@pytest.mark.parametrize("architecture", [None, "GQA", "MLA"])
def test_direct_builder_preserves_empty_metadata(monkeypatch, architecture):
    spec = make_speculator(architecture)
    metadata: dict[str, SimpleNamespace] = {}
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", MagicMock(return_value=metadata))
    assert spec._build_draft_attn_metadata(num_reqs=0, num_reqs_padded=1, num_tokens_padded=5, step=5) is metadata


@pytest.mark.parametrize("architecture", ["GQA", "MLA"])
@pytest.mark.parametrize("local_reqs,padded_reqs", [(1, 2), (2, 2), (1, 4)])
def test_eager_dp_metadata_covers_padded_tokens(monkeypatch, architecture, local_reqs, padded_reqs):
    spec = make_speculator(architecture)
    spec.num_query_per_req = 3
    query_metadata = SimpleNamespace(actual_seq_lengths_q=[3] * local_reqs)
    metadata = {"draft": SimpleNamespace(decode=query_metadata) if architecture == "MLA" else query_metadata}
    builder = MagicMock(return_value=metadata)
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", builder)

    result = spec._build_draft_attn_metadata(
        num_reqs=local_reqs, num_reqs_padded=local_reqs, num_tokens_padded=padded_reqs * 3, step=3
    )

    builder.assert_called_once_with(
        num_reqs=local_reqs, num_reqs_padded=padded_reqs, num_tokens_padded=padded_reqs * 3, step=3
    )
    assert result is metadata
    assert query_metadata.actual_seq_lengths_q == [3 * (i + 1) for i in range(padded_reqs)]


@pytest.mark.parametrize("architecture", ["GQA", "MLA"])
def test_dp_padded_queries_require_padded_request_boundaries(monkeypatch, architecture):
    spec = make_speculator(architecture)
    query = torch.zeros(10, 1, 8)
    original_boundaries = []

    def build(*, num_reqs, num_reqs_padded, num_tokens_padded, **kwargs):
        assert num_reqs == 1
        assert num_reqs_padded == 2
        assert num_tokens_padded == query.shape[0]
        # Upstream clamps padded request boundaries to the real query count.
        starts = torch.arange(num_reqs_padded + 1).clamp(max=num_reqs) * spec.num_query_per_req
        boundaries = starts[1:].tolist()
        original_boundaries.extend(boundaries)
        metadata = SimpleNamespace(actual_seq_lengths_q=boundaries)
        return {"draft": SimpleNamespace(decode=metadata) if architecture == "MLA" else metadata}

    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", staticmethod(build))
    result = spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=query.shape[0], step=5)
    metadata = result["draft"].decode if architecture == "MLA" else result["draft"]
    assert original_boundaries == [5, 5]
    assert original_boundaries[-1] != query.shape[0]
    assert metadata.actual_seq_lengths_q == [5, 10]
    assert metadata.actual_seq_lengths_q[-1] == query.shape[0]


@pytest.mark.parametrize("architecture", ["GQA", "MLA"])
def test_draft_query_tokens_must_be_divisible_by_query_width(monkeypatch, architecture):
    spec = make_speculator(architecture)
    builder = MagicMock()
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", builder)
    with pytest.raises(AssertionError, match="whole query groups"):
        spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=2, num_tokens_padded=9, step=5)
    builder.assert_not_called()
