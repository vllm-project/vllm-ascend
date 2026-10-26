# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_metadata import KDARecoverSSMCommitMetadata, KimiK3KDAMetadataBuilder

from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder


def test_graph_padding_is_excluded_from_sampler_commit_rows(monkeypatch):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.use_full_cuda_graph = True
    starts = torch.tensor([0, 6, 12, 18, 24], dtype=torch.int32)
    drafts = torch.full((4,), 5, dtype=torch.int32)
    m = SimpleNamespace(num_reqs=4, num_actual_tokens=24, query_start_loc_cpu=starts, query_start_loc=starts)
    m.replace = lambda **updates: SimpleNamespace(**(vars(m) | updates))

    def parent_build(self, prefix, common, accepted, draft_counts, fast_build):
        assert self.use_full_cuda_graph is False
        assert common.query_start_loc_cpu.tolist() == common.query_start_loc.tolist() == [0, 6, 12, 12, 12]
        assert common.num_actual_tokens == 12
        assert draft_counts.tolist() == [5, 5, -1, -1]
        return SimpleNamespace(num_spec_decodes=0, num_prefills=0, num_decodes=0)

    monkeypatch.setattr(KimiK3KDAMetadataBuilder, "build", parent_build)
    builder.build(0, m, num_decode_draft_tokens_cpu=drafts, num_actual_reqs=2)
    assert builder.use_full_cuda_graph is True
    assert drafts.tolist() == [5, 5, 5, 5]
    assert starts.tolist() == [0, 6, 12, 18, 24]


def test_graph_staging_reuses_addresses_and_clears_prior_rows():
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.use_full_cuda_graph = True
    builder.spec_state_indices_tensor = torch.full((4, 1), 99, dtype=torch.int32)
    builder.spec_query_start_loc = torch.full((5,), 99, dtype=torch.int32)
    builder.num_accepted_tokens = torch.full((4,), 99, dtype=torch.int32)
    addresses = [
        t.data_ptr()
        for t in (builder.spec_state_indices_tensor, builder.spec_query_start_loc, builder.num_accepted_tokens)
    ]
    for blocks, lengths in [([3, 1], [6, 2]), ([2], [1])]:
        starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        indices = torch.tensor(blocks, dtype=torch.int32).reshape(-1, 1)
        metadata = SimpleNamespace(
            num_spec_decodes=len(blocks),
            num_prefills=0,
            num_decodes=0,
            spec_state_indices_tensor=indices,
            spec_query_start_loc=starts,
            recoverssm_commit=KDARecoverSSMCommitMetadata(indices, starts, None, None),
        )
        builder._stage_spec(metadata, 4)
        assert metadata.spec_state_indices_tensor[:, 0].tolist() == blocks + [0] * (4 - len(blocks))
        assert metadata.spec_query_start_loc.tolist() == starts.tolist() + [starts[-1].item()] * (4 - len(blocks))
        assert metadata.num_accepted_tokens.tolist() == [1] * 4
        assert metadata.recoverssm_commit.state_indices is metadata.spec_state_indices_tensor
        assert metadata.recoverssm_commit.query_start_loc is metadata.spec_query_start_loc
        assert addresses == [
            t.data_ptr()
            for t in (metadata.spec_state_indices_tensor, metadata.spec_query_start_loc, metadata.num_accepted_tokens)
        ]


@pytest.mark.parametrize("actual_reqs", [0, 1, 2, 3])
def test_padding_offsets_stay_on_device_without_host_copy(monkeypatch, actual_reqs):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.use_full_cuda_graph = True
    cpu_starts = torch.tensor([0, 1, 5, 11, 17], dtype=torch.int32)
    device_starts = cpu_starts.clone()
    original = device_starts.clone()
    m = SimpleNamespace(num_reqs=4, num_actual_tokens=17, query_start_loc_cpu=cpu_starts, query_start_loc=device_starts)
    m.replace = lambda **updates: SimpleNamespace(**(vars(m) | updates))
    expected = original.clone()
    expected[actual_reqs + 1 :] = original[actual_reqs]

    def forbidden_host_copy(*args, **kwargs):
        raise AssertionError("Padding offsets must not use Tensor.to")

    def parent_build(self, prefix, common, accepted, draft_counts, fast_build):
        torch.testing.assert_close(common.query_start_loc, expected, rtol=0, atol=0)
        torch.testing.assert_close(common.query_start_loc_cpu, expected, rtol=0, atol=0)
        assert common.num_actual_tokens == int(original[actual_reqs])
        return SimpleNamespace(num_spec_decodes=0, num_prefills=0, num_decodes=0)

    monkeypatch.setattr(KimiK3KDAMetadataBuilder, "build", parent_build)
    monkeypatch.setattr(torch.Tensor, "to", forbidden_host_copy)
    builder.build(0, m, num_actual_reqs=actual_reqs)
    torch.testing.assert_close(device_starts, original, rtol=0, atol=0)
    torch.testing.assert_close(cpu_starts, original, rtol=0, atol=0)


def test_mixed_prefill_cpu_offsets_match_upstream_decode_partition():
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.num_spec = 5
    m = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 4, 10, 11, 11, 18], dtype=torch.int32),
        is_prefilling=torch.tensor([True, False, False, False, False]),
    )
    # Zero-draft decode rows still use RecoverSSM; over-capacity profiler rows
    # remain on the non-spec path. Zero-length padding is omitted.
    starts = builder._non_spec_cpu_starts(
        m, SimpleNamespace(num_spec_decodes=2), torch.tensor([-1, 5, -1, -1, -1], dtype=torch.int32)
    )
    assert starts.tolist() == [0, 4, 11]
