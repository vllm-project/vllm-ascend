# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_metadata import KDARecoverSSMCommitMetadata, KimiK3KDAMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext


@pytest.mark.parametrize("graph_mode", [False, True])
def test_align_indices_follow_sequence_position_and_refresh_each_step(monkeypatch, graph_mode):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    builder.use_full_cuda_graph = graph_mode
    builder.kv_cache_spec = MambaSpec(
        block_size=8,
        shapes=((3, 4), (2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
        num_speculative_blocks=1,
    )
    table = torch.tensor([[9, 7, 5, 3], [2, 4, 6, 8], [11, 12, 13, 14], [0, 0, 0, 0]], dtype=torch.int32)
    original = table.clone()
    starts = torch.tensor([0, 6, 12, 18, 24], dtype=torch.int32)
    expected = None

    def parent_build(self, prefix, common, accepted, draft_counts, fast_build):
        assert not self.use_full_cuda_graph
        torch.testing.assert_close(self.mamba_aligned_state_indices, expected, atol=0, rtol=0)
        return SimpleNamespace(num_spec_decodes=0, num_prefills=0, num_decodes=0)

    monkeypatch.setattr(KimiK3KDAMetadataBuilder, "build", parent_build)
    for lengths, blocks in [
        ([8, 9, 17, 0, 999], [[9, 7], [4, 6], [13, 14], [0, 0]]),
        ([9, 17, 8, 0, 999], [[7, 5], [6, 8], [11, 12], [0, 0]]),
    ]:
        expected = torch.tensor(blocks, dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=4,
            num_actual_tokens=24,
            query_start_loc=starts,
            query_start_loc_cpu=starts,
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            block_table_tensor=table,
        )
        common.replace = lambda common=common, **updates: SimpleNamespace(**(vars(common) | updates))
        builder.build(0, common, num_actual_reqs=3 if graph_mode else None)
        assert builder.use_full_cuda_graph is graph_mode
        torch.testing.assert_close(table, original, atol=0, rtol=0)


def test_upstream_context_creation_uses_ascend_patch_and_caches(monkeypatch):
    from vllm.models.kimi_k3.nvidia.ops import recoverssm

    monkeypatch.setattr(recoverssm, "KDARecoverSSMCommitContext", None)
    patch = runpy.run_path(str(Path(__file__).parents[3] / "vllm_ascend/patch/worker/patch_kda_recoverssm.py"))
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.recoverssm_context = None
    builder.layer_names = ["layer0", "layer1"]
    layers = [SimpleNamespace(recoverssm_backend="triton") for _ in builder.layer_names]
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next")),
        cache_config=SimpleNamespace(use_replayssm=True),
        compilation_config=SimpleNamespace(static_forward_context=dict(zip(builder.layer_names, layers))),
        scheduler_config=SimpleNamespace(max_num_seqs=12),
        num_speculative_tokens=5,
    )
    patch["patch_kda_recoverssm"](builder.vllm_config)
    assert recoverssm.KDARecoverSSMCommitContext is KDARecoverSSMCommitContext
    context = object()
    create = Mock(return_value=context)
    monkeypatch.setattr(KDARecoverSSMCommitContext, "create", create)
    assert builder._get_recoverssm_context() is context
    assert builder._get_recoverssm_context() is context
    create.assert_called_once()
    args, kwargs = create.call_args
    assert args == (layers,)
    assert kwargs["spec_query_len"] == 6
    assert kwargs["max_num_reqs"] == 12
    # vLLM main forwards the commit backend; 0.32.0 does not.
    assert kwargs.get("backend", "triton") == "triton"


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text", "kimi_k3", None])
@pytest.mark.parametrize("enabled", [False, True])
def test_context_patch_is_limited_to_enabled_glm(monkeypatch, model_type, enabled):
    from vllm.models.kimi_k3.nvidia.ops import recoverssm

    original = object()
    monkeypatch.setattr(recoverssm, "KDARecoverSSMCommitContext", original)
    patch = runpy.run_path(str(Path(__file__).parents[3] / "vllm_ascend/patch/worker/patch_kda_recoverssm.py"))
    assert recoverssm.KDARecoverSSMCommitContext is original
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)) if model_type else None,
        cache_config=SimpleNamespace(use_replayssm=enabled),
    )
    patch["patch_kda_recoverssm"](config)
    expected = KDARecoverSSMCommitContext if enabled and model_type in ("glm5_next", "glm5_next_text") else original
    assert recoverssm.KDARecoverSSMCommitContext is expected


def test_graph_padding_is_excluded_from_sampler_commit_rows(monkeypatch):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
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
    builder.decode_cudagraph_max_bs = 8
    builder.spec_state_indices_tensor = torch.full((4, 1), 99, dtype=torch.int32)
    builder.spec_query_start_loc = torch.full((builder.decode_cudagraph_max_bs + 1,), 99, dtype=torch.int32)
    builder.num_accepted_tokens = torch.full((builder.decode_cudagraph_max_bs,), 99, dtype=torch.int32)
    addresses = [
        t.data_ptr()
        for t in (builder.spec_state_indices_tensor, builder.spec_query_start_loc, builder.num_accepted_tokens)
    ]
    for blocks, lengths in [([3, 1], [6, 2]), ([2], [1])]:
        starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        indices = torch.tensor(blocks, dtype=torch.int32).reshape(-1, 1)
        metadata = SimpleNamespace(
            num_spec_decodes=len(blocks),
            num_spec_decode_tokens=sum(lengths),
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


@pytest.mark.parametrize(
    "batch,count,state_capacity,capture_capacity",
    [
        (64, 64, 64, 24),
        (32, 3, 64, 24),
        (5, 5, 64, 24),
        (4, 3, 3, 18),
    ],
)
def test_eager_fallback_keeps_metadata_when_graph_staging_does_not_fit(batch, count, state_capacity, capture_capacity):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.use_full_cuda_graph = True
    builder.decode_cudagraph_max_bs = capture_capacity
    builder.spec_state_indices_tensor = torch.full((state_capacity, 1), 99, dtype=torch.int32)
    builder.spec_query_start_loc = torch.full((capture_capacity + 1,), 99, dtype=torch.int32)
    builder.num_accepted_tokens = torch.full((capture_capacity,), 99, dtype=torch.int32)
    states = torch.arange(1, count + 1, dtype=torch.int32).reshape(-1, 1)
    starts = torch.arange(count + 1, dtype=torch.int32) * 6
    accepted = torch.full((count,), 2, dtype=torch.int32)
    commit = KDARecoverSSMCommitMetadata(states, starts, None, None)
    metadata = SimpleNamespace(
        num_spec_decodes=count,
        num_spec_decode_tokens=count * 6,
        num_prefills=0,
        num_decodes=0,
        spec_state_indices_tensor=states,
        spec_query_start_loc=starts,
        num_accepted_tokens=accepted,
        recoverssm_commit=commit,
    )

    builder._build_conv_metadata(SimpleNamespace(num_reqs=batch), metadata, None)

    assert metadata.spec_state_indices_tensor is states
    assert metadata.spec_query_start_loc is starts
    assert metadata.num_accepted_tokens is accepted
    assert metadata.recoverssm_commit is commit
    conv = metadata.spec_decode_metadata.spec_causal_conv1d
    assert conv.query_start_loc is starts and conv.num_accepted_tokens is accepted
    assert torch.equal(conv.cache_indices, states[:, 0])
    for buffer in (builder.spec_state_indices_tensor, builder.spec_query_start_loc, builder.num_accepted_tokens):
        assert bool((buffer == 99).all())


@pytest.mark.parametrize("actual_reqs", [0, 1, 2, 3])
def test_padding_offsets_stay_on_device_without_host_copy(monkeypatch, actual_reqs):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
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


def test_inherited_capture_builds_speculative_arguments(monkeypatch):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.decode_cudagraph_max_bs = 12
    m = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=12,
        query_start_loc=torch.tensor([0, 6, 12], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6, 12], dtype=torch.int32),
    )
    build = Mock(return_value=object())
    monkeypatch.setattr(builder, "build", build)
    assert builder.build_for_cudagraph_capture(m) is build.return_value
    build.assert_called_once()
    prefix, common, accepted, drafts = build.call_args.args
    assert prefix == 0 and common is m
    assert accepted.tolist() == [6, 6]
    assert drafts.tolist() == [5, 5]


@pytest.mark.parametrize(
    "num_reqs,num_tokens,cpu_starts",
    [(13, 12, [0, 6, 12]), (2, 13, [0, 6, 12]), (2, 12, [0, 6])],
)
def test_inherited_capture_preserves_upstream_guards(monkeypatch, num_reqs, num_tokens, cpu_starts):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    builder.decode_cudagraph_max_bs = 12
    build = Mock()
    monkeypatch.setattr(builder, "build", build)
    m = SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, 6, 12], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(cpu_starts, dtype=torch.int32),
    )
    with pytest.raises(AssertionError):
        builder.build_for_cudagraph_capture(m)
    build.assert_not_called()


def test_prefill_chunk_offsets_exclude_ordinary_decode(monkeypatch):
    builder = AscendKDARecoverSSMMetadataBuilder.__new__(AscendKDARecoverSSMMetadataBuilder)
    starts = torch.tensor([0, 1, 5, 11], dtype=torch.int32)
    states = torch.tensor([4, 5, 6], dtype=torch.int32)
    initial = torch.tensor([True, False, True])
    metadata = SimpleNamespace(
        num_spec_decodes=0,
        num_decodes=1,
        num_prefills=2,
        num_decode_tokens=1,
        non_spec_query_start_loc=starts,
        non_spec_state_indices_tensor=states,
        has_initial_state=initial,
    )
    chunk = object()
    prepare_chunk = Mock(return_value=chunk)
    monkeypatch.setattr(
        "vllm_ascend.models.glm5next.recoverssm_metadata._build_non_spec_chunked_prefill_metadata", prepare_chunk
    )
    m = SimpleNamespace(query_start_loc_cpu=starts, query_start_loc=starts)
    builder._build_conv_metadata(m, metadata, None)
    assert metadata.prefill_query_start_loc.tolist() == [0, 4, 10]
    assert metadata.prefill_state_indices.tolist() == [5, 6]
    assert metadata.prefill_has_initial_state.tolist() == [False, True]
    assert metadata.non_spec_prefill_metadata.chunk is chunk
    assert metadata.non_spec_prefill_metadata.causal_conv1d.cache_indices is states
    assert metadata.non_spec_decode_metadata is None and metadata.spec_decode_metadata is None
    args, kwargs = prepare_chunk.call_args
    assert args[0] is builder and args[1].tolist() == [0, 4, 10] and args[2] == starts.device
    assert kwargs == {"build_device_metadata": False}
