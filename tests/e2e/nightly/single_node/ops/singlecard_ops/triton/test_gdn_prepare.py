# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from copy import deepcopy
from dataclasses import is_dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode

from tests.ut.ops.test_gdn_attn_builder import BatchSpec, _make_builder, create_common_attn_metadata
from vllm_ascend.ops import gdn_attn_builder as builder_module
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder
from vllm_ascend.ops.triton.mamba.gdn_prepare import _prepare_gdn_kernel


def _assert_metadata_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif is_dataclass(expected):
        assert vars(actual).keys() == vars(expected).keys()
        for name, value in vars(expected).items():
            _assert_metadata_equal(getattr(actual, name), value)
    else:
        assert actual == expected


@pytest.mark.parametrize("live_prefix", [False, True])
def test_grouped_q6_reuses_graph_buffers_and_matches_individual_launches(live_prefix):
    device = torch.device("npu:0")
    builders = [
        _make_builder(
            device=device,
            num_heads=32,
            num_speculative_tokens=5,
            builder_cls=AscendGDNHostMetadataBuilder,
            mamba_cache_mode="align",
            num_speculative_blocks=5,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        for _ in range(3)
    ]
    for builder in builders:
        builder.vllm_config.model_config.dtype = torch.bfloat16
        builder.compilation_config.static_forward_context = {
            "layer0": SimpleNamespace(
                use_contiguous_conv_indices=True, use_shared_conv_output=True, local_projection_size=11
            )
        }
    common = create_common_attn_metadata(BatchSpec([15, 32], [6, 6]), 16, device)
    common.is_prefilling = torch.zeros(2, dtype=torch.bool)
    common_by_group = [
        replace(common, block_table_tensor=torch.arange(32, dtype=torch.int32, device=device).view(2, 16) + index)
        for index in (1, 33, 65)
    ]
    accepted = torch.tensor([2, 5], dtype=torch.int32, device=device)
    drafts = torch.full((2,), 5, dtype=torch.int32)
    originals = [
        builder.build(0, group_common, accepted, drafts, num_actual_reqs=2)
        for builder, group_common in zip(builders, common_by_group, strict=True)
    ]
    expected = deepcopy(originals)
    if live_prefix:
        for metadata in expected:
            output = metadata.spec_decode_metadata.spec_causal_conv1d.output_buffer
            assert output.shape[0] > common.num_actual_tokens
            output[common.num_actual_tokens :].fill_(1)
    for builder in builders:
        builder.spec_state_indices_tensor.fill_(-1)
        builder._spec_conv_indices.fill_(-1)
        builder._conv_output_buffer.fill_(1)
    grouped = builders[0].begin_grouped_spec_gdn()
    with patch.object(_prepare_gdn_kernel, "run", wraps=_prepare_gdn_kernel.run) as run:
        actual = [
            builder.build_grouped_spec_gdn(group_common, accepted, grouped)
            for builder, group_common in zip(builders, common_by_group, strict=True)
        ]
        run.assert_not_called()
        grouped.launch(clear_conv_tokens=common.num_actual_tokens if live_prefix else None)
        run.assert_called_once()
    for before, after, oracle in zip(originals, actual, expected, strict=True):
        _assert_metadata_equal(after, oracle)
        assert after.spec_state_indices_tensor.data_ptr() == before.spec_state_indices_tensor.data_ptr()
        assert after.spec_query_start_loc.data_ptr() == before.spec_query_start_loc.data_ptr()
        assert after.num_accepted_tokens.data_ptr() == before.num_accepted_tokens.data_ptr()
        before_conv = before.spec_decode_metadata.spec_causal_conv1d
        after_conv = after.spec_decode_metadata.spec_causal_conv1d
        assert after_conv.cache_indices.data_ptr() == before_conv.cache_indices.data_ptr()
        assert after_conv.output_buffer.data_ptr() == before_conv.output_buffer.data_ptr()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_adaptive_device_boundaries_match_generic_metadata(mixed, graph_mode):
    device = torch.device("npu:0")
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=3,
        enable_adaptive_verification=True,
        builder_cls=AscendGDNHostMetadataBuilder,
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        cudagraph_mode=graph_mode,
    )
    builder.compilation_config.static_forward_context = {"layer0": SimpleNamespace(use_contiguous_conv_indices=True)}
    host_queries = [2, 2, 9] if mixed else [2, 2]
    common = create_common_attn_metadata(BatchSpec([64, 64, 9] if mixed else [64, 64], host_queries), 16, device)
    common.is_prefilling = torch.tensor([False, False, True] if mixed else [False, False])
    common.query_start_loc = torch.tensor([0, 1, 4, 13] if mixed else [0, 1, 4], dtype=torch.int32, device=device)
    common.block_table_tensor = torch.arange(len(host_queries) * 16, dtype=torch.int32, device=device).view(-1, 16) + 1
    accepted = torch.tensor([4, 3, 1] if mixed else [4, 3], dtype=torch.int32, device=device)
    drafts = torch.tensor([3, 3, -1] if mixed else [3, 3], dtype=torch.int32)
    with patch.object(builder, "_build_fused_metadata", return_value=None):
        expected = deepcopy(builder.build(0, common, accepted, drafts, num_actual_reqs=len(host_queries)))
    with patch.object(builder_module, "prepare_gdn", wraps=builder_module.prepare_gdn) as prepare:
        actual = builder.build(0, common, accepted, drafts, num_actual_reqs=len(host_queries))
    _assert_metadata_equal(actual, expected)
    if mixed:
        prepare.assert_not_called()
    else:
        prepare.assert_called_once()
    assert actual.spec_query_start_loc[:3].cpu().tolist() == [0, 1, 4]
    assert actual.num_accepted_tokens[:2].cpu().tolist() == [4, 3]
    assert actual.spec_decode_metadata.actual_seq_lengths is None


@pytest.mark.parametrize(
    "query_len,num_spec,projection_size",
    [
        (1, 0, 0),
        (1, 0, 11),
        (1, 3, 11),
        (2, 1, 11),
        (4, 3, 11),
        (5, 4, 11),
        ([1, 2, 4, 3], 3, 11),  # Different draft lengths, including zero drafts.
        ([1, 1, 17, 257], 0, 11),  # Ordinary decode + fresh/chunked prefill.
        ([4, 257, 2, 1], 3, 11),  # Speculative decode + prefill + ordinary decode.
    ],
)
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_fused_decode_matches_generic_metadata(query_len, num_spec, projection_size, graph_mode):
    device = torch.device("npu:0")
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_spec,
        builder_cls=AscendGDNHostMetadataBuilder,
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
        cudagraph_mode=graph_mode,
    )
    builder.vllm_config.model_config.dtype = torch.bfloat16
    builder.compilation_config.static_forward_context = {
        "layer0": SimpleNamespace(
            use_contiguous_conv_indices=True,
            use_shared_conv_output=projection_size > 0,
            local_projection_size=projection_size,
        )
    }
    if graph_mode == CUDAGraphMode.NONE:
        # Eager batches can exceed the buffers reserved for small graph sizes.
        for name in ("spec_state_indices_tensor", "non_spec_state_indices_tensor", "num_accepted_tokens"):
            setattr(builder, name, getattr(builder, name)[:2])
        for name in ("spec_query_start_loc", "non_spec_query_start_loc"):
            setattr(builder, name, getattr(builder, name)[:3])
        builder.spec_token_indx = builder.spec_token_indx[: 2 * (num_spec + 1)]
    queries = [query_len] * 4 if isinstance(query_len, int) else query_len
    draft_counts = [q - 1 if num_spec and q <= num_spec + 1 else -1 for q in queries]
    if isinstance(query_len, int) and query_len == 1:
        draft_counts = [-1] * 4
    if queries == [4, 257, 2, 1]:
        draft_counts[-1] = -1
    pure_decode = all(q <= num_spec + 1 for q in queries)
    # Exercise cache-block boundaries, strided inputs and shrinking batches.
    for count in (4, 1, 3):
        rows = 4 if graph_mode == CUDAGraphMode.FULL_DECODE_ONLY and pure_decode else count
        common = create_common_attn_metadata(
            BatchSpec([max(q, n) for q, n in zip(queries[:rows], [15, 16, 17, 32])], queries[:rows]), 16, device
        )
        context = SimpleNamespace() if count == 1 else None
        common.pcp_context = context
        common.is_prefilling = torch.tensor([q > 1 and d < 0 for q, d in zip(queries[:rows], draft_counts[:rows])])
        common.num_actual_tokens = sum(queries[:count])
        common.block_table_tensor = torch.arange(rows * 64, dtype=torch.int32, device=device).view(rows, 64)[:, ::2]
        common.seq_lens = torch.stack((common.seq_lens, common.seq_lens), dim=1)[:, 0]
        accepted = torch.tensor([min(q, 2) for q in queries for _ in range(2)], dtype=torch.int32, device=device)[::2][
            :rows
        ]
        drafts = torch.tensor(draft_counts[:rows], dtype=torch.int32) if num_spec else None
        if not any(d > 0 for d in draft_counts[:count]):
            accepted = None
        # The generic implementation is the oracle for every metadata field.
        with patch.object(builder, "_build_fused_metadata", return_value=None):
            generic = builder.build(0, common, accepted, drafts, num_actual_reqs=count)
        expected = deepcopy(generic)
        for name in (
            "spec_state_indices_tensor",
            "non_spec_state_indices_tensor",
            "spec_sequence_masks",
            "spec_query_start_loc",
            "non_spec_query_start_loc",
            "spec_actual_seq_lengths",
            "non_spec_actual_seq_lengths",
            "num_accepted_tokens",
            "spec_token_indx",
            "_spec_conv_indices",
        ):
            getattr(builder, name).fill_(12345)
        if builder._conv_output_buffer is not None:
            builder._conv_output_buffer.fill_(float("nan"))
        with patch.object(builder, "_build_fused_metadata", wraps=builder._build_fused_metadata) as fused:
            actual = builder.build(0, common, accepted, drafts, num_actual_reqs=count)
        fused.assert_called_once()
        _assert_metadata_equal(actual, expected)
        assert actual.pcp_context is context
        for phase in (actual.spec_decode_metadata, actual.non_spec_decode_metadata):
            if phase is not None:
                conv = phase.spec_causal_conv1d if phase is actual.spec_decode_metadata else phase.causal_conv1d
                assert conv.output_buffer is builder._conv_output_buffer
        if graph_mode == CUDAGraphMode.FULL_DECODE_ONLY and actual.num_prefills == 0:
            for name, value in vars(generic).items():
                if isinstance(value, torch.Tensor):
                    assert getattr(actual, name).data_ptr() == value.data_ptr(), name
        if query_len == 1 and num_spec and graph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
            assert not torch.any(builder.spec_actual_seq_lengths[: rows + 1])
            assert not torch.any(builder.num_accepted_tokens[:rows])


@pytest.mark.parametrize("mode", ["decode", "spec", "prefill", "mixed"])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_fused_metadata_reuses_kernel_across_batch_counts(mode, graph_mode):
    device = torch.device("npu:0")
    num_spec = 3 if mode in ("spec", "mixed") else 0
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_spec,
        builder_cls=AscendGDNHostMetadataBuilder,
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
        cudagraph_mode=graph_mode,
    )
    builder.vllm_config.model_config.dtype = torch.bfloat16
    builder.compilation_config.static_forward_context = {
        "layer0": SimpleNamespace(
            use_contiguous_conv_indices=True,
            use_shared_conv_output=True,
            local_projection_size=11,
        )
    }
    compiled_kernels = []
    run = _prepare_gdn_kernel.run

    def record_kernel(*args, **kwargs):
        compiled = run(*args, **kwargs)
        compiled_kernels.append(compiled)
        return compiled

    # Include 1 and aligned counts to catch implicit integer specialization.
    counts = (2, 3, 8, 16, 4) if mode == "mixed" else (1, 2, 3, 8, 16, 4)
    for count in counts:
        rows = 16 if graph_mode == CUDAGraphMode.FULL_DECODE_ONLY and mode in ("decode", "spec") else count
        if mode == "decode":
            queries = [1] * rows
        elif mode == "spec":
            queries = [4] * rows
        else:
            # Vary both decode counts and the prefill boundary in the same phase.
            queries = [4 if num_spec else 1] * (count // 2) + [17] * (count - count // 2)
        common = create_common_attn_metadata(BatchSpec([max(q, 32) for q in queries], queries), 16, device)
        common.num_actual_tokens = sum(queries[:count])
        # Keep physical strides fixed so only batch counts and grid sizes vary.
        common.block_table_tensor = torch.arange(rows * 32, dtype=torch.int32, device=device).view(rows, 32)
        drafts = torch.tensor([3 if q == 4 else -1 for q in queries], dtype=torch.int32) if num_spec else None
        accepted = torch.ones(rows, dtype=torch.int32, device=device) if num_spec else None
        with patch.object(builder, "_build_fused_metadata", return_value=None):
            expected = deepcopy(builder.build(0, common, accepted, drafts, num_actual_reqs=count))
        with patch.object(_prepare_gdn_kernel, "run", side_effect=record_kernel):
            actual = builder.build(0, common, accepted, drafts, num_actual_reqs=count)
        _assert_metadata_equal(actual, expected)

    assert len(compiled_kernels) == len(counts)
    assert compiled_kernels[0] is not None
    assert all(kernel is compiled_kernels[0] for kernel in compiled_kernels)


def test_grouped_q6_cached_views_follow_current_block_tables_and_accepted_counts():
    device = torch.device("npu:0")
    builders = [
        _make_builder(
            device=device,
            num_heads=32,
            num_speculative_tokens=5,
            builder_cls=AscendGDNHostMetadataBuilder,
            mamba_cache_mode="align",
            num_speculative_blocks=5,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        for _ in range(2)
    ]
    for builder in builders:
        builder.vllm_config.model_config.dtype = torch.bfloat16
        builder.compilation_config.static_forward_context = {
            "layer0": SimpleNamespace(
                use_contiguous_conv_indices=True, use_shared_conv_output=True, local_projection_size=11
            )
        }

    def step(seq_lens, table_start, accepted_counts):
        num_reqs = len(seq_lens)
        common = create_common_attn_metadata(BatchSpec(seq_lens, [6] * num_reqs), 16, device)
        common.is_prefilling.zero_()
        table = torch.arange(num_reqs * 16, dtype=torch.int32, device=device).view(num_reqs, 16) + table_start
        per_group = [replace(common, block_table_tensor=table + 100 * index) for index in range(len(builders))]
        accepted = torch.tensor(accepted_counts, dtype=torch.int32, device=device)
        drafts = torch.full((num_reqs,), 5, dtype=torch.int32)
        expected = [
            deepcopy(builder.build(0, metadata, accepted, drafts, num_actual_reqs=num_reqs))
            for builder, metadata in zip(builders, per_group, strict=True)
        ]
        for builder in builders:
            builder.spec_state_indices_tensor.fill_(-1)
            builder.num_accepted_tokens.fill_(-1)
        grouped = builders[0].begin_grouped_spec_gdn()
        actual = [
            builder.build_grouped_spec_gdn(metadata, accepted, grouped)
            for builder, metadata in zip(builders, per_group, strict=True)
        ]
        grouped.launch()
        for metadata, oracle in zip(actual, expected, strict=True):
            _assert_metadata_equal(metadata, oracle)
        return actual

    first = step([15, 32], 1, [2, 5])
    second = step([17, 34], 61, [1, 3])
    assert all(before is after for before, after in zip(first, second, strict=True))
    assert all(metadata.num_accepted_tokens.cpu().tolist() == [1, 3] for metadata in second)
    narrower = step([34], 19, [4])
    assert all(before is not after for before, after in zip(second, narrower, strict=True))
