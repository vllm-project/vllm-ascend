# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
    AscendMlaDCPMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata, AscendMLAMetadataBuilder
from vllm_ascend.worker.v2 import attn_utils, dcp
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize("speculative", [False, True])
def test_mla_dcp_passes_upstream_runner_capability_check(monkeypatch, speculative):
    from vllm.distributed import parallel_state
    from vllm.v1.worker import cp_utils

    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0))
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    monkeypatch.setattr(cp_utils, "get_layers_from_vllm_config", lambda *_: {"mla": SimpleNamespace(impl=impl)})
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=2, cp_kv_cache_interleave_size=128
        ),
        speculative_config=SimpleNamespace() if speculative else None,
    )
    cp_utils.check_attention_cp_compatibility(config)


def _builder(monkeypatch, size=2, interleave=4):
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.dcp_size = size
    builder.dcp_rank = 0
    builder.cp_local_block_size = interleave
    builder.decode_threshold = 4
    builder.vllm_config = SimpleNamespace()
    monkeypatch.setattr(dcp, "is_pd_decode_recompute_scheduler_enabled", lambda *_: False)
    # Exercise V2's actual common-metadata construction and DCP adapter;
    # attention kernels are tested independently against dense attention.
    builder.build = lambda **kwargs: kwargs["common_attn_metadata"]
    builder.build_for_cudagraph_capture = lambda common, **_: common
    return builder


def _kwargs(builder, queries, lengths, prefilling):
    starts = torch.tensor([0, *np.cumsum(queries)], dtype=torch.int32)
    num_tokens = sum(queries)
    num_reqs = len(queries)
    return dict(
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda _: builder, layer_names=["mla"])]],
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=starts,
        query_start_loc_cpu=starts,
        max_query_len=max(queries),
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        max_seq_len=4096,
        block_tables=[torch.zeros(num_reqs, 8, dtype=torch.int32)],
        slot_mappings=torch.zeros(1, num_tokens, dtype=torch.int32),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
        seq_lens_np=np.array(lengths, dtype=np.int32),
        is_prefilling=torch.tensor(prefilling),
    )


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 128])
def test_mixed_batch_uses_full_decode_and_prefill_history(monkeypatch, size, interleave):
    builder = _builder(monkeypatch, size, interleave)
    common = attn_utils.build_attn_metadata(
        **_kwargs(builder, [1, 3, 2, 9], [129, 131, 130, 265], [False, False, True, True])
    )["mla"]
    # Enumerate ownership independently of the production vector formula.
    expected = [
        [sum((pos // interleave) % size == rank for pos in range(length)) for rank in range(size)]
        for length in [129, 131, 128, 256]
    ]
    assert common.context_parallel_metadata.num_computed_tokens_of_dcp.tolist() == expected
    assert common.num_computed_tokens_cpu.tolist() == [128, 128, 128, 256]
    assert common.context_parallel_metadata.query_lens_cpu.tolist() == [1, 3, 2, 9]


def test_draft_uses_actual_lengths_instead_of_upper_bound(monkeypatch):
    builder = _builder(monkeypatch)
    kwargs = _kwargs(builder, [1, 1, 1], [7, 9, 0], [False] * 3)
    kwargs.pop("seq_lens_np")
    kwargs["seq_lens_cpu_upper_bound"] = torch.tensor([10, 12, 0])
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    common = attn_utils.build_attn_metadata(**kwargs)["mla"]
    assert common.seq_lens_cpu.tolist() == [7, 9, 0]
    assert common.context_parallel_metadata.num_computed_tokens_of_dcp.tolist() == [[4, 3], [5, 4], [0, 0]]


def test_capture_uses_synthetic_host_lengths(monkeypatch):
    builder = _builder(monkeypatch)
    kwargs = _kwargs(builder, [1, 1], [9, 10], [False, False])
    kwargs.pop("seq_lens_np")
    kwargs["seq_lens"] = torch.empty(2, device="meta", dtype=torch.int32)
    kwargs["seq_lens_cpu_upper_bound"] = torch.tensor([5, 5], dtype=torch.int32)
    kwargs["for_cudagraph_capture"] = True
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    common = attn_utils.build_attn_metadata(**kwargs)["mla"]
    assert common.context_parallel_metadata.num_computed_tokens_of_dcp.tolist() == [[4, 1], [4, 1]]


def test_dspark_capture_without_cpu_upper_bound(monkeypatch):
    builder = _builder(monkeypatch)
    kwargs = _kwargs(builder, [3, 3], [3, 3], [False, False])
    kwargs.pop("seq_lens_np")
    kwargs.pop("is_prefilling")
    kwargs["seq_lens"] = torch.empty(2, device="meta", dtype=torch.int32)
    kwargs["for_cudagraph_capture"] = True
    common = attn_utils.build_attn_metadata(**kwargs)["mla"]
    assert common.seq_lens_cpu.tolist() == [3, 3]
    assert common.is_prefilling.tolist() == [False, False]
    assert common.context_parallel_metadata.num_computed_tokens_of_dcp.tolist() == [[3, 0], [3, 0]]


def test_padded_dspark_queries_are_decodes_even_after_target_prefill(monkeypatch):
    builder = _builder(monkeypatch)
    kwargs = _kwargs(builder, [3, 0], [11, 0], [True, False])
    kwargs["num_tokens"] = 6
    kwargs["slot_mappings"] = torch.zeros(1, 6, dtype=torch.int32)
    with (
        attn_utils.build_attn_metadata_wrapper(),
        attn_utils.build_draft_attn_metadata_factory(torch.arange(6), 6, torch.tensor([True]), uniform_mla_query=True),
    ):
        common = attn_utils._BUILD_ATTN_METADATA_MODULE.build_attn_metadata(**kwargs)["mla"]
    assert common.query_start_loc_cpu.tolist() == [0, 3, 6]
    assert common.is_prefilling.tolist() == [False, False]
    assert common.context_parallel_metadata.num_computed_tokens_of_dcp.tolist() == [[7, 4], [0, 0]]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave", [1, 128])
@pytest.mark.parametrize("advance", [False, True])
def test_full_graph_steps_start_from_rewound_gpu_lengths(rank, interleave, advance):
    base = AscendMLAMetadata.__new__(AscendMLAMetadata)
    base.decode = AscendMLADCPDecodeMetadata.__new__(AscendMLADCPDecodeMetadata)
    base.decode.block_table = torch.zeros(3, 2, dtype=torch.int32)
    base.decode.actual_seq_lengths_q = [1, 2, 3]
    base.query_lens = [1, 1, 1]
    base.seq_lens = torch.tensor([129, 255, 0])
    seen = []

    def build(**kwargs):
        seen.append(kwargs["seq_lens_cpu"].clone())
        return {"draft": base, "draft2": base}

    spec = SimpleNamespace(
        model_state=SimpleNamespace(attn_metadata={"draft": base}),
        draft_attn_layer_names={"draft"},
        attn_architecture="MLA",
        block_tables=SimpleNamespace(cp_size=2, cp_rank=rank, cp_interleave=interleave),
        input_batch=SimpleNamespace(num_reqs=2),
        input_buffers=SimpleNamespace(seq_lens=torch.tensor([129, 255, 999])),
        num_speculative_steps=4,
        max_model_len=256,
        advance_draft_positions=advance,
        _build_draft_attn_metadata=build,
    )
    result = AscendAutoRegressiveSpeculator.build_draft_attn_metadatas(spec, 3, 3, False)
    assert len(seen) == 1
    assert seen[0].tolist() == [129, 255, 0]
    owned = lambda n: sum((pos // interleave) % 2 == rank for pos in range(n))
    for step, groups in enumerate(result):
        metadata = groups["draft"]
        lengths = [129 + step if advance else 129, min(255 + step, 256) if advance else 255, 0]
        assert metadata is groups["draft2"]
        assert metadata is not base
        assert metadata.decode is not base.decode
        assert metadata.seq_lens_cpu.tolist() == lengths
        assert metadata.decode.cp_seq_len == [owned(n) for n in lengths]
        assert metadata.decode.cp_history_seq_len == [owned(max(n - 1, 0)) for n in lengths]
        assert metadata.decode.block_table is base.decode.block_table
        assert metadata.decode.actual_seq_lengths_q is base.decode.actual_seq_lengths_q
        assert metadata.seq_lens is base.seq_lens
    assert result[0]["draft"].decode is not result[1]["draft"].decode
    assert result[0]["draft"].seq_lens_cpu.data_ptr() != result[1]["draft"].seq_lens_cpu.data_ptr()


def test_dspark_updates_nested_mla_query_metadata():
    metadata = AscendMLAMetadata.__new__(AscendMLAMetadata)
    metadata.decode = AscendMLADCPDecodeMetadata.__new__(AscendMLADCPDecodeMetadata)
    metadata.decode.actual_seq_lengths_q = [3, 3]
    metadata.query_lens = [3, 0]
    spec = SimpleNamespace(num_query_per_req=3)
    AscendDSparkSpeculator._update_draft_attn_metadata(spec, {"mla": metadata}, 2)
    assert metadata.decode.actual_seq_lengths_q == [3, 6]
    assert metadata.query_lens == [3, 3]
    assert not hasattr(metadata, "actual_seq_lengths_q")


def test_smaller_target_batch_clears_all_padded_cpu_lengths():
    runner = SimpleNamespace(
        speculator=None,
        req_states=SimpleNamespace(req_id_to_index={"req": 2}, num_computed_tokens_cpu=torch.tensor([1, 2, 128])),
        input_buffers=SimpleNamespace(seq_lens_cpu=torch.tensor([300, 400, 500, 600])),
    )
    output = SimpleNamespace(num_scheduled_tokens={"req": 1})
    NPUModelRunner._update_seq_lens_cpu(runner, output, ["req"])
    assert runner.input_buffers.seq_lens_cpu.tolist() == [129, 0, 0, 0]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave", [1, 128])
def test_decode_builder_excludes_current_queries_from_history(monkeypatch, rank, interleave):
    builder = _builder(monkeypatch, interleave=interleave)
    builder.dcp_rank = rank
    builder.num_decodes = 3
    builder.query_lens = torch.tensor([1, 3, 1], dtype=torch.int32)
    common = attn_utils.build_attn_metadata(**_kwargs(builder, [1, 3, 1], [129, 131, 0], [False] * 3))["mla"]
    decode = AscendMLADCPDecodeMetadata.__new__(AscendMLADCPDecodeMetadata)
    decode.actual_seq_lengths_q = [1, 4, 5]
    monkeypatch.setattr(AscendMLAMetadataBuilder, "build_decode_metadata", lambda *_: decode)
    result = builder.build_decode_metadata(0, common)
    owned = lambda n: sum((pos // interleave) % 2 == rank for pos in range(n))
    assert result.cp_seq_len == [owned(129), owned(131), 0]
    assert result.cp_history_seq_len == [owned(128), owned(128), 0]
    assert result.actual_seq_lengths_q == [1, 4, 5]


@pytest.mark.parametrize("query_width", [1, 3, 8])
def test_dspark_dcp_graph_update_uses_request_lengths(monkeypatch, query_width):
    from vllm_ascend.attention.context_parallel import mla_cp

    num_reqs = 3
    num_tokens = num_reqs * query_width
    q = torch.empty(num_reqs, query_width, 2, 4)
    decode = SimpleNamespace(cp_seq_len=[65, 128], actual_seq_lengths_q=[query_width * i for i in range(1, 4)])
    event = Mock()
    params = SimpleNamespace(
        attn_params={num_tokens: [(q, q, q, q, 2, 1, "BSND", None, 0, 1.0, None, 128, [1, 1, 1], [1, 1, 1], q, q)]},
        handles={num_tokens: [object()]},
        events={num_tokens: [event]},
        workspaces={num_tokens: None},
    )
    monkeypatch.setattr(mla_cp, "get_draft_graph_params", lambda: params)
    monkeypatch.setattr(mla_cp, "_EXTRA_CTX", SimpleNamespace(is_draft_model=True, is_draft_model_prefill=False))
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.npu, "graph_task_update_begin", Mock())
    monkeypatch.setattr(torch.npu, "graph_task_update_end", Mock())
    op = Mock()
    monkeypatch.setattr(mla_cp.torch_npu, "npu_fused_infer_attention_score", SimpleNamespace(out=op))
    stream = object()
    AscendMlaDCPImpl.update_graph_params(
        stream,
        SimpleNamespace(),
        num_tokens,
        draft_attn_metadatas=[{"draft": SimpleNamespace(decode=decode)}],
    )
    assert op.call_args.kwargs["actual_seq_lengths_kv"] == [65, 128, 0]
    assert op.call_args.kwargs["actual_seq_lengths"] == [query_width] * num_reqs
    assert op.call_args.kwargs["sparse_mode"] == 0
    assert op.call_args.kwargs["atten_mask"] is None
    event.record.assert_called_once_with(stream)
