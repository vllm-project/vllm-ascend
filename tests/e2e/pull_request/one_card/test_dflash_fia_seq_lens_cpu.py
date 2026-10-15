# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.attention.attention_v1 import AscendMetadata, FIAParamProvider
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, _select_seq_lens
from vllm_ascend.compilation.updatable_graph import UpdatableGraph, register_task
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


@pytest.mark.parametrize(
    "initial,next_lengths",
    [([73], [74]), ([73, 81], [8, 82]), ([1, 128], [8, 129])],
)
def test_exact_dflash_mirror_keeps_fia_output_after_rejection(initial, next_lengths):
    device = torch.device("npu:0")
    num_reqs = len(initial)
    padded = 4
    width = 8
    num_heads, num_kv_heads, head_size, block_size = 4, 2, 64, 128
    torch.manual_seed(0)
    query = torch.randn(num_reqs * width, num_heads, head_size, dtype=torch.float16, device=device)
    key = torch.randn(num_reqs * 2, num_kv_heads, block_size, head_size, dtype=query.dtype, device=device)
    value = torch.randn_like(key)
    block_table = torch.arange(num_reqs * 2, dtype=torch.int32, device=device).view(num_reqs, 2)
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.input_buffers = SimpleNamespace(seq_lens=torch.full((padded,), 99999, dtype=torch.int32, device=device))
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5")),
    )
    speculative = SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False)
    query_start_loc_cpu = torch.arange(num_reqs + 1, dtype=torch.int32) * width
    query_start_loc = query_start_loc_cpu.to(device)

    for lengths in (initial, next_lengths):
        spec.input_buffers.seq_lens[:num_reqs].copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        mirror = spec._prepare_draft_seq_lens_cpu(num_reqs, padded)
        assert mirror.tolist() == lengths + [0] * (padded - num_reqs)
        common = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=spec.input_buffers.seq_lens,
            seq_lens_cpu=mirror,
            seq_lens_cpu_is_exact=True,
            _seq_lens_cpu=torch.full((padded,), 99999, dtype=torch.int32),
            num_reqs=num_reqs,
            num_actual_tokens=num_reqs * width,
            max_query_len=width,
            max_seq_len=max(lengths),
            block_table_tensor=block_table,
            slot_mapping=torch.arange(num_reqs * width, dtype=torch.int32, device=device),
        )
        actual_lengths = _select_seq_lens(common, None, speculative, config)
        assert actual_lengths.device.type == "cpu"
        assert actual_lengths.tolist() == lengths
        kwargs = dict(
            input_layout="TND",
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            scale=head_size**-0.5,
            sparse_mode=0,
            block_table=block_table,
            block_size=block_size,
            actual_seq_lengths=query_start_loc_cpu[1:].tolist(),
        )
        expected, _ = torch_npu.npu_fused_infer_attention_score(
            query, key, value, actual_seq_lengths_kv=lengths, **kwargs
        )
        actual, _ = torch_npu.npu_fused_infer_attention_score(
            query, key, value, actual_seq_lengths_kv=actual_lengths.tolist(), **kwargs
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_dflash_builder_chain_shares_and_refreshes_exact_npu_lengths(mode, monkeypatch):
    device = torch.device("npu:0")
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._use_cpu_seq_lens = True
    spec._prepared_draft_attn_metadata = None
    spec.num_query_per_req = 8
    spec.max_model_len = spec.draft_max_seq_len = 256
    spec.input_batch = SimpleNamespace(num_reqs=2)
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([73, 81, 99999, 99999], dtype=torch.int32, device=device),
        positions=torch.arange(32, device=device),
        query_start_loc=torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32, device=device),
    )
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5")),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
    )
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    speculative = SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False)

    class RecordingBuilder:
        def build(self, common_prefix_len, common_attn_metadata):
            common = common_attn_metadata
            selected = _select_seq_lens(common, None, speculative, spec.vllm_config)
            assert selected.device.type == "cpu"
            return SimpleNamespace(
                seq_lens_list=selected.tolist(),
                actual_seq_lengths_q=common.query_start_loc_cpu[1:].tolist(),
                common=common,
            )

    groups = [SimpleNamespace(kv_cache_spec=object(), layer_names=[f"draft{i}"]) for i in range(2)]
    spec.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    spec.attn_groups = [
        [SimpleNamespace(layer_names=g.layer_names, get_metadata_builder=lambda _index: RecordingBuilder())]
        for g in groups
    ]
    spec.block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros((4, 2), dtype=torch.int32, device=device) for _ in groups],
        slot_mappings=torch.zeros((2, 32), dtype=torch.int64, device=device),
        cp_size=1,
    )
    copies = []
    prepare = spec._prepare_draft_seq_lens_cpu

    def track_copy(*args):
        copies.append(args)
        return prepare(*args)

    monkeypatch.setattr(spec, "_prepare_draft_seq_lens_cpu", track_copy)
    descriptor = BatchExecutionDescriptor(cg_mode=mode, num_tokens=32, num_reqs=4)
    upper_bound = torch.tensor([100, 100], dtype=torch.int32)
    for lengths in ([73, 81], [74, 60]):
        spec.input_buffers.seq_lens[:2].copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        with build_attn_metadata_wrapper():
            metadata = spec._build_attn_metadata(
                2, descriptor, np.array([0, 8, 16], dtype=np.int32), upper_bound, 8, False
            )
        for item in metadata.values():
            assert item.seq_lens_list == lengths + [0, 0]
            assert item.common.seq_lens_cpu_is_exact
        assert metadata["draft0"].common.seq_lens_cpu.data_ptr() == metadata["draft1"].common.seq_lens_cpu.data_ptr()
        if mode == CUDAGraphMode.FULL:
            assert spec.build_draft_attn_metadatas(4, upper_bound)[0] is metadata
    assert copies == [(2, 4), (2, 4)]
    assert upper_bound.tolist() == [100, 100]


def _make_prefetch_speculator(source):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.input_buffers = SimpleNamespace(seq_lens=source)
    spec._draft_seq_lens_cpu = None
    spec._draft_seq_lens_copy_stream = None
    spec._draft_seq_lens_copy_event = None
    spec._draft_seq_lens_copy_count = None
    return spec


def test_async_exact_lengths_refresh_when_batch_shrinks_and_grows():
    source = torch.full((4,), 99999, dtype=torch.int32, device="npu:0")
    spec = _make_prefetch_speculator(source)
    buffer_address = None
    for lengths in ([73, 81], [8], [74, 60, 128], [1, 2]):
        source[: len(lengths)].copy_(torch.tensor(lengths, dtype=source.dtype, device=source.device))
        spec._start_draft_seq_lens_copy(len(lengths))
        mirror = spec._prepare_draft_seq_lens_cpu(len(lengths), 4)
        assert mirror.tolist() == list(lengths) + [0] * (4 - len(lengths))
        assert mirror.is_pinned()
        if buffer_address is None:
            buffer_address = mirror.data_ptr()
        assert mirror.data_ptr() == buffer_address


@torch.inference_mode()
def test_early_draft_graph_uses_current_exact_lengths_after_rejection():
    device = torch.device("npu:0")
    torch.manual_seed(0)
    width, num_heads, num_kv_heads, head_size, block_size = 8, 4, 2, 64, 128
    raw_query = torch.randn(2 * width, num_heads, head_size, dtype=torch.float16, device=device)
    query = torch.empty_like(raw_query)
    key = torch.randn(4, num_kv_heads, block_size, head_size, dtype=query.dtype, device=device)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    lse = torch.empty(1, dtype=query.dtype, device=device)
    block_table = torch.arange(4, dtype=torch.int32, device=device).reshape(2, 2)
    source = torch.tensor([73, 81, 99999, 99999], dtype=torch.int32, device=device)
    spec = _make_prefetch_speculator(source)
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        block_table=block_table,
        block_size=block_size,
        input_layout="TND",
        sparse_mode=0,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=head_size**-0.5,
        actual_seq_lengths=[width, 2 * width],
        actual_seq_lengths_kv=[73, 81],
    )
    torch.mul(raw_query, 1.01, out=query)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    torch_npu.npu_fused_infer_attention_score.out(**kwargs, workspace=workspace, out=[output, lse])
    torch.npu.synchronize()
    graph = UpdatableGraph()
    with torch.npu.graph(graph):
        # Independent compute prefix; FIA must remain behind its update event.
        torch.mul(raw_query, 1.01, out=query)
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            {**kwargs, "workspace": workspace, "out": [output, lse]},
            FIAParamProvider("draft", None, is_draft_model=True),
        )
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=16, num_reqs=2)
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.graphs = {desc: graph}
    manager.update_stream = torch.npu.Stream(device=device)
    spec.attn_backends = {"draft": object()}
    spec.input_batch = SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([100, 100]))
    consumed = []

    def build(*args):
        mirror = spec._prepare_draft_seq_lens_cpu(2, 4)
        lengths = mirror[:2].tolist()
        consumed.append(lengths)
        return [
            {
                "draft": AscendMetadata(
                    seq_lens_list=lengths,
                    actual_seq_lengths_q=[width, 2 * width],
                    block_tables=block_table,
                )
            }
        ]

    spec.build_draft_attn_metadatas = build
    manager.speculator = spec
    for iteration, lengths in enumerate(([73, 81], [8, 82], [128, 1], [74, 60])):
        # Rebind physical cache rows as well as changing accepted lengths.
        block_table = torch.arange(4, dtype=torch.int32, device=device).roll(iteration).reshape(2, 2)
        source[:2].copy_(torch.tensor(lengths, dtype=source.dtype, device=device))
        spec._start_draft_seq_lens_copy(2)
        spec._deferred_draft_attn_metadata = object()
        manager.run_fullgraph(desc)
        torch.npu.synchronize()
        expected, _ = torch_npu.npu_fused_infer_attention_score(
            **{**kwargs, "actual_seq_lengths_kv": lengths, "block_table": block_table}
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert consumed[-1] == list(lengths)
