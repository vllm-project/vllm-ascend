# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.context_parallel import mla_cp
from vllm_ascend.attention.context_parallel.mla_cp import MLASplitAttentionKind
from vllm_ascend.attention.utils import split_decodes_and_prefills


@pytest.mark.parametrize(
    "draft,causal,query_lens,num_decodes,mode,capturing,expected",
    [
        (True, True, [1, 1], 2, CUDAGraphMode.NONE, False, False),
        (True, True, [1, 1], 2, CUDAGraphMode.PIECEWISE, False, False),
        (True, True, [1, 1, 128], 2, CUDAGraphMode.NONE, False, False),
        (True, True, [4, 4], 2, CUDAGraphMode.NONE, False, True),
        (True, True, [1, 3], 2, CUDAGraphMode.NONE, False, True),
        (True, False, [5, 5], 2, CUDAGraphMode.NONE, False, False),
        (True, False, [5, 5], 2, CUDAGraphMode.FULL, True, False),
        (False, True, [6, 6], 2, CUDAGraphMode.FULL, False, True),
        (False, True, [1, 1], 2, CUDAGraphMode.NONE, False, True),
        (True, True, [1, 1], 2, CUDAGraphMode.FULL, False, True),
        (True, True, [6, 6], 2, CUDAGraphMode.FULL, True, True),
        (True, True, [1, 1], 2, CUDAGraphMode.NONE, True, True),
    ],
)
def test_draft_decode_split_routing(
    monkeypatch, draft, causal, query_lens, num_decodes, mode, capturing, expected
):
    monkeypatch.setattr(mla_cp, "_EXTRA_CTX", SimpleNamespace(is_draft_model=draft, capturing=capturing))
    monkeypatch.setattr(mla_cp, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    impl = mla_cp.AscendMlaDCPImpl.__new__(mla_cp.AscendMlaDCPImpl)
    metadata = SimpleNamespace(causal=causal, query_lens=query_lens, num_decodes=num_decodes)
    assert impl._use_history_current_split_decode(metadata) is expected
    assert impl._decode_requires_current_kv(metadata) is expected


@pytest.mark.parametrize("prefilling,expected", [(True, (0, 1, 0, 4)), (False, (1, 0, 4, 0))])
def test_draft_first_pass_classification(monkeypatch, prefilling, expected):
    # Equal query lengths need not mean the same phase: an initial short
    # prefill stays prefill, while a later first draft pass may be decode.
    from vllm_ascend.attention import utils

    monkeypatch.setattr(utils, "is_pd_decode_recompute_scheduler_enabled", lambda: False)
    metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc_cpu=torch.tensor([0, 4]),
        is_prefilling=torch.tensor([prefilling]),
        context_parallel_metadata=SimpleNamespace(query_lens_cpu=torch.tensor([4]), max_query_len=4),
    )
    assert split_decodes_and_prefills(metadata, decode_threshold=6, treat_short_extends_as_decodes=False) == expected


@pytest.mark.parametrize(
    "draft,prefill,step_kinds",
    [
        (False, False, [[True, True]]),
        (False, False, [[False, False]]),
        (True, False, [[True, True], [True, True], [True, True]]),
        (True, False, [[False, False], [False, False], [False, False]]),
        (True, False, [[True, True], [False, False], [False, False]]),
        (True, False, [[True, False], [True, False], [True, False]]),
        (True, True, [[True, True], [False, False]]),
        (True, False, [[False, True], [False, True]]),
    ],
)
def test_graph_update_counts_layer_invocations(monkeypatch, draft, prefill, step_kinds):
    # The actual proposer invokes all layers of step 0 before step 1.
    # A split layer still consumes exactly one metadata entry for that step.
    keys = ["layer_b", "layer_a"]
    metadata = [
        {
            key: SimpleNamespace(
                decode=SimpleNamespace(
                    actual_seq_lengths_q=[2, 4, 6],
                    cp_history_seq_len=[100 * step + layer + 10] * 3,
                    cp_seq_len=[100 * step + layer + 20] * 3,
                    block_table=object(),
                )
            )
            for layer, key in enumerate(keys)
        }
        for step in range(len(step_kinds))
    ]
    params, expected = [], []
    for step, split_layers in enumerate(step_kinds):
        # Named split records must select metadata by their recorded layer,
        # even when capture order differs from the metadata dictionary order.
        layer_order = [1, 0] if all(split_layers) else [0, 1]
        for layer in layer_order:
            split = split_layers[layer]
            key = keys[layer]
            decode = metadata[step][key].decode
            for kind in ([MLASplitAttentionKind.HISTORY, MLASplitAttentionKind.CURRENT] if split else [None]):
                table = object()
                # Split FIA uses cumulative TND boundaries. The ordinary
                # BSND call has three requests with two queries each.
                query_lengths = [2, 4, 6] if split else [2, 2, 2]
                layout = "TND" if split else "BSND"
                sparse_mode = 3 if kind is MLASplitAttentionKind.CURRENT else 0
                block_size = 0 if kind is MLASplitAttentionKind.CURRENT else 128
                args = (0, 1, 2, 3, 4, 1, layout, None, sparse_mode, 1.0, table, block_size,
                        query_lengths, [7, 7, 7], 14, 15)
                params.append(mla_cp.MLASplitAttentionGraphParams(args, kind, key) if split else args)
                expected.append(
                    (
                        decode.actual_seq_lengths_q if split else [2, 2, 2],
                        decode.cp_history_seq_len
                        if kind == MLASplitAttentionKind.HISTORY
                        else decode.actual_seq_lengths_q if split else decode.cp_seq_len + [0, 0, 0],
                        decode.block_table if kind == MLASplitAttentionKind.HISTORY else None if split else table,
                    )
                )
    workspace = object()
    events = [Mock() for _ in params]
    handles = list(range(len(params)))
    graph = SimpleNamespace(
        attn_params={6: params}, events={6: events}, handles={6: handles}, workspaces={6: workspace}
    )
    getters = {}
    for name in ("get_graph_params", "get_draft_graph_params", "get_draft_graph_prefill_params"):
        getters[name] = Mock(return_value=graph)
        monkeypatch.setattr(mla_cp, name, getters[name])
    monkeypatch.setattr(mla_cp, "_EXTRA_CTX", SimpleNamespace(is_draft_model=draft, is_draft_model_prefill=prefill))
    monkeypatch.setattr(mla_cp.torch.npu, "stream", lambda _stream: nullcontext())
    begin, end, fia = Mock(), Mock(), Mock()
    monkeypatch.setattr(mla_cp.torch.npu, "graph_task_update_begin", begin)
    monkeypatch.setattr(mla_cp.torch.npu, "graph_task_update_end", end)
    monkeypatch.setattr(mla_cp.torch_npu, "npu_fused_infer_attention_score", SimpleNamespace(out=fia))
    mla_cp.AscendMlaDCPImpl.update_graph_params(
        "stream", SimpleNamespace(attn_metadata=metadata[0]), 6, draft_attn_metadatas=metadata
    )
    selected = (
        "get_draft_graph_prefill_params" if prefill else "get_draft_graph_params" if draft else "get_graph_params"
    )
    for name, getter in getters.items():
        assert getter.call_count == int(name == selected)
    assert begin.call_count == end.call_count == fia.call_count == len(params)
    assert [call.args[1] for call in begin.call_args_list] == handles
    for call, (query, kv, table) in zip(fia.call_args_list, expected):
        assert call.kwargs["actual_seq_lengths"] == query
        assert call.kwargs["actual_seq_lengths_kv"] == kv
        assert call.kwargs["block_table"] is table
        assert call.kwargs["workspace"] is workspace
    for event in events:
        event.record.assert_called_once_with("stream")


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_single_token_draft_uses_cached_kv_without_current_projection(monkeypatch, mode):
    monkeypatch.setattr(
        mla_cp, "_EXTRA_CTX", SimpleNamespace(is_draft_model=True, capturing=False, is_draft_model_prefill=False)
    )
    monkeypatch.setattr(mla_cp, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    monkeypatch.setattr(mla_cp, "get_draft_graph_params", lambda: None)
    impl = mla_cp.AscendMlaDCPImpl.__new__(mla_cp.AscendMlaDCPImpl)
    impl.dcp_size, impl.num_heads, impl.num_kv_heads = 2, 2, 1
    impl.kv_lora_rank, impl.qk_rope_head_dim = 3, 2
    impl.scale = 1.0
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl._forward_decode_split_attention = Mock(side_effect=AssertionError("Unexpected split attention"))
    impl._merge_dcp_attention_output = lambda output, _lse, _rank: output
    impl._v_up_proj_batch_major = lambda output: output
    expected = torch.randn(2, 1, 4, 3)
    fia = Mock(return_value=(expected, torch.randn(2, 4, 1, 1)))
    monkeypatch.setattr(mla_cp.torch_npu, "npu_fused_infer_attention_score", fia)
    decode = mla_cp.AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(2),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        seq_lens=torch.tensor([2, 3]),
        max_seq_lens=3,
        seq_lens_list=[2, 3],
        cp_seq_len=torch.tensor([2, 3]),
        dcp_mtp_attn_mask=None,
    )
    metadata = SimpleNamespace(
        causal=True,
        query_lens=[1, 1, 128],
        num_decodes=2,
        decode=decode,
        attn_state=mla_cp.AscendAttentionState.SpecDecoding,
    )
    preprocess = mla_cp.DecodeMLAPreprocessResult(
        torch.randn(2, 4, 3), torch.randn(2, 4, 2), torch.randn(2, 1, 4, 3), torch.randn(2, 1, 4, 2)
    )
    assert not impl._decode_requires_current_kv(metadata)
    result = impl._forward_decode(preprocess, 4, metadata)
    impl._forward_decode_split_attention.assert_not_called()
    assert fia.call_args.args[0].shape == (2, 1, 4, 3)
    assert fia.call_args.kwargs["actual_seq_lengths"] == [1, 1]
    assert fia.call_args.kwargs["actual_seq_lengths_kv"] is decode.cp_seq_len
    assert fia.call_args.kwargs["block_table"] is decode.block_table
    assert fia.call_args.kwargs["atten_mask"] is None
    torch.testing.assert_close(result, expected.view(2, 4, 3))
