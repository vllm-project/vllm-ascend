# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendMetadataForDecode,
    DCPFIAParamProvider,
    build_dcp_fia_params,
)
from vllm_ascend.attention.context_parallel.common_cp import CPKVScope, use_history_current_split_decode
from vllm_ascend.compilation.updatable_graph import ContextSource, GraphUpdateTask, SharedSource, UpdatableGraph
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


@pytest.mark.parametrize("size,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("interleave", [1, 8, 16])
def test_history_is_partitioned_after_removing_global_current_chunk(size, interleave, rank):
    total = torch.tensor([0, 1, 9, 16, 33, 69], dtype=torch.int32)
    query = torch.tensor([1, 1, 4, 3, 2, 4], dtype=torch.int32)
    decode = AscendMetadataForDecode()
    decode.update_dcp_seq_lens_cpu(
        total,
        get_dcp_local_seq_lens(total, dcp_size=size, dcp_rank=rank, cp_kv_cache_interleave_size=interleave),
        query,
        dcp_size=size,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
    )
    # Independent position ownership reference, including interleave boundaries.
    expected = [
        sum((position // interleave) % size == rank for position in range(max(0, int(s - q))))
        for s, q in zip(total, query)
    ]
    assert decode.cp_history_seq_len == expected
    assert decode.num_computed_tokens_of_dcp.sum(axis=1).tolist() == total.tolist()


def make_metadata(total=(13, 23), query=(4, 2), rank=1, interleave=1):
    decode = AscendMetadataForDecode(
        block_tables=torch.arange(1, 2 * len(query) + 1, dtype=torch.int32).reshape(len(query), 2),
        actual_seq_lengths_q=torch.tensor(query).cumsum(0).tolist(),
        seq_lens_list=list(total),
    )
    decode.update_dcp_seq_lens_cpu(
        torch.tensor(total),
        get_dcp_local_seq_lens(torch.tensor(total), dcp_size=2, dcp_rank=rank, cp_kv_cache_interleave_size=interleave),
        torch.tensor(query),
        dcp_size=2,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
    )
    return SimpleNamespace(
        num_decodes=len(query),
        actual_seq_lengths_q=torch.tensor(query).cumsum(0).tolist(),
        decode=decode,
        causal=True,
        attn_mask=torch.ones((8, 8), dtype=torch.bool),
    )


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 5])
def test_plain_eager_paged_fia_uses_inner_precise_without_padding(monkeypatch, num_reqs):
    from vllm_ascend.attention.context_parallel import attention_cp

    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl._layer_name = "layer"
    impl.kv_sharing_target_layer_name = None
    impl.dcp_rank = 1
    impl.num_kv_heads = 1
    impl.scale = 0.1
    impl.key_cache = torch.zeros((4, 128, 1, 128))
    metadata = make_metadata(total=tuple(range(13, 13 + num_reqs)), query=(1,) * num_reqs)
    query = torch.randn(num_reqs, 8, 128)
    key = value = impl.key_cache.view(4, 128, 128)
    seen: dict[str, Any] = {}

    def fia(q, k, v, **kwargs):
        seen.update(query=q, key=k, value=v, **kwargs)
        return q.clone(), torch.zeros((*q.shape[:2], 1))

    monkeypatch.setattr(attention_cp, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    monkeypatch.setattr(attention_cp.torch_npu, "npu_fused_infer_attention_score", fia)
    output, lse = impl._run_dcp_attention(query, key, value, metadata, CPKVScope.FULL, 8)
    assert seen["actual_seq_lengths"] == list(range(1, num_reqs + 1))
    assert seen["actual_seq_lengths_kv"] == metadata.decode.num_computed_tokens_of_dcp[:, 1].tolist()
    assert seen["inner_precise"] == 1
    torch.testing.assert_close(seen["query"], query)
    torch.testing.assert_close(seen["block_table"], metadata.decode.block_tables)
    assert seen["key"] is key and seen["value"] is value
    torch.testing.assert_close(output, query)
    assert lse.shape == (num_reqs, 8, 1)
    assert len(metadata.decode.actual_seq_lengths_q) == num_reqs


def test_history_current_graph_tasks_have_distinct_tnd_parameters():
    first, second = make_metadata(), make_metadata(total=(14, 24), query=(1, 1))
    params = build_dcp_fia_params("layer", first, 1, use_spec_decode=True) + build_dcp_fia_params(
        "layer", second, 1, use_spec_decode=True
    )
    history = DCPFIAParamProvider("layer", 1, CPKVScope.HISTORY)
    current = DCPFIAParamProvider("layer", 1, CPKVScope.CURRENT)
    histories, currents = SharedSource(params).get(history), SharedSource(params).get(current)
    assert len(histories) == len(currents) == 2
    assert histories[0]["actual_seq_lengths"] == [4, 6]
    assert histories[0]["actual_seq_lengths_kv"] == [4, 10]
    assert currents[0]["actual_seq_lengths_kv"] == [4, 6]
    assert currents[0]["block_table"] is None
    assert histories[1]["actual_seq_lengths_kv"] == [6, 11]
    assert currents[1]["actual_seq_lengths_kv"] == [1, 2]
    assert ContextSource({"layer": first}).get(history)[0] == histories[0]


@pytest.mark.parametrize(
    "causal,use_spec_decode,is_draft,is_prefill,query,expected_split",
    [
        pytest.param(True, False, False, False, (1, 1), False, id="decode"),
        pytest.param(True, False, False, False, (1, 4), True, id="multi-token-decode"),
        pytest.param(True, True, False, False, (1, 1), True, id="speculative-target"),
        pytest.param(True, True, True, False, (1, 1), False, id="draft-decode"),
        pytest.param(True, True, True, False, (1, 4), True, id="multi-token-draft"),
        pytest.param(True, False, True, True, (1, 1), True, id="draft-prefill"),
        pytest.param(False, True, True, True, (4, 2), False, id="noncausal"),
    ],
)
def test_dcp_draft_path_selection_and_graph_parameters_agree(
    causal, use_spec_decode, is_draft, is_prefill, query, expected_split
):
    metadata = make_metadata(query=query)
    metadata.causal = causal
    assert (
        use_history_current_split_decode(
            metadata,
            is_draft_model=is_draft,
            is_draft_model_prefill=is_prefill,
            use_spec_decode=use_spec_decode,
        )
        is expected_split
    )
    params = build_dcp_fia_params(
        "layer",
        metadata,
        1,
        is_draft_model=is_draft,
        is_draft_model_prefill=is_prefill,
        use_spec_decode=use_spec_decode,
    )
    kinds = (CPKVScope.HISTORY, CPKVScope.CURRENT) if expected_split else (CPKVScope.FULL,)
    assert [param["layer_name"] for param in params] == [("layer", kind) for kind in kinds]
    if not expected_split:
        assert params[0]["actual_seq_lengths_kv"] == [6, 11]


def test_prefill_only_does_not_require_current_decode_kv():
    metadata = SimpleNamespace(causal=True, decode=None)
    assert not use_history_current_split_decode(
        metadata,
        use_spec_decode=True,
        is_draft_model=True,
        is_draft_model_prefill=True,
    )


def test_single_token_draft_reads_complete_cache_without_current_attention():
    impl = object.__new__(AscendAttentionDCPImpl)
    impl.dcp_size, impl.num_heads, impl.head_size = 2, 2, 8
    impl.vllm_config = SimpleNamespace(speculative_config=object())
    impl.dcp_group = SimpleNamespace(unique_name="group")
    impl.key_cache = torch.empty(2, 16, 1, 8)
    impl.value_cache = torch.empty_like(impl.key_cache)
    query = torch.empty(2, 2, 8)
    gathered = torch.empty(2, 4, 8)
    impl._dcp_all_gather = MagicMock(return_value=gathered)
    cached = (torch.ones_like(gathered), torch.zeros(2, 4, 1))
    impl._run_dcp_attention = MagicMock(return_value=cached)
    merged = torch.full_like(query, 42)
    merged[1].fill_(float("nan"))
    impl._merge_dcp_attention_output = MagicMock(return_value=merged)
    module = "vllm_ascend.attention.context_parallel.attention_cp"
    with (
        patch(module + "._EXTRA_CTX", SimpleNamespace(is_draft_model=True, is_draft_model_prefill=False)),
        patch(module + ".cp_decode_comm_stream", side_effect=AssertionError("No split stream for draft decode")),
        patch(module + ".fused_dcp_lse_combine", side_effect=AssertionError("No history/current merge")),
    ):
        actual = impl._forward_decode_dcp(query, make_metadata(query=(1, 1)))
    assert actual is merged
    assert impl._run_dcp_attention.call_count == 1
    assert impl._run_dcp_attention.call_args.args[4:] == (CPKVScope.FULL, 4)
    impl._merge_dcp_attention_output.assert_called_once_with(*cached)


def test_capture_registers_tnd_history_and_current_with_separate_providers():
    impl = object.__new__(AscendAttentionDCPImpl)
    impl.dcp_rank = 1
    impl.num_kv_heads = 1
    impl.scale = 0.125
    impl.key_cache = torch.empty(2, 16, 1, 8)
    impl._use_max_workspace_for_fia_graph = False
    impl._graph_metadata_layer_name = MagicMock(return_value="layer")
    query = torch.empty(6, 4, 8)
    metadata = make_metadata()
    module = "vllm_ascend.attention.context_parallel.attention_cp"
    with (
        patch(module + "._EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch(module + ".get_capture_resource", return_value=torch.empty(1)) as resource,
        patch(module + ".register_task") as register,
    ):
        for kind in (CPKVScope.HISTORY, CPKVScope.CURRENT):
            impl._run_dcp_attention(query, query, query, metadata, kind, 4)
    assert register.call_count == 2
    history, current = [call.args for call in register.call_args_list]
    assert history[1]["input_layout"] == current[1]["input_layout"] == "TND"
    assert history[1]["atten_mask"] is None
    assert history[1]["sparse_mode"] == 0
    assert history[1]["inner_precise"] == 1
    assert current[1]["atten_mask"] is metadata.attn_mask
    assert current[1]["sparse_mode"] == 3
    assert "inner_precise" not in current[1]
    assert history[1]["actual_seq_lengths_kv"] == [4, 10]
    assert current[1]["actual_seq_lengths_kv"] == [4, 6]
    assert history[2] != current[2]
    assert resource.call_args_list[0].args[0] != resource.call_args_list[1].args[0]


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_decode_only_historical_shards_are_exchanged_current_contributes_once(dcp_size):
    impl = object.__new__(AscendAttentionDCPImpl)
    impl.dcp_size, impl.num_heads, impl.head_size = dcp_size, 2, 8
    impl.vllm_config = SimpleNamespace(speculative_config=object())
    impl.dcp_group = SimpleNamespace(unique_name="group")
    # Preserve the non-contiguous paged K/V layout that broke BSND .out.
    cache = torch.empty(2, 2, 16, 1, 8)
    impl.key_cache, impl.value_cache = cache[:, 0], cache[:, 1]
    assert not impl.key_cache.is_contiguous()
    query = torch.empty(6, 2, 8)
    history_query = torch.empty(6, 2 * dcp_size, 8)
    impl._dcp_all_gather = MagicMock(return_value=history_query)
    history = (torch.ones_like(history_query), torch.zeros(6, 2 * dcp_size, 1))
    current = (torch.full_like(query, 2), torch.ones(6, 2, 1))
    main, attn = MagicMock(), MagicMock()
    main.record_event.return_value = "history_ready"
    attn.record_event.return_value = "current_done"
    events = []
    active = ["main"]

    @contextmanager
    def on_stream(stream):
        assert stream is attn
        active[0] = "attn"
        yield
        active[0] = "main"

    def attention(*args):
        kind = args[4]
        assert active[0] == ("main" if kind == CPKVScope.HISTORY else "attn")
        events.append(kind)
        return history if kind == CPKVScope.HISTORY else current

    def communicate(*args, **kwargs):
        assert active[0] == "main"
        events.append("a2a")
        return "history_recv"

    def merge(*args, **kwargs):
        assert events[-1] == ("wait", "current_done")
        events.append("merge")
        return query

    attn.wait_event.side_effect = lambda event: events.append(("wait", event))
    main.wait_event.side_effect = lambda event: events.append(("wait", event))
    impl._run_dcp_attention = MagicMock(side_effect=attention)
    module = "vllm_ascend.attention.context_parallel.attention_cp"
    with (
        patch(module + "._EXTRA_CTX", SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False)),
        patch(module + ".cp_decode_comm_stream", return_value=attn),
        patch.object(torch.npu, "current_stream", return_value=main),
        patch.object(torch.npu, "stream", side_effect=on_stream),
        patch.object(torch.Tensor, "record_stream", autospec=True) as record_stream,
        patch("torch.ops.vllm.dcp_a2a_fused", side_effect=communicate) as a2a,
        patch(module + ".fused_dcp_lse_combine", side_effect=merge) as combine,
    ):
        result = impl._forward_decode_dcp(query, make_metadata(), query, query)
    assert result is query
    a2a.assert_called_once_with(*history, dcp_size, 1, "group" if dcp_size > 1 else "", defer_combine=True)
    assert combine.call_args.kwargs["local_output"] is current[0]
    assert combine.call_args.kwargs["local_lse"] is current[1]
    calls = impl._run_dcp_attention.call_args_list
    assert calls[0].args[4:] == (CPKVScope.HISTORY, 2 * dcp_size)
    assert not calls[0].args[1].is_contiguous()
    assert calls[1].args[4:] == (CPKVScope.CURRENT, 2)
    assert record_stream.call_count == 6
    assert events == [
        CPKVScope.HISTORY,
        ("wait", "history_ready"),
        CPKVScope.CURRENT,
        "a2a",
        ("wait", "current_done"),
        "merge",
    ]


@pytest.mark.parametrize("interleave", [1, 8])
def test_v2_draft_steps_update_history_and_current_independently(interleave):
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "GQA"
    speculator.max_model_len = 32
    speculator.use_dcp = True
    prepare_local = MagicMock(
        side_effect=lambda lengths: get_dcp_local_seq_lens(
            lengths, dcp_size=2, dcp_rank=1, cp_kv_cache_interleave_size=interleave
        )
    )
    speculator.dcp_manager = SimpleNamespace(dcp_world_rank=1, prepare_dcp_local_seq_lens_cpu=prepare_local)
    speculator.draft_vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, cp_kv_cache_interleave_size=interleave)
    )
    speculator._get_seq_lens_cpu = MagicMock(return_value=torch.tensor([10, 31, 0], dtype=torch.int32))
    metadata = SimpleNamespace(seq_lens_cpu=torch.zeros(3, dtype=torch.int32), decode=AscendMetadataForDecode())
    speculator._update_decode_attn_metadata({"draft": metadata}, step=2, num_reqs=2)
    expected_history = [sum((position // interleave) % 2 == 1 for position in range(length)) for length in [11, 31, 0]]
    assert metadata.decode.cp_history_seq_len == expected_history
    assert metadata.decode.actual_seq_lengths_q == [1, 2, 3]
    assert metadata.decode.seq_lens_list == [12, 32, 0]
    assert prepare_local.call_count == 1
    assert torch.equal(prepare_local.call_args.args[0], torch.tensor([12, 32, 0]))
    expected_local = get_dcp_local_seq_lens(
        torch.tensor([12, 32, 0]), dcp_size=2, dcp_rank=1, cp_kv_cache_interleave_size=interleave
    )
    assert metadata.decode.num_computed_tokens_of_dcp[:, 1].tolist() == expected_local.tolist()


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave", [1, 8])
def test_v2_graph_shared_source_preserves_steps_layers_and_padding(is_prefill, causal, rank, interleave):
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator.use_dcp = True
    speculator.attn_architecture = "GQA"
    speculator.dcp_manager = SimpleNamespace(dcp_world_rank=rank)
    speculator.draft_vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=interleave,
        )
    )
    speculator.input_batch = SimpleNamespace(num_reqs=2, seq_lens_np=[13, 31])
    speculator.num_speculative_steps = 3
    speculator.max_model_len = 32
    query = (4, 2, 1) if is_prefill else (1, 1, 1)
    steps = [
        {
            layer: make_metadata(total=(14 + step, 32, 0), query=query, rank=rank, interleave=interleave)
            for layer in ("layer", "other_layer")
        }
        for step in range(1 if is_prefill else 2)
    ]
    speculator.draft_attn_layer_names = set(steps[0])
    for step_index, metadata in enumerate(steps):
        for layer_index, (layer, item) in enumerate(metadata.items()):
            item.causal = causal
            # Draft steps share each layer's block table, as in the metadata builder.
            if step_index == 0:
                item.decode.block_tables = item.decode.block_tables + 10 * layer_index
            else:
                item.decode.block_tables = steps[0][layer].decode.block_tables
            if not is_prefill:
                # Decode lengths come from the common seq_lens loop, not cached metadata.
                item.decode.cp_history_seq_len = [-1, -1, -1]
                item.decode.num_computed_tokens_of_dcp.fill(-1)
    kinds = (CPKVScope.HISTORY, CPKVScope.CURRENT) if is_prefill and causal else (CPKVScope.FULL,)
    providers = [DCPFIAParamProvider(layer, rank, kind) for layer in steps[0] for kind in kinds]
    tasks = [
        GraphUpdateTask(MagicMock(), {"step": step}, provider, step, None, None)
        for step in range(len(steps))
        for provider in providers
    ]
    graph = SimpleNamespace(tasks=tasks, provider_sizes={provider: len(steps) for provider in providers})
    with patch.object(
        AscendMetadataForDecode,
        "update_dcp_seq_lens_cpu",
        side_effect=AssertionError("FIA parameters must not mutate attention metadata"),
    ):
        params = speculator.build_fia_params(3, steps[0], is_prefill)
        source = SharedSource(params)
        resolved = UpdatableGraph.resolve_tasks(graph, source)

    for task in resolved:
        layer_metadata = steps[task.provider_index][task.provider.metadata_layer_name]
        assert task.kwargs["actual_seq_lengths"] == layer_metadata.decode.actual_seq_lengths_q
        if task.provider.attention_kind == CPKVScope.CURRENT:
            assert task.kwargs["actual_seq_lengths_kv"] == layer_metadata.decode.actual_seq_lengths_q
            assert task.kwargs["block_table"] is None
        else:
            lengths = layer_metadata.decode.seq_lens_list
            if task.provider.attention_kind == CPKVScope.HISTORY:
                lengths = [max(total - current, 0) for total, current in zip(lengths, query)]
            expected = [sum((position // interleave) % 2 == rank for position in range(length)) for length in lengths]
            assert task.kwargs["actual_seq_lengths_kv"] == expected
            assert task.kwargs["actual_seq_lengths_kv"][-1] == 0
            assert task.kwargs["block_table"] is layer_metadata.decode.block_tables
        assert "layer_name" not in task.kwargs
    assert all(task.kwargs == {"step": task.provider_index} for task in tasks)


def test_v2_draft_metadata_does_not_alias_history_between_steps():
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "GQA"
    speculator.use_dcp = True
    speculator.input_batch = SimpleNamespace(num_reqs=2, seq_lens_cpu_upper_bound=torch.tensor([13, 23]))
    speculator.input_buffers = SimpleNamespace(draft_seq_lens_cpus=[torch.tensor([14, 24]), torch.tensor([15, 25])])
    metadata = make_metadata(total=(13, 23), query=(1, 1))
    speculator._build_uniform_attn_metadata = MagicMock(return_value={"layer": metadata})
    steps = speculator._init_decode_draft_attn_metadatas({"layer": metadata}, 2)
    assert steps[0]["layer"].decode is not steps[1]["layer"].decode
    steps[0]["layer"].decode.cp_history_seq_len = [6, 11]
    steps[1]["layer"].decode.cp_history_seq_len = [7, 12]
    assert metadata.decode.cp_history_seq_len == [6, 11]
    assert steps[0]["layer"].decode.cp_history_seq_len == [6, 11]


@pytest.mark.parametrize("rank", [0, 1])
def test_v1_draft_builder_owns_history_and_padding_without_manager_override(rank):
    from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPMetadataBuilder
    from vllm_ascend.worker.dcp_utils import DCPManager

    builder = object.__new__(AscendAttentionDCPMetadataBuilder)
    builder.pcp_enabled = False
    builder.dcp_size, builder.dcp_rank = 2, rank
    builder.decode_threshold = 4
    builder.speculative_config = SimpleNamespace(parallel_drafting=False)
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
    )
    builder.kv_cache_spec = object()
    builder.device = torch.device("cpu")
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *args: None)
    advanced = torch.tensor([12, 13, 14, 15], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=16,
        num_actual_tokens=4,
        max_query_len=1,
        query_start_loc_cpu=torch.arange(17, dtype=torch.int32),
        query_start_loc=torch.arange(17, dtype=torch.int32),
        seq_lens=advanced,
        dcp_local_seq_lens_cpu=get_dcp_local_seq_lens(advanced, dcp_size=2, dcp_rank=rank),
        _seq_lens_cpu=advanced,
        seq_lens_cpu=advanced,
        block_table_tensor=torch.zeros(4, 1, dtype=torch.int32),
        slot_mapping=torch.arange(16, dtype=torch.int32),
        attn_state=None,
        causal=True,
        is_prefilling=torch.ones(16, dtype=torch.bool),
        context_parallel_metadata=object(),
    )
    # Production __init__ pre-allocates this graph-replay buffer; the fixture
    # builds the builder without __init__, so provide it explicitly.
    builder._seq_lens_device_buffer = torch.empty(common.num_reqs + 1, dtype=torch.int32)
    manager = object.__new__(DCPManager)
    manager.dcp_world_size, manager.dcp_world_rank = 2, rank
    manager.vllm_config = builder.vllm_config
    manager._get_dcp_local_seq_lens = MagicMock(side_effect=AssertionError("builder owns GQA lengths"))
    with (
        patch.object(DCPManager, "_is_mla_kv_cache_spec", return_value=False),
        patch.object(DCPManager, "_is_sfa_dcp_metadata_builder", return_value=False),
        patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
    ):
        manager.prepare_spec_decode_drafting_cp_metadata(common, object())
        assert common.context_parallel_metadata is None
        metadata = builder.build(0, common)
        decode = metadata.decode
        original_history = decode.cp_history_seq_len
        original_total = decode.num_computed_tokens_of_dcp
        manager.update_spec_decode_drafting_cp_metadata(
            metadata, object(), torch.tensor([10, 11, 12, 13]), draft_index=99
        )
    expected = [sum(position % 2 == rank for position in range(length)) for length in [11, 12, 13, 14]]
    assert metadata.num_decodes == 16
    assert decode.cp_history_seq_len == expected + [0] * 12
    assert decode.block_tables.shape == (16, 1)
    assert decode.num_computed_tokens_of_dcp.shape == (16, 2)
    assert decode.cp_history_seq_len is original_history
    assert decode.num_computed_tokens_of_dcp is original_total
    manager._get_dcp_local_seq_lens.assert_not_called()


def test_dcp_forward_preserves_graph_padding_query_span():
    from unittest.mock import Mock

    from vllm_ascend.attention.context_parallel.attention_cp import (
        AscendAttentionDCPImpl,
        AscendAttentionDCPMetadata,
        AscendMetadataForDecode,
    )

    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    query = torch.arange(8).reshape(4, 1, 2).float()
    impl._forward_decode_dcp = Mock(side_effect=lambda q, *_args: q)
    metadata = AscendAttentionDCPMetadata(
        num_decodes=4,
        num_prefills=0,
        num_decode_tokens=1,
        num_actual_tokens=1,
        decode=AscendMetadataForDecode(actual_seq_lengths_q=[1, 2, 3, 4]),
    )
    output = torch.zeros_like(query)
    actual = impl.forward_impl(query, query, query, (), metadata, output)
    torch.testing.assert_close(actual, query)
    assert impl._forward_decode_dcp.call_args.args[0].shape[0] == 4
