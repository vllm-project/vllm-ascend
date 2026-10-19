# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.workspace import WorkspaceManager

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.ops.triton.turboquant_prefill import current_workspace_shapes, workspace_shapes


@pytest.mark.parametrize(
    "requests,length,queries,budget",
    [
        (0, 30000, 256, 64),
        (1, 0, 256, 64),
        (1, 30000, 1, 64),
        (1, 30000, 83, 64),
        (2, 30000, 256, 64),
        (1, 30000, 256, 1),
        (1, 4096, 256, 0),
    ],
)
def test_workspace_fallback_for_prefix_budget_or_empty(requests, length, queries, budget):
    assert workspace_shapes(requests, length, queries, budget * 1024**2) is None


def test_workspace_budget_includes_table_and_alignment():
    data_bytes = 128 * (512 + 64) * 2
    assert workspace_shapes(1, 128, 32, data_bytes) is None
    assert workspace_shapes(1, 128, 32, data_bytes + 256) is not None


def test_profile_reserves_one_shared_workspace_for_all_layers():
    manager = WorkspaceManager(torch.device("cpu"))
    layers = [object.__new__(AscendSFAImpl) for _ in range(2)]
    with patch("vllm_ascend.attention.sfa_v1.current_workspace_manager", return_value=manager):
        pointers = []
        for layer in layers:
            layer.enable_sparse_sfa_turboquant = True
            layer.turboquant = MagicMock()
            layer.tq_prefill_workspace_bytes = 2 * 1024**2
            layer.forward("layer", torch.empty(1), (), None, output=torch.empty(1))
            pointers.append(manager.get_simultaneous(((1,), torch.uint8))[0].data_ptr())
        assert pointers[0] == pointers[1]
        manager.lock()
        shapes = workspace_shapes(1, 1025, 32, 2 * 1024**2)
        views = manager.get_simultaneous(*shapes)
        assert views[0].data_ptr() == pointers[0]


@pytest.mark.parametrize("prefills,decodes,budget", [(0, 1, 64), (1, 1, 64), (1, 0, 0)])
def test_decode_mixed_and_disabled_do_not_allocate(prefills, decodes, budget):
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = budget * 1024**2
    metadata = SimpleNamespace(num_prefills=prefills, num_decodes=decodes)
    with patch("vllm_ascend.attention.sfa_v1.current_workspace_manager") as manager:
        result = layer._try_tq_prefill_workspace(None, None, None, None, metadata, None, None, None)
    assert result is None
    manager.assert_not_called()


@pytest.mark.parametrize("folded", [False, True])
def test_workspace_uses_existing_bf16_attention_dispatch(folded):
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = 1024**2
    layer._tq_rotations_folded = folded
    layer.scale = 576**-0.5
    layer.turboquant = MagicMock()
    layer.turboquant.forward.side_effect = lambda x: x + 1
    layer.turboquant.inverse.side_effect = lambda x: x + 2
    metadata = SimpleNamespace(num_prefills=1, num_decodes=0, max_seq_len=128)
    query = torch.zeros(32, 64, 512, dtype=torch.bfloat16)
    rope = torch.zeros(32, 64, 64, dtype=torch.bfloat16)
    cache = torch.empty(1, 128, 1, 386, dtype=torch.int8)
    indices = torch.zeros(32, 1, 2048, dtype=torch.int32)
    qlens, klens = torch.tensor([32]), torch.tensor([128])
    table = torch.zeros(1, 1, dtype=torch.int32)
    output = torch.ones_like(query)
    manager = WorkspaceManager(torch.device("cpu"))
    with (
        patch("vllm_ascend.attention.sfa_v1.current_workspace_manager", return_value=manager),
        patch("vllm_ascend.attention.sfa_v1.expand_turboquant_pages") as expand,
        patch("torch.ops._C_ascend.npu_sparse_flash_attention", create=True, return_value=(output,)) as attention,
    ):
        result = layer._try_tq_prefill_workspace(query, rope, cache, indices, metadata, qlens, klens, table)
    expand.assert_called_once()
    nope, expanded_rope, expanded_table = expand.call_args.args[-3:]
    attention.assert_called_once()
    args = attention.call_args.kwargs
    assert args["key"] is args["value"] is nope
    assert args["key_rope"] is expanded_rope
    assert args["block_table"] is expanded_table
    assert args["actual_seq_lengths_query"] is qlens
    assert args["actual_seq_lengths_kv"] is klens
    assert args["sparse_mode"] == 3
    assert args["return_softmax_lse"] is False
    torch.testing.assert_close(args["query"], query if folded else query + 1)
    torch.testing.assert_close(result, output if folded else output + 2)


def test_current_workspace_budget_counts_send_and_receive_buffers():
    shapes = current_workspace_shapes(1, 4096, 256, 4096, 64 * 1024**2)
    assert shapes is not None and len(shapes) == 5
    manager = WorkspaceManager(torch.device("cpu"))
    views = manager.get_simultaneous(*shapes)
    required = sum(((t.numel() * t.element_size() + 255) // 256) * 256 for t in views)
    assert current_workspace_shapes(1, 4096, 256, 4096, required) == shapes
    assert current_workspace_shapes(1, 4096, 256, 4096, required - 1) is None
    assert views[3].data_ptr() != views[4].data_ptr()
    assert len(current_workspace_shapes(1, 4096, 4096, 4096, 64 * 1024**2)) == 4


@pytest.mark.parametrize("rank", [0, 1])
def test_current_workspace_decision_is_uniform_on_padded_tp_ranks(rank):
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = 8 * 1024**2
    layer.tq_prefill_bf16_current = True
    layer.local_num_heads = 64
    metadata = SimpleNamespace(
        num_prefills=1, num_decodes=0, max_seq_len=4096,
        block_table=torch.zeros(1, 32, dtype=torch.int32), seq_lens=torch.tensor([4096]),
        # Rank 1 can have no query for a particular request. The gate must use
        # padded global counts, never rank-local has_local_query/sequence lengths.
        dsa_cp_context=SimpleNamespace(num_tokens_pad=64, local_start=rank * 32),
    )
    cache = torch.empty(32, 128, 1, 386, dtype=torch.int8)
    manager = WorkspaceManager(torch.device("cpu"))
    with (
        patch("vllm_ascend.attention.sfa_v1.current_workspace_manager", return_value=manager),
        patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=SimpleNamespace(world_size=2)),
    ):
        views = layer._prepare_tq_current_workspace(torch.empty(32, 128, dtype=torch.bfloat16), cache, metadata)
    assert views is not None
    assert views[3].shape == (32, 576) and views[4].shape == (64, 576)


@pytest.mark.parametrize("enabled,budget,prefills,decodes", [
    (False, 64, 1, 0), (True, 0, 1, 0), (True, 64, 0, 1), (True, 64, 1, 1),
])
def test_current_workspace_fallback_never_enters_collective(enabled, budget, prefills, decodes):
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = budget * 1024**2
    layer.tq_prefill_bf16_current = enabled
    metadata = SimpleNamespace(num_prefills=prefills, num_decodes=decodes)
    with patch("vllm_ascend.attention.sfa_v1.current_workspace_manager") as manager:
        assert layer._prepare_tq_current_workspace(None, None, metadata) is None
    manager.assert_not_called()


def test_current_attention_uses_global_history_lengths_and_existing_storage():
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = 8 * 1024**2
    layer._tq_rotations_folded = True
    layer.turboquant = MagicMock()
    metadata = SimpleNamespace(
        num_prefills=1, num_decodes=0, max_seq_len=4096,
        cum_query_lens=torch.tensor([2048]), seq_lens=torch.tensor([4096]),
    )
    shapes = current_workspace_shapes(1, 4096, 128, 2048, 8 * 1024**2)
    views = tuple(torch.empty(shape, dtype=dtype) for shape, dtype in shapes)
    q = torch.empty(128, 64, 512, dtype=torch.bfloat16)
    klens = torch.tensor([2176])  # This rank only consumes the start of the fresh suffix.
    with (
        patch("vllm_ascend.attention.sfa_v1.current_workspace_manager") as manager,
        patch("vllm_ascend.attention.sfa_v1.expand_turboquant_pages") as expand,
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.execute_sparse_flash_attention_process", return_value=q),
    ):
        result = layer._try_tq_prefill_workspace(
            q, torch.empty(128, 64, 64), torch.empty(32, 128, 1, 386, dtype=torch.int8),
            torch.zeros(128, 1, 2048), metadata, torch.tensor([128]), klens,
            torch.zeros(1, 32), current_workspace=views,
        )
    assert result is q
    manager.assert_not_called()
    assert expand.call_args.args[2] is klens
    assert expand.call_args.kwargs["full_seq_lens"] is metadata.seq_lens
    assert expand.call_args.kwargs["current_query_lens"] is metadata.cum_query_lens
    assert expand.call_args.kwargs["current_kv"] is views[4]


@pytest.mark.parametrize("has_query", [False, True])
def test_current_forward_waits_for_gather_and_keeps_cache_notification_order(has_query):
    from vllm_ascend.attention.sfa_v1 import PreprocessType

    layer = object.__new__(AscendSFAImpl)
    layer.enable_sparse_sfa_turboquant = True
    layer.turboquant = MagicMock()
    layer.qk_rope_head_dim, layer.kv_lora_rank, layer.q_lora_rank = 64, 512, 8
    layer.num_kv_heads, layer.local_num_heads, layer.v_head_dim = 1, 64, 128
    layer.preprocess_type = PreprocessType.NATIVE
    layer.g_proj = layer.layerwise_kv_cache_hook = None
    layer.has_indexer = True
    layer.skip_topk = layer.use_index_cache = False
    layer.layer_name = "test.attn"
    layer.kv_a_layernorm = SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-5)
    layer.q_a_layernorm = lambda x: x
    hidden = torch.ones(32, 8, dtype=torch.bfloat16)
    cache = (torch.zeros(1, 128, 1, 386, dtype=torch.int8),)
    metadata = SimpleNamespace(
        num_input_tokens=32, num_actual_tokens=32 if has_query else 0,
        cos=None, sin=None, block_table=torch.zeros(1, 1), dsa_cp_context=object(),
    )
    events = []
    current = (None, None, None, torch.empty(32, 576), torch.empty(64, 576))
    layer._compose_sfa_kv_cache = MagicMock(return_value=cache)
    layer._get_sfa_kv_slot_mapping = MagicMock(return_value=torch.arange(32))
    layer._get_indexer_attn_metadata = MagicMock(return_value=object())
    layer._get_parallel_forward_context = MagicMock(return_value=SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([32]), actual_seq_lengths_key=torch.tensor([64]),
        gather_full_o_proj=False, topk_num_tokens=32,
    ))
    layer._prepare_native_hidden_states = lambda x, m: x
    layer.fused_qkv_a_proj = MagicMock(return_value=(torch.zeros(32, 584),))
    layer._prepare_tq_current_workspace = MagicMock(return_value=current)
    layer._prepare_kv_for_parallel = MagicMock(return_value=(None, []))
    layer._q_proj_and_k_up_proj = MagicMock(return_value=(torch.empty(32, 64, 512), torch.empty(32, 64, 64)))
    layer.rope_single = lambda x, *args: x
    layer._record_query_gather_context = MagicMock()
    layer._store_parallel_kv = MagicMock(side_effect=lambda *args: (events.append("cache_written"), None))
    layer.indexer = MagicMock(side_effect=lambda *args, **kwargs: (events.append("indexer"), torch.zeros(1))[1])
    layer._prepare_indexer_metadata = MagicMock()
    layer._try_tq_prefill_workspace = MagicMock(side_effect=lambda *args, **kwargs: (
        events.append("attention"), torch.empty(32, 64, 512)
    )[1])
    layer._execute_sparse_flash_attention_process = MagicMock()
    layer._v_up_proj = MagicMock(return_value=torch.empty(32, 8192))
    layer._finalize_o_proj = MagicMock()
    handle = SimpleNamespace(wait=lambda: events.append("gather_wait"))
    with (
        patch("vllm_ascend.attention.sfa_v1.turboquant_kv_rmsnorm_rope", return_value=(None, None, None)) as kv,
        patch("vllm_ascend.attention.sfa_v1.all_gather_async", side_effect=lambda *args, **kwargs: (
            events.append("gather_start"), handle
        )) as gather,
        patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=object()),
        patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written", side_effect=lambda *args: events.append("notify")),
        patch("vllm_ascend.attention.sfa_v1.attention_transfer_window"),
        patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"),
    ):
        layer.forward(layer.layer_name, hidden, cache, metadata, output=torch.empty_like(hidden))
    assert kv.call_args.kwargs["current_kv"] is current[3]
    assert gather.call_args.kwargs["output"] is current[4]
    assert events == ["gather_start", "cache_written", "indexer", "notify", "gather_wait"] + (
        ["attention"] if has_query else []
    )
    layer._execute_sparse_flash_attention_process.assert_not_called()
