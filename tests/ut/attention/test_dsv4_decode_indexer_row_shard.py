# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.models.deepseek_v4.decode_row_shard import make_decode_indexer_row_plan
from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerOps, gather_decode_indexer_topk
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


def make_builder(rows_per_request=1):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v4", index_n_heads=64, index_head_dim=128, index_topk=4),
        enforce_eager=True,
    )
    builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        use_v2_model_runner=False,
    )
    builder.compressor_ratio = 4
    builder.speculative_config = (
        None if rows_per_request == 1 else SimpleNamespace(method="dspark", enable_adaptive_verification=False)
    )
    builder.decode_threshold = max(rows_per_request, 1)
    builder.num_prefills = 0
    builder.num_decodes = 13
    builder.num_decode_tokens = 13 * rows_per_request
    builder.num_actual_tokens = 13 * rows_per_request
    builder.device = torch.device("cpu")
    builder.block_table = torch.tensor([[20 - i, 40 + i] for i in range(13)], dtype=torch.int32)
    builder.seq_lens = torch.arange(103, 116, dtype=torch.int32)
    builder.decode_row_shard_seq_lens = torch.zeros(32, dtype=torch.int32)
    builder.decode_row_shard_k = torch.zeros(32, dtype=torch.int32)
    builder.decode_row_shard_residual_k = torch.zeros(32, dtype=torch.int32)
    builder.decode_row_shard_qli_metadata_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.decode_row_shard_plans = {}
    common = SimpleNamespace(
        num_reqs=13,
        causal=True,
        query_start_loc_cpu=torch.arange(14) * rows_per_request,
        num_input_tokens=builder.num_actual_tokens + 7,
    )
    return builder, common


def prepare(builder, common, full_graph=False, enabled=True, legacy_cp=False):
    with (
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=8),
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_rank", return_value=3),
        patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_config",
            return_value=SimpleNamespace(enable_dsa_cp=legacy_cp, enable_dsv4_decode_indexer_tp_shard=enabled),
        ),
    ):
        return builder._prepare_decode_row_shard(common, builder.seq_lens, common.num_reqs, full_graph)


@pytest.mark.parametrize("rows_per_request", [1, 2, 6])
def test_local_metadata_indexes_requests_and_preserves_all_dspark_rows(rows_per_request):
    builder, common = make_builder(rows_per_request)
    shard = prepare(builder, common)
    assert shard.plan.request_ids == (3, 11)
    assert shard.query_rows.numel() == 2 * rows_per_request
    assert torch.equal(shard.block_table, builder.block_table[[3, 11]])
    with (
        patch("vllm_ascend.attention.dsa_v1.DeviceOperator.get_dsa_indexer_quant_mode", return_value=2),
        patch.object(
            torch.ops._C_ascend,
            "npu_quant_lightning_indexer_v2_metadata",
            create=True,
            return_value=torch.ones(1024, dtype=torch.int32),
        ) as metadata_op,
    ):
        builder._build_decode_row_shard_qli_metadata(shard)
        kwargs = metadata_op.call_args.kwargs
        assert kwargs["batch_size"] == 2
        assert kwargs["max_seqlen_q"] == rows_per_request
        assert kwargs["max_seqlen_k"] == 114 // 4
        assert torch.equal(kwargs["cu_seqlens_q"], torch.tensor([0, rows_per_request, 2 * rows_per_request]))
        assert torch.equal(shard.qli_seqused_k, builder.seq_lens[[3, 11]] // 4)
        assert torch.equal(shard.qli_cmp_residual_k, builder.seq_lens[[3, 11]] % 4)
        builder.seq_lens.add_(2)
        next_shard = prepare(builder, common)
        assert next_shard.qli_metadata is shard.qli_metadata
        assert next_shard.query_rows is shard.query_rows
        builder._build_decode_row_shard_qli_metadata(next_shard)
        assert torch.equal(next_shard.qli_cmp_residual_k, builder.seq_lens[[3, 11]] % 4)


@pytest.mark.parametrize(
    "reason",
    [
        "graph",
        "full_graph",
        "adaptive",
        "noncausal",
        "prefill",
        "pcp",
        "dcp",
        "ragged",
        "v41",
        "disabled",
        "small_batch",
        "unsupported_spec",
        "legacy_cp",
    ],
)
def test_unsupported_layouts_fall_back(reason):
    builder, common = make_builder(6)
    if reason == "graph":
        builder.model_config.enforce_eager = False
    elif reason == "adaptive":
        builder.speculative_config.enable_adaptive_verification = True
    elif reason == "noncausal":
        common.causal = False
    elif reason == "prefill":
        builder.num_prefills = 1
    elif reason == "pcp":
        builder.vllm_config.parallel_config.prefill_context_parallel_size = 2
    elif reason == "dcp":
        builder.vllm_config.parallel_config.decode_context_parallel_size = 2
    elif reason == "ragged":
        common.query_start_loc_cpu[2] += 1
    elif reason == "v41":
        builder.model_config.hf_config.model_type = "deepseek_v41"
    elif reason == "small_batch":
        builder.num_decodes = common.num_reqs = 7
        builder.num_decode_tokens = builder.num_actual_tokens = 42
    elif reason == "unsupported_spec":
        builder.speculative_config.method = "eagle"
    assert prepare(builder, common, reason == "full_graph", reason != "disabled", reason == "legacy_cp") is None


@pytest.mark.parametrize("rows_per_request", [1, 6])
def test_collective_padding_and_inverse_mapping(rows_per_request):
    num_requests = 13
    plans = [make_decode_indexer_row_plan(num_requests, rows_per_request, 8, rank) for rank in range(8)]
    reference = torch.arange(num_requests * rows_per_request * 4, dtype=torch.int32).reshape(-1, 1, 4)
    cap = plans[0].max_local_rows
    gathered = torch.full((8 * cap, 1, 4), -1, dtype=torch.int32)
    for rank, plan in enumerate(plans):
        gathered[rank * cap : rank * cap + len(plan.query_rows)] = reference[list(plan.query_rows)]
    for rank, plan in enumerate(plans):
        group = SimpleNamespace(world_size=8, rank_in_group=rank, all_gather=MagicMock(return_value=gathered))
        with patch("vllm_ascend.models.deepseek_v4.indexer.get_tp_group", return_value=group):
            actual = gather_decode_indexer_topk(reference[list(plan.query_rows)], plan, torch.tensor(plan.restore_rows))
        assert torch.equal(actual, reference)
        sent = group.all_gather.call_args.args[0]
        assert sent.shape == (cap, 1, 4)
        assert torch.all(sent[len(plan.query_rows) :] == -1)


@pytest.mark.parametrize("rows_per_request", [1, 6])
def test_selection_keeps_complete_keys_and_uses_local_query_rows(rows_per_request):
    builder, common = make_builder(rows_per_request)
    shard = prepare(builder, common)
    num_rows = 13 * rows_per_request
    query = torch.arange(num_rows * 4, dtype=torch.int16).to(torch.int8).reshape(num_rows, 1, 4)
    weights = torch.arange(num_rows, dtype=torch.float16).reshape(num_rows, 1)
    scale = weights + 1
    key = torch.zeros((5, 8, 1, 4), dtype=torch.int8)
    key_scale = torch.ones((5, 8, 1, 1), dtype=torch.float16)
    local = torch.zeros((len(shard.plan.query_rows), 1, 4), dtype=torch.int32)
    expected = torch.zeros((num_rows, 1, 4), dtype=torch.int32)
    ops = AscendIndexerOps.__new__(AscendIndexerOps)
    ops.index_topk = 4
    ops.device_operator = SimpleNamespace(
        prepare_dsa_indexer_weights=lambda x: x,
        prepare_dsa_indexer_query_scale=lambda x: x,
        prepare_dsa_indexer_key_scale=lambda x: x,
        get_dsa_indexer_quant_mode=lambda: 2,
    )
    with (
        patch("vllm_ascend.models.deepseek_v4.indexer.wait_for_device_metadata") as wait,
        patch.object(
            torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", create=True, return_value=(local, None)
        ) as qli,
        patch("vllm_ascend.models.deepseek_v4.indexer.gather_decode_indexer_topk", return_value=expected) as gather,
    ):
        assert (
            ops.select_topk(query, weights, scale, key, key_scale, SimpleNamespace(decode_row_shard=shard)) is expected
        )
    kwargs = qli.call_args.kwargs
    rows = list(shard.plan.query_rows)
    assert torch.equal(kwargs["query"], query[rows])
    assert torch.equal(kwargs["weights"], weights[rows])
    assert torch.equal(kwargs["query_dequant_scale"], scale[rows])
    assert kwargs["key"] is key
    assert kwargs["block_table"] is shard.block_table
    wait.assert_called_once_with(DeviceMetadataStage.INDEXER, id(shard.qli_metadata))
    gather.assert_called_once_with(local, shard.plan, shard.restore_rows)


def test_deferred_producer_refreshes_local_metadata_and_preserves_consumer_key():
    builder, common = make_builder(6)
    common.query_start_loc = common.query_start_loc_cpu.to(torch.int32)
    builder.common_ratio_to_sas_metadata = {}
    builder.model_config.hf_config.sliding_window = 4096
    builder.start_pos_prefill = torch.zeros(13, dtype=torch.int32)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.empty(0, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.empty(0, dtype=torch.int32)
    builder._device_metadata_enabled = True
    builder.qli_metadata_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.qli_seqused_k = torch.zeros(32, dtype=torch.int32)
    builder.qli_cmp_residual_k = torch.zeros(32, dtype=torch.int32)
    builder.sas_metadata_buffer = torch.zeros(1024, dtype=torch.int32)
    builder.compressor_metadata_buffers = None
    builder.storage_block_size = 128
    builder.cache_group_key = "indexer-cache"
    cos = sin = torch.zeros((common.num_input_tokens, 1, 1, 4))
    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_config",
            return_value=SimpleNamespace(enable_dsa_cp=False, enable_dsv4_decode_indexer_tp_shard=True),
        ),
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=8),
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_rank", return_value=3),
        patch("vllm_ascend.attention.dsa_v1.DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv"),
        patch("vllm_ascend.attention.dsa_v1.DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv"),
        patch("vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa", return_value=(cos, sin)),
        patch.object(builder, "_build_sas_metadata", return_value=builder.sas_metadata_buffer),
        patch.object(builder, "_build_qli_metadata", return_value=builder.qli_metadata_buffer),
        patch.object(
            torch.ops._C_ascend,
            "npu_quant_lightning_indexer_v2_metadata",
            create=True,
            return_value=torch.ones(1024, dtype=torch.int32),
        ) as metadata_op,
    ):
        req_metadata = builder.build_req_metadata(common, builder.seq_lens, 13, cos, sin)
        shard = req_metadata.decode_row_shard
        assert shard is not None
        metadata_op.assert_not_called()
        tasks = builder.take_device_metadata_tasks()
        local_task = next(task for task in tasks if task.group_id == id(shard.qli_metadata))
        assert local_task.stage == DeviceMetadataStage.INDEXER
        local_task.run()
    assert torch.equal(shard.qli_seqused_k, builder.seq_lens[[3, 11]] // 4)
    assert torch.equal(shard.qli_cmp_residual_k, builder.seq_lens[[3, 11]] % 4)
    assert torch.all(shard.qli_metadata == 1)
    assert builder.take_device_metadata_tasks() == ()


def test_unsharded_selection_keeps_original_operator_metadata():
    query = torch.ones((3, 1, 4), dtype=torch.int8)
    weights = scale = torch.ones((3, 1), dtype=torch.float16)
    key = torch.zeros((5, 8, 1, 4), dtype=torch.int8)
    key_scale = torch.ones((5, 8, 1, 1), dtype=torch.float16)
    metadata = SimpleNamespace(
        qli_cu_seqlens_q=torch.arange(4, dtype=torch.int32),
        qli_seqused_k=torch.full((3,), 10, dtype=torch.int32),
        qli_cmp_residual_k=torch.zeros(3, dtype=torch.int32),
        block_table=torch.zeros((3, 1), dtype=torch.int32),
        qli_metadata=torch.ones(1024, dtype=torch.int32),
    )
    ops = AscendIndexerOps.__new__(AscendIndexerOps)
    ops.index_topk = 4
    ops.device_operator = SimpleNamespace(
        prepare_dsa_indexer_weights=lambda value: value,
        prepare_dsa_indexer_query_scale=lambda value: value,
        prepare_dsa_indexer_key_scale=lambda value: value,
        get_dsa_indexer_quant_mode=lambda: 2,
    )
    expected = torch.zeros((3, 1, 4), dtype=torch.int32)
    with (
        patch("vllm_ascend.models.deepseek_v4.indexer.wait_for_device_metadata") as wait,
        patch.object(
            torch.ops._C_ascend,
            "npu_quant_lightning_indexer_v2",
            create=True,
            return_value=(expected, None),
        ) as qli,
        patch("vllm_ascend.models.deepseek_v4.indexer.get_tp_group") as tp_group,
    ):
        assert ops.select_topk(query, weights, scale, key, key_scale, metadata) is expected
    assert qli.call_args.kwargs["query"] is query
    assert qli.call_args.kwargs["block_table"] is metadata.block_table
    assert qli.call_args.kwargs["metadata"] is metadata.qli_metadata
    wait.assert_called_once_with(DeviceMetadataStage.INDEXER, id(metadata.qli_metadata))
    tp_group.assert_not_called()


def test_cache_update_receives_all_rows_before_sharded_selection():
    builder, common = make_builder(6)
    shard = prepare(builder, common)
    query = torch.ones((78, 1, 4), dtype=torch.float16)
    key = torch.ones((26, 1, 4), dtype=torch.float16)
    weights = torch.ones((78, 1), dtype=torch.float16)
    key_cache = torch.zeros((32, 8, 1, 4), dtype=torch.int8)
    scale_cache = torch.ones((32, 8, 1, 1), dtype=torch.float16)
    slots = torch.arange(26, dtype=torch.int32)
    quantized = query.to(torch.int8)
    q_scale = torch.ones((78, 1), dtype=torch.float16)
    ops = AscendIndexerOps.__new__(AscendIndexerOps)
    scatter = MagicMock(return_value=(quantized, q_scale, None, None))
    ops.device_operator = SimpleNamespace(indexer_quant_scatter=scatter)
    metadata = SimpleNamespace(decode_row_shard=shard)
    with patch.object(ops, "select_topk", return_value=object()) as select:
        ops.quantize_update_cache_and_select_topk(query, key, weights, key_cache, scale_cache, None, slots, metadata)
    scatter.assert_called_once_with(query, key, key_cache, scale_cache, None, slots)
    select.assert_called_once_with(quantized, weights, q_scale, key_cache, scale_cache, metadata)
