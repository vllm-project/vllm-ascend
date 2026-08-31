from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import (
    AscendDSAImpl,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


def _make_decode_builder(compressor_ratio: int, enabled: bool):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": query_start_loc,
        "input_positions": torch.arange(2),
        "cos": torch.ones((2, 1)),
        "sin": torch.zeros((2, 1)),
        "query_start_loc_cpu": query_start_loc,
        "max_seq_lens": 9,
        "seq_lens_list": [8, 9],
        "max_seqlen_kv": 9,
        "max_seqlen_q": 1,
        "start_pos_decode": torch.tensor([7, 8], dtype=torch.int32),
    }
    builder.compressor_ratio = compressor_ratio
    builder.num_decodes = 2
    builder.num_decode_tokens = 2
    builder.seq_lens = torch.tensor([8, 9], dtype=torch.int32)
    builder.start_pos_decode = torch.zeros(2, dtype=torch.int32)
    builder.block_table = torch.zeros((2, 2), dtype=torch.int32)
    builder.slot_mapping = torch.zeros((2, 2), dtype=torch.int32)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_attention_heads=8,
            index_topk=512,
            index_n_heads=64,
            index_head_dim=128,
            sliding_window=4096,
        ),
        get_head_size=lambda: 192,
    )
    builder.seqused_q = torch.empty(0)
    builder.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.empty(0, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.empty(0, dtype=torch.int32)
    builder._device_metadata_enabled = enabled
    builder._device_metadata_tasks = ()
    builder.block_size = 128
    builder.cache_group_key = "group"
    builder.get_block_table_size = MagicMock(return_value=2)
    builder._num_compressor_metadata_rows = MagicMock(return_value=2)
    return builder


@pytest.mark.parametrize("compressor_ratio", [1, 4, 128])
@pytest.mark.parametrize("enabled", [False, True])
def test_decode_metadata_defers_device_work(
    compressor_ratio: int,
    enabled: bool,
):
    builder = _make_decode_builder(compressor_ratio, enabled)
    sas_output = torch.full((1024,), 3, dtype=torch.int32)
    qli_output = torch.full((1024,), 4, dtype=torch.int32)
    sas_op = MagicMock(return_value=sas_output)

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(torch.ones(1), torch.zeros(1)),
        ),
        patch.object(
            DeviceOperator,
            "pad_dsa_decode_slot_mapping",
            return_value=torch.zeros((2, 2), dtype=torch.int32),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=torch.tensor([0, 8, 17], dtype=torch.int32),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_cmp_kv",
            return_value=None,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=sas_op,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_kwargs",
            return_value={},
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            create=True,
            return_value=qli_output,
        ) as qli_op,
    ):
        metadata = builder.build_decode_metadata(0, SimpleNamespace(), 2)
        tasks = builder.take_device_metadata_tasks()
        assert builder.take_device_metadata_tasks() == ()

        if enabled:
            expected = (
                [DeviceMetadataStage.INDEXER, DeviceMetadataStage.ATTENTION]
                if compressor_ratio == 4
                else [DeviceMetadataStage.ATTENTION]
            )
            assert [task.stage for task in tasks] == expected
            assert [task.group_id for task in tasks] == (
                [id(builder.decode_qli_metadata), id(builder.decode_sas_metadata)]
                if compressor_ratio == 4
                else [id(builder.decode_sas_metadata)]
            )
            sas_op.assert_not_called()
            qli_op.assert_not_called()
            for task in tasks:
                task.run()
        else:
            assert tasks == ()

        sas_op.assert_called_once()
        assert sas_op.call_args.kwargs["cmp_ratio"] == compressor_ratio
        if enabled and compressor_ratio != 4:
            qli_op.assert_not_called()
        else:
            qli_op.assert_called_once()
        assert metadata.sas_metadata is builder.decode_sas_metadata
        assert metadata.qli_metadata is builder.decode_qli_metadata
        assert torch.equal(builder.decode_sas_metadata, sas_output)
        if not enabled or compressor_ratio == 4:
            assert torch.equal(builder.decode_qli_metadata, qli_output)


def _make_prefill_builder(compressor_ratio: int, enabled: bool):
    builder = _make_decode_builder(compressor_ratio, enabled)
    builder.prefill_ratio_to_sas_metadata = {
        "input_positions": torch.arange(3),
        "max_query_len": 2,
        "max_seq_lens": 3,
        "prefill_input_positions": torch.tensor([1, 2]),
        "prefill_query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "cos": torch.ones((2, 1)),
        "sin": torch.zeros((2, 1)),
        "prefill_seq_lens": torch.tensor([3], dtype=torch.int32),
        "num_prefill": 1,
    }
    builder.decode_ratio_to_sas_metadata = {}
    builder.num_decodes = 1
    builder.num_decode_tokens = 1
    builder.num_prefill_tokens = 2
    builder.num_actual_tokens = 3
    builder.query_lens = torch.tensor([1, 2], dtype=torch.int32)
    builder.seq_lens = torch.tensor([1, 3], dtype=torch.int32)
    builder.start_pos_prefill = torch.zeros(2, dtype=torch.int32)
    builder.block_table = torch.zeros((2, 2), dtype=torch.int32)
    builder.slot_mapping = torch.zeros((3, 2), dtype=torch.int32)
    builder.prefill_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.prefill_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    return builder


@pytest.mark.parametrize("compressor_ratio", [1, 4, 128])
@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_metadata_defers_device_work(
    compressor_ratio: int,
    enabled: bool,
):
    builder = _make_prefill_builder(compressor_ratio, enabled)
    sas_output = torch.full((1024,), 5, dtype=torch.int32)
    qli_output = torch.full((1024,), 6, dtype=torch.int32)
    sas_op = MagicMock(return_value=sas_output)
    common_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(torch.ones(1), torch.zeros(1)),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=sas_op,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_kwargs",
            return_value={},
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            create=True,
            return_value=qli_output,
        ) as qli_op,
    ):
        metadata = builder.build_prefill_metadata(0, common_metadata, 2)
        tasks = builder.take_device_metadata_tasks()

        if enabled:
            expected_stages = (
                [DeviceMetadataStage.INDEXER, DeviceMetadataStage.ATTENTION]
                if compressor_ratio == 4
                else [DeviceMetadataStage.ATTENTION]
            )
            expected_groups = (
                [id(builder.prefill_qli_metadata), id(builder.prefill_sas_metadata)]
                if compressor_ratio == 4
                else [id(builder.prefill_sas_metadata)]
            )
            assert [task.stage for task in tasks] == expected_stages
            assert [task.group_id for task in tasks] == expected_groups
            sas_op.assert_not_called()
            qli_op.assert_not_called()
            for task in tasks:
                task.run()
        else:
            assert tasks == ()

        sas_op.assert_called_once()
        assert sas_op.call_args.kwargs["cmp_ratio"] == compressor_ratio
        assert ("cmp_mask_mode" in sas_op.call_args.kwargs) == (compressor_ratio > 1)
        assert ("cmp_topk" in sas_op.call_args.kwargs) == (compressor_ratio == 4)
        if enabled and compressor_ratio != 4:
            qli_op.assert_not_called()
        else:
            qli_op.assert_called_once()
            assert qli_op.call_args.kwargs["max_seqlen_q"] == 2
            assert qli_op.call_args.kwargs["max_seqlen_k"] == 3
        if enabled:
            assert metadata.sas_metadata is builder.prefill_sas_metadata
            assert metadata.qli_metadata is builder.prefill_qli_metadata
            assert torch.equal(builder.prefill_sas_metadata, sas_output)
            if compressor_ratio == 4:
                assert torch.equal(builder.prefill_qli_metadata, qli_output)
        else:
            assert metadata.sas_metadata is sas_output
            assert metadata.qli_metadata is qli_output


def test_mixed_metadata_keeps_prefill_and_decode_groups_isolated():
    builder = _make_prefill_builder(4, True)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "input_positions": torch.arange(1),
        "cos": torch.ones((1, 1)),
        "sin": torch.zeros((1, 1)),
        "query_start_loc_cpu": torch.tensor([0, 1], dtype=torch.int32),
        "max_seq_lens": 1,
        "seq_lens_list": [1],
        "max_seqlen_kv": 1,
        "max_seqlen_q": 1,
        "start_pos_decode": torch.zeros(1, dtype=torch.int32),
    }
    sas_op = MagicMock(return_value=torch.ones(1024, dtype=torch.int32))
    qli_op = MagicMock(return_value=torch.ones(1024, dtype=torch.int32))

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(torch.ones(1), torch.zeros(1)),
        ),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_op", return_value=sas_op),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=torch.tensor([0, 1], dtype=torch.int32),
        ),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", return_value=None),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            create=True,
            new=qli_op,
        ),
    ):
        builder.build_prefill_metadata(
            0,
            SimpleNamespace(query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32)),
            2,
        )
        builder.build_decode_metadata(0, SimpleNamespace(), 2)
        tasks = builder.take_device_metadata_tasks()
        for task in tasks:
            task.run()

    assert [task.stage for task in tasks] == [
        DeviceMetadataStage.INDEXER,
        DeviceMetadataStage.ATTENTION,
        DeviceMetadataStage.INDEXER,
        DeviceMetadataStage.ATTENTION,
    ]
    assert [task.group_id for task in tasks] == [
        id(builder.prefill_qli_metadata),
        id(builder.prefill_sas_metadata),
        id(builder.decode_qli_metadata),
        id(builder.decode_sas_metadata),
    ]
    assert len(set(task.group_id for task in tasks)) == 4
    assert sas_op.call_count == 2
    assert qli_op.call_count == 2


@pytest.mark.parametrize("with_prefill", [False, True])
def test_indexer_waits_for_qli_consumer(with_prefill: bool):
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.index_topk = 512
    qli_metadata = torch.zeros(1024, dtype=torch.int32)
    metadata_value = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        qli_metadata=qli_metadata,
    )
    metadata = SimpleNamespace(
        decode=None if with_prefill else metadata_value,
        prefill=metadata_value if with_prefill else None,
    )
    calls = []

    with (
        patch.object(
            DeviceOperator,
            "prepare_dsa_indexer_weights",
            side_effect=lambda value: value,
        ),
        patch.object(
            DeviceOperator,
            "prepare_dsa_indexer_query_scale",
            side_effect=lambda value: value,
        ),
        patch.object(
            DeviceOperator,
            "prepare_dsa_indexer_key_scale",
            side_effect=lambda value: value,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.wait_for_device_metadata",
            side_effect=lambda stage, group_id: calls.append((stage, group_id)),
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer",
            create=True,
            side_effect=lambda **kwargs: (calls.append("indexer") or torch.zeros(1), None),
        ),
    ):
        impl._indexer_qli(
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones(1),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            metadata,
            with_prefill=with_prefill,
        )

    assert calls == [(DeviceMetadataStage.INDEXER, id(qli_metadata)), "indexer"]


@pytest.mark.parametrize("compressor_ratio", [1, 4, 128])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_consumers_wait_for_their_metadata(compressor_ratio: int, phase: str):
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.compress_ratio = compressor_ratio
    impl.multistream_dsv4_dsa_overlap = True
    impl.skip_topk = False
    impl.use_index_cache = False
    impl.window_size = 4096
    impl.compressor_overlap = False
    impl.compressor_wkv = SimpleNamespace(weight=torch.ones(1))
    impl.compressor_wgate = SimpleNamespace(weight=torch.ones(1))
    impl.compressor_ape = torch.ones(1)
    impl.compressor_norm = SimpleNamespace(weight=torch.ones(1))
    impl.rope_head_dim = 1
    impl.compressor_norm_eps = 1e-6
    impl.attn_sink = None
    impl.softmax_scale = 1.0
    impl.indexer_softmax_scale = 1.0
    impl.indexer_heads = 1
    impl.index_topk = 512
    impl.weights_proj = MagicMock(return_value=torch.ones((1, 1)))
    impl._mla_prolog_multistream = MagicMock(return_value=(torch.ones((1, 1, 1)), torch.ones((1, 1)), torch.ones(1)))
    impl.cv_indexer_select_qli = MagicMock(return_value=torch.ones((1, 1)))
    impl._compute_compressor_metadata = MagicMock(
        return_value=(torch.ones((1, 1)), torch.ones((1, 1)), torch.zeros(1, dtype=torch.int32))
    )

    sas_metadata = torch.zeros(1024, dtype=torch.int32)
    other_sas_metadata = torch.ones(1024, dtype=torch.int32)
    qli_metadata = torch.full((1024,), 2, dtype=torch.int32)
    common = SimpleNamespace(
        cos={"layer": torch.ones((1, 1))},
        sin={"layer": torch.ones((1, 1))},
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        start_pos=torch.zeros(1, dtype=torch.int32),
        sas_metadata=sas_metadata,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        cu_c4_cmp_seqlen_list=None,
        cu_c128_cmp_seqlen_list=None,
    )
    swa_values = dict(vars(common))
    swa_values.update(
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        ori_win_left=None,
        ori_win_right=None,
        dspark_swa_indices=None,
        sas_metadata=sas_metadata if compressor_ratio == 1 else other_sas_metadata,
    )
    swa = SimpleNamespace(**swa_values)
    compressor = SimpleNamespace(**vars(common))
    compressor_state = SimpleNamespace(block_table=common.block_table)
    indexer = SimpleNamespace(
        query_start_loc=common.query_start_loc,
        seq_lens=common.seq_lens,
        block_table=common.block_table,
        qli_metadata=qli_metadata,
    )

    def wrap(value):
        return SimpleNamespace(**{phase: value}, num_decode_tokens=1)

    if compressor_ratio == 1:
        attn_metadata = [wrap(swa)]
    elif compressor_ratio == 4:
        attn_metadata = [wrap(compressor), wrap(compressor_state), wrap(common), wrap(indexer), wrap(swa)]
    else:
        attn_metadata = [wrap(compressor), wrap(compressor_state), wrap(swa)]

    events = []
    stream = MagicMock()
    stream.record_event.return_value = object()
    attn_op = MagicMock(side_effect=lambda *args, **kwargs: (events.append("attention") or torch.ones(1),))

    with (
        patch.object(
            DeviceOperator,
            "unpack_dsa_forward_kv_cache",
            return_value=tuple(torch.ones((1, 1)) for _ in range(6)),
        ),
        patch.object(DeviceOperator, "dsa_kv_compress_scatter"),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_op", return_value=attn_op),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_base_kwargs", return_value={}),
        patch.object(DeviceOperator, "add_dsa_sparse_attn_extra_kwargs"),
        patch.object(DeviceOperator, "indexer_quantize_query", return_value=(torch.ones(1), torch.ones(1))),
        patch.object(DeviceOperator, "prepare_dsa_indexer_weights", side_effect=lambda value: value),
        patch.object(DeviceOperator, "prepare_dsa_indexer_query_scale", side_effect=lambda value: value),
        patch.object(DeviceOperator, "prepare_dsa_indexer_key_scale", side_effect=lambda value: value),
        patch.object(torch.npu, "current_stream", return_value=stream),
        patch("vllm_ascend.attention.dsa_v1.dsv4_dsa_overlap_stream", return_value=MagicMock()),
        patch("vllm_ascend.attention.dsa_v1.npu_stream_switch", return_value=nullcontext()),
        patch("vllm_ascend.attention.dsa_v1.notify_kv_cache_written"),
        patch(
            "vllm_ascend.attention.dsa_v1.record_attention_compute_start",
            side_effect=lambda: events.append("attention-start"),
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.wait_for_device_metadata",
            side_effect=lambda stage, group_id: events.append((stage, group_id)),
        ),
        patch.object(
            torch.ops._C_ascend,
            "compressor",
            create=True,
            return_value=torch.ones((1, 1)),
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer",
            create=True,
            side_effect=lambda **kwargs: (events.append("indexer") or torch.ones(1), None),
        ),
    ):
        getattr(impl, f"_forward_{phase}")("layer", torch.ones((1, 1)), tuple(), attn_metadata)

    expected_sas = sas_metadata if compressor_ratio == 1 else compressor.sas_metadata
    expected = []
    if compressor_ratio == 4:
        expected.extend([(DeviceMetadataStage.INDEXER, id(qli_metadata)), "indexer"])
    expected.extend(
        [
            (DeviceMetadataStage.ATTENTION, id(expected_sas)),
            "attention-start",
            "attention",
        ]
    )
    assert events == expected
