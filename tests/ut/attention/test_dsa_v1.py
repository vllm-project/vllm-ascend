from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPImpl,
    AscendDSACPMetadataBuilder,
    DSACPMetadata,
)
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSAMetadata as AscendDSACPMetadata,
)
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSAReqMetadata as AscendDSACPReqMetadata,
)
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAImpl,
    AscendDSAMetadataBuilder,
    build_compressor_metadata_out,
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
    builder.prefill_compressor_metadata_buffers = None
    builder.decode_compressor_metadata_buffers = None
    if enabled and compressor_ratio > 1:
        builder.prefill_compressor_metadata_buffers = tuple(torch.empty((8, 2), dtype=torch.int32) for _ in range(3))
        builder.decode_compressor_metadata_buffers = tuple(torch.empty((8, 2), dtype=torch.int32) for _ in range(3))
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
                list(DeviceMetadataStage)
                if compressor_ratio == 4
                else [DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.ATTENTION]
                if compressor_ratio > 1
                else [DeviceMetadataStage.ATTENTION]
            )
            assert [task.stage for task in tasks] == expected
            expected_groups = []
            if compressor_ratio > 1:
                assert builder.decode_compressor_metadata_buffers is not None
                expected_groups.append(id(builder.decode_compressor_metadata_buffers[0]))
            if compressor_ratio == 4:
                expected_groups.append(id(builder.decode_qli_metadata))
            expected_groups.append(id(builder.decode_sas_metadata))
            assert [task.group_id for task in tasks] == expected_groups
            sas_op.assert_not_called()
            qli_op.assert_not_called()
            with patch("vllm_ascend.attention.dsa_v1.build_compressor_metadata_out"):
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
                list(DeviceMetadataStage)
                if compressor_ratio == 4
                else [DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.ATTENTION]
                if compressor_ratio > 1
                else [DeviceMetadataStage.ATTENTION]
            )
            expected_groups = []
            if compressor_ratio > 1:
                assert builder.prefill_compressor_metadata_buffers is not None
                expected_groups.append(id(builder.prefill_compressor_metadata_buffers[0]))
            if compressor_ratio == 4:
                expected_groups.append(id(builder.prefill_qli_metadata))
            expected_groups.append(id(builder.prefill_sas_metadata))
            assert [task.stage for task in tasks] == expected_stages
            assert [task.group_id for task in tasks] == expected_groups
            sas_op.assert_not_called()
            qli_op.assert_not_called()
            with patch("vllm_ascend.attention.dsa_v1.build_compressor_metadata_out"):
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
        with patch("vllm_ascend.attention.dsa_v1.build_compressor_metadata_out"):
            for task in tasks:
                task.run()

    ordered_tasks = sorted(tasks, key=lambda task: task.stage)
    assert [task.stage for task in ordered_tasks] == [
        DeviceMetadataStage.COMPRESSOR,
        DeviceMetadataStage.COMPRESSOR,
        DeviceMetadataStage.INDEXER,
        DeviceMetadataStage.INDEXER,
        DeviceMetadataStage.ATTENTION,
        DeviceMetadataStage.ATTENTION,
    ]
    assert {task.group_id for task in tasks} == {
        id(builder.prefill_compressor_metadata_buffers[0]),
        id(builder.decode_compressor_metadata_buffers[0]),
        id(builder.prefill_qli_metadata),
        id(builder.decode_qli_metadata),
        id(builder.prefill_sas_metadata),
        id(builder.decode_sas_metadata),
    }
    assert sas_op.call_count == 2
    assert qli_op.call_count == 2


def test_build_compressor_metadata_out_uses_fixed_outputs():
    metadata = SimpleNamespace(
        full_compress_cos=torch.ones((8, 1, 1, 4)),
        full_compress_sin=torch.zeros((8, 1, 1, 4)),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        start_pos=torch.tensor([1], dtype=torch.int32),
        block_table=torch.tensor([[3]], dtype=torch.int32),
        block_size=128,
        num_reqs_actual=1,
    )
    outputs = (
        torch.empty((2, 1, 1, 4)),
        torch.empty((2, 1, 1, 4)),
        torch.empty((2, 2), dtype=torch.int32),
    )

    with (
        patch.object(DeviceOperator, "get_dsa_compressor_slot_mapping_format", return_value=2),
        patch.object(torch.ops._C_ascend, "compressor_metadata_out", create=True) as metadata_out,
    ):
        build_compressor_metadata_out(metadata, 4, outputs)

    assert metadata_out.call_args.args[-3:] == outputs


@pytest.mark.parametrize(
    ("mode", "allocates_buffers"),
    [
        (CUDAGraphMode.NONE, True),
        (CUDAGraphMode.FULL_AND_PIECEWISE, True),
        (CUDAGraphMode.FULL, False),
    ],
)
def test_enable_device_metadata_keeps_pure_full_compressor_legacy(
    mode: CUDAGraphMode,
    allocates_buffers: bool,
):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder._device_metadata_enabled = False
    builder.compressor_ratio = 4
    builder.device = torch.device("cpu")
    builder.slot_mapping_shape = (8, 2)
    builder.model_config = SimpleNamespace(hf_config=SimpleNamespace(qk_rope_head_dim=4))
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    builder.prefill_compressor_metadata_buffers = None
    builder.decode_compressor_metadata_buffers = None

    builder.enable_device_metadata()

    assert builder._device_metadata_enabled
    assert (builder.prefill_compressor_metadata_buffers is not None) is allocates_buffers
    assert (builder.decode_compressor_metadata_buffers is not None) is allocates_buffers
    if allocates_buffers:
        assert builder.prefill_compressor_metadata_buffers is not None
        assert builder.decode_compressor_metadata_buffers is not None
        for buffers in (
            builder.prefill_compressor_metadata_buffers,
            builder.decode_compressor_metadata_buffers,
        ):
            assert buffers[0].shape == (8, 1, 1, 4)
            assert buffers[1].shape == (8, 1, 1, 4)
            assert buffers[2].shape == (8, 2)
            assert buffers[0].dtype == buffers[1].dtype == torch.float32
            assert buffers[2].dtype == torch.int32


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_full_graph_compressor_uses_stable_padded_extent(phase: str):
    builder = _make_prefill_builder(4, True)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": torch.tensor([0, 1, 2], dtype=torch.int32),
        "input_positions": torch.arange(2),
        "cos": torch.ones((2, 1)),
        "sin": torch.zeros((2, 1)),
        "query_start_loc_cpu": torch.tensor([0, 1, 2], dtype=torch.int32),
        "max_seq_lens": 9,
        "seq_lens_list": [8, 9],
        "max_seqlen_kv": 9,
        "max_seqlen_q": 1,
        "start_pos_decode": torch.tensor([7, 8], dtype=torch.int32),
    }
    common = SimpleNamespace(
        num_input_tokens=8,
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
            return_value=MagicMock(return_value=torch.ones(1024, dtype=torch.int32)),
        ),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=torch.tensor([0, 8, 17], dtype=torch.int32),
        ),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", return_value=None),
    ):
        if phase == "prefill":
            capture_metadata = builder.build_prefill_metadata(0, common, 2, full_graph_mode=True)
            metadata = builder.build_prefill_metadata(0, common, 1, full_graph_mode=True)
            buffers = builder.prefill_compressor_metadata_buffers
            expected_rows = 3
            expected_reqs = 1
        else:
            builder.num_decodes = 2
            builder.num_decode_tokens = 2
            capture_metadata = builder.build_decode_metadata(0, common, 2, full_graph_mode=True)
            metadata = builder.build_decode_metadata(0, common, 1, full_graph_mode=True)
            buffers = builder.decode_compressor_metadata_buffers
            expected_rows = 4
            expected_reqs = 2

    assert buffers is not None
    assert metadata.num_compressed_tokens == expected_rows
    assert metadata.num_reqs_actual == expected_reqs
    assert metadata.compressor_metadata is not None
    assert metadata.compressor_metadata[0].shape[0] == expected_rows
    assert metadata.compressor_metadata[0].data_ptr() == buffers[0].data_ptr()
    assert capture_metadata.compressor_metadata is not None
    assert capture_metadata.compressor_metadata[0].shape == metadata.compressor_metadata[0].shape
    assert capture_metadata.compressor_metadata[0].data_ptr() == metadata.compressor_metadata[0].data_ptr()


def test_compressor_consumer_waits_only_for_precomputed_metadata():
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.compress_ratio = 4
    outputs = (torch.ones(1), torch.zeros(1), torch.zeros(1, dtype=torch.int32))
    precomputed = SimpleNamespace(
        compressor_metadata=outputs,
        compressor_metadata_group_id=17,
    )
    legacy = SimpleNamespace(
        compressor_metadata=None,
        compressor_metadata_group_id=None,
    )

    with (
        patch("vllm_ascend.attention.dsa_v1.wait_for_device_metadata") as wait,
        patch(
            "vllm_ascend.attention.dsa_v1.get_or_compute_compressor_metadata",
            return_value=(torch.ones(1), torch.ones(1), torch.ones(1)),
        ) as legacy_compute,
    ):
        assert impl._compute_compressor_metadata(precomputed) is outputs
        impl._compute_compressor_metadata(legacy)

    wait.assert_called_once_with(DeviceMetadataStage.COMPRESSOR, 17)
    legacy_compute.assert_called_once_with(legacy, 4)


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


@pytest.mark.parametrize(
    ("compressor_ratio", "enabled", "expected_stages"),
    [
        (1, True, [DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.ATTENTION]),
        (4, True, [DeviceMetadataStage.COMPRESSOR, *list(DeviceMetadataStage)]),
        (128, True, [DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.ATTENTION]),
        (4, False, []),
    ],
)
def test_dsa_cp_defers_device_metadata(
    compressor_ratio: int,
    enabled: bool,
    expected_stages: list[DeviceMetadataStage],
):
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.num_prefills = 1
    builder.num_actual_tokens = 3
    builder.compressor_ratio = compressor_ratio
    builder.block_size = 128
    builder.cache_group_key = "group"
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.seq_lens_cpu = builder.seq_lens
    builder.block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    builder.start_pos_prefill = torch.zeros(2, dtype=torch.int32)
    builder.slot_mapping = torch.zeros((3, 2), dtype=torch.int32)
    builder.compressor_metadata_buffers = (
        (torch.empty((3, 1)), torch.empty((3, 1)), builder.slot_mapping) if enabled and compressor_ratio > 1 else None
    )
    builder.req_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder._device_metadata_enabled = enabled
    builder._device_metadata_tasks = ()
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_attention_heads=64, index_topk=512),
        get_head_size=lambda: 512,
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    builder.common_ratio_to_sas_metadata = {
        "_cpu_local": {
            "qsl_cpu": query_start_loc,
            "sl_cpu": builder.seq_lens,
        }
    }
    events = []

    def build_local() -> None:
        events.append("local")
        builder.start_pos_prefill.fill_(1)

    build_local_metadata = MagicMock(side_effect=build_local) if enabled else None
    builder._ensure_device_local_metadata = MagicMock(
        return_value=(0, 3, 3, 3, query_start_loc, builder.seq_lens, build_local_metadata)
    )
    builder._get_cmp_seqlens_for_metadata = MagicMock(return_value=None)
    builder._build_sas_metadata = MagicMock(side_effect=lambda **_: events.append("sas") or builder.req_sas_metadata)
    builder._build_qli_metadata = MagicMock(
        side_effect=lambda **_: events.append("qli") or (builder.req_qli_metadata if compressor_ratio == 4 else None)
    )
    common_metadata = SimpleNamespace(
        num_reqs=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )

    with patch(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_full_cos_and_sin_dsa",
        return_value=(torch.ones(1), torch.zeros(1)),
    ):
        metadata = builder.build_req_metadata(
            common_metadata,
            input_positions=None,
            num_input_tokens=3,
            num_reqs_actual=1,
            attn_state=MagicMock(),
            cos=MagicMock(),
            sin=MagicMock(),
        )

    tasks = builder.take_device_metadata_tasks()
    assert builder.take_device_metadata_tasks() == ()
    assert [task.stage for task in tasks] == expected_stages
    expected_group_ids = [id(builder.req_sas_metadata)] * len(tasks)
    if builder.compressor_metadata_buffers is not None:
        expected_group_ids[1] = id(builder.compressor_metadata_buffers[0])
    if compressor_ratio == 4 and enabled:
        expected_group_ids[-2] = id(builder.req_qli_metadata)
    assert [task.group_id for task in tasks] == expected_group_ids
    assert metadata.device_local_metadata_group_id == (id(builder.req_sas_metadata) if enabled else None)
    assert metadata.sas_metadata is builder.req_sas_metadata
    assert (metadata.qli_metadata is builder.req_qli_metadata) is (compressor_ratio == 4)
    assert (metadata.compressor_metadata is not None) is (builder.compressor_metadata_buffers is not None)
    if enabled:
        build_local_metadata.assert_not_called()
        builder._build_sas_metadata.assert_not_called()
        builder._build_qli_metadata.assert_not_called()
        with patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.build_compressor_metadata_out",
            side_effect=lambda *_: events.append("compressor"),
        ):
            for task in tasks:
                task.run()
        build_local_metadata.assert_called_once_with()
        assert torch.equal(builder.start_pos_prefill, torch.tensor([1, 0], dtype=torch.int32))
        expected_events = ["local"]
        if compressor_ratio > 1:
            expected_events.append("compressor")
        if compressor_ratio == 4:
            expected_events.append("qli")
        expected_events.append("sas")
        assert events == expected_events
    builder._build_sas_metadata.assert_called_once()
    if compressor_ratio == 4:
        builder._build_qli_metadata.assert_called_once()
        assert builder._build_qli_metadata.call_args.kwargs["max_seqlen_q"] == 2
        assert builder._build_qli_metadata.call_args.kwargs["max_seqlen_k"] == 8
    else:
        builder._build_qli_metadata.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "ratio", "allocates_buffers"),
    [
        (CUDAGraphMode.FULL, 4, False),
        (CUDAGraphMode.PIECEWISE, 4, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, 4, True),
        (CUDAGraphMode.FULL_AND_PIECEWISE, 128, True),
        (CUDAGraphMode.NONE, 1, False),
    ],
)
def test_dsa_cp_enable_device_metadata_allocates_compressor_buffers(
    mode: CUDAGraphMode,
    ratio: int,
    allocates_buffers: bool,
):
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder._device_metadata_enabled = False
    builder.compressor_ratio = ratio
    builder.compressor_metadata_buffers = None
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    builder.model_config = SimpleNamespace(hf_config=SimpleNamespace(qk_rope_head_dim=4))
    builder.device = torch.device("cpu")
    builder.slot_mapping = torch.empty((8, 2), dtype=torch.int32)

    builder.enable_device_metadata()

    assert builder._device_metadata_enabled
    assert (builder.compressor_metadata_buffers is not None) is allocates_buffers
    if allocates_buffers:
        assert builder.compressor_metadata_buffers is not None
        cos, sin, slot = builder.compressor_metadata_buffers
        assert cos.shape == sin.shape == (8, 1, 1, 4)
        assert cos.dtype == sin.dtype == torch.float32
        assert slot is builder.slot_mapping
        assert slot.shape == (8, 2)
        assert slot.dtype == torch.int32


def test_dsa_cp_full_graph_compressor_uses_stable_bucket_extent():
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.num_prefills = 0
    builder.num_actual_tokens = 4
    builder.compressor_ratio = 4
    builder.block_size = 128
    builder.cache_group_key = "group"
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.seq_lens_cpu = builder.seq_lens
    builder.block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    builder.start_pos_prefill = torch.zeros(2, dtype=torch.int32)
    builder.slot_mapping = torch.zeros((4, 2), dtype=torch.int32)
    builder.compressor_metadata_buffers = (
        torch.empty((4, 1)),
        torch.empty((4, 1)),
        builder.slot_mapping,
    )
    base_pointers = tuple(buffer.data_ptr() for buffer in builder.compressor_metadata_buffers)
    builder.req_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder._device_metadata_enabled = True
    builder._device_metadata_tasks = ()
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_attention_heads=64, index_topk=512),
        get_head_size=lambda: 512,
    )
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    builder.common_ratio_to_sas_metadata = {"_cpu_local": {"qsl_cpu": query_start_loc, "sl_cpu": builder.seq_lens}}
    builder._ensure_device_local_metadata = MagicMock(
        return_value=(0, 4, 4, 4, query_start_loc, builder.seq_lens, MagicMock())
    )
    builder._get_cmp_seqlens_for_metadata = MagicMock(return_value=None)
    builder._build_sas_metadata = MagicMock(return_value=builder.req_sas_metadata)
    builder._build_qli_metadata = MagicMock(return_value=builder.req_qli_metadata)
    common_metadata = SimpleNamespace(
        num_reqs=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )

    with patch(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_full_cos_and_sin_dsa",
        return_value=(torch.ones(1), torch.zeros(1)),
    ):
        capture = builder.build_req_metadata(
            common_metadata, None, 4, 2, MagicMock(), MagicMock(), MagicMock(), full_graph_mode=True
        )
        builder.num_actual_tokens = 3
        runtime = builder.build_req_metadata(
            common_metadata, None, 4, 1, MagicMock(), MagicMock(), MagicMock(), full_graph_mode=True
        )

    assert capture.num_compressed_tokens == runtime.num_compressed_tokens == 3
    assert capture.num_reqs_actual == runtime.num_reqs_actual == 2
    assert capture.compressor_metadata is not None
    assert runtime.compressor_metadata is not None
    for index, (capture_output, runtime_output) in enumerate(
        zip(capture.compressor_metadata, runtime.compressor_metadata)
    ):
        assert capture_output.shape == runtime_output.shape == (3, 1 if index < 2 else 2)
        assert capture_output.data_ptr() == runtime_output.data_ptr() == base_pointers[index]


def test_dsa_cp_device_local_metadata_is_deferred_and_reused():
    cache = {}

    def make_builder():
        builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
        builder._device_metadata_enabled = True
        builder.common_ratio_to_sas_metadata = cache
        builder.local_query_start_loc = torch.zeros(3, dtype=torch.int32)
        builder.local_seq_lens = torch.zeros(2, dtype=torch.int32)
        builder.start_pos_prefill = torch.zeros(2, dtype=torch.int32)
        return builder

    first_builder = make_builder()
    second_builder = make_builder()
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    first_addresses = (
        first_builder.local_query_start_loc.data_ptr(),
        first_builder.local_seq_lens.data_ptr(),
        first_builder.start_pos_prefill.data_ptr(),
    )

    with patch(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        return_value=SimpleNamespace(world_size=2, rank_in_group=0),
    ):
        first = first_builder._ensure_device_local_metadata(2, 4, query_start_loc, seq_lens)
        second = second_builder._ensure_device_local_metadata(2, 4, query_start_loc, seq_lens)
        first[-1]()
        second[-1]()

    assert first[:4] == (0, 2, 2, 4)
    assert torch.equal(first_builder.local_query_start_loc, torch.tensor([0, 2, 2], dtype=torch.int32))
    assert torch.equal(first_builder.local_seq_lens, torch.tensor([2, 0], dtype=torch.int32))
    assert torch.equal(first_builder.start_pos_prefill, torch.tensor([0, 2], dtype=torch.int32))
    assert torch.equal(second_builder.local_query_start_loc, first_builder.local_query_start_loc)
    assert torch.equal(second_builder.local_seq_lens, first_builder.local_seq_lens)
    assert torch.equal(second_builder.start_pos_prefill, first_builder.start_pos_prefill)
    assert cache["_device_local"]["qsl"].data_ptr() == first_addresses[0]
    assert first_addresses == (
        first_builder.local_query_start_loc.data_ptr(),
        first_builder.local_seq_lens.data_ptr(),
        first_builder.start_pos_prefill.data_ptr(),
    )


def test_dsa_cp_qli_metadata_uses_host_maxima():
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.compressor_ratio = 4
    builder.common_ratio_to_sas_metadata = {}
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.seqused_q = torch.empty(0)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
        )
    )
    seq_lens = MagicMock()
    seq_lens.clone.return_value = torch.tensor([8, 6], dtype=torch.int32)
    generated_metadata = torch.arange(1024, dtype=torch.int32)

    with patch.object(
        torch.ops._C_ascend,
        "npu_vllm_quant_lightning_indexer_metadata",
        create=True,
        return_value=generated_metadata,
    ) as metadata_op:
        builder._build_qli_metadata(
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            seq_lens=seq_lens,
            num_reqs=2,
            max_seqlen_q=2,
            max_seqlen_k=8,
        )

    seq_lens.max.assert_not_called()
    assert metadata_op.call_args.kwargs["max_seqlen_q"] == 2
    assert metadata_op.call_args.kwargs["max_seqlen_k"] == 8


def test_dsa_cp_legacy_compressor_waits_for_local_metadata():
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.compress_ratio = 4
    metadata = SimpleNamespace(
        compressor_metadata=None,
        compressor_metadata_group_id=None,
        device_local_metadata_group_id=23,
    )
    result = (torch.ones(1), torch.zeros(1), torch.zeros(1, dtype=torch.int32))
    events = []

    with (
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.wait_for_device_metadata",
            side_effect=lambda stage, group_id: events.append((stage, group_id)),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.get_or_compute_compressor_metadata",
            side_effect=lambda *_: events.append("compressor") or result,
        ),
    ):
        assert impl._compute_compressor_metadata(metadata) is result

    assert events == [(DeviceMetadataStage.COMPRESSOR, 23), "compressor"]


def test_dsa_cp_compressor_waits_for_precomputed_metadata():
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    outputs = (torch.ones(1), torch.zeros(1), torch.zeros(1, dtype=torch.int32))
    metadata = SimpleNamespace(
        compressor_metadata=outputs,
        compressor_metadata_group_id=29,
        device_local_metadata_group_id=23,
    )

    with (
        patch("vllm_ascend.attention.context_parallel.dsa_cp.wait_for_device_metadata") as wait,
        patch("vllm_ascend.attention.context_parallel.dsa_cp.get_or_compute_compressor_metadata") as legacy,
    ):
        assert impl._compute_compressor_metadata(metadata) is outputs

    wait.assert_called_once_with(DeviceMetadataStage.COMPRESSOR, 29)
    legacy.assert_not_called()


def _make_dsa_cp_metadata(sas_metadata: torch.Tensor) -> AscendDSACPMetadata:
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)
    cp_metadata = DSACPMetadata(
        local_query_start_loc=query_start_loc,
        local_seq_lens=seq_lens,
        local_start=0,
        local_end=1,
        tokens_per_rank=1,
        num_tokens_pad=1,
        local_sin={"layer": torch.zeros((1, 1, 1, 2))},
        local_cos={"layer": torch.ones((1, 1, 1, 2))},
    )
    req_metadata = AscendDSACPReqMetadata(
        input_positions=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=seq_lens,
        slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
        block_size=128,
        query_start_loc=query_start_loc,
        cp_metadata=cp_metadata,
        sin={"layer": torch.zeros((1, 1, 1, 2))},
        cos={"layer": torch.ones((1, 1, 1, 2))},
        start_pos=torch.zeros(1, dtype=torch.int32),
        sas_metadata=sas_metadata,
        qli_metadata=torch.zeros(1024, dtype=torch.int32),
        cu_cmp_seqlen_list=torch.tensor([0, 1], dtype=torch.int32),
    )
    return AscendDSACPMetadata(
        num_actual_tokens=1,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_tables=req_metadata.block_table,
        sin=req_metadata.sin,
        cos=req_metadata.cos,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        req_metadata=req_metadata,
    )


def test_dsa_cp_indexer_waits_before_qli_consumer():
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.inderxer_wq_b = MagicMock(return_value=torch.ones((1, 2)))
    impl.inderxer_wq_b.quant_method = MagicMock()
    impl.indexer_heads = 1
    impl.indexcom_head_dim = 2
    impl.rope_head_dim = 1
    impl.weights_proj = MagicMock(return_value=torch.ones((1, 1)))
    impl.indexer_softmax_scale = 1.0
    impl.index_topk = 1
    qli_metadata = torch.zeros(1024, dtype=torch.int32)
    indexer_metadata = _make_dsa_cp_metadata(torch.zeros(1024, dtype=torch.int32))
    assert indexer_metadata.req_metadata is not None
    indexer_metadata.req_metadata.qli_metadata = qli_metadata
    indexer_metadata.hadamard = torch.eye(2)
    attn_metadata = [None, None, None, indexer_metadata, None]
    events = []

    with (
        patch.object(
            DeviceOperator,
            "unpack_dsa_indexer_kv_cache",
            return_value=(torch.empty(0),) * 4,
        ),
        patch.object(DeviceOperator, "indexer_quantize_query", return_value=(torch.ones((1, 1, 2)), torch.ones(1))),
        patch.object(DeviceOperator, "prepare_dsa_indexer_weights", side_effect=lambda value: value),
        patch.object(DeviceOperator, "prepare_dsa_indexer_query_scale", side_effect=lambda value: value),
        patch.object(DeviceOperator, "prepare_dsa_indexer_key_scale", side_effect=lambda value: value),
        patch.object(torch.ops._C_ascend, "inplace_partial_rotary_mul", create=True),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer",
            create=True,
            side_effect=lambda **kwargs: (events.append("indexer") or torch.zeros((1, 1, 1)), None),
        ),
        patch("vllm_ascend.attention.context_parallel.dsa_cp.rotate_activation", side_effect=lambda value, _: value),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.wait_for_device_metadata",
            side_effect=lambda stage, group_id: events.append((stage, group_id)),
        ),
    ):
        impl._indexer_select_topk(
            x=torch.ones((1, 2)),
            qr=torch.ones((1, 2)),
            kv_cache=(torch.empty(0),),
            attn_metadata=attn_metadata,
            cos=torch.ones((1, 1, 1, 2)),
            sin=torch.zeros((1, 1, 1, 2)),
            actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([1], dtype=torch.int32),
        )

    assert events == [(DeviceMetadataStage.INDEXER, id(qli_metadata)), "indexer"]


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_dsa_cp_attention_waits_before_memcache_fence(compress_ratio: int):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.compress_ratio = compress_ratio
    impl.num_heads = 1
    impl.head_dim = 2
    impl.nope_head_dim = 1
    impl.rope_head_dim = 1
    impl.window_size = 16
    impl.softmax_scale = 1.0
    impl.eps = 1e-6
    impl.attn_sink = None
    impl.compressor_overlap = False
    impl.wq_a = MagicMock(return_value=torch.ones((1, 2)))
    impl.q_norm = MagicMock(side_effect=lambda value: value)
    impl.wq_b = MagicMock(return_value=torch.ones((1, 2)))
    impl.wq_b.quant_method = MagicMock()
    impl.q_norm_without_weight = MagicMock()
    impl.wkv = MagicMock(return_value=torch.ones((1, 2)))
    impl.kv_norm = MagicMock(side_effect=lambda value: value)
    impl._maybe_all_gather_o_proj_full_weight = MagicMock(return_value=None)
    impl.compressor_wkv = SimpleNamespace(weight=torch.empty(0))
    impl.compressor_wgate = SimpleNamespace(weight=torch.empty(0))
    impl.compressor_ape = torch.empty(0)
    impl.compressor_norm = SimpleNamespace(weight=torch.empty(0))
    impl.compressor_norm_eps = 1e-6

    swa_sas = torch.zeros(1024, dtype=torch.int32)
    compressor_sas = torch.ones(1024, dtype=torch.int32)
    swa_metadata = _make_dsa_cp_metadata(swa_sas)
    compressor_metadata = _make_dsa_cp_metadata(compressor_sas)
    if compress_ratio == 1:
        attn_metadata = [swa_metadata]
        expected_sas = swa_sas
    elif compress_ratio == 4:
        attn_metadata = [
            compressor_metadata,
            compressor_metadata,
            compressor_metadata,
            compressor_metadata,
            swa_metadata,
        ]
        expected_sas = compressor_sas
    else:
        attn_metadata = [compressor_metadata, compressor_metadata, swa_metadata]
        expected_sas = compressor_sas
    events = []

    def run_attention(*args, **kwargs):
        assert kwargs["metadata"] is expected_sas
        events.append("attention")
        return (torch.ones((1, 1, 2)),)

    def add_extra_kwargs(extra_kwargs: dict[str, Any], **kwargs) -> None:
        extra_kwargs.update(kwargs)

    with (
        patch.object(
            DeviceOperator,
            "unpack_dsa_forward_kv_cache",
            return_value=(torch.empty(0), torch.empty(0), torch.zeros((1, 1, 1)), None, None, None),
        ),
        patch.object(DeviceOperator, "apply_dsa_q_rms", side_effect=lambda value, *_: value),
        patch.object(DeviceOperator, "dsa_kv_compress_scatter"),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_op", return_value=run_attention),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_base_kwargs", return_value={}),
        patch.object(DeviceOperator, "add_dsa_sparse_attn_extra_kwargs", side_effect=add_extra_kwargs),
        patch.object(
            torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", create=True, side_effect=lambda value, _: value
        ),
        patch.object(torch.ops._C_ascend, "inplace_partial_rotary_mul", create=True),
        patch.object(torch.ops._C_ascend, "compressor", create=True, return_value=torch.empty(0)),
        patch.object(impl, "_update_indexer_cache"),
        patch.object(impl, "_indexer_select_topk", return_value=torch.zeros((1, 1, 1), dtype=torch.int32)),
        patch.object(
            impl,
            "_compute_compressor_metadata",
            return_value=(
                torch.ones((1, 2)),
                torch.zeros((1, 2)),
                torch.zeros((1, 2), dtype=torch.int32),
            ),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.wait_for_device_metadata",
            side_effect=lambda stage, group_id: events.append((stage, group_id)),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.notify_kv_cache_written",
            side_effect=lambda *_: events.append("cache-written"),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.record_attention_compute_start",
            side_effect=lambda: events.append("attention-start"),
        ),
    ):
        impl._forward(
            "layer",
            torch.ones((1, 2)),
            (torch.empty(0),),
            attn_metadata,
        )

    assert events == [
        "cache-written",
        (DeviceMetadataStage.ATTENTION, id(expected_sas)),
        "attention-start",
        "attention",
    ]
