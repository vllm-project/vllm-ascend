#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPImpl,
    AscendDSACPLayerMetadata,
    AscendDSACPMetadataBuilder,
    AscendDSAPCPImpl,
    AscendDSAPCPMetadata,
    AscendDSAPCPMetadataBuilder,
    DSACPMetadata,
)
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSAMetadata as AscendDSACPMetadata,
)
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSAReqMetadata as AscendDSACPReqMetadata,
)
from vllm_ascend.attention.dsa_v1 import (
    DSA_METADATA_BUFFER_SIZE,
    AscendDSABackend,
    AscendDSAImpl,
    AscendDSALayerMetadata,
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
    AscendDSAReqMetadata,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import (
    AscendIndexerMetadata,
    IndexerOverlapPlan,
)
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
)
from vllm_ascend.worker.v2.pcp_manager import (
    AscendPCPAttentionContext,
    AscendPCPManager,
)


def _mock_dsa_kv_plan(**method_returns) -> MagicMock:
    plan = MagicMock()
    plan.layout_kv = "PA_ND"
    for name, value in method_returns.items():
        getattr(plan, name).return_value = value
    return plan


def _make_builder(compressor_ratio: int = 4) -> AscendDSAMetadataBuilder:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="test",
            num_attention_heads=64,
            index_topk=512,
            index_n_heads=64,
            index_head_dim=128,
            sliding_window=4096,
        ),
        enable_sleep_mode=False,
        get_head_size=lambda: 512,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    physical_block_size = 128
    logical_compress_ratio = 128 if compressor_ratio > 4 else compressor_ratio
    kv_cache_spec = SimpleNamespace(
        compress_ratio=compressor_ratio,
        block_size=physical_block_size * logical_compress_ratio,
        storage_block_size=physical_block_size,
    )
    builder = AscendDSAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["model.layers.0.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    # These caches are supplied by model_runner through build() in production.
    builder.common_ratio_to_sas_metadata = {}
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.num_decodes = 2
    return builder


@pytest.mark.parametrize(
    ("compressor_ratio", "num_tokens", "num_reqs", "expected_rows"),
    [
        (1, 13, 3, 13),
        (4, 13, 3, 6),
        (128, 13, 3, 3),
    ],
)
def test_num_compressor_metadata_rows(
    compressor_ratio: int,
    num_tokens: int,
    num_reqs: int,
    expected_rows: int,
):
    builder = _make_builder(compressor_ratio)
    builder.num_actual_tokens = num_tokens

    assert builder._num_compressor_metadata_rows(num_reqs) == expected_rows


@pytest.mark.parametrize(
    ("compressor_ratio", "expected_cmp_ratio", "expected_cmp_topk", "expected_has_cmp_kv"),
    [
        (1, 1, 0, False),
        (4, 4, 512, True),
        (8, 128, 0, True),
        (128, 128, 0, True),
    ],
)
def test_build_sas_metadata_parameters_cache_and_builder_buffer(
    compressor_ratio,
    expected_cmp_ratio,
    expected_cmp_topk,
    expected_has_cmp_kv,
):
    builder = _make_builder(compressor_ratio)
    metadata_cache: dict[str, torch.Tensor] = {}
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    cu_seqlens_ori_kv = torch.tensor([0, 8, 14], dtype=torch.int32)
    cu_seqlens_cmp_kv = torch.tensor([0, 2, 4], dtype=torch.int32)
    generated_metadata = torch.arange(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    metadata_op = MagicMock(return_value=generated_metadata)
    plan = _mock_dsa_kv_plan(
        get_dsa_sparse_attn_metadata_op=metadata_op,
        get_dsa_sparse_attn_metadata_kwargs={"device": "cpu"},
    )

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", return_value=plan),
    ):
        result = builder._build_sas_metadata(
            metadata_cache=metadata_cache,
            layer_name=f"c{compressor_ratio}",
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_seqlen_q=2,
            max_seqlen_kv=8,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        )
        cached_result = builder._build_sas_metadata(
            metadata_cache=metadata_cache,
            layer_name=f"c{compressor_ratio}",
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_seqlen_q=2,
            max_seqlen_kv=8,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        )

    assert result is builder.sas_metadata_buffer
    assert cached_result is builder.sas_metadata_buffer
    assert torch.equal(builder.sas_metadata_buffer, generated_metadata)
    metadata_op.assert_called_once()
    call_kwargs = metadata_op.call_args.kwargs
    assert call_kwargs["device"] == "cpu"
    assert call_kwargs["num_heads_q"] == 32
    assert call_kwargs["cmp_ratio"] == expected_cmp_ratio
    assert call_kwargs["cmp_topk"] == expected_cmp_topk
    assert call_kwargs["has_cmp_kv"] is expected_has_cmp_kv
    assert call_kwargs["cu_seqlens_q"] is query_start_loc
    assert call_kwargs["cu_seqlens_ori_kv"] is cu_seqlens_ori_kv
    assert call_kwargs["cu_seqlens_cmp_kv"] is cu_seqlens_cmp_kv
    assert call_kwargs["seqused_kv"] is seq_lens


def test_build_qli_metadata_parameters_cache_and_builder_buffer():
    builder = _make_builder()
    metadata_cache: dict[str, torch.Tensor] = {}
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    generated_metadata = torch.arange(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)

    with patch.object(
        torch.ops._C_ascend,
        "npu_vllm_quant_lightning_indexer_metadata",
        create=True,
        return_value=generated_metadata,
    ) as metadata_op:
        result = builder._build_qli_metadata(
            metadata_cache=metadata_cache,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_seqlen_q=2,
            max_seqlen_kv=8,
        )
        cached_result = builder._build_qli_metadata(
            metadata_cache=metadata_cache,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_seqlen_q=2,
            max_seqlen_kv=8,
        )

    assert result is builder.qli_metadata_buffer
    assert cached_result is builder.qli_metadata_buffer
    assert torch.equal(builder.qli_metadata_buffer, generated_metadata)
    metadata_op.assert_called_once()
    call_kwargs = metadata_op.call_args.kwargs
    assert torch.equal(call_kwargs["actual_seq_lengths_query"], query_start_loc[1:])
    assert torch.equal(call_kwargs["actual_seq_lengths_key"], seq_lens)
    assert call_kwargs["max_seqlen_q"] == 2
    assert call_kwargs["max_seqlen_k"] == 8
    assert call_kwargs["cmp_ratio"] == 4


@pytest.mark.parametrize("num_prefills", [0, 1])
def test_build_req_metadata_uses_for_prefill_and_decode(
    num_prefills: int,
):
    builder = _make_builder()
    metadata_cache: dict[str, Any] = {}
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([8, 6], dtype=torch.int32)
    input_positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        num_input_tokens=3,
        positions=input_positions,
        seq_lens=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu=None,
        slot_mapping=torch.arange(3, dtype=torch.int32),
        block_table_tensor=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        attn_state=MagicMock(),
        causal=True,
    )
    sas_metadata = builder.sas_metadata_buffer.fill_(1)
    qli_metadata = builder.qli_metadata_buffer.fill_(2)
    builder._build_sas_metadata = MagicMock(return_value=sas_metadata)
    builder._build_qli_metadata = MagicMock(return_value=qli_metadata)
    decode_cu_seqlens_ori_kv = torch.tensor([0, 8, 14], dtype=torch.int32)
    decode_cu_seqlens_cmp_kv = torch.tensor([0, 2, 4], dtype=torch.int32)
    full_compress_cos = torch.ones((4, 1, 1, 2))
    full_compress_sin = torch.zeros((4, 1, 1, 2))
    cos = torch.ones((3, 1, 1, 2))
    sin = torch.zeros((3, 1, 1, 2))

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.split_decodes_and_prefills",
            return_value=(2, 0, 3, 0) if num_prefills == 0 else (1, 1, 1, 2),
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan",
            return_value=_mock_dsa_kv_plan(
                format_dsa_slot_mapping=torch.zeros((3, 2), dtype=torch.int32),
            ),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=decode_cu_seqlens_ori_kv,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_cmp_kv",
            return_value=decode_cu_seqlens_cmp_kv,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(full_compress_cos, full_compress_sin),
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_cos_and_sin_dsa",
            return_value=(cos, sin),
        ) as rope_op,
    ):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            common_ratio_to_sas_metadata=metadata_cache,
        )
        cached_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            common_ratio_to_sas_metadata=metadata_cache,
        )

    rope_op.assert_called_once()
    assert torch.equal(rope_op.call_args.args[0], input_positions)
    assert rope_op.call_args.kwargs["use_cache"] is (num_prefills == 0)
    assert metadata_cache["cos"] is cos
    assert metadata_cache["sin"] is sin
    req_metadata = cast(AscendDSAReqMetadata, metadata.req_metadata)
    cached_req_metadata = cast(AscendDSAReqMetadata, cached_metadata.req_metadata)
    assert req_metadata.cos is cos
    assert req_metadata.sin is sin
    assert cached_req_metadata.cos is cos
    assert cached_req_metadata.sin is sin
    sas_kwargs = builder._build_sas_metadata.call_args.kwargs
    qli_kwargs = builder._build_qli_metadata.call_args.kwargs
    assert sas_kwargs["metadata_cache"] is metadata_cache
    assert torch.equal(sas_kwargs["query_start_loc"], query_start_loc)
    assert torch.equal(sas_kwargs["seq_lens"], builder.seq_lens)
    assert sas_kwargs["max_seqlen_q"] == 2
    assert sas_kwargs["max_seqlen_kv"] == 8
    assert qli_kwargs["metadata_cache"] is metadata_cache
    assert torch.equal(qli_kwargs["query_start_loc"], query_start_loc)
    assert torch.equal(qli_kwargs["seq_lens"], builder.seq_lens)
    assert qli_kwargs["max_seqlen_q"] == 2
    assert qli_kwargs["max_seqlen_kv"] == 8
    if num_prefills:
        assert torch.equal(sas_kwargs["cu_seqlens_ori_kv"], query_start_loc)
        assert sas_kwargs["cu_seqlens_cmp_kv"] is None
    else:
        assert sas_kwargs["cu_seqlens_ori_kv"] is decode_cu_seqlens_ori_kv
        assert sas_kwargs["cu_seqlens_cmp_kv"] is decode_cu_seqlens_cmp_kv

    assert req_metadata.sas_metadata is sas_metadata
    assert req_metadata.qli_metadata is qli_metadata
    assert torch.equal(req_metadata.query_start_loc, query_start_loc)
    assert torch.equal(req_metadata.block_table, builder.block_table)
    assert torch.equal(
        req_metadata.start_pos,
        torch.tensor([6, 5], dtype=torch.int32),
    )
    assert req_metadata.num_compressed_tokens == 2
    assert req_metadata.slot_mapping is None


@pytest.mark.parametrize(
    ("compressor_ratio", "expected_stages"),
    [
        (4, [DeviceMetadataStage.INDEXER, DeviceMetadataStage.ATTENTION]),
        (128, [DeviceMetadataStage.ATTENTION]),
    ],
)
def test_build_req_metadata_defers_device_work_to_fixed_buffers(
    compressor_ratio: int,
    expected_stages: list[DeviceMetadataStage],
):
    builder = _make_builder(compressor_ratio)
    builder.enable_device_metadata()
    builder.num_actual_tokens = 3
    builder.num_prefills = 1
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    builder.common_ratio_to_sas_metadata = {}
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=3,
        positions=torch.arange(3),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        causal=True,
    )
    builder._build_sas_metadata = MagicMock()
    builder._build_qli_metadata = MagicMock()

    with patch(
        "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
        return_value=(torch.ones(1), torch.zeros(1)),
    ):
        req_metadata = builder.build_req_metadata(
            common_attn_metadata=common_attn_metadata,
            seq_lens_cpu=builder.seq_lens,
            num_actual_reqs=None,
            cos=torch.ones(3),
            sin=torch.zeros(3),
        )

    assert req_metadata.sas_metadata is builder.sas_metadata_buffer
    assert (req_metadata.qli_metadata is builder.qli_metadata_buffer) is (compressor_ratio == 4)
    builder._build_sas_metadata.assert_not_called()
    builder._build_qli_metadata.assert_not_called()

    tasks = builder.take_device_metadata_tasks()
    assert builder.take_device_metadata_tasks() == ()
    assert [task.stage for task in tasks] == expected_stages
    assert all(task.group_id in (id(builder.qli_metadata_buffer), id(builder.sas_metadata_buffer)) for task in tasks)
    for task in tasks:
        task.run()
    builder._build_sas_metadata.assert_called_once()
    if compressor_ratio == 4:
        builder._build_qli_metadata.assert_called_once()
    else:
        builder._build_qli_metadata.assert_not_called()


@pytest.mark.parametrize(
    ("compressor_ratio", "device_metadata_enabled", "expected_stages"),
    [
        (4, True, [DeviceMetadataStage.INDEXER, DeviceMetadataStage.ATTENTION]),
        (128, True, [DeviceMetadataStage.ATTENTION]),
        (4, False, []),
    ],
)
def test_dsa_cp_device_metadata_tasks(
    compressor_ratio: int,
    device_metadata_enabled: bool,
    expected_stages: list[DeviceMetadataStage],
):
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.num_prefills = 1
    builder.num_actual_tokens = 3
    builder.compressor_ratio = compressor_ratio
    builder.storage_block_size = 128
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.seq_lens_cpu = builder.seq_lens
    builder.block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    builder.start_pos_prefill = torch.zeros(2, dtype=torch.int32)
    builder.slot_mapping = torch.zeros((3, 2), dtype=torch.int32)
    builder.req_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder._device_metadata_enabled = device_metadata_enabled
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
    builder._ensure_device_local_metadata = MagicMock(return_value=(0, 3, 3, 3, query_start_loc, builder.seq_lens))
    builder._get_cmp_seqlens_for_metadata = MagicMock(return_value=None)
    builder._num_compressor_metadata_rows = MagicMock(return_value=2)
    builder._build_sas_metadata = MagicMock(return_value=builder.req_sas_metadata)
    builder._build_qli_metadata = MagicMock(return_value=builder.req_qli_metadata)
    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )

    with patch(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_full_cos_and_sin_dsa",
        return_value=(torch.ones(1), torch.zeros(1)),
    ):
        req_metadata = builder.build_req_metadata(
            common_attn_metadata,
            input_positions=None,
            num_input_tokens=3,
            num_actual_reqs=None,
            attn_state=MagicMock(),
            cos=MagicMock(),
            sin=MagicMock(),
        )

    tasks = builder.take_device_metadata_tasks()
    assert builder.take_device_metadata_tasks() == ()
    assert [task.stage for task in tasks] == expected_stages
    assert all(task.group_id in (id(builder.req_qli_metadata), id(builder.req_sas_metadata)) for task in tasks)
    assert req_metadata.sas_metadata is builder.req_sas_metadata
    assert (req_metadata.qli_metadata is builder.req_qli_metadata) is (compressor_ratio == 4)
    if device_metadata_enabled:
        builder._build_sas_metadata.assert_not_called()
        builder._build_qli_metadata.assert_not_called()
        for task in tasks:
            task.run()
    builder._build_sas_metadata.assert_called_once()
    if compressor_ratio == 4:
        builder._build_qli_metadata.assert_called_once()
        assert builder._build_qli_metadata.call_args.kwargs["max_seqlen_q"] == 2
        assert builder._build_qli_metadata.call_args.kwargs["max_seqlen_k"] == 8
    else:
        builder._build_qli_metadata.assert_not_called()


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
        local_sin=cast(Any, {"layer": torch.zeros((1, 1, 1, 2))}),
        local_cos=cast(Any, {"layer": torch.ones((1, 1, 1, 2))}),
    )
    req_metadata = AscendDSACPReqMetadata(
        input_positions=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=seq_lens,
        slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
        storage_block_size=128,
        query_start_loc=query_start_loc,
        cp_metadata=cp_metadata,
        sin=cast(Any, {"layer": torch.zeros((1, 1, 1, 2))}),
        cos=cast(Any, {"layer": torch.ones((1, 1, 1, 2))}),
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
        hadamard=torch.eye(2),
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
    indexer_cache = SimpleNamespace(
        req_metadata=SimpleNamespace(
            qli_metadata=qli_metadata,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
        ),
        hadamard=torch.eye(2),
    )
    layer_metadata = AscendDSACPLayerMetadata(
        swa=cast(Any, object()),
        indexer_cache=cast(Any, indexer_cache),
    )
    waited = False

    def record_wait(stage: DeviceMetadataStage, group_id: int) -> None:
        nonlocal waited
        assert (stage, group_id) == (DeviceMetadataStage.INDEXER, id(qli_metadata))
        waited = True

    def run_indexer(**kwargs):
        assert waited
        return torch.zeros((1, 1, 1), dtype=torch.int32), None

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
            side_effect=run_indexer,
        ),
        patch("vllm_ascend.attention.context_parallel.dsa_cp.rotate_activation", side_effect=lambda value, _: value),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.wait_for_device_metadata",
            side_effect=record_wait,
        ),
    ):
        impl._indexer_select_topk(
            x=torch.ones((1, 2)),
            qr=torch.ones((1, 2)),
            kv_cache=(torch.empty(0),),
            metadata=layer_metadata,
            cos=torch.ones((1, 1, 1, 2)),
            sin=torch.zeros((1, 1, 1, 2)),
            actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([1], dtype=torch.int32),
        )

    assert waited


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_dsa_cp_attention_waits_before_sas_consumer(compress_ratio: int):
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
    impl._maybe_all_gather_o_proj_full_weight = MagicMock()
    impl.compressor_wkv = SimpleNamespace(weight=torch.empty(0))
    impl.compressor_wgate = SimpleNamespace(weight=torch.empty(0))
    impl.compressor_ape = torch.empty(0)
    impl.compressor_norm = SimpleNamespace(weight=torch.empty(0))
    impl.compressor_norm_eps = 1e-6

    swa_sas = torch.zeros(1024, dtype=torch.int32)
    compressor_sas = torch.ones(1024, dtype=torch.int32)
    swa_metadata = _make_dsa_cp_metadata(swa_sas)
    compressor_metadata = _make_dsa_cp_metadata(compressor_sas)
    layer_metadata = AscendDSACPLayerMetadata(
        swa=swa_metadata,
        compressor_cache=compressor_metadata if compress_ratio > 1 else None,
        compressor_state=compressor_metadata if compress_ratio > 1 else None,
        indexer_cache=compressor_metadata if compress_ratio == 4 else None,
        indexer_state=compressor_metadata if compress_ratio == 4 else None,
    )
    expected_sas = swa_sas if compress_ratio <= 1 else compressor_sas
    waited = False

    def record_wait(stage: DeviceMetadataStage, group_id: int) -> None:
        nonlocal waited
        assert (stage, group_id) == (DeviceMetadataStage.ATTENTION, id(expected_sas))
        waited = True

    def run_attention(*args, **kwargs):
        assert waited
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
        patch.object(torch.ops._C_ascend, "inplace_partial_rotary_mul", create=True),
        patch.object(torch.ops._C_ascend, "compressor", create=True, return_value=torch.empty(0)),
        patch.object(impl, "_split_full_hidden_states_for_cp", side_effect=lambda value, _: value),
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
        patch("vllm_ascend.attention.context_parallel.dsa_cp.notify_kv_cache_written"),
        patch("vllm_ascend.attention.context_parallel.dsa_cp.record_attention_compute_start"),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.wait_for_device_metadata",
            side_effect=record_wait,
        ),
    ):
        impl._forward(
            "layer",
            torch.ones((1, 2)),
            (torch.empty(0),),
            layer_metadata,
        )

    assert waited


@pytest.mark.parametrize("for_drafting", [False, True])
def test_build_classifies_short_speculative_extends_as_decodes(
    for_drafting: bool,
):
    builder = _make_builder(compressor_ratio=1)
    builder.decode_threshold = 8
    query_start_loc = torch.tensor([0, 7, 14], dtype=torch.int32)
    seq_lens = torch.tensor([20, 14], dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=14,
        num_input_tokens=14,
        max_query_len=7,
        context_parallel_metadata=None,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        positions=torch.arange(14, dtype=torch.int64),
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens,
        seq_lens_cpu=None,
        is_prefilling=torch.tensor([False, True]),
        slot_mapping=torch.arange(14, dtype=torch.int32),
        block_table_tensor=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        attn_state=MagicMock(),
    )
    req_metadata = MagicMock()
    builder.build_req_metadata = MagicMock(return_value=req_metadata)
    builder.build_req_metadata_for_drafting = MagicMock(return_value=req_metadata)
    builder.spec_slot_mapping = [torch.zeros((16, 2), dtype=torch.int32)]

    with (
        patch(
            "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
            return_value=False,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan",
            return_value=_mock_dsa_kv_plan(
                format_dsa_slot_mapping=torch.zeros((14, 2), dtype=torch.int32),
            ),
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_cos_and_sin_dsa",
            return_value=(torch.ones(14), torch.zeros(14)),
        ),
    ):
        if for_drafting:
            metadata = builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                draft_index=1,
            )
        else:
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                common_ratio_to_sas_metadata={},
            )

    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 14
    assert metadata.num_prefills == 0


def test_build_req_metadata_preserves_zero_max_sequence_lengths():
    builder = _make_builder(compressor_ratio=1)
    builder.common_ratio_to_sas_metadata = {}
    builder.num_actual_tokens = 0
    builder.num_prefills = 1
    builder.seq_lens = torch.zeros(2, dtype=torch.int32)
    builder.block_table = torch.zeros((2, 2), dtype=torch.int32)
    query_start_loc = torch.zeros(3, dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=0,
        positions=torch.empty(0, dtype=torch.int64),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )
    builder._build_sas_metadata = MagicMock(return_value=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32))
    builder._build_qli_metadata = MagicMock(return_value=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32))

    metadata = builder.build_req_metadata(
        common_attn_metadata=common_attn_metadata,
        seq_lens_cpu=torch.zeros(2, dtype=torch.int32),
        num_actual_reqs=None,
        cos=torch.empty(0),
        sin=torch.empty(0),
    )

    sas_kwargs = builder._build_sas_metadata.call_args.kwargs
    qli_kwargs = builder._build_qli_metadata.call_args.kwargs
    assert sas_kwargs["max_seqlen_q"] == 0
    assert sas_kwargs["max_seqlen_kv"] == 0
    assert qli_kwargs["max_seqlen_q"] == 0
    assert qli_kwargs["max_seqlen_kv"] == 0
    assert metadata.num_compressed_tokens == 0


def test_build_req_metadata_clears_graph_padding_rows():
    builder = _make_builder(compressor_ratio=1)
    builder.common_ratio_to_sas_metadata = {}
    builder.num_actual_tokens = 2
    builder.num_prefills = 1
    builder.seq_lens = torch.tensor([8, 6, 7], dtype=torch.int32)
    builder.block_table = torch.tensor(
        [[1, 2], [3, 4], [5, 6]],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 2, 2], dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        num_reqs=3,
        num_input_tokens=2,
        positions=torch.tensor([0, 1], dtype=torch.int64),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )
    builder._build_sas_metadata = MagicMock(return_value=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32))
    builder._build_qli_metadata = MagicMock(return_value=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32))

    metadata = builder.build_req_metadata(
        common_attn_metadata=common_attn_metadata,
        seq_lens_cpu=torch.tensor([8, 6, 7], dtype=torch.int32),
        num_actual_reqs=1,
        cos=torch.ones(2),
        sin=torch.zeros(2),
    )

    assert metadata.num_actual_reqs == 1
    assert torch.equal(
        metadata.start_pos,
        torch.tensor([6, 0, 0], dtype=torch.int32),
    )
    assert torch.equal(metadata.block_table[0], torch.tensor([1, 2]))
    assert torch.count_nonzero(metadata.block_table[1:]).item() == 0


def test_build_req_metadata_for_drafting_uses_decode_buffer_and_cpu_lengths():
    builder = _make_builder(compressor_ratio=1)
    builder.num_actual_tokens = 3
    builder.num_prefills = 0
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    spec_slot_mapping = [torch.arange(16, dtype=torch.int32).reshape(8, 2)]
    spec_sas_metadata = [torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)]
    builder.spec_slot_mapping = spec_slot_mapping
    builder.spec_sas_metadata = spec_sas_metadata
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        _seq_lens_cpu=torch.tensor([9, 7], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([99, 99], dtype=torch.int32),
        seq_lens=torch.tensor([8, 6], dtype=torch.int32),
        causal=True,
    )
    decode_cu_seqlens_ori_kv = torch.tensor([0, 9, 16], dtype=torch.int32)
    decode_cu_seqlens_cmp_kv = torch.tensor([0, 2, 4], dtype=torch.int32)
    generated_metadata = torch.arange(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    metadata_op = MagicMock(return_value=generated_metadata)
    plan = _mock_dsa_kv_plan(
        get_dsa_sparse_attn_metadata_op=metadata_op,
        get_dsa_sparse_attn_metadata_kwargs={"device": "cpu"},
    )
    cos = torch.ones((3, 1, 1, 2))
    sin = torch.zeros((3, 1, 1, 2))

    with (
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=decode_cu_seqlens_ori_kv,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_cmp_kv",
            return_value=decode_cu_seqlens_cmp_kv,
        ),
        patch("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", return_value=plan),
    ):
        metadata = builder.build_req_metadata_for_drafting(
            draft_index=1,
            common_attn_metadata=common_attn_metadata,
            cos=cos,
            sin=sin,
        )

    call_kwargs = metadata_op.call_args.kwargs
    assert call_kwargs["max_seqlen_q"] == 2
    assert call_kwargs["max_seqlen_kv"] == 9
    assert call_kwargs["cu_seqlens_ori_kv"] is decode_cu_seqlens_ori_kv
    assert call_kwargs["cu_seqlens_cmp_kv"] is decode_cu_seqlens_cmp_kv
    assert metadata.sas_metadata is spec_sas_metadata[0]
    assert metadata.sas_metadata is not None
    assert torch.equal(metadata.sas_metadata, generated_metadata)
    assert torch.equal(
        metadata.slot_mapping,
        spec_slot_mapping[0][: builder.num_actual_tokens],
    )
    assert torch.equal(
        metadata.start_pos,
        torch.tensor([6, 5], dtype=torch.int32),
    )
    assert metadata.qli_metadata is None
    assert metadata.cos is cos
    assert metadata.sin is sin


def _make_req_metadata() -> AscendDSAReqMetadata:
    return AscendDSAReqMetadata(
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        slot_mapping=None,
        storage_block_size=128,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        sin=cast(Any, {"layer": torch.zeros(1)}),
        cos=cast(Any, {"layer": torch.ones(1)}),
    )


def _make_impl(
    impl_cls: type[AscendDSAImpl] = AscendDSAImpl,
) -> AscendDSAImpl:
    linear = MagicMock()
    with (
        patch(
            "vllm_ascend.attention.dsa_v1.CVLinearWrapper",
            side_effect=lambda layer: layer,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_config",
            return_value=SimpleNamespace(multistream_dsv4_dsa_overlap=False),
        ),
    ):
        return impl_cls(
            n_heads=1,
            scale=1.0,
            n_local_heads=1,
            q_lora_rank=2,
            o_lora_rank=2,
            head_dim=2,
            rope_head_dim=1,
            nope_head_dim=1,
            n_groups=1,
            n_local_groups=1,
            window_size=16,
            compress_ratio=1,
            vllm_config=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto")),
            wq_a=linear,
            wq_b=linear,
            wkv=linear,
            q_norm=linear,
            q_norm_without_weight=linear,
            kv_norm=linear,
            indexer=None,
            compressor=None,
            wo_a=linear,
            wo_b=linear,
            eps=1e-6,
            attn_sink=None,
            swa_cache_layer=SimpleNamespace(prefix="swa_cache"),
        )


def test_forward_runs_mixed_prefill_and_decode_in_one_attention_call():
    impl = _make_impl()
    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    unified_output = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
    output = torch.empty((5, 2), dtype=torch.float32)
    metadata = AscendDSAMetadata(
        num_actual_tokens=5,
        num_decodes=1,
        num_decode_tokens=2,
        num_prefills=1,
        req_metadata=_make_req_metadata(),
    )
    captured_o_proj_input: list[torch.Tensor] = []

    def fake_o_proj(
        o_proj_input: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> torch.Tensor:
        captured_o_proj_input.append(o_proj_input.clone())
        output_tensor.copy_(o_proj_input.view(5, 2))
        return output_tensor

    with (
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(num_tokens=5),
        ),
        patch("vllm_ascend.attention.dsa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.dsa_v1.maybe_save_kv_layer_to_connector"),
        patch.object(
            torch.ops.vllm,
            "maybe_all_gather_and_maybe_unpad",
            create=True,
            side_effect=lambda tensor, _: tensor,
        ),
        patch.object(
            torch.ops._C_ascend,
            "inplace_partial_rotary_mul",
            create=True,
        ),
        patch.object(
            impl,
            "_forward_attention",
            return_value=unified_output,
        ) as forward_attention,
        patch.object(impl, "_forward_o_proj", side_effect=fake_o_proj),
    ):
        actual = impl.forward(
            layer_name="layer",
            hidden_states=hidden_states,
            kv_cache=(torch.empty(0),),
            attn_metadata={"swa_cache": metadata},
            output=output,
        )

    assert actual is output
    assert len(captured_o_proj_input) == 1
    assert torch.equal(captured_o_proj_input[0], unified_output)
    forward_attention.assert_called_once()
    call = forward_attention.call_args
    assert torch.equal(call.args[1], hidden_states)
    assert call.args[3].attention is None
    assert call.args[3].swa is metadata
    assert call.args[4] is False
    assert not call.kwargs


@pytest.mark.parametrize(
    ("num_prefills", "num_decodes", "num_decode_tokens"),
    [
        (0, 2, 3),
        (2, 0, 0),
        (1, 1, 1),
    ],
    ids=["decode_only", "prefill_only", "mixed"],
)
def test_forward_attention_routes_unified_req_metadata(
    num_prefills: int,
    num_decodes: int,
    num_decode_tokens: int,
):
    impl = _make_impl()
    impl.multistream_dsv4_dsa_overlap = True
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    q = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
    attention_output = torch.full((3, 1, 2), 7.0)
    cos = torch.ones((5, 1, 1, 2))
    sin = torch.zeros((5, 1, 1, 2))
    slot_mapping = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    sas_metadata = torch.arange(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    req_metadata = AscendDSAReqMetadata(
        block_table=torch.tensor([[1], [2]], dtype=torch.int32),
        seq_lens=torch.tensor([8, 6], dtype=torch.int32),
        slot_mapping=slot_mapping,
        storage_block_size=128,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        sin=cast(Any, {"layer": sin}),
        cos=cast(Any, {"layer": cos}),
        sas_metadata=sas_metadata,
        ori_win_left=7,
        ori_win_right=0,
    )
    metadata = AscendDSAMetadata(
        num_actual_tokens=3,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        req_metadata=req_metadata,
    )
    swa_kv_cache = torch.empty(0)
    sparse_attn_op = MagicMock(return_value=(attention_output,))

    def add_extra_kwargs(extra_kwargs: dict[str, Any], **kwargs) -> None:
        extra_kwargs.update(kwargs)

    plan = _mock_dsa_kv_plan(
        get_dsa_sparse_attn_op=sparse_attn_op,
        get_dsa_sparse_attn_base_kwargs={},
    )
    plan.add_dsa_sparse_attn_extra_kwargs.side_effect = add_extra_kwargs

    with (
        patch.object(
            DeviceOperator,
            "unpack_dsa_forward_kv_cache",
            return_value=(
                torch.empty(0),
                swa_kv_cache,
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                None,
            ),
        ),
        patch.object(
            impl,
            "_mla_prolog_multistream",
            return_value=(q, torch.empty(0), None),
        ) as mla_prolog,
        patch("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", return_value=plan),
        patch("vllm_ascend.attention.dsa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.dsa_v1.record_attention_compute_start"),
    ):
        layer_metadata = impl._get_layer_metadata(
            "layer",
            {"swa_cache": metadata},
        )
        actual = impl._forward_attention(
            layer_name="layer",
            hidden_states=hidden_states,
            kv_cache=(torch.empty(0),),
            layer_metadata=layer_metadata,
        )

    assert actual is attention_output
    mla_call = mla_prolog.call_args
    assert torch.equal(mla_call.args[0], hidden_states)
    assert torch.equal(mla_call.args[1], cos[:3])
    assert torch.equal(mla_call.args[2], sin[:3])
    assert mla_call.args[3] is swa_kv_cache
    assert mla_call.args[4] is slot_mapping
    assert mla_call.kwargs["is_prefill"] is bool(num_prefills)
    sparse_call = sparse_attn_op.call_args
    assert sparse_call.args[0] is q
    assert sparse_call.kwargs["ori_kv"] is swa_kv_cache
    assert sparse_call.kwargs["ori_block_table"] is req_metadata.block_table
    assert sparse_call.kwargs["cu_seqlens_q"] is req_metadata.query_start_loc
    assert sparse_call.kwargs["seqused_kv"] is req_metadata.seq_lens
    assert sparse_call.kwargs["metadata"] is sas_metadata
    if num_prefills:
        assert sparse_call.kwargs["cu_seqlens_ori_kv"] is req_metadata.query_start_loc
    else:
        assert "cu_seqlens_ori_kv" not in sparse_call.kwargs


class TestAscendDSAComponentMetadata:
    def test_routes_c4_metadata_by_cache_prefix(self):
        impl = _make_impl()
        impl.compress_ratio = 4
        impl.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="compressor.state_cache"))
        impl.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="indexer.k_cache"),
            compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="indexer.compressor.state_cache")),
        )
        attention_metadata = cast(Any, object())
        compressor_state_metadata = cast(Any, object())
        indexer_cache_metadata = cast(Any, object())
        indexer_state_metadata = cast(Any, object())
        swa_metadata = cast(Any, object())

        actual = impl._get_layer_metadata(
            "layer",
            {
                "layer": attention_metadata,
                "compressor.state_cache": compressor_state_metadata,
                "indexer.k_cache": indexer_cache_metadata,
                "indexer.compressor.state_cache": indexer_state_metadata,
                "swa_cache": swa_metadata,
            },
        )

        assert actual.attention is attention_metadata
        assert actual.swa is swa_metadata
        assert actual.compressor is not None
        assert actual.compressor.cache is attention_metadata
        assert actual.compressor.state is compressor_state_metadata
        assert actual.indexer is not None
        assert actual.indexer.compressor.cache is indexer_cache_metadata
        assert actual.indexer.compressor.state is indexer_state_metadata


class TestAscendDSACompressedCacheRouting:
    def test_c4_delegates_through_indexer_overlap_plan(self):
        impl = _make_impl()
        impl.compress_ratio = 4
        impl.multistream_dsv4_dsa_overlap = False
        compressed_kv = torch.ones((1, 1, 4))
        compress_slot_mapping = torch.zeros((1, 2), dtype=torch.int32)
        impl.compressor = MagicMock(return_value=(compressed_kv, compress_slot_mapping))
        topk_indices = torch.tensor([[[1, 2, 3]]], dtype=torch.int32)
        impl.indexer = MagicMock(return_value=topk_indices)
        compressor_metadata = AscendCompressorMetadata(
            cache=cast(Any, object()),
            state=cast(Any, object()),
        )
        indexer_metadata = AscendIndexerMetadata(compressor=compressor_metadata)
        layer_metadata = AscendDSALayerMetadata(
            attention=cast(Any, object()),
            swa=cast(Any, object()),
            compressor=compressor_metadata,
            indexer=indexer_metadata,
        )
        hidden_states = torch.ones((1, 4))
        qr = torch.ones((1, 4))
        kv_cache = (torch.empty(0),)
        compress_kv_cache = torch.empty(0)
        state_cache = torch.empty(0)
        plan = _mock_dsa_kv_plan()

        with patch("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", return_value=plan):
            actual = impl._maybe_update_compressed_caches_and_select_topk(
                layer_name="model.layers.0.self_attn.attn",
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                layer_metadata=layer_metadata,
                qr_pertoken_scale=None,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
            )
            indexer_call = impl.indexer.call_args
            overlap_plan = indexer_call.kwargs["overlap_plan"]
            actual_kv, actual_mapping = overlap_plan.compute_attention_compressed_kv()
            overlap_plan.scatter_attention_compressed_kv(actual_kv, actual_mapping)

        assert actual is topk_indices
        assert indexer_call.kwargs["layer_name"] == "model.layers.0.self_attn.attn"
        assert indexer_call.kwargs["hidden_states"] is hidden_states
        assert indexer_call.kwargs["qr"] is qr
        assert indexer_call.kwargs["kv_cache"] is kv_cache
        assert indexer_call.kwargs["metadata"] is indexer_metadata
        assert indexer_call.kwargs["write_cache"] is True
        assert isinstance(overlap_plan, IndexerOverlapPlan)
        assert overlap_plan.aux_stream is None
        impl.compressor.assert_called_once_with(
            hidden_states=hidden_states,
            state_cache=state_cache,
            metadata=compressor_metadata,
        )
        plan.dsa_kv_compress_scatter.assert_called_once_with(
            compress_kv_cache,
            compressed_kv,
            compress_slot_mapping,
        )

    @pytest.mark.parametrize("write_cache", [True, False], ids=["normal", "prepared"])
    def test_c128_delegates_directly_to_compressor(self, write_cache: bool):
        impl = _make_impl()
        impl.compress_ratio = 128
        compressed_kv = torch.ones((1, 1, 4))
        compress_slot_mapping = torch.zeros((1, 2), dtype=torch.int32)
        impl.compressor = MagicMock(return_value=(compressed_kv, compress_slot_mapping))
        compressor_metadata = AscendCompressorMetadata(
            cache=cast(Any, object()),
            state=cast(Any, object()),
        )
        layer_metadata = AscendDSALayerMetadata(
            attention=cast(Any, object()),
            swa=cast(Any, object()),
            compressor=compressor_metadata,
        )
        hidden_states = torch.ones((1, 4))
        state_cache = torch.empty(0)
        compress_kv_cache = torch.empty(0)
        plan = _mock_dsa_kv_plan()

        with patch("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", return_value=plan):
            actual = impl._maybe_update_compressed_caches_and_select_topk(
                layer_name="layer",
                hidden_states=hidden_states,
                qr=torch.ones((1, 4)),
                kv_cache=(torch.empty(0),),
                layer_metadata=layer_metadata,
                qr_pertoken_scale=None,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
                write_cache=write_cache,
            )

        assert actual is None
        if write_cache:
            impl.compressor.assert_called_once_with(
                hidden_states=hidden_states,
                state_cache=state_cache,
                metadata=compressor_metadata,
            )
            plan.dsa_kv_compress_scatter.assert_called_once_with(
                compress_kv_cache,
                compressed_kv,
                compress_slot_mapping,
            )
        else:
            impl.compressor.assert_not_called()
            plan.dsa_kv_compress_scatter.assert_not_called()


@pytest.mark.parametrize(
    ("compress_ratio", "cache_is_prepared"),
    [(4, False), (128, False), (4, True)],
    ids=["c4_normal", "c128_normal", "c4_prepared"],
)
def test_forward_attention_sets_compressed_kv_args(
    compress_ratio: int,
    cache_is_prepared: bool,
):
    impl = _make_impl()
    impl.compress_ratio = compress_ratio
    impl.multistream_dsv4_dsa_overlap = False
    hidden_states = torch.ones((2, 4))
    q = torch.ones((2, 1, 2))
    qr = torch.ones((2, 2))
    attention_output = torch.full((2, 1, 2), 9.0)
    cos = torch.ones((2, 1, 1, 2))
    sin = torch.zeros((2, 1, 1, 2))
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    common_req = AscendDSAReqMetadata(
        block_table=torch.tensor([[1]], dtype=torch.int32),
        seq_lens=seq_lens,
        slot_mapping=None,
        storage_block_size=128,
        query_start_loc=query_start_loc,
        sin=cast(Any, {"layer": sin}),
        cos=cast(Any, {"layer": cos}),
        sas_metadata=torch.zeros(DSA_METADATA_BUFFER_SIZE, dtype=torch.int32),
        cu_cmp_seqlen_list=torch.tensor([0, 1], dtype=torch.int32),
    )
    swa_req = AscendDSAReqMetadata(
        block_table=torch.tensor([[5]], dtype=torch.int32),
        seq_lens=seq_lens,
        slot_mapping=torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
        storage_block_size=128,
        query_start_loc=query_start_loc,
    )
    attention_metadata = AscendDSAMetadata(2, 0, 0, 1, req_metadata=common_req)
    compressor_metadata = AscendCompressorMetadata(
        cache=attention_metadata,
        state=cast(Any, object()),
    )
    layer_metadata = AscendDSALayerMetadata(
        attention=attention_metadata,
        swa=AscendDSAMetadata(2, 0, 0, 1, req_metadata=swa_req),
        compressor=compressor_metadata,
    )
    compress_kv_cache = torch.empty(0)
    swa_kv_cache = torch.empty(0)
    state_cache = torch.empty(0)
    topk_indices = torch.tensor([[[1, 2, 3]]], dtype=torch.int32) if compress_ratio == 4 else None
    sparse_attn_op = MagicMock(return_value=(attention_output,))

    def add_extra_kwargs(extra_kwargs: dict[str, Any], **kwargs) -> None:
        extra_kwargs.update(kwargs)

    plan = _mock_dsa_kv_plan(
        get_dsa_sparse_attn_op=sparse_attn_op,
        get_dsa_sparse_attn_base_kwargs={},
    )
    plan.add_dsa_sparse_attn_extra_kwargs.side_effect = add_extra_kwargs

    with (
        patch.object(
            DeviceOperator,
            "unpack_dsa_forward_kv_cache",
            return_value=(
                compress_kv_cache,
                swa_kv_cache,
                state_cache,
                None,
                None,
                None,
            ),
        ),
        patch.object(
            impl,
            "_mla_prolog_single_stream",
            return_value=(q, qr, None),
        ) as single_stream_prolog,
        patch.object(
            impl,
            "_maybe_update_compressed_caches_and_select_topk",
            return_value=topk_indices,
        ) as update_compressed_caches,
        patch("vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan", return_value=plan),
        patch("vllm_ascend.attention.dsa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.dsa_v1.record_attention_compute_start"),
    ):
        actual = impl._forward_attention(
            "layer",
            hidden_states,
            (torch.empty(0),),
            layer_metadata,
            cache_is_prepared,
        )

    assert actual is attention_output
    assert single_stream_prolog.call_args.kwargs["write_swa_cache"] is not cache_is_prepared
    update_call = update_compressed_caches.call_args
    assert update_call.kwargs["layer_name"] == "layer"
    assert update_call.kwargs["hidden_states"] is hidden_states
    assert update_call.kwargs["layer_metadata"] is layer_metadata
    assert update_call.kwargs["compress_kv_cache"] is compress_kv_cache
    assert update_call.kwargs["state_cache"] is state_cache
    assert update_call.kwargs["write_cache"] is not cache_is_prepared
    sparse_kwargs = sparse_attn_op.call_args.kwargs
    assert sparse_kwargs["cmp_kv"] is compress_kv_cache
    assert sparse_kwargs["cmp_block_table"] is common_req.block_table
    assert sparse_kwargs["cmp_mask_mode"] == 3
    assert sparse_kwargs["cmp_ratio"] == compress_ratio
    assert sparse_kwargs["cu_seqlens_cmp_kv"] is common_req.cu_cmp_seqlen_list
    if compress_ratio == 4:
        assert sparse_kwargs["cmp_sparse_indices"] is topk_indices
    else:
        assert "cmp_sparse_indices" not in sparse_kwargs


def test_a5_bf16_o_proj_uses_transpose_batchmatmul():
    impl = _make_impl()
    impl.support_fp8_attention = True
    impl.wo_a = SimpleNamespace(weight=torch.randn(1, 2, 2), weight_scale=None)
    impl.wo_b = lambda x: x
    o_proj_input = torch.randn(4, 1, 2)
    output = torch.empty(4, 2)
    projected = torch.randn(4, 1, 2)

    with (
        patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        patch("vllm_ascend.attention.dsa_v1.olora_tp_enable", return_value=False),
        patch("vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_batchmatmul", return_value=projected) as batched,
        patch("vllm_ascend.attention.dsa_v1.torch_npu.npu_dynamic_mx_quant") as quant,
    ):
        impl._forward_o_proj(o_proj_input, output)

    batched.assert_called_once()
    quant.assert_not_called()
    torch.testing.assert_close(output, projected.reshape(4, -1))


def test_prepared_cache_rejects_multistream_before_cache_access():
    impl = _make_impl()
    impl.multistream_dsv4_dsa_overlap = True

    with (
        patch.object(DeviceOperator, "unpack_dsa_forward_kv_cache") as unpack_cache,
        pytest.raises(
            RuntimeError,
            match="Prepared DSA caches require single-stream attention",
        ),
    ):
        impl._forward_attention(
            "layer",
            torch.empty((1, 4)),
            (torch.empty(0),),
            cast(Any, None),
            True,
        )

    unpack_cache.assert_not_called()


def test_dsa_backend_selects_pcp_and_rejects_legacy_cp():
    with (
        patch("vllm_ascend.attention.dsa_v1.enable_pcp", return_value=True),
        patch("vllm_ascend.utils.enable_dsa_cp", return_value=False),
    ):
        assert AscendDSABackend.get_builder_cls() is AscendDSAPCPMetadataBuilder
        assert AscendDSABackend.get_impl_cls() is AscendDSAPCPImpl

    with (
        patch("vllm_ascend.attention.dsa_v1.enable_pcp", return_value=True),
        patch("vllm_ascend.utils.enable_dsa_cp", return_value=True),
    ):
        for get_backend_cls in (
            AscendDSABackend.get_builder_cls,
            AscendDSABackend.get_impl_cls,
        ):
            with pytest.raises(ValueError, match="cannot be enabled at the same time"):
                get_backend_cls()


def test_pcp_metadata_builds_from_manager_global_view():
    """Build rank-local metadata from the manager's scheduler-global view."""
    builder = AscendDSAPCPMetadataBuilder.__new__(AscendDSAPCPMetadataBuilder)
    builder._pcp_world_size = 2
    builder._pcp_rank = 1
    builder.model_config = SimpleNamespace(get_head_size=lambda: 512)

    raw_slot_mapping = torch.arange(6, dtype=torch.int64)
    hidden_restore_idx = torch.arange(5, dtype=torch.int64)
    global_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=5,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        seq_lens=torch.tensor([4, 7], dtype=torch.int32),
        seq_lens_np=np.array([4, 7], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8, 8], dtype=torch.int32),
        num_computed_tokens_np=np.array([2, 4], dtype=np.int32),
        num_scheduled_tokens=np.array([2, 3], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(5, dtype=torch.int64),
        attn_state=object(),
        is_prefilling_np=np.array([True, True]),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        num_reqs_after_padding=2,
    )
    global_block_table = torch.tensor([[1], [2]], dtype=torch.int32)
    global_block_tables = (torch.empty(0), global_block_table)
    global_slot_mappings = torch.arange(14, dtype=torch.int64).view(2, 7)
    global_slot_mapping = global_slot_mappings[1, : global_batch.num_tokens]
    local_common = SimpleNamespace(
        attn_state="local",
        num_actual_tokens=2,
    )
    common_attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=raw_slot_mapping,
        max_seq_len=8,
        causal=True,
        replace=MagicMock(return_value=local_common),
    )
    gather_block_tables = MagicMock(return_value=global_block_tables)
    pcp_manager = AscendPCPManager.__new__(AscendPCPManager)
    pcp_manager._global_batch = global_batch
    pcp_manager._block_tables = SimpleNamespace(
        gather_block_tables=gather_block_tables,
    )
    pcp_manager._global_batch_slot_mappings = global_slot_mappings
    pcp_manager._hidden_restore_idx = hidden_restore_idx
    pcp_context = pcp_manager.build_attention_context(
        SimpleNamespace(is_dummy=False, num_tokens_after_padding=3),
        (),
        torch.empty(0),
    )
    global_metadata = AscendDSAMetadata(
        num_actual_tokens=5,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
    )
    local_metadata = AscendDSAMetadata(
        num_actual_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
    )
    builder._global_metadata_builder = SimpleNamespace(
        build=MagicMock(return_value=global_metadata),
    )

    with patch.object(
        AscendDSAMetadataBuilder,
        "build",
        autospec=True,
        return_value=local_metadata,
    ) as build_local:
        actual = builder.build(
            0,
            common_attn_metadata,
            pcp_context=pcp_context,
            pcp_cache_group_idx=1,
            common_ratio_to_sas_metadata={"local": True},
            num_actual_reqs=7,
        )

    assert isinstance(actual, AscendDSAPCPMetadata)
    assert actual.num_actual_tokens == local_metadata.num_actual_tokens
    assert actual.global_dsa_metadata is global_metadata
    assert actual.hidden_restore_idx is hidden_restore_idx
    assert actual.local_num_tokens_after_padding == 3
    gather_block_tables.assert_called_once_with(
        global_batch.idx_mapping,
        global_batch.num_reqs_after_padding,
    )
    common_attn_metadata.replace.assert_called_once()
    replace_kwargs = common_attn_metadata.replace.call_args.kwargs
    assert torch.equal(replace_kwargs["slot_mapping"], raw_slot_mapping.view(2, 3)[1])
    assert replace_kwargs["num_input_tokens"] == 3
    global_call = builder._global_metadata_builder.build.call_args
    global_common = global_call.args[1]
    assert global_common.num_actual_tokens == global_batch.num_tokens
    assert global_common.block_table_tensor is global_block_table
    assert torch.equal(global_common.slot_mapping, global_slot_mapping)
    assert global_common.attn_state is global_batch.attn_state
    assert global_call.kwargs["num_actual_reqs"] == global_batch.num_reqs
    assert global_call.kwargs["common_ratio_to_sas_metadata"] == {}
    assert build_local.call_args.args[2] is local_common
    assert build_local.call_args.kwargs["num_actual_reqs"] == 7


def test_pcp_metadata_provider_discards_unused_global_tasks():
    builder = AscendDSAPCPMetadataBuilder.__new__(AscendDSAPCPMetadataBuilder)
    local_task = DeviceMetadataTask(DeviceMetadataStage.INDEXER, MagicMock(), 1)
    global_task = DeviceMetadataTask(DeviceMetadataStage.ATTENTION, MagicMock(), 2)
    builder._device_metadata_tasks = (local_task,)
    builder._global_metadata_builder = SimpleNamespace(
        take_device_metadata_tasks=MagicMock(return_value=(global_task,)),
    )

    assert builder.take_device_metadata_tasks() == (local_task,)
    assert builder._device_metadata_tasks == ()
    builder._global_metadata_builder.take_device_metadata_tasks.assert_called_once_with()


def test_pcp_metadata_provider_enables_local_and_global_builders():
    builder = AscendDSAPCPMetadataBuilder.__new__(AscendDSAPCPMetadataBuilder)
    builder._device_metadata_enabled = False
    builder._global_metadata_builder = SimpleNamespace(enable_device_metadata=MagicMock())

    builder.enable_device_metadata()

    assert builder._device_metadata_enabled
    builder._global_metadata_builder.enable_device_metadata.assert_called_once_with()


@pytest.mark.parametrize("local_num_actual_tokens", [2, 0], ids=["local_tokens", "empty_rank"])
def test_pcp_forward_updates_global_caches_before_local_attention(
    local_num_actual_tokens: int,
):
    """Exercise batched cache preparation, local attention, and empty ranks."""
    impl = _make_impl(AscendDSAPCPImpl)
    impl.compress_ratio = 4
    impl.compressor = SimpleNamespace(
        state_cache=SimpleNamespace(prefix="compressor.state_cache"),
    )
    impl.indexer = SimpleNamespace(
        skip_topk=False,
        k_cache=SimpleNamespace(prefix="indexer.k_cache"),
        compressor=SimpleNamespace(
            state_cache=SimpleNamespace(prefix="indexer.compressor.state_cache"),
        ),
        update_cache=MagicMock(),
    )

    global_num_tokens = 2
    restore_idx = torch.tensor([0, 5], dtype=torch.int64)
    cache_prefixes = [
        "layer",
        "compressor.state_cache",
        "indexer.compressor.state_cache",
        "indexer.k_cache",
        "swa_cache",
    ]
    global_dsa_metadata_by_prefix = {
        cache_prefix: AscendDSAMetadata(
            num_actual_tokens=global_num_tokens,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
        )
        for cache_prefix in cache_prefixes
    }
    req_metadata = AscendDSAReqMetadata(
        block_table=torch.tensor([[1]], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        slot_mapping=None,
        storage_block_size=32,
        query_start_loc=torch.tensor([0, local_num_actual_tokens], dtype=torch.int32),
        sin=cast(Any, {"layer": torch.zeros((4, 1, 1, 2))}),
        cos=cast(Any, {"layer": torch.ones((4, 1, 1, 2))}),
    )
    attn_metadata = {
        cache_prefix: AscendDSAPCPMetadata(
            num_actual_tokens=local_num_actual_tokens,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=int(local_num_actual_tokens > 0),
            req_metadata=req_metadata if local_num_actual_tokens else None,
            local_num_tokens_after_padding=4,
            hidden_restore_idx=restore_idx,
            global_dsa_metadata=global_dsa_metadata_by_prefix[cache_prefix],
        )
        for cache_prefix in cache_prefixes
    }
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    gathered_hidden_states = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    attention_output = torch.arange(
        local_num_actual_tokens * 2,
        dtype=torch.float32,
    ).reshape(local_num_actual_tokens, 1, 2)
    output = torch.full((4, 2), -1.0)
    captured_o_proj_inputs: list[torch.Tensor] = []
    pcp_group = SimpleNamespace(
        all_gather=MagicMock(return_value=gathered_hidden_states),
    )

    def fake_o_proj(o_proj_input: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
        captured_o_proj_inputs.append(o_proj_input.clone())
        output_tensor[:local_num_actual_tokens].copy_(
            o_proj_input[:local_num_actual_tokens].reshape(local_num_actual_tokens, 2)
        )
        return output_tensor

    caches = tuple(torch.empty(0) for _ in range(6))
    with (
        patch.object(
            torch.ops.vllm,
            "maybe_all_gather_and_maybe_unpad",
            create=True,
            side_effect=lambda tensor, _: tensor,
        ),
        patch.object(
            torch.ops._C_ascend,
            "inplace_partial_rotary_mul",
            create=True,
        ),
        patch.object(
            DeviceOperator,
            "unpack_dsa_forward_kv_cache",
            return_value=caches,
        ),
        patch("vllm_ascend.attention.context_parallel.dsa_cp.get_pcp_group", return_value=pcp_group),
        patch.object(impl, "_update_global_swa_cache") as update_swa,
        patch.object(impl, "_update_global_compressor_cache") as update_compressor,
        patch.object(
            impl,
            "_forward_attention",
            return_value=attention_output,
        ) as forward_attention,
        patch.object(impl, "_forward_o_proj", side_effect=fake_o_proj),
        patch("vllm_ascend.attention.dsa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.dsa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.dsa_v1.maybe_save_kv_layer_to_connector"),
    ):
        actual = impl.forward(
            layer_name="layer",
            hidden_states=hidden_states,
            kv_cache=caches,
            attn_metadata=attn_metadata,
            output=output,
        )

    assert actual is output
    assert torch.equal(pcp_group.all_gather.call_args.args[0], hidden_states)
    assert pcp_group.all_gather.call_args.kwargs == {"dim": 0}
    update_swa.assert_called_once()
    update_compressor.assert_called_once()
    impl.indexer.update_cache.assert_called_once()
    expected_global_hidden = gathered_hidden_states.index_select(0, restore_idx)
    assert torch.equal(update_swa.call_args.args[1], expected_global_hidden)
    assert torch.equal(update_compressor.call_args.args[0], expected_global_hidden)
    update_indexer_call = impl.indexer.update_cache.call_args
    assert torch.equal(
        update_indexer_call.kwargs["hidden_states"],
        expected_global_hidden,
    )
    assert update_indexer_call.kwargs["kv_cache"] is caches
    if local_num_actual_tokens == 0:
        forward_attention.assert_not_called()
        assert not captured_o_proj_inputs
        assert torch.count_nonzero(output) == 0
    else:
        assert torch.equal(
            forward_attention.call_args.args[1],
            hidden_states[:local_num_actual_tokens],
        )
        assert forward_attention.call_args.args[4] is True
        assert not forward_attention.call_args.kwargs
        assert len(captured_o_proj_inputs) == 1
        assert torch.count_nonzero(captured_o_proj_inputs[0][local_num_actual_tokens:]) == 0
        assert torch.equal(
            output[:local_num_actual_tokens],
            attention_output.view(local_num_actual_tokens, 2),
        )
