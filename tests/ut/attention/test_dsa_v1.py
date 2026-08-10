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
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.device.device_op import DeviceOperator


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
    )
    kv_cache_spec = SimpleNamespace(compress_ratio=compressor_ratio)
    builder = AscendDSAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["model.layers.0.self_attn.attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    # These caches are supplied by model_runner through build() in production.
    builder.prefill_ratio_to_sas_metadata = {}
    builder.decode_ratio_to_sas_metadata = {}
    builder.seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    builder.num_decodes = 2
    return builder


@pytest.mark.parametrize(
    ("compressor_ratio", "expected_cmp_ratio", "expected_cmp_topk", "expected_has_cmp_kv"),
    [
        (1, 1, 0, False),
        (4, 4, 512, True),
        (8, 128, 0, True),
        (128, 128, 0, True),
    ],
)
def test_build_sas_metadata_parameters_cache_and_output_buffer(
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
    generated_metadata = torch.arange(1024, dtype=torch.int32)
    output_buffer = torch.zeros_like(generated_metadata)
    metadata_op = MagicMock(return_value=generated_metadata)

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=metadata_op,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_kwargs",
            return_value={"device": "cpu"},
        ),
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
            output_buffer=output_buffer,
        )

    assert result is generated_metadata
    assert cached_result is output_buffer
    assert torch.equal(output_buffer, generated_metadata)
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


def test_build_qli_metadata_parameters_cache_and_output_buffer():
    builder = _make_builder()
    metadata_cache: dict[str, torch.Tensor] = {}
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    generated_metadata = torch.arange(1024, dtype=torch.int32)
    output_buffer = torch.zeros_like(generated_metadata)

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
            output_buffer=output_buffer,
        )

    assert result is generated_metadata
    assert cached_result is output_buffer
    assert torch.equal(output_buffer, generated_metadata)
    metadata_op.assert_called_once()
    call_kwargs = metadata_op.call_args.kwargs
    assert torch.equal(call_kwargs["actual_seq_lengths_query"], query_start_loc[1:])
    assert torch.equal(call_kwargs["actual_seq_lengths_key"], seq_lens)
    assert call_kwargs["max_seqlen_q"] == 2
    assert call_kwargs["max_seqlen_k"] == 8
    assert call_kwargs["cmp_ratio"] == 4


def test_prefill_and_decode_sas_wrappers_route_phase_specific_inputs():
    builder = _make_builder()
    prefill_result = torch.full((1024,), 1, dtype=torch.int32)
    decode_result = torch.full((1024,), 2, dtype=torch.int32)
    builder._build_sas_metadata = MagicMock(side_effect=[prefill_result, decode_result])
    prefill_query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    prefill_seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    seq_lens_q = torch.tensor([2, 1], dtype=torch.int32)
    decode_query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    decode_seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    decode_cu_seqlens_ori_kv = torch.tensor([0, 8, 14], dtype=torch.int32)
    decode_cu_seqlens_cmp_kv = torch.tensor([0, 2, 4], dtype=torch.int32)

    prefill_actual = builder._build_prefill_sas_metadata(
        "c4",
        prefill_query_start_loc,
        prefill_seq_lens,
        seq_lens_q,
    )
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
    ):
        decode_actual = builder._build_decode_sas_metadata(
            "c4",
            decode_query_start_loc,
            decode_seq_lens,
            max_seqlen_q=1,
            max_seqlen_kv=8,
        )

    assert prefill_actual is prefill_result
    assert decode_actual is decode_result
    prefill_kwargs = builder._build_sas_metadata.call_args_list[0].kwargs
    assert prefill_kwargs["metadata_cache"] is builder.prefill_ratio_to_sas_metadata
    assert prefill_kwargs["cu_seqlens_ori_kv"] is prefill_query_start_loc
    assert prefill_kwargs["cu_seqlens_cmp_kv"] is None
    assert "output_buffer" not in prefill_kwargs
    decode_kwargs = builder._build_sas_metadata.call_args_list[1].kwargs
    assert decode_kwargs["metadata_cache"] is builder.decode_ratio_to_sas_metadata
    assert decode_kwargs["cu_seqlens_ori_kv"] is decode_cu_seqlens_ori_kv
    assert decode_kwargs["cu_seqlens_cmp_kv"] is decode_cu_seqlens_cmp_kv
    assert decode_kwargs["output_buffer"] is builder.decode_sas_metadata


def test_prefill_and_decode_qli_wrappers_route_decode_output_buffer():
    builder = _make_builder()
    prefill_result = torch.full((1024,), 1, dtype=torch.int32)
    decode_result = torch.full((1024,), 2, dtype=torch.int32)
    builder._build_qli_metadata = MagicMock(side_effect=[prefill_result, decode_result])
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([8, 6], dtype=torch.int32)
    seq_lens_q = torch.tensor([2, 1], dtype=torch.int32)

    prefill_actual = builder._build_prefill_qli_metadata(
        query_start_loc,
        seq_lens,
        seq_lens_q,
    )
    decode_actual = builder._build_decode_qli_metadata(
        query_start_loc,
        seq_lens,
        max_seqlen_q=2,
        max_seqlen_kv=8,
    )

    assert prefill_actual is prefill_result
    assert decode_actual is decode_result
    prefill_kwargs = builder._build_qli_metadata.call_args_list[0].kwargs
    assert prefill_kwargs["metadata_cache"] is builder.prefill_ratio_to_sas_metadata
    assert prefill_kwargs["max_seqlen_q"] == 2
    assert prefill_kwargs["max_seqlen_kv"] == 8
    assert "output_buffer" not in prefill_kwargs
    decode_kwargs = builder._build_qli_metadata.call_args_list[1].kwargs
    assert decode_kwargs["metadata_cache"] is builder.decode_ratio_to_sas_metadata
