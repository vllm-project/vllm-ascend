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

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import (
    DSA_METADATA_BUFFER_SIZE,
    AscendDSAImpl,
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
    AscendDSAReqMetadata,
)
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
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    kv_cache_spec = SimpleNamespace(compress_ratio=compressor_ratio, block_size=128)
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
    )
    sas_metadata = torch.full((DSA_METADATA_BUFFER_SIZE,), 1, dtype=torch.int32)
    qli_metadata = torch.full((DSA_METADATA_BUFFER_SIZE,), 2, dtype=torch.int32)
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
        patch.object(
            DeviceOperator,
            "format_dsa_slot_mapping",
            return_value=torch.zeros((3, 2), dtype=torch.int32),
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
        num_reqs_actual=None,
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
        num_reqs_actual=1,
        cos=torch.ones(2),
        sin=torch.zeros(2),
    )

    assert metadata.num_reqs_actual == 1
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
        block_size=128,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        sin=cast(Any, {"layer": torch.zeros(1)}),
        cos=cast(Any, {"layer": torch.ones(1)}),
    )


def _make_impl() -> AscendDSAImpl:
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
        patch(
            "vllm_ascend.attention.dsa_v1.get_current_vllm_config",
            return_value=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace(use_index_cache=False))
            ),
        ),
    ):
        return AscendDSAImpl(
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
            attn_metadata=[metadata],
            output=output,
        )

    assert actual is output
    assert len(captured_o_proj_input) == 1
    assert torch.equal(captured_o_proj_input[0], unified_output)
    forward_attention.assert_called_once()
    call = forward_attention.call_args
    assert torch.equal(call.args[1], hidden_states)
    assert call.args[3] == [metadata]
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
        block_size=128,
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
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_op",
            return_value=sparse_attn_op,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_base_kwargs",
            return_value={},
        ),
        patch.object(
            DeviceOperator,
            "add_dsa_sparse_attn_extra_kwargs",
            side_effect=add_extra_kwargs,
        ),
        patch("vllm_ascend.attention.dsa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.dsa_v1.record_attention_compute_start"),
    ):
        actual = impl._forward_attention(
            layer_name="layer",
            hidden_states=hidden_states,
            kv_cache=(torch.empty(0),),
            attn_metadata=[metadata],
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
