# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.device.device_op import DeviceOperator


def _common_metadata(num_reqs: int = 2, num_tokens: int = 6):
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32) * (num_tokens // num_reqs)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_input_tokens=num_tokens,
        num_actual_tokens=num_tokens,
        positions=torch.arange(num_tokens),
        seq_lens=torch.arange(num_reqs, dtype=torch.int32) + 8,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int32),
        graph_pad_size=0,
        block_table_tensor=torch.zeros((num_reqs, 2), dtype=torch.int32),
        attn_state=None,
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_target_reuses_common_rope_only_for_pure_decode(mixed):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder.decode_threshold = 4
    builder.metadata_cls = SimpleNamespace
    builder.model_config = SimpleNamespace(get_head_size=lambda: 192)
    builder.slot_mapping = torch.zeros((8, 2), dtype=torch.int32)
    builder.get_block_table_size = MagicMock(return_value=2)
    builder.build_prefill_metadata = MagicMock(return_value=object())
    builder.build_decode_metadata = MagicMock(return_value=object())
    builder.set_num_actual_tokens = MagicMock(side_effect=lambda common: setattr(builder, "num_actual_tokens", 6))
    common = _common_metadata()
    split = (1, 1, 3, 3) if mixed else (2, 0, 6, 0)
    common_cos, common_sin = object(), object()

    with (
        patch("vllm_ascend.attention.dsa_v1.split_decodes_and_prefills", return_value=split),
        patch(
            "vllm_ascend.attention.dsa_v1.get_cos_and_sin_dsa",
            return_value=(common_cos, common_sin),
        ),
        patch.object(DeviceOperator, "format_dsa_slot_mapping", return_value=torch.zeros((6, 2))),
    ):
        builder.build(
            0,
            common,
            common_ratio_to_sas_metadata={},
            prefill_ratio_to_sas_metadata={},
            decode_ratio_to_sas_metadata={},
        )

    decode_kwargs = builder.build_decode_metadata.call_args.kwargs
    assert decode_kwargs["common_cos"] is (None if mixed else common_cos)
    assert decode_kwargs["common_sin"] is (None if mixed else common_sin)


@pytest.mark.parametrize("mixed", [False, True])
def test_draft_reuses_common_rope_only_for_pure_decode(mixed):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder.compressor_ratio = 1
    builder.decode_threshold = 4
    builder.block_size = 8
    builder.rope_layer_names = ("mtp.0.self_attn.attn",)
    builder.spec_slot_mapping = [torch.zeros((8, 2), dtype=torch.int32)]
    builder.metadata_cls = SimpleNamespace
    builder.model_config = SimpleNamespace(get_head_size=lambda: 192)
    builder.build_prefill_metadata_for_drafting = MagicMock(return_value=object())
    builder.build_decode_metadata_for_drafting = MagicMock(return_value=object())
    common = _common_metadata()
    split = (1, 1, 3, 3) if mixed else (2, 0, 6, 0)
    common_cos, common_sin = object(), object()

    with (
        patch("vllm_ascend.attention.dsa_v1.split_decodes_and_prefills", return_value=split),
        patch(
            "vllm_ascend.attention.dsa_v1.get_cos_and_sin_dsa",
            return_value=(common_cos, common_sin),
        ) as get_rope,
        patch.object(DeviceOperator, "format_dsa_slot_mapping", return_value=torch.zeros((6, 2))),
    ):
        builder.build_for_drafting(common, draft_index=1)

    assert get_rope.call_args.kwargs["layer_names"] == builder.rope_layer_names
    decode_kwargs = builder.build_decode_metadata_for_drafting.call_args.kwargs
    assert decode_kwargs["common_cos"] is (None if mixed else common_cos)
    assert decode_kwargs["common_sin"] is (None if mixed else common_sin)


def test_dsa_cp_draft_filters_rope_to_consuming_layers():
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.compressor_ratio = 1
    builder.decode_threshold = 2
    builder.block_size = 8
    builder.rope_layer_names = ("mtp.0.self_attn.attn",)
    builder.spec_slot_mapping = [torch.zeros((8, 2), dtype=torch.int32)]
    builder.metadata_cls = SimpleNamespace
    builder.model_config = SimpleNamespace(get_head_size=lambda: 192)
    req_metadata = object()
    builder.build_req_metadata_for_drafting = MagicMock(return_value=req_metadata)
    common = _common_metadata()
    common._seq_lens_cpu = common.seq_lens
    common.seq_lens_cpu = common.seq_lens
    common.causal = True
    cos, sin = object(), object()

    with (
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.split_decodes_and_prefills",
            return_value=(2, 0, 6, 0),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.get_cos_and_sin_dsa",
            return_value=(cos, sin),
        ) as get_rope,
        patch.object(DeviceOperator, "format_dsa_slot_mapping", return_value=torch.zeros((6, 2))),
    ):
        metadata = builder.build_for_drafting(common, draft_index=1)

    get_rope.assert_called_once()
    assert torch.equal(get_rope.call_args.args[0], common.positions.long())
    assert get_rope.call_args.kwargs == {
        "use_cache": False,
        "layer_names": builder.rope_layer_names,
    }
    assert metadata.cos is cos
    assert metadata.sin is sin
    assert metadata.req_metadata is req_metadata
