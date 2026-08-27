# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.attention.dsa_v1 import (
    get_or_compute_compressor_metadata,
    reset_compressor_metadata_cache,
)
from vllm_ascend.device.device_op import DeviceOperator


def _make_forward_context() -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
    )


def test_reuses_by_cache_group_and_resets_between_substeps():
    class PrefillMetadata(SimpleNamespace):
        pass

    class DecodeMetadata(SimpleNamespace):
        pass

    metadata = PrefillMetadata(
        cache_group_key="model.layers.0.self_attn.attn",
        full_compress_cos=torch.zeros((2, 1, 1, 4)),
        full_compress_sin=torch.zeros((2, 1, 1, 4)),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=32,
        num_compressed_tokens=1,
        num_reqs_actual=1,
    )
    same_group_metadata = PrefillMetadata(**vars(metadata))
    same_group_decode_metadata = DecodeMetadata(**vars(metadata))
    other_group_metadata = PrefillMetadata(
        **{
            **vars(metadata),
            "cache_group_key": "model.layers.0.self_attn.indexer.k_cache",
        }
    )
    outputs = [(torch.full((1,), value),) * 3 for value in range(5)]

    with (
        patch.object(
            DeviceOperator,
            "get_dsa_compressor_slot_mapping_format",
            return_value=0,
        ),
        patch.object(
            torch.ops._C_ascend,
            "compressor_metadata",
            create=True,
            side_effect=outputs,
        ) as metadata_op,
    ):
        with override_forward_context(_make_forward_context()):
            first = get_or_compute_compressor_metadata(metadata, 4)
            reused = get_or_compute_compressor_metadata(same_group_metadata, 4)
            decode = get_or_compute_compressor_metadata(same_group_decode_metadata, 4)
            isolated = get_or_compute_compressor_metadata(other_group_metadata, 4)
            reset_compressor_metadata_cache()
            next_substep = get_or_compute_compressor_metadata(metadata, 4)
        with override_forward_context(_make_forward_context()):
            next_forward = get_or_compute_compressor_metadata(metadata, 4)

    assert first is reused
    assert decode is not first
    assert isolated is not first
    assert next_substep is not first
    assert next_forward is not first
    assert metadata_op.call_count == 5
