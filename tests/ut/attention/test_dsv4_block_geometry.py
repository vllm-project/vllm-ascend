# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.attention.dsa_v1 import get_or_compute_compressor_metadata
from vllm_ascend.device.device_op import DeviceOperator


@pytest.mark.parametrize(
    "compress_ratio",
    [4, 128],
)
def test_compressor_metadata_uses_physical_storage_geometry(
    monkeypatch,
    compress_ratio,
):
    logical_block_table = torch.tensor([[7, 11]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    start_pos = torch.tensor([508], dtype=torch.int32)
    metadata = SimpleNamespace(
        cache_group_key="model.layers.0.self_attn.attn",
        full_compress_cos=torch.zeros((8, 1, 1, 64), dtype=torch.bfloat16),
        full_compress_sin=torch.zeros((8, 1, 1, 64), dtype=torch.bfloat16),
        query_start_loc=query_start_loc,
        start_pos=start_pos,
        block_table=logical_block_table,
        storage_block_size=128,
        num_compressed_tokens=2,
        num_actual_reqs=1,
    )
    captured = {}
    expected = (
        torch.empty((2, 64)),
        torch.empty((2, 64)),
        torch.empty((2, 2), dtype=torch.int32),
    )

    def fake_compressor_metadata(*args):
        captured["args"] = args
        return expected

    monkeypatch.setattr(
        DeviceOperator,
        "get_dsa_compressor_slot_mapping_format",
        staticmethod(lambda: 2),
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "compressor_metadata",
        fake_compressor_metadata,
        raising=False,
    )
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
    )
    with override_forward_context(forward_context):
        result = get_or_compute_compressor_metadata(metadata, compress_ratio)

    assert result is expected
    args = captured["args"]
    assert args[2] is query_start_loc
    assert args[3] is start_pos
    assert args[4] is logical_block_table
    assert args[5] == 128
    assert args[6] == 2
    assert args[7] == compress_ratio
    assert args[8] == 2
    assert args[9] == 1
