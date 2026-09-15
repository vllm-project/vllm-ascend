# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram native lookup and pinned staging require the Ascend runtime."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors.torch import save_file

from vllm_ascend.models.deepseek_v41 import engram_hbm as hbm


@pytest.mark.parametrize("width,count", [(32, 0), (32, 1), (256, 255), (256, 256), (256, 521)])
@pytest.mark.parametrize("pin_output", [False, True])
def test_fused_int8_offload_matches_torch(width, count, pin_output):
    table = hbm.NodeShardedEngram(257, width, SimpleNamespace(size=1, rank=0), storage_format="int8", cpu_offload=True)
    generator = torch.Generator().manual_seed(20260913)
    table.weight.data.copy_(torch.randint(-128, 128, table.weight.shape, generator=generator, dtype=torch.int8))
    table.weight_scale.copy_(torch.rand(table.weight_scale.shape, generator=generator) * 16)
    table.weight.data[0].copy_(torch.arange(width).to(torch.int8))
    table.weight_scale[0].fill_(1.00390625)
    table.weight_scale[1].fill_(-0.0)
    ids = (torch.arange(count * 2) % 257)[::2]
    reference = hbm.dequantize_engram_rows(table.weight[ids], table.weight_scale[ids])
    actual = table.lookup_local(ids, pin_output=pin_output)
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))


def test_fused_cpu_operator_rejects_invalid_contracts():
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    weight = torch.ones(5, 32, dtype=torch.int8)
    scale = torch.ones(5, 1)
    ids = torch.tensor([0, 4])
    output = torch.empty(2, 32, dtype=torch.bfloat16)
    for args in (
        (weight.float(), scale, ids, output),
        (weight, scale, ids.int(), output),
        (weight, scale, ids, output[:1]),
        (weight[:, ::2], scale, ids, output),
    ):
        with pytest.raises(RuntimeError, match="Engram CPU lookup"):
            torch.ops._C_ascend.engram_int8_lookup_cpu(*args)


@pytest.mark.parametrize("storage", ["bf16", "int8", "fp8", "mxfp8"])
def test_offload_loader_and_pinned_reuse(tmp_path, storage):
    from unittest.mock import Mock

    key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    codes = (torch.arange(19 * 32).reshape(19, 32) % 31 - 15).to(
        torch.bfloat16 if storage == "bf16" else torch.int8 if storage == "int8" else torch.float8_e4m3fn
    )
    if storage == "bf16":
        codes.view(torch.int16)[15, :4] = torch.tensor([0, -32768, 32705, 1], dtype=torch.int16)
    scales = (2.0 ** (torch.arange(19)[:, None] % 5 - 2)).to(
        torch.float32 if storage == "int8" else torch.float8_e8m0fnu
    )
    tensors = {key: codes}
    if storage != "bf16":
        tensors[scale_key] = scales
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"other": "missing"}}))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors", scale_key: "weights.safetensors"}})
    )
    query = SimpleNamespace(size=4, rank=3)
    with torch.device("meta"):
        table = hbm.NodeShardedEngram(19, 32, query, device="meta", storage_format=storage, cpu_offload=True)
    assert table.weight.device.type == "cpu"
    if storage != "bf16":
        assert table.weight_scale.device.type == "cpu"
    assert not table.weight.is_pinned()
    table.load_checkpoint(tmp_path, key, chunk_rows=2)
    assert torch.equal(table.weight.view(torch.uint8), codes[15:19].view(torch.uint8))
    ids = torch.tensor([0, 3, 0])
    reference = codes[ids + 15] if storage == "bf16" else (codes.float() * scales.float()).bfloat16()[ids + 15]
    first = table.lookup_local(ids, pin_output=True)
    assert first.is_pinned()
    torch.testing.assert_close(first.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0)
    event = Mock()
    table._offload_events[first.data_ptr()] = event
    second = table.lookup_local(ids, pin_output=True)
    event.synchronize.assert_not_called()
    third = table.lookup_local(ids, pin_output=True)
    event.synchronize.assert_called_once_with()
    assert first.data_ptr() == third.data_ptr() != second.data_ptr()
    # Force eviction and verify a still-used pinned source is fenced first.
    event = Mock()
    table._offload_events[third.data_ptr()] = event
    table._offload_buffer_bytes_limit = 1
    table.lookup_local(ids[:1], pin_output=True)
    event.synchronize.assert_called_once_with()
    assert len(table._offload_buffers) == 1
    assert table.lookup_local(torch.empty(0, 2, dtype=torch.int64)).shape == (0, 2, 32)
    for invalid in (-1, table.weight.shape[0]):
        with pytest.raises(IndexError):
            table.lookup_local(torch.tensor([invalid]))
