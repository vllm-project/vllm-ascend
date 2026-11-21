# SPDX-License-Identifier: Apache-2.0
"""Exact bytes, padding, and replay checks for the managed MXFP4 index cache."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.ops.triton.msa.msa_m3_mxfp4_cache import scatter_mxfp4_cache


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available():
        pytest.skip("MiniMax M3 MXFP4 integration requires Ascend 950")
    check_ascend_device_type()
    if not is_950():
        pytest.skip("MiniMax M3 MXFP4 integration requires Ascend 950")


def _reference(packed, scales, slots, keys, cache_scales):
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            keys.view(-1, packed.shape[-1])[slot] = packed[row].cpu()
            cache_scales.view(-1, scales.shape[-1])[slot] = scales[row].cpu()


@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("tokens", [0, 1, 8, 224, 1664])
def test_incremental_cache_preserves_untouched_bytes(width, tokens):
    pages, page_size = 17, 128
    keys = torch.full((pages, page_size, width // 2), 0x35, dtype=torch.uint8, device="npu")
    scale_bytes = torch.full((pages, page_size, width // 64, 2), 127, dtype=torch.uint8, device="npu")
    packed = torch.randint(0, 256, (tokens, width // 2), dtype=torch.uint8, device="npu")
    scales = torch.randint(0, 256, (tokens, width // 32), dtype=torch.uint8, device="npu")
    slots = torch.randperm(pages * page_size, device="npu")[:tokens].clone()
    slots[1::3] = -1
    expected_keys, expected_scales = keys.cpu(), scale_bytes.cpu()
    _reference(packed, scales, slots, expected_keys, expected_scales)
    scatter_mxfp4_cache(packed, scales, slots, keys, scale_bytes.view(torch.float8_e8m0fnu))
    torch.npu.synchronize()
    assert torch.equal(keys.cpu(), expected_keys)
    assert torch.equal(scale_bytes.cpu(), expected_scales)


@pytest.mark.parametrize("width", [64, 128])
def test_cache_graph_replay_reads_changed_inputs_and_slots(width):
    keys = torch.zeros((3, 128, width // 2), dtype=torch.uint8, device="npu")
    scale_bytes = torch.full((3, 128, width // 64, 2), 127, dtype=torch.uint8, device="npu")
    packed = torch.ones((4, width // 2), dtype=torch.uint8, device="npu")
    scales = torch.ones((4, width // 32), dtype=torch.uint8, device="npu")
    slots = torch.tensor([0, 129, -1, 383], device="npu")

    def run():
        scatter_mxfp4_cache(packed, scales, slots, keys, scale_bytes.view(torch.float8_e8m0fnu))

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    torch.npu.synchronize()
    expected_keys, expected_scales = keys.cpu(), scale_bytes.cpu()
    for value in (3, 11, 23):
        packed.fill_(value)
        scales.fill_(127 + value)
        slots.copy_(torch.tensor([value, 129, -1, 383 - value], device="npu"))
        _reference(packed, scales, slots, expected_keys, expected_scales)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(keys.cpu(), expected_keys)
        assert torch.equal(scale_bytes.cpu(), expected_scales)
