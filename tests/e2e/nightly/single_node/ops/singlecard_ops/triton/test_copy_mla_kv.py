# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.copy_mla_kv import copy_mla_kv


@pytest.mark.parametrize("tokens", [4, 8, 16])
@torch.inference_mode()
def test_copy_current_kv_to_owned_history_graph(tokens):
    source_storage = torch.randn((4, 2, 128, 576), device="npu", dtype=torch.bfloat16)
    destination_storage = torch.full((4, 3, 128, 576), 42, device="npu", dtype=torch.bfloat16)
    source, destination = source_storage[:, 1], destination_storage[:, 1]
    source_slots = torch.arange(tokens, device="npu", dtype=torch.int64) + 127
    destination_slots = torch.arange(tokens, device="npu", dtype=torch.int64) + 255

    def run():
        copy_mla_kv(source, destination, source_slots, destination_slots)

    def expected_copy():
        expected = destination_storage.cpu().clone()
        src = source.cpu()
        for src_slot, dst_slot in zip(source_slots.cpu().tolist(), destination_slots.cpu().tolist()):
            if src_slot >= 0 and dst_slot >= 0:
                expected[dst_slot // 128, 1, dst_slot % 128] = src[src_slot // 128, src_slot % 128]
        return expected

    source_slots[-1] = -1
    destination_slots[::2] = -1
    expected = expected_copy()
    run()
    torch.testing.assert_close(destination_storage.cpu(), expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()

    # Metadata and input contents change without changing captured addresses.
    for owner_mask in ("odd", "even", "none"):
        destination_storage.fill_(42)
        source_storage.normal_()
        source_slots.copy_(torch.arange(tokens, device="npu", dtype=torch.int64) + 255)
        source_slots[-1] = -1
        destination_slots.copy_(torch.arange(tokens, device="npu", dtype=torch.int64) + 127)
        if owner_mask == "none":
            destination_slots.fill_(-1)
        else:
            destination_slots[0 if owner_mask == "odd" else 1 :: 2] = -1
        expected = expected_copy()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(destination_storage.cpu(), expected, rtol=0, atol=0)
