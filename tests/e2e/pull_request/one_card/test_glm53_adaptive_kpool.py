# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder
from vllm_ascend.ops.triton.glm5_next_cann_keypool import build_keypool_metadata, build_tail_inputs


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "acl_graph"])
@torch.inference_mode()
def test_device_partition_masks_stale_padding_and_retains_tail(graph_mode):
    builder = AscendIndexerKPoolMetadataBuilder.__new__(AscendIndexerKPoolMetadataBuilder)
    builder.device = torch.device("npu")
    builder._metadata_buffers = {}
    builder._max_num_metadata_reqs = 4
    builder._max_num_batched_tokens = 8
    common = SimpleNamespace(
        num_reqs=4,
        num_input_tokens=8,
        num_actual_tokens=8,
        query_start_loc=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32, device="npu"),
        seq_lens=torch.tensor([9, 17, 2, 2], dtype=torch.int32, device="npu"),
        positions=torch.zeros(8, dtype=torch.int64, device="npu"),
        slot_mapping=torch.ones(8, dtype=torch.int64, device="npu"),
        block_table_tensor=torch.tensor([[1], [2], [0], [0]], dtype=torch.int32, device="npu"),
    )
    buffers = builder._get_metadata_buffers(common)
    tail_buffers = {
        "history": torch.empty(4, 4, dtype=torch.int64, device="npu"),
        **{name: torch.empty(4, 9, dtype=torch.int64, device="npu") for name in ("writes", "state", "retained")},
    }

    def run():
        build_keypool_metadata(common, buffers, 8, 8, 128, 4)
        build_tail_inputs(common, 4, 9, 16, 6, tail_buffers)

    graph = None
    if graph_mode:
        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
    for lengths in ([1, 6, 0, 0], [6, 1, 0, 0], [0, 0, 0, 0]):
        starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        histories = [7, 15, 0, 0]
        positions = [pos for history, length in zip(histories, lengths) for pos in range(history, history + length)]
        slots = [
            128 * (r + 1) + pos
            for r, (history, length) in enumerate(zip(histories, lengths))
            for pos in range(history, history + length)
        ]
        # Poison the graph tail with valid-looking pool-boundary slots.
        positions += [3] * (8 - len(positions))
        slots += [999] * (8 - len(slots))
        common.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32))
        common.seq_lens.copy_(torch.tensor([h + n if n else 0 for h, n in zip(histories, lengths)], dtype=torch.int32))
        common.positions.copy_(torch.tensor(positions))
        common.slot_mapping.copy_(torch.tensor(slots))
        run() if graph is None else graph.replay()
        torch.testing.assert_close(buffers.query_start_loc[:5].cpu(), torch.tensor(starts, dtype=torch.int32))
        expected_slots = [
            slot // 128 * 32 + slot % 128 // 4 if i < sum(lengths) and (pos + 1) % 4 == 0 else -1
            for i, (slot, pos) in enumerate(zip(slots, positions))
        ]
        torch.testing.assert_close(buffers.slot_mapping.cpu(), torch.tensor(expected_slots))
        expected_compact = [-1] + [
            slot // 128 * 32 + slot % 128 // 4
            for slot, pos in zip(slots[: sum(lengths)], positions[: sum(lengths)])
            if (pos + 1) % 4 == 0
        ]
        expected_compact += [-1] * (7 - len(expected_compact))
        torch.testing.assert_close(buffers.pooled_slot_mapping[:7].cpu(), torch.tensor(expected_compact))
        writes = tail_buffers["writes"].flatten()[:24].view(4, 6).cpu()
        retained = tail_buffers["retained"].flatten()[:24].view(4, 6).cpu()
        for r, length in enumerate(lengths):
            for col in range(6):
                source = starts[r + 1] - 6 + col
                assert retained[r, col] == max(source, 0)
                expected = (r + 1) * 16 + positions[source] % 9 if source >= starts[r] and length else -1
                assert writes[r, col] == expected
