# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu

from vllm_ascend.ops.bf16_sharded_indexer import bf16_paged_indexer_candidates


@torch.inference_mode()
def test_bf16_sharded_indexer_npu_matches_cpu():
    torch_npu.npu.set_device(0)
    generator = torch.Generator().manual_seed(37)
    query = torch.randn((4, 32, 128), generator=generator).bfloat16()
    query[1].zero_()
    weights = torch.randn((4, 32), generator=generator).bfloat16()
    keys = torch.randn((24, 128, 1, 128), generator=generator).bfloat16()
    table = torch.tensor([[*range(2, 20)], [*range(23, 5, -1)]], dtype=torch.int32)
    ends = torch.tensor([2, 3], dtype=torch.int32)
    visible = torch.tensor([127, 2049, 2101, 0], dtype=torch.int32)
    inputs = (query, weights, keys, table, ends, visible)
    expected_ids, expected_scores = bf16_paged_indexer_candidates(*inputs, 2048, first_valid_page=2)
    device_inputs = tuple(tensor.npu() for tensor in inputs)
    ids, scores = bf16_paged_indexer_candidates(*device_inputs, 2048, first_valid_page=2)
    ids, scores = ids.cpu(), scores.cpu()
    # Different FP32 reduction order can reorder nearly tied stochastic rows.
    # Demand exact IDs for deterministic ties and padding, and compare the
    # sorted score distributions plus ID set where the cutoff is separated.
    torch.testing.assert_close(ids[[1, 3]], expected_ids[[1, 3]])
    torch.testing.assert_close(ids[0].sort().values, expected_ids[0].sort().values)
    torch.testing.assert_close(scores, expected_scores, atol=5e-3, rtol=5e-4)
