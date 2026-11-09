# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU input preparation tests for the V2 model runner."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.worker.gpu.input_batch import combine_sampled_and_draft_tokens

from vllm_ascend.worker.v2.input_batch import AscendInputBuffers


@pytest.fixture(autouse=True)
def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU input preparation requires an available NPU")
    torch.npu.set_device(0)


@pytest.mark.parametrize("num_new", [0, 1])
@pytest.mark.parametrize("capture", [False, True])
def test_target_logits_workspace_matches_core_with_changing_inputs(num_new, capture):
    owners = [AscendInputBuffers(3, 32, torch.device("npu:0")) for _ in range(2)]
    for owner in owners:
        owner.bind_logits_indices_workspace()
    assert owners[0].logits_indices_workspace.data_ptr() != owners[1].logits_indices_workspace.data_ptr()
    count_sets = (
        (((1, 4), (4, 1), (2, 3)), ((1, 1),) * 3)
        if num_new
        else (
            ((0, 3), (3, 0), (1, 2)),
            ((0, 1), (1, 0), (0, 1)),
        )
    )
    graphs, retained_metadata = [], []
    try:
        for owner_index, owner in enumerate(owners):
            cpu = {
                "mapping": torch.tensor([2, 0], dtype=torch.int32),
                "sampled": torch.empty(3, dtype=torch.int32),
                "starts": torch.tensor([0, 4, 8], dtype=torch.int32),
                "lengths": torch.tensor([16, 4], dtype=torch.int32),
                "prefill": torch.tensor([4, 20, 9], dtype=torch.int32),
                "drafts": torch.empty((3, 3), dtype=torch.int32),
                "cumulative": torch.empty(3, dtype=torch.int32),
            }
            device = {name: value.to("npu") for name, value in cpu.items()}
            retained_metadata.append((cpu, device))

            def update(change, counts, owner_index, owner, cpu, device):
                salt = owner_index * 100 + change * 10
                cpu["mapping"].copy_(torch.tensor([2, 0] if change % 2 == 0 else [0, 2], dtype=torch.int32))
                cpu["sampled"].copy_(torch.arange(3, dtype=torch.int32) + 1000 + salt)
                cpu["drafts"].copy_((torch.arange(9, dtype=torch.int32) + 2000 + salt).reshape(3, 3))
                cpu["cumulative"].copy_(torch.tensor([0, counts[0], sum(counts)], dtype=torch.int32))
                for name, value in cpu.items():
                    device[name].copy_(value)
                initial = torch.arange(32, dtype=torch.int32) + 70 + salt
                owner.input_ids.copy_(initial)
                owner.logits_indices_workspace.fill_(-777)
                return initial

            for counts_sequence in count_sets:
                num_logits = sum(counts_sequence[0])
                update(0, counts_sequence[0], owner_index, owner, cpu, device)
                output = owner.combine_sampled_and_draft_tokens(owner.input_ids, *device.values(), num_logits, num_new)
                torch.npu.synchronize()
                graph = None
                if capture:
                    graph = torch.npu.NPUGraph()
                    graphs.append(graph)
                    with torch.npu.graph(graph):
                        output = owner.combine_sampled_and_draft_tokens(
                            owner.input_ids, *device.values(), num_logits, num_new
                        )
                for change, counts in enumerate(counts_sequence):
                    initial = update(change, counts, owner_index, owner, cpu, device)
                    if graph is None:
                        output = owner.combine_sampled_and_draft_tokens(
                            owner.input_ids, *device.values(), num_logits, num_new
                        )
                    else:
                        graph.replay()
                    baseline_ids = initial.to("npu")
                    baseline = combine_sampled_and_draft_tokens(baseline_ids, *device.values(), num_logits, num_new)
                    torch.testing.assert_close(output.cpu(), baseline.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(owner.input_ids.cpu(), baseline_ids.cpu(), rtol=0, atol=0)
                    assert output.data_ptr() == owner.logits_indices_workspace.data_ptr()
                    assert torch.all(owner.logits_indices_workspace[num_logits:].cpu() == -777)
            empty = dict(device)
            empty["mapping"] = device["mapping"][:0]
            empty["starts"] = device["starts"][:1]
            empty["lengths"] = device["lengths"][:0]
            empty["cumulative"] = device["cumulative"][:1]
            assert owner.combine_sampled_and_draft_tokens(owner.input_ids, *empty.values(), 0, num_new).numel() == 0
            assert combine_sampled_and_draft_tokens(owner.input_ids, *empty.values(), 0, num_new).numel() == 0
            with pytest.raises(ValueError, match="target input capacity"):
                owner.combine_sampled_and_draft_tokens(owner.input_ids, *device.values(), 33, num_new)
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
