# SPDX-License-Identifier: Apache-2.0
"""Real FLA conv calls across prefill, speculation and ordinary decode."""

import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update

from tests.e2e.pull_request.one_card.triton.test_gdn_recover import _padded
from vllm_ascend.ops.triton.mamba.gdn_recover import gdn_commit_conv_history, gdn_save_conv_history


def _oracle(x, weights, initial, lengths):
    outputs, histories, start = [], [], 0
    for state, length in zip(initial, lengths):
        snapshots = [state.clone()]
        for token in x[start : start + length]:
            window = torch.cat((state, token[None]))
            outputs.append(torch.nn.functional.silu((window.float() * weights.float()).sum(0)).bfloat16())
            state = window[1:].clone()
            snapshots.append(state)
        histories.append(snapshots)
        start += length
    return torch.stack(outputs), histories


def test_real_conv_prefill_verify_commit_and_decode():
    torch.manual_seed(37)
    _, conv = _padded((5, 6, 256), dtype=torch.bfloat16)
    _, saved = _padded((5, 3, 256), dtype=torch.bfloat16)
    conv.zero_()
    ids = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    weights = (torch.randn(4, 256) * 0.2).bfloat16()
    weights_device = weights.to("npu")
    x = (torch.randn(12, 256) * 0.2).bfloat16()
    expected, snapshots = _oracle(x, weights, [torch.zeros(3, 256).bfloat16()] * 2, [7, 5])
    output = causal_conv1d_fn(
        x.to("npu"),
        weights_device,
        None,
        conv_states=conv,
        query_start_loc=torch.tensor([0, 7, 12], dtype=torch.int32, device="npu"),
        cache_indices=ids,
        has_initial_state=torch.zeros(2, dtype=torch.bool, device="npu"),
        activation="silu",
        pad_slot_id=-1,
        null_block_id=0,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=2e-2, atol=1e-3)
    history = torch.stack([s[-1] for s in snapshots])
    assert torch.equal(conv[ids.long(), :3].cpu(), history)
    for lengths, counts in (([4, 2], [1, 2]), ([2, 4], [2, 3])):
        cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device="npu")
        x = (torch.randn(sum(lengths), 256) * 0.2).bfloat16()
        expected, snapshots = _oracle(x, weights, history, lengths)
        gdn_save_conv_history(conv, saved, ids, cu)
        output = causal_conv1d_update(
            x.to("npu"),
            conv,
            weights_device,
            activation="silu",
            conv_state_indices=ids,
            num_accepted_tokens=torch.ones_like(ids),
            query_start_loc=cu,
            max_query_len=4,
            null_block_id=0,
            out=torch.empty_like(x, device="npu"),
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=2e-2, atol=1e-3)
        gdn_commit_conv_history(conv, saved, ids, cu, torch.tensor(counts, dtype=torch.int32, device="npu"))
        history = torch.stack([s[n] for s, n in zip(snapshots, counts)])
        assert torch.equal(conv[ids.long(), :3].cpu(), history)
        # Drafter skipping/all drafts trimmed takes this same ordinary path.
        x = (torch.randn(2, 256) * 0.2).bfloat16()
        expected, snapshots = _oracle(x, weights, history, [1, 1])
        output = causal_conv1d_update(
            x.to("npu"),
            conv,
            weights_device,
            activation="silu",
            conv_state_indices=ids,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
            max_query_len=1,
            null_block_id=0,
            out=torch.empty_like(x, device="npu"),
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=2e-2, atol=1e-3)
        history = torch.stack([s[-1] for s in snapshots])
        assert torch.equal(conv[ids.long(), :3].cpu(), history)
    assert torch.count_nonzero(conv[[0, 2, 4]]).cpu() == 0
