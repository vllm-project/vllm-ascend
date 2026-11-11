# SPDX-License-Identifier: Apache-2.0
"""NPU differential tests for the experimental greedy verifier."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("triton")
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU required", allow_module_level=True)

from vllm_ascend.worker.v2.spec_decode.greedy_verify_experiment import greedy_verify
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


@pytest.mark.parametrize("edge", ["finite", "tie", "nan", "inf"])
@pytest.mark.parametrize("vocab", [1024, 8201])
def test_greedy_verifier_matches_rejection_sampler(edge, vocab):
    # Requests have 0, 1, 2 and 3 drafts; include accept, reject, and -1 placeholder.
    torch.manual_seed(1234)
    lengths = [1, 2, 3, 4]
    cu = [0, 1, 3, 6, 10]
    rows = sum(lengths)
    logits = torch.randn(rows, vocab, dtype=torch.float32)
    draft = torch.zeros(rows, dtype=torch.int64)
    for start, length in zip(cu, lengths):
        for step in range(length - 1):
            draft[start + step + 1] = logits[start + step].argmax()
    draft[cu[2] + 1] = -1
    draft[cu[3] + 2] = (int(draft[cu[3] + 2]) + 1) % vocab
    if edge == "tie":
        logits[0, 0] = logits[0, vocab - 1] = 1000
    elif edge == "nan":
        logits[0, 0] = float("nan")
    elif edge == "inf":
        logits[0, 0] = float("inf")
    device = "npu"
    logits = logits.to(device)
    draft = draft.to(device)
    cu_tensor = torch.tensor(cu, dtype=torch.int32, device=device)
    mapping = torch.arange(len(lengths), dtype=torch.int32, device=device)
    expanded = torch.tensor([i for i, n in enumerate(lengths) for _ in range(n)], dtype=torch.int32, device=device)
    local_pos = torch.tensor([i for n in lengths for i in range(n)], dtype=torch.int32, device=device)
    reference, ref_count = rejection_sample(
        target_logits=logits, draft_logits=None, draft_sampled=draft, cu_num_logits=cu_tensor,
        pos=torch.arange(rows, dtype=torch.int32, device=device), idx_mapping=mapping,
        expanded_idx_mapping=expanded, expanded_local_pos=local_pos,
        temperature=torch.zeros(len(lengths), dtype=torch.float32, device=device),
        seed=torch.zeros(len(lengths), dtype=torch.int64, device=device),
        num_speculative_steps=3, use_entropy_verify=False,
    )
    actual, actual_count = greedy_verify(logits, draft, cu_tensor, 3)
    torch.npu.synchronize()
    torch.testing.assert_close(actual_count.cpu(), ref_count.cpu(), rtol=0, atol=0)
    for idx, count in enumerate(ref_count.cpu().tolist()):
        torch.testing.assert_close(actual[idx, :count].cpu(), reference[idx, :count].cpu(), rtol=0, atol=0)
