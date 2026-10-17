# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.recoverssm import KDARecoverSSMAdapter
from vllm_ascend.ops.kda import run_recurrent_kda


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_sample_counts_commit_and_next_verify(dtype):
    torch.manual_seed(61566)
    batch, heads, dim, length = 3, 8, 128, 6
    device = "npu"
    indices = torch.arange(1, batch * length + 1, device=device, dtype=torch.int32).reshape(batch, length)
    original = torch.randn(batch * length + 1, heads, dim, dim, device=device).to(dtype) * 0.05
    native, probe = original.clone(), original.clone()
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * length
    previous = torch.ones(batch, device=device, dtype=torch.int32)
    a_log, bias = (
        torch.randn(heads, device=device) * 0.1,
        torch.randn(heads * dim, device=device) * 0.1,
    )
    metadata = SimpleNamespace(spec_sequence_masks=torch.tensor([False, True, False, True, True], device=device))
    adapter = KDARecoverSSMAdapter(length)
    spec_rows = [1, 3, 4]
    adapter.commit(None)  # Prefill/non-spec steps have no pending verification.
    for accepted_list in [[1, 3, 6], [4, 2, 5], [6, 1, 3], [2, 6, 1]]:
        q, k, v, gate = [
            torch.randn(1, batch * length, heads, dim, dtype=torch.bfloat16, device=device) for _ in range(4)
        ]
        beta = torch.randn(1, batch * length, heads, dtype=torch.bfloat16, device=device)
        expected = run_recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            native,
            starts,
            indices,
            a_log,
            bias,
            lower_bound=-5.0,
            beta_is_preprocessed=False,
            num_accepted_tokens=previous,
        )
        actual = adapter.verify(
            q,
            k,
            v,
            gate,
            beta,
            probe,
            starts,
            indices,
            a_log,
            bias,
            -5.0,
            previous,
            metadata.spec_sequence_masks,
        )
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
        with pytest.raises(RuntimeError, match="not been committed"):
            adapter.verify(
                q,
                k,
                v,
                gate,
                beta,
                probe,
                starts,
                indices,
                a_log,
                bias,
                -5.0,
                previous,
                metadata.spec_sequence_masks,
            )
        sampled_ids = torch.full((5, length), -1, device=device, dtype=torch.int64)
        sampled_ids[:, 0] = 1
        for row, count in zip(spec_rows, accepted_list):
            sampled_ids[row, :count] = torch.arange(count, device=device)
        adapter.commit(sampled_ids)
        assert adapter.pending is None
        accepted = torch.tensor(accepted_list, device=device, dtype=torch.int32)
        for row, count in enumerate(accepted_list):
            selected = native[indices[row, count - 1].long()]
            for position in range(length):
                torch.testing.assert_close(
                    probe[indices[row, position].long()].float(),
                    selected.float(),
                    atol=2e-2,
                    rtol=2e-2,
                )
        previous = accepted
