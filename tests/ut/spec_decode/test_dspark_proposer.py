# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

import vllm_ascend.spec_decode.dspark_proposer as dspark_proposer_module
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


class _FakeDSparkModel:
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.long)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        vocab_size = 5
        bias = torch.zeros(
            (markov_embed.numel(), vocab_size),
            dtype=torch.float32,
            device=markov_embed.device,
        )
        next_ids = (markov_embed.to(torch.long) + 1) % vocab_size
        bias.scatter_(1, next_ids.view(-1, 1), 10.0)
        return bias


def test_dspark_sample_sequential_uses_previous_draft_token(monkeypatch):
    monkeypatch.setattr(
        dspark_proposer_module,
        "greedy_sample",
        lambda logits: logits.argmax(dim=-1),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=3,
        model=_FakeDSparkModel(),
        _dspark_seed_buffer=torch.tensor([1, 3], dtype=torch.int64),
        _dspark_draft_buffer=torch.zeros((2, 3), dtype=torch.int64),
    )
    head_hidden = torch.zeros((6, 5), dtype=torch.float32)
    token_indices = torch.arange(6, dtype=torch.int32)

    draft_tokens = AscendDSparkProposer._sample_sequential(
        proposer,
        num_reqs=2,
        head_hidden=head_hidden,
        token_indices_to_sample=token_indices,
    )

    assert draft_tokens.data_ptr() == proposer._dspark_draft_buffer.data_ptr()
    torch.testing.assert_close(
        draft_tokens,
        torch.tensor([[2, 3, 4], [4, 0, 1]], dtype=torch.int64),
    )
