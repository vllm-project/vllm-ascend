# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_ascend._310p.sample.rejection_sampler import AscendRejectionSampler310


class _RecordingSampler:
    def __init__(self) -> None:
        self.gather_args: tuple[torch.Tensor, int, torch.Tensor] | None = None

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        max_num_logprobs: int,
        token_ids: torch.Tensor,
    ):
        self.gather_args = (logprobs, max_num_logprobs, token_ids)
        return "gathered"


def _make_sampler(*, processed: bool):
    sampler = AscendRejectionSampler310.__new__(AscendRejectionSampler310)
    torch.nn.Module.__init__(sampler)
    recording_sampler = _RecordingSampler()
    object.__setattr__(sampler, "sampler", recording_sampler)
    sampler.is_logits_logprobs_mode = True
    sampler.is_processed_logprobs_mode = processed
    return sampler, recording_sampler


def _metadata():
    return SimpleNamespace(
        cu_num_sampled_tokens=torch.tensor([6], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2, 3, 4]),
        bonus_logits_indices=torch.tensor([5]),
    )


def test_raw_speculative_logprobs_avoid_index_writes() -> None:
    sampler, recording_sampler = _make_sampler(processed=False)

    logits = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    sampled_token_ids = torch.tensor([[1, 2, -1, -1, -1, -1]], dtype=torch.int32)

    output = sampler._get_logprobs_tensors(
        max_num_logprobs=5,
        metadata=_metadata(),
        logits=logits,
        target_logits=logits[:5] + 100,
        bonus_logits=logits[5:] + 200,
        sampled_token_ids=sampled_token_ids,
    )

    assert output == "gathered"
    assert recording_sampler.gather_args is not None
    gathered_logits, max_num_logprobs, token_ids = recording_sampler.gather_args
    assert max_num_logprobs == 5
    assert torch.equal(token_ids, torch.tensor([1, 2, 0, 0, 0, 0]))
    assert torch.equal(gathered_logits, logits)


def test_processed_speculative_logprobs_use_index_copy() -> None:
    sampler, recording_sampler = _make_sampler(processed=True)

    logits = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    target_logits = logits[:5] + 100
    bonus_logits = logits[5:] + 200

    output = sampler._get_logprobs_tensors(
        max_num_logprobs=5,
        metadata=_metadata(),
        logits=logits,
        target_logits=target_logits,
        bonus_logits=bonus_logits,
        sampled_token_ids=torch.tensor([[1, 2, -1, -1, -1, -1]], dtype=torch.int32),
    )

    assert output == "gathered"
    assert recording_sampler.gather_args is not None
    gathered_logits, _, token_ids = recording_sampler.gather_args
    assert torch.equal(token_ids, torch.tensor([1, 2, 0, 0, 0, 0]))
    assert torch.equal(gathered_logits[:5], target_logits)
    assert torch.equal(gathered_logits[5:], bonus_logits)
