"""DSA sparse-cache extensions for scheduler output dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import vllm.v1.core.sched.output as sched_output

_BaseNewRequestData = sched_output.NewRequestData
_BaseCachedRequestData = sched_output.CachedRequestData
_BaseSchedulerOutput = sched_output.SchedulerOutput


@dataclass
class NewRequestData(_BaseNewRequestData):
    block_hashes: list[Any] | None = None

    @classmethod
    def from_request(
        cls,
        request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
            block_hashes=list(request.block_hashes),
        )


@dataclass
class CachedRequestData(_BaseCachedRequestData):
    # ``block_hashes[i]`` is a suffix starting at
    # ``block_hash_starts[i]``, not a full request-lifetime snapshot.
    block_hash_starts: list[int] = field(default_factory=list)
    block_hashes: list[list[Any]] = field(default_factory=list)

    def anon_repr(self) -> str:
        base_repr = super().anon_repr()
        insert_at = base_repr.rfind(")")
        block_hash_deltas = list(zip(
            self.block_hash_starts,
            [len(hashes) for hashes in self.block_hashes],
        ))
        if insert_at == -1:
            return f"{base_repr},block_hash_deltas={block_hash_deltas}"
        return (
            f"{base_repr[:insert_at]},"
            f"block_hash_deltas={block_hash_deltas}"
            f"{base_repr[insert_at:]}"
        )

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
            block_hash_starts=[],
            block_hashes=[],
        )


@dataclass
class SchedulerOutput(_BaseSchedulerOutput):
    req_dsa_stage: dict[str, int] | None = None
    req_dsa_resident_valid_seq_len: dict[str, int] | None = None
    req_dsa_sparse_budget_tokens: dict[str, int] | None = None
    req_dsa_target_resident_budget_tokens: dict[str, int] | None = None

    @classmethod
    def make_empty(cls) -> SchedulerOutput:
        output = super().make_empty()
        output.req_dsa_stage = {}
        output.req_dsa_resident_valid_seq_len = {}
        output.req_dsa_sparse_budget_tokens = {}
        output.req_dsa_target_resident_budget_tokens = {}
        return output


sched_output.NewRequestData = NewRequestData
sched_output.CachedRequestData = CachedRequestData
sched_output.SchedulerOutput = SchedulerOutput
