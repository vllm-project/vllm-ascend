# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact v0.33.0 cached-request ABI for testing release-only patches on main.

Copied from vLLM tag v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5):
v1/core/sched/output.py:CachedRequestData and
v1/core/sched/scheduler.py:Scheduler._make_cached_request_data.
The current main scheduler never receives this fixture or these V1 patches.
Main's native payload is tested separately; this fixture keeps the original
release-patch assertions executable after upstream removes their V1 ABI.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request


def release_cached_request_abi() -> tuple[type[Any], type[Any]]:
    @dataclass
    class CachedRequestData:
        req_ids: list[str]
        # For request ids not in resumed_req_ids, new_block_ids will be appended to
        # the request's block IDs. For those in the set, new_block_ids will be used as the
        # request's block IDs instead of appending to the existing block IDs.
        resumed_req_ids: set[str]
        # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
        # When PP is not used, new_token_ids will be empty.
        new_token_ids: list[list[int]]
        # MRV1-only: For requests not scheduled in the last step, propagate the token ids
        # to the connector. Won't contain requests scheduled in the prior step.
        all_token_ids: dict[str, list[int]]
        new_block_ids: list[tuple[list[int], ...] | None]
        num_computed_tokens: list[int]
        num_output_tokens: list[int]

        # Version of dataclass repr with token IDs obfuscated.
        def anon_repr(self) -> str:
            new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
            all_token_ids_lens = {req_id: len(toks) for req_id, toks in self.all_token_ids.items()}
            return (
                f"CachedRequestData("
                f"req_ids={self.req_ids},"
                f"resumed_req_ids={self.resumed_req_ids},"
                f"new_token_ids_lens={new_token_ids_lens},"
                f"all_token_ids_lens={all_token_ids_lens},"
                f"new_block_ids={self.new_block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"num_output_tokens={self.num_output_tokens}"
                f")"
            )

        def __repr__(self) -> str:
            return self.anon_repr()

        @property
        def num_reqs(self) -> int:
            return len(self.req_ids)

        @cached_property
        def _req_id_to_num_output_tokens(self) -> dict[str, int]:
            """Cache mapping of req_id to num_output_tokens for O(1) lookup.

            This cached property is safe because CachedRequestData instances
            are created fresh each scheduling iteration and not mutated during
            computation of iteration details.
            """
            return dict(zip(self.req_ids, self.num_output_tokens))

        def is_context_phase(self, req_id: str) -> bool:
            num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
            return num_output_tokens is not None and num_output_tokens == 0

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
            )

    class Scheduler:
        use_pp: bool
        scheduler_config: Any
        use_v2_model_runner: bool
        prev_step_scheduled_req_ids: set[str]

        def _make_cached_request_data(
            self,
            running_reqs: list[Request],
            resumed_reqs: list[Request],
            num_scheduled_tokens: dict[str, int],
            spec_decode_tokens: dict[str, list[int]],
            req_to_new_blocks: dict[str, KVCacheBlocks],
        ) -> CachedRequestData:
            req_ids: list[str] = []
            new_token_ids: list[list[int]] = []
            new_block_ids: list[tuple[list[int], ...] | None] = []
            all_token_ids: dict[str, list[int]] = {}
            num_computed_tokens: list[int] = []
            num_output_tokens: list[int] = []
            resumed_req_ids = set()

            num_running_reqs = len(running_reqs)
            for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
                req_id = req.request_id
                req_ids.append(req_id)
                # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
                # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
                # omit this payload.
                if self.use_pp and not self.scheduler_config.async_scheduling:
                    # When using PP, the scheduler sends the sampled tokens back,
                    # because there's no direct communication between the first-
                    # stage worker and the last-stage worker. Otherwise, we don't
                    # need to send the sampled tokens back because the model runner
                    # will cache them.
                    num_tokens = num_scheduled_tokens[req_id] - len(spec_decode_tokens.get(req_id, ()))
                    token_ids = req.all_token_ids[req.num_computed_tokens : req.num_computed_tokens + num_tokens]
                    new_token_ids.append(token_ids)
                if idx >= num_running_reqs:
                    resumed_req_ids.add(req_id)
                if not self.use_v2_model_runner:  # noqa: SIM102
                    if req_id not in self.prev_step_scheduled_req_ids:
                        all_token_ids[req_id] = req.all_token_ids.copy()
                new_block_ids.append(req_to_new_blocks[req_id].get_block_ids(allow_none=True))
                num_computed_tokens.append(req.num_computed_tokens)
                num_output_tokens.append(req.num_output_tokens + req.num_output_placeholders)

            return CachedRequestData(
                req_ids=req_ids,
                resumed_req_ids=resumed_req_ids,
                new_token_ids=new_token_ids,
                all_token_ids=all_token_ids,
                new_block_ids=new_block_ids,
                num_computed_tokens=num_computed_tokens,
                num_output_tokens=num_output_tokens,
            )

    return Scheduler, CachedRequestData
