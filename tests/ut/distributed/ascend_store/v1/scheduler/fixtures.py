"""Scheduler requests, allocation records, and observable block-pool effects."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec, SlidingWindowSpec

from tests.ut.distributed.ascend_store.v1.helpers import make_pool_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import KVPoolConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LookupResult,
    TailKeyBoundary,
)


class FakeLookupClient:
    def __init__(self, available_end_token: int = 0, *, tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()) -> None:
        self.available_end_token = available_end_token
        self.tail_key_boundaries = tail_key_boundaries
        self.queries: list[tuple[tuple[int, int], tuple[int, ...], tuple[bytes, ...]]] = []
        self.closed = False

    def lookup(self, request) -> LookupResult:
        self.queries.append(
            ((request.start_token, request.end_token), request.transfer_group_ids, request.block_hashes)
        )
        return LookupResult(self.available_end_token, self.tail_key_boundaries)

    def close(self) -> None:
        self.closed = True


class FakeBlocks:
    def __init__(self, *groups: list[int]) -> None:
        self.groups = groups

    def get_block_ids(self):
        return self.groups


class FakeBlockPool:
    def __init__(self, block_count: int = 64) -> None:
        self.blocks = [SimpleNamespace(block_id=block_id) for block_id in range(block_count)]
        self.touched: list[tuple[int, ...]] = []
        self.freed: list[tuple[int, ...]] = []

    def touch(self, blocks) -> None:
        self.touched.append(tuple(block.block_id for block in blocks))

    def free_blocks(self, blocks) -> None:
        self.freed.append(tuple(block.block_id for block in blocks))


def make_config(
    *,
    granularity: int = 4,
    groups: tuple[int, ...] = (0,),
    align_groups: frozenset[int] = frozenset(),
    load: bool = True,
    store: bool = False,
    save_decode: bool = False,
    eagle: bool = False,
    private: bool = False,
    workers: int = 1,
    recycling: bool = False,
) -> KVPoolConfig:
    group_specs = []
    for group_id in range(max(groups, default=-1) + 1 + int(private)):
        spec: KVCacheSpec
        if group_id in align_groups:
            spec = MambaSpec(block_size=granularity, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align")
        elif recycling:
            spec = SlidingWindowSpec(
                block_size=granularity, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=granularity
            )
        else:
            spec = FullAttentionSpec(block_size=granularity, num_kv_heads=1, head_size=1, dtype=torch.float32)
        group_specs.append(spec)
    return make_pool_config(
        cache_transfer_granularity=granularity,
        hash_block_size=4,
        transfer_group_ids=groups,
        align_state_group_ids=align_groups,
        load_enabled=load,
        store_enabled=store,
        save_decode_cache=save_decode,
        use_eagle=eagle,
        expected_worker_count=workers,
        group_specs=tuple(group_specs),
    )


def make_request(
    request_id: str = "request",
    *,
    prompt_tokens: int = 8,
    tokens: int | None = None,
    hashes: int | None = None,
):
    token_count = prompt_tokens if tokens is None else tokens
    hash_count = (token_count + 3) // 4 if hashes is None else hashes
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_tokens,
        num_tokens=token_count,
        block_hashes=[bytes((index + 1,)) for index in range(hash_count)],
    )


def make_output(
    *,
    new=(),
    cached=(),
    resumed=(),
    scheduled_tokens=None,
    finished=(),
    preempted=(),
    checkpoints=None,
):
    cached = tuple(cached)
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[entry[0] for entry in cached],
            resumed_req_ids=set(resumed),
            new_block_ids=[entry[1] for entry in cached],
            num_computed_tokens=[entry[2] for entry in cached],
        ),
        num_scheduled_tokens={} if scheduled_tokens is None else scheduled_tokens,
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        kv_connector_block_state=(
            None if checkpoints is None else SimpleNamespace(boundary_state_offloads=checkpoints)
        ),
    )


def new_request_data(request_id: str, block_ids, num_computed_tokens: int):
    return SimpleNamespace(req_id=request_id, block_ids=block_ids, num_computed_tokens=num_computed_tokens)
