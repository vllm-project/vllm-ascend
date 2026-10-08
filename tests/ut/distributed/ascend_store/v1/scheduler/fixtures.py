"""Scheduler requests, allocation records, and observable block-pool effects."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import LookupResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    SchedulerConfig,
)


class FakeRemoteLookup:
    def __init__(self, available_end_token: int = 0) -> None:
        self.available_end_token = available_end_token
        self.queries: list[tuple[TokenRange, tuple[int, ...], tuple[bytes, ...]]] = []
        self.closed = False

    def query(self, query_range, transfer_group_ids, block_hashes) -> LookupResult:
        self.queries.append((query_range, transfer_group_ids, block_hashes))
        return LookupResult(self.available_end_token)

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
) -> SchedulerConfig:
    return SchedulerConfig(
        cache_transfer_granularity=granularity,
        hash_block_size=4,
        transfer_group_ids=groups,
        align_state_group_ids=align_groups,
        load_enabled=load,
        store_enabled=store,
        save_decode_cache=save_decode,
        discard_partial_chunks=True,
        use_eagle_block_drop=eagle,
        has_private_state=private,
        expected_worker_count=workers,
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
    return SimpleNamespace(
        req_id=request_id,
        block_ids=block_ids,
        num_computed_tokens=num_computed_tokens,
    )
