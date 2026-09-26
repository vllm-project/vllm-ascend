"""Build executable Lookup tasks from content hashes and topology."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ..coordinator import LookupChunkSelection
from .request import WorkerLookupRequest


@dataclass(frozen=True, slots=True)
class LookupTask:
    """Backend keys and result layout required by one cache-group Lookup."""

    group_id: int
    chunk_ends: tuple[int, ...]
    chunk_hashes: tuple[BlockHash | str, ...]
    backend_keys: tuple[str, ...]
    num_ranks: int


class LookupTaskBuilder:
    """Resolve Lookup inputs into rank-expanded Backend keys."""

    def __init__(self, token_database: ChunkedTokenDatabase, num_head_ranks: int, pp_size: int, dcp_size: int) -> None:
        self.token_database = token_database
        self.num_head_ranks = num_head_ranks
        self.pp_size = pp_size
        self.dcp_size = dcp_size

    def build(self, request: WorkerLookupRequest, selection: LookupChunkSelection) -> LookupTask:
        group_id = selection.group_id
        block_size = self.token_database.get_block_size(group_id)
        chunks = list(
            self.token_database.process_token_key_strings(
                request.lookup_end_token,
                list(request.block_hashes),
                mask_num=selection.query_start_token,
                kv_cache_group_id=group_id,
                chunk_filter=lambda start: selection.includes(start, block_size),
            )
        )
        if not chunks:
            return LookupTask(group_id, (), (), (), 0)

        keys = [key for _, _, key, _ in chunks]
        rank_keys = self._expand_rank_keys(keys)
        return LookupTask(
            group_id=group_id,
            chunk_ends=tuple(end for _, end, _, _ in chunks),
            chunk_hashes=tuple(chunk_hash for _, _, _, chunk_hash in chunks),
            backend_keys=tuple(rank_keys),
            num_ranks=len(rank_keys) // len(keys),
        )

    def _expand_rank_keys(self, keys: list[str]) -> list[str]:
        rank_keys = []
        # Keep each rank's chunks contiguous so exists results remain [rank][chunk].
        for pp_rank in range(self.pp_size):
            for dcp_rank in range(self.dcp_size):
                for head_rank in range(self.num_head_ranks):
                    for key in keys:
                        rank_key = self._replace_key_rank(key, "dcp", dcp_rank)
                        rank_key = self._replace_key_rank(rank_key, "head_or_tp_rank", head_rank)
                        rank_keys.append(self._replace_key_rank(rank_key, "pp_rank", pp_rank))
        return rank_keys

    @staticmethod
    def _replace_key_rank(key: str, field: str, rank: int) -> str:
        marker = f"@{field}:"
        value_start = key.index(marker) + len(marker)
        value_end = key.index("@", value_start)
        return f"{key[:value_start]}{rank}{key[value_end:]}"
