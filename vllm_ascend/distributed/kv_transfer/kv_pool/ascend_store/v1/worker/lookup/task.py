"""Build executable classic Lookup tasks from content hashes and topology."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase


@dataclass(frozen=True, slots=True)
class LookupTask:
    """Backend keys and result layout required by one Worker Lookup."""

    chunk_ends: tuple[int, ...]
    backend_keys: tuple[str, ...]
    num_ranks: int


class LookupTaskBuilder:
    """Resolve Lookup inputs into rank-expanded Backend keys."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        tp_size: int,
        pp_size: int,
        dcp_size: int,
        num_kv_heads: int,
    ) -> None:
        self.token_database = token_database
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dcp_size = dcp_size
        self.num_kv_heads = num_kv_heads

    def build(self, token_len: int, block_hashes: list[BlockHash] | list[str]) -> LookupTask:
        chunks = list(self.token_database.process_token_key_strings(token_len, block_hashes))
        if not chunks:
            return LookupTask((), (), 0)

        keys = [key for _, _, key, _ in chunks]
        rank_keys = self._expand_rank_keys(keys)
        return LookupTask(
            chunk_ends=tuple(end for _, end, _, _ in chunks),
            backend_keys=tuple(rank_keys),
            num_ranks=len(rank_keys) // len(keys),
        )

    def _expand_rank_keys(self, keys: list[str]) -> list[str]:
        rank_keys = []
        num_head_ranks = min(self.tp_size, self.num_kv_heads)
        # Keep each rank's chunks contiguous so exists results remain [rank][chunk].
        for pp_rank in range(self.pp_size):
            for dcp_rank in range(self.dcp_size):
                for head_rank in range(num_head_ranks):
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
