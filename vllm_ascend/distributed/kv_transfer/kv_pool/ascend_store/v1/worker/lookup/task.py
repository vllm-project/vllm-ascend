"""Build executable Lookup tasks from content hashes and topology."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from ..projection import KVObjectProjection


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

    def __init__(self, num_head_ranks: int, pp_size: int, dcp_size: int) -> None:
        self.num_head_ranks = num_head_ranks
        self.pp_size = pp_size
        self.dcp_size = dcp_size

    def build(self, projection: KVObjectProjection) -> LookupTask:
        if not projection.objects:
            return LookupTask(projection.group_id, (), (), (), 0)

        keys = [kv_object.backend_key for kv_object in projection.objects]
        rank_keys = self._expand_rank_keys(keys)
        return LookupTask(
            group_id=projection.group_id,
            chunk_ends=tuple(kv_object.token_range.end_token for kv_object in projection.objects),
            chunk_hashes=tuple(kv_object.content_hash for kv_object in projection.objects),
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
