"""Own finer request hashes when the engine grain cannot name v1 objects."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request


class RequestHashes:
    """Reuse engine hashes, or incrementally hash the same append-only request.

    A shallow request view preserves vLLM's token, multimodal, LoRA, and salt
    inputs. Only its hash list and prompt-embedding hash cache are v1-owned.
    Request completion and preemption discard the corresponding entry.
    """

    def __init__(self, block_hasher: Callable[[Request], list[BlockHash]] | None = None) -> None:
        self._block_hasher = block_hasher
        self._requests: dict[str, tuple[Request, Request]] = {}

    def get(self, request: Request) -> tuple[BlockHash, ...]:
        if self._block_hasher is None:
            return tuple(request.block_hashes)
        previous = self._requests.get(request.request_id)
        view = copy(request)
        if previous is not None and previous[0] is request:
            view.block_hashes = previous[1].block_hashes
            view._prompt_embeds_per_block_hashes = previous[1]._prompt_embeds_per_block_hashes
        else:
            view.block_hashes = []
            view._prompt_embeds_per_block_hashes = {}
        view.block_hashes.extend(self._block_hasher(view))
        self._requests[request.request_id] = (request, view)
        return tuple(view.block_hashes)

    def discard(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def clear(self) -> None:
        self._requests.clear()
