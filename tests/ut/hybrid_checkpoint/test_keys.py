#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for frontier / locator prefix-cache keys."""

import hashlib
from types import SimpleNamespace

from vllm_ascend.hybrid_checkpoint.keys import (
    FRONTIER_LOCATOR_TOKENS,
    frontier_key,
    frontier_locator_key,
    frontier_locator_unit,
)

HASH_BLOCK = 4


def _hash_fn(value) -> bytes:
    return hashlib.sha256(repr(value).encode()).digest()


def _request(tokens: list[int], block_hashes: list[bytes] | None = None):
    blocks = len(tokens) // HASH_BLOCK
    return SimpleNamespace(
        all_token_ids=tokens,
        block_hashes=block_hashes or [b"blk%d" % i for i in range(blocks)],
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )


def test_frontier_key_is_deterministic():
    tokens = list(range(100))
    first = frontier_key(_hash_fn, _request(tokens), 18, HASH_BLOCK)
    second = frontier_key(_hash_fn, _request(tokens), 18, HASH_BLOCK)

    assert first == second
    assert len(first) == 32


def test_frontier_key_depends_on_length():
    tokens = list(range(100))
    assert frontier_key(_hash_fn, _request(tokens), 18, HASH_BLOCK) != frontier_key(
        _hash_fn, _request(tokens), 19, HASH_BLOCK
    )


def test_frontier_key_depends_on_parent_block_hash():
    tokens = list(range(100))
    left = frontier_key(_hash_fn, _request(tokens), 18, HASH_BLOCK)
    # num_tokens=18 makes the parent block_hashes[3].
    changed = [b"blk0", b"blk1", b"blk2", b"other"] + [b"blk%d" % i for i in range(4, 25)]
    right = frontier_key(_hash_fn, _request(tokens, changed), 18, HASH_BLOCK)

    assert left != right


def test_frontier_key_last_token_override_changes_the_key():
    tokens = list(range(100))
    plain = frontier_key(_hash_fn, _request(tokens), 18, HASH_BLOCK)
    replaced = frontier_key(_hash_fn, _request(tokens), 18, HASH_BLOCK, last_token=999)

    assert plain != replaced


def test_locator_unit_sits_inside_the_frontier():
    assert frontier_locator_unit(FRONTIER_LOCATOR_TOKENS, HASH_BLOCK) == 0
    assert frontier_locator_unit(FRONTIER_LOCATOR_TOKENS + 3, HASH_BLOCK) == 0
    assert frontier_locator_unit(2 * FRONTIER_LOCATOR_TOKENS, HASH_BLOCK) == FRONTIER_LOCATOR_TOKENS // HASH_BLOCK


def test_locator_key_is_shared_by_requests_with_the_same_window():
    tokens = list(range(100))
    left = frontier_locator_key(_hash_fn, _request(tokens), 1, HASH_BLOCK)
    right = frontier_locator_key(_hash_fn, _request(tokens), 1, HASH_BLOCK)
    other = frontier_locator_key(_hash_fn, _request(tokens), 2, HASH_BLOCK)

    assert left == right
    assert left != other
