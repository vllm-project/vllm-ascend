# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Verify P-side hash truncation against independent full-prompt rebuilding."""

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeBaseConnectorScheduler


@pytest.fixture(autouse=True)
def initialize_hash_state():
    init_none_hash(sha256)


@pytest.fixture
def require_native_hash_truncation():
    if not callable(getattr(Request, "truncate_block_hashes", None)):
        pytest.skip("Requires vLLM Request.truncate_block_hashes")


def make_hash_scheduler(block_size, lookahead):
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.need_truncate = True
    scheduler.kv_cache_config = SimpleNamespace(hash_block_size=block_size)
    scheduler.vllm_config = SimpleNamespace(num_hash_lookahead_tokens=lookahead)
    return scheduler


def make_hashed_request(n, block_size, lookahead, hash_fn=sha256, embeds=False):
    rng = random.Random(9017 + n)
    kwargs = {"prompt_embeds": torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)} if embeds else {}
    request = Request(
        "hash-regression",
        None if embeds else [rng.randrange(1, 100000) for _ in range(n)],
        SamplingParams(max_tokens=32, extra_args={"kv_transfer_params": {"do_remote_decode": True}}),
        None,
        block_hasher=get_request_block_hasher(block_size, hash_fn, lookahead),
        **kwargs,
    )
    return request


def truncate_with_full_rebuild(request):
    if request.prompt_token_ids is not None:
        request.prompt_token_ids.pop()
    else:
        request.prompt_embeds = request.prompt_embeds[:-1]
    request._all_token_ids.pop()
    request.num_prompt_tokens -= 1
    request.max_tokens = 1
    request.block_hashes.clear()
    request.update_block_hashes()
    request.kv_transfer_params["_p_side_truncated"] = True


@pytest.mark.parametrize("block_size", [1, 2, 16, 32])
@pytest.mark.parametrize("lookahead", [0, 1, 5])
@pytest.mark.parametrize("n", [2, 3, 15, 16, 17, 31, 32, 33, 129])
def test_hashes_equal_independent_full_rebuild(block_size, lookahead, n):
    request = make_hashed_request(n, block_size, lookahead)
    reference = make_hashed_request(n, block_size, lookahead)
    existing = list(request.block_hashes)
    make_hash_scheduler(block_size, lookahead).on_new_request(request)
    truncate_with_full_rebuild(reference)
    assert request.block_hashes == reference.block_hashes
    assert request.prompt_token_ids == reference.prompt_token_ids
    assert request.num_prompt_tokens == n - 1 and request.max_tokens == 1
    if callable(getattr(request, "truncate_block_hashes", None)):
        valid = max(0, (n - 1 - lookahead) // block_size)
        assert all(a is b for a, b in zip(request.block_hashes[:valid], existing[:valid]))


@pytest.mark.parametrize("block_size", [2, 16])
@pytest.mark.parametrize("lookahead", [0, 1, 5])
@pytest.mark.parametrize("n", [2, 3, 16, 17, 33])
def test_prompt_embeddings_match_full_rebuild(block_size, lookahead, n):
    request = make_hashed_request(n, block_size, lookahead, embeds=True)
    reference = make_hashed_request(n, block_size, lookahead, embeds=True)
    make_hash_scheduler(block_size, lookahead).on_new_request(request)
    truncate_with_full_rebuild(reference)
    assert request.block_hashes == reference.block_hashes
    assert torch.equal(request.prompt_embeds, reference.prompt_embeds)


@pytest.mark.parametrize("n", [131154, 131155])
@pytest.mark.parametrize("lookahead", [0, 1, 5])
def test_long_context_hashes_match_full_rebuild(n, lookahead):
    counted = Mock(side_effect=sha256)
    request = make_hashed_request(n, 2, lookahead, counted)
    counted.reset_mock()
    make_hash_scheduler(2, lookahead).on_new_request(request)
    if callable(getattr(request, "truncate_block_hashes", None)):
        assert counted.call_count == 0
    else:
        assert counted.call_count == len(request.block_hashes)
    reference = make_hashed_request(n, 2, lookahead)
    truncate_with_full_rebuild(reference)
    assert request.block_hashes == reference.block_hashes


@pytest.mark.parametrize("lookahead", [0, 1, 5])
def test_incomplete_hash_list_only_fills_missing_suffix(lookahead):
    request = make_hashed_request(129, 2, lookahead)
    del request.block_hashes[3:]
    reference = make_hashed_request(129, 2, lookahead)
    make_hash_scheduler(2, lookahead).on_new_request(request)
    truncate_with_full_rebuild(reference)
    assert request.block_hashes == reference.block_hashes


def test_repeated_hook_is_idempotent():
    scheduler = make_hash_scheduler(2, 0)
    request = make_hashed_request(33, 2, 0)
    scheduler.on_new_request(request)
    state = (list(request._all_token_ids), list(request.block_hashes))
    scheduler.on_new_request(request)
    assert state == (request._all_token_ids, request.block_hashes)


def test_embedding_hash_memo_is_invalidated_like_upstream_api(require_native_hash_truncation):
    request = make_hashed_request(33, 2, 0, embeds=True)
    assert request._prompt_embeds_per_block_hashes
    make_hash_scheduler(2, 0).on_new_request(request)
    assert not request._prompt_embeds_per_block_hashes


@pytest.mark.parametrize("block_size", [2, 16])
@pytest.mark.parametrize("lookahead", [0, 1, 5])
def test_uses_native_truncation_api(block_size, lookahead, require_native_hash_truncation):
    request = make_hashed_request(33, block_size, lookahead)
    request.truncate_block_hashes = Mock(wraps=request.truncate_block_hashes)
    reference = make_hashed_request(33, block_size, lookahead)

    make_hash_scheduler(block_size, lookahead).on_new_request(request)
    truncate_with_full_rebuild(reference)

    request.truncate_block_hashes.assert_called_once_with(block_size, lookahead)
    assert request.block_hashes == reference.block_hashes


@pytest.mark.parametrize("native_api", ["missing", None, False])
@pytest.mark.parametrize("embeds", [False, True])
@pytest.mark.parametrize("lookahead", [0, 1, 5])
def test_unavailable_native_api_uses_safe_full_rebuild(monkeypatch, native_api, embeds, lookahead):
    counted = Mock(side_effect=sha256)
    request = make_hashed_request(33, 2, lookahead, counted, embeds=embeds)
    reference = make_hashed_request(33, 2, lookahead, embeds=embeds)
    if native_api == "missing":
        monkeypatch.delattr(Request, "truncate_block_hashes", raising=False)
        assert not hasattr(request, "truncate_block_hashes")
    else:
        monkeypatch.setattr(request, "truncate_block_hashes", native_api, raising=False)
    counted.reset_mock()

    scheduler = make_hash_scheduler(2, lookahead)
    scheduler.on_new_request(request)
    truncate_with_full_rebuild(reference)

    assert counted.call_count > 0
    assert request.block_hashes == reference.block_hashes
    assert request._all_token_ids == reference._all_token_ids
    assert request.num_prompt_tokens == 32 and request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True
    counted.reset_mock()
    scheduler.on_new_request(request)
    assert counted.call_count == 0
    assert request.block_hashes == reference.block_hashes
    assert request.num_prompt_tokens == 32


def test_consumer_is_unchanged():
    scheduler = make_hash_scheduler(2, 0)
    request = make_hashed_request(33, 2, 0)
    request.kv_transfer_params = {"do_remote_prefill": True}
    state = (list(request._all_token_ids), list(request.block_hashes))
    scheduler.on_new_request(request)
    assert state == (request._all_token_ids, request.block_hashes)


@pytest.mark.parametrize("num_tokens", [0, 1])
def test_short_requests_are_unchanged(num_tokens):
    request = make_hashed_request(num_tokens, 2, 0)
    before = (list(request._all_token_ids), list(request.block_hashes), request.max_tokens)
    make_hash_scheduler(2, 0).on_new_request(request)
    assert before == (request._all_token_ids, request.block_hashes, request.max_tokens)
    assert "_p_side_truncated" not in request.kv_transfer_params
