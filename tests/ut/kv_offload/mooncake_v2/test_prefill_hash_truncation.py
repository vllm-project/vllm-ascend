# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeBaseConnectorScheduler


@pytest.mark.parametrize("hash_block_size", [2, 16])
@pytest.mark.parametrize("lookahead", [0, 3])
@pytest.mark.parametrize("boundary_offset", [-1, 0, 1])
def test_truncation_preserves_valid_hash_prefix(hash_block_size, lookahead, boundary_offset):
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(hash_block_size, hash_fn, lookahead)

    def make_request(tokens):
        return Request(
            request_id="hash-boundary",
            prompt_token_ids=list(tokens),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=hasher,
        )

    tokens = list(range(2 * hash_block_size + lookahead + boundary_offset))
    request = make_request(tokens)
    request.kv_transfer_params = {"do_remote_decode": True}
    original_hashes = list(request.block_hashes)
    hash_list = request.block_hashes
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.need_truncate = True
    scheduler.hash_block_size = hash_block_size
    scheduler.vllm_config = SimpleNamespace(num_hash_lookahead_tokens=lookahead)

    with patch.object(request, "update_block_hashes", wraps=request.update_block_hashes) as rehash:
        scheduler.on_new_request(request)
        scheduler.on_new_request(request)
        rehash.assert_not_called()

    reference = make_request(tokens[:-1])
    assert request.prompt_token_ids == tokens[:-1]
    assert request.block_hashes is hash_list
    assert request.block_hashes == reference.block_hashes
    assert request.block_hashes == original_hashes[: len(reference.block_hashes)]
    if boundary_offset == 0:
        assert len(request.block_hashes) == len(original_hashes) - 1

    # New hashes must chain from the preserved prefix after the boundary moves.
    suffix = [100 + i for i in range(hash_block_size + lookahead)]
    request.append_output_token_ids(suffix)
    assert request.block_hashes == make_request(tokens[:-1] + suffix).block_hashes
