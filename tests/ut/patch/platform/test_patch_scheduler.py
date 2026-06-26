# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.patch.platform import patch_scheduler


class _KVCacheManager:
    def __init__(self, block_ids_by_req):
        self.block_ids_by_req = block_ids_by_req

    def get_block_ids(self, req_id):
        return self.block_ids_by_req[req_id]


def test_update_requests_with_invalid_blocks_handles_hybrid_groups() -> None:
    request = SimpleNamespace(request_id="req", num_computed_tokens=256)
    scheduler = SimpleNamespace(
        block_size=128,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=256)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
            ],
        ),
        kv_cache_manager=_KVCacheManager({"req": ([10], [20, 21])}),
    )

    affected_req_ids, total_affected_tokens, blocks_to_evict = patch_scheduler._update_requests_with_invalid_blocks(
        scheduler,
        [request],
        {10, 21},
        {},
    )

    assert affected_req_ids == {"req"}
    assert request.num_computed_tokens == 0
    assert total_affected_tokens == 256
    assert blocks_to_evict == {10, 20, 21}
