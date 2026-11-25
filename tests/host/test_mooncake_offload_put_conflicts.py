# SPDX-License-Identifier: Apache-2.0
import ast
import threading
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.fixture
def production():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/planes/block_key_offload.py"
    )
    tree = ast.parse(path.read_text())
    namespace = dict(
        copy=copy,
        require_aligned_batch_results=lambda op, keys, results: results,
        get_partial_block_index=lambda *args: None,
        get_block_hashes=lambda hashes, *_: hashes,
        block_hash_to_str=bytes.hex,
        selected=lambda mask, index: mask is None or mask[index],
    )
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.If))]
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec', flags=0x1000000), namespace)
    return namespace


@pytest.mark.parametrize('scenario', ['success', 'public_conflict', 'private_conflict', 'capacity', 'committed'])
def test_conflict_protocol(production, scenario):
    calls = []
    def put(keys, sizes):
        calls.append(keys)
        if scenario == 'success' or (scenario == 'public_conflict' and len(calls) == 2):
            return [0] * len(keys)
        return [-1 if scenario == 'capacity' else -705] * len(keys)
    def missing(keys):
        return [] if scenario == 'committed' and calls else list(keys)
    worker = NS(_filter_pool_existing_keys=missing, _layerwise_key_batches=lambda keys: [keys],
                m_store=NS(batch_put_start=put), _put_started_keys=set(),
                _put_started_keys_lock=threading.Lock(), _queue_layerwise_revoke_keys=Mock())
    key = 'offload_v1:private' if scenario == 'private_conflict' else 'public'
    request = NS(req_id='r', layerwise_request_id='inc', target_token_len=128,
                 save_block_keys=[None], save_last_block_key=None)
    if scenario in ('private_conflict', 'capacity'):
        with pytest.raises(RuntimeError):
            production['_open_puts'](worker, [(request, key, 0, 128)])
        assert len(calls) == 1
        return
    if scenario == 'public_conflict':
        other = NS(layerwise_request_id='other', save_block_keys=[None])
        fallbacks = production['_open_puts'](worker, [(request, key, 0, 128), (other, key, 0, 128)])
        assert calls == [['public'], ['public@offload_v1:fallback:inc', 'public@offload_v1:fallback:other']]
        assert fallbacks == [(request, 0, calls[1][0]), (other, 0, calls[1][1])]
        assert request.save_block_keys == [calls[1][0]]
        assert other.save_block_keys == [calls[1][1]]
        return
    duplicate = NS(save_block_keys=[None])
    production['_open_puts'](worker, [(request, key, 0, 128), (duplicate, key, 0, 128)])
    assert duplicate.save_block_keys == [None]
    assert len(calls) == 1
    if scenario == 'committed':
        assert request.save_block_keys == [None]
    else:
        assert request.save_block_keys == ['public']


def test_put_rejection_revokes_only_owned_sessions(production):
    calls = []
    def put(keys, sizes):
        calls.append(keys)
        return [0, -705] if len(calls) == 1 else [-705]
    worker = NS(_filter_pool_existing_keys=lambda keys: list(keys),
                _layerwise_key_batches=lambda keys: [keys], m_store=NS(batch_put_start=put),
                _put_started_keys=set(), _put_started_keys_lock=threading.Lock(),
                _queue_layerwise_revoke_keys=Mock())
    request = NS(req_id='r', layerwise_request_id='inc', target_token_len=128,
                 save_block_keys=[None, None])
    with pytest.raises(RuntimeError):
        production['_open_puts'](worker, [(request, 'owned', 0, 128), (request, 'competing', 1, 128)])
    assert calls == [['owned', 'competing'], ['competing@offload_v1:fallback:inc']]
    revoked = {key for call in worker._queue_layerwise_revoke_keys.call_args_list for key in call.args[0]}
    assert revoked == {'owned'}


@pytest.mark.parametrize("window", [False, True])
def test_continuation_uses_conflict_fallback(production, window):
    windows = {0: 128} if window else {}
    worker = NS(
        _layer_load_aborted=threading.Event(),
        _snapshot_keys={},
        _reused_layers_by_group={0: [0]},
        _is_layerwise_save_owner=lambda group: True,
        _make_layerwise_full_key=lambda group, key: key,
        _compute_reachable_load_masks=lambda request, aligned: None,
        _open_layerwise_get_sessions=Mock(),
        _check_hybrid_load_errors=Mock(),
        _filter_pool_existing_keys=list,
        _layerwise_key_batches=lambda keys: [keys] if keys else [],
        m_store=NS(batch_put_start=Mock(side_effect=[[-705], [0]])),
        _put_started_keys=set(),
        _put_started_keys_lock=threading.Lock(),
        _queue_layerwise_revoke_keys=Mock(),
        cacheable_group_ids=[0],
        cache_transfer_granularity=128,
        grouped_block_size=[128],
        hash_block_size=128,
        group_uses_align_state=[False],
        group_block_len=[[128]],
        kv_cache_config=NS(block_key_prefill_windows=windows),
    )
    request = NS(
        req_id="r", layerwise_request_id="inc", is_last_chunk=False, load_spec=None,
        target_token_len=128, num_prompt_tokens=256, can_save=True, save_start_token=0,
        block_hashes=[b"a"], block_ids_by_group=[[1]], store_masks=None,
    )
    groups = production["prepare_offload_sessions"](worker, [request])
    private_key = "61@offload_v1:fallback:inc"
    assert groups[0][0].save_block_keys == [private_key]
    assert worker._snapshot_keys == {"r": {0: {0: private_key}}}
    if window:
        assert worker._offload_pending_pages == {("r", "inc", 0): (128, {0: private_key})}
        worker._offload_active_pages = worker._offload_pending_pages

    request.target_token_len = 256
    request.load_spec = NS(can_load=True, layerwise_previous_step=True, kvpool_cached_tokens=128)
    assert production["_load_entries"](worker, request, 0, [b"a"], None) == [(private_key, 0)]
    request.load_spec.layerwise_previous_step = False
    assert production["_load_entries"](worker, request, 0, [b"a"], None) == [("61", 0)]
