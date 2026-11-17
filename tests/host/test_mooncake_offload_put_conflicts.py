# SPDX-License-Identifier: Apache-2.0
import ast
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.fixture
def production():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_offload.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_open_puts', '_load_entries', 'snapshot_key')]
    namespace = dict(_OBJECT_ALREADY_EXISTS=-705, require_aligned_batch_results=lambda op, keys, results: results,
                     offload_event=Mock(), block_hash_to_str=lambda value: value)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec', flags=0x1000000), namespace)
    return namespace


@pytest.mark.parametrize('scenario', ['success', 'public_conflict', 'private_conflict', 'capacity', 'committed'])
def test_conflict_protocol(production, scenario):
    calls = []
    def put(keys, sizes):
        calls.append(keys)
        if len(calls) > 1:
            return [0] * len(keys)
        return [0 if scenario == 'success' else -1 if scenario == 'capacity' else -705] * len(keys)
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
    production['_open_puts'](worker, [(request, key, 0, 128)])
    if scenario == 'public_conflict':
        fallback = 'public@offload_v1:fallback:inc'
        assert request.save_block_keys == [fallback]
        worker.group_uses_align_state = [False]
        worker.grouped_block_size = [128]
        worker.use_eagle = False
        worker._layerwise_load_snapshot_start = lambda *args: None
        worker._make_layerwise_full_key = lambda group, key: key
        request.block_ids_by_group = [[7]]
        request.load_spec = NS(can_load=True, layerwise_previous_step=True,
                               kvpool_cached_tokens=128, kvpool_store_skip_tokens=None)
        assert production['_load_entries'](worker, request, 0, ['public'], None) == [(fallback, 0)]
        # Another incarnation must never inherit this request's private copy.
        request.layerwise_request_id = 'other'
        assert production['_load_entries'](worker, request, 0, ['public'], None) == [('public', 0)]
    elif scenario == 'committed':
        assert request.save_block_keys == [None]
    else:
        assert request.save_block_keys == ['public']


def test_fallback_failure_revokes_only_owned_sessions(production):
    calls = []
    def put(keys, sizes):
        calls.append(keys)
        return [0, -705] if len(calls) == 1 else [-1]
    worker = NS(_filter_pool_existing_keys=lambda keys: list(keys),
                _layerwise_key_batches=lambda keys: [keys], m_store=NS(batch_put_start=put),
                _put_started_keys=set(), _put_started_keys_lock=threading.Lock(),
                _queue_layerwise_revoke_keys=Mock())
    request = NS(req_id='r', layerwise_request_id='inc', target_token_len=128,
                 save_block_keys=[None, None])
    with pytest.raises(RuntimeError):
        production['_open_puts'](worker, [(request, 'owned', 0, 128), (request, 'competing', 1, 128)])
    revoked = {key for call in worker._queue_layerwise_revoke_keys.call_args_list for key in call.args[0]}
    assert revoked == {'owned'}
