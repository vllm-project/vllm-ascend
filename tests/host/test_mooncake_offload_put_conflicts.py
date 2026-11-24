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
        / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/planes/block_key_offload.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_open_puts"]
    namespace = dict(require_aligned_batch_results=lambda op, keys, results: results)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec', flags=0x1000000), namespace)
    return namespace


@pytest.mark.parametrize('scenario', ['success', 'public_conflict', 'private_conflict', 'capacity', 'committed'])
def test_conflict_protocol(production, scenario):
    calls = []
    def put(keys, sizes):
        calls.append(keys)
        return [0 if scenario == 'success' else -1 if scenario == 'capacity' else -705] * len(keys)
    def missing(keys):
        return [] if scenario == 'committed' and calls else list(keys)
    worker = NS(_filter_pool_existing_keys=missing, _layerwise_key_batches=lambda keys: [keys],
                m_store=NS(batch_put_start=put), _put_started_keys=set(),
                _put_started_keys_lock=threading.Lock(), _queue_layerwise_revoke_keys=Mock())
    key = 'offload_v1:private' if scenario == 'private_conflict' else 'public'
    request = NS(req_id='r', layerwise_request_id='inc', target_token_len=128,
                 save_block_keys=[None], save_last_block_key=None)
    if scenario in ('public_conflict', 'private_conflict', 'capacity'):
        with pytest.raises(RuntimeError):
            production['_open_puts'](worker, [(request, key, 0, 128)])
        assert len(calls) == 1
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
        return [0, -705]
    worker = NS(_filter_pool_existing_keys=lambda keys: list(keys),
                _layerwise_key_batches=lambda keys: [keys], m_store=NS(batch_put_start=put),
                _put_started_keys=set(), _put_started_keys_lock=threading.Lock(),
                _queue_layerwise_revoke_keys=Mock())
    request = NS(req_id='r', layerwise_request_id='inc', target_token_len=128,
                 save_block_keys=[None, None])
    with pytest.raises(RuntimeError):
        production['_open_puts'](worker, [(request, 'owned', 0, 128), (request, 'competing', 1, 128)])
    assert calls == [['owned', 'competing']]
    revoked = {key for call in worker._queue_layerwise_revoke_keys.call_args_list for key in call.args[0]}
    assert revoked == {'owned'}
