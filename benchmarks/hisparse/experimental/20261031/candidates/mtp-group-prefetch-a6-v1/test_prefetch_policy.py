"""CPU call-order checks; these do not emulate an NPU graph or DMA."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('actual,decode,flag,group', [
    (8, 8, True, True), (32, 32, False, True), (4, 4, False, True),
    (40, 32, False, False), (32, 0, False, False), (0, 0, False, False),
    (33, 33, False, False), (8, 9, False, False), (-1, -1, False, False),
])
@pytest.mark.parametrize('dependency', ['topk_event', 'no_event', 'piecewise'])
@pytest.mark.parametrize('mirror', [True, False])
def test_current_metadata_and_existing_fences(actual, decode, flag, group, dependency, mirror):
    source = Path(__file__).resolve().parents[6] / 'vllm_ascend/distributed/kv_transfer/hisparse/runtime_adapter.py'
    tree = ast.parse(source.read_bytes())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_resolve_and_stage_group')
    events = []

    class Stream:
        def wait_event(self, event):
            events.append(('wait_topk', event))

        def wait_stream(self, stream):
            events.append(('wait_compute', stream))

        def __enter__(self):
            events.append(('enter_copy',))

        def __exit__(self, *exc):
            events.append(('leave_copy',))

    class Event:
        def record(self, stream):
            events.append(('record_layer_ready', stream))

    def runtime(name):
        return SimpleNamespace(
            _swap_rows=lambda rows: events.append(('gather', name, rows)),
            _layer_ready_event=None, _swap_staged=False,
        )

    leader = runtime('leader')
    followers = [runtime('follower1'), runtime('follower2')]
    stream = Stream()
    leader.index_group = SimpleNamespace(
        copy_stream=stream, logical_topk_ready=None if dependency == 'no_event' else 'current_topk',
        followers=followers,
    )
    leader.max_swap_rows = 32
    kwargs_seen = []
    leader._resolve_residency = lambda **kwargs: (kwargs_seen.append(kwargs), events.append(('resolve',)))
    resident = SimpleNamespace(decode_batch=flag, num_actual_tokens=actual, num_decode_tokens=decode,
                               host_mirror_required=mirror, slot_mapping=object(), draft_layer=False)
    original = vars(resident).copy()
    scope = dict(native_current_stream=lambda: 'compute', in_piecewise_cudagraph=lambda: dependency == 'piecewise',
                 create_event=Event)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), scope)
    kwargs = dict(resident=resident, req_id_per_token=object(), block_table=object(),
                  logical_topk_indices=object(), block_size=128, return_valid_counts=True,
                  shared_rows=slice(0, max(actual, 0)), attention_indices_out=object(),
                  valid_counts_out=object(), num_valid_rows=object())
    scope['_resolve_and_stage_group'](leader, **kwargs)
    expected_resolve = dict(kwargs)
    expected_resolve['topk_indices'] = expected_resolve.pop('logical_topk_indices')
    assert kwargs_seen == [expected_resolve]
    assert vars(resident) == original
    assert events[0][0] == ('wait_topk' if dependency == 'topk_event' else 'wait_compute')
    assert [x[1] for x in events if x[0] == 'gather'] == (
        ['leader', 'follower1', 'follower2'] if group else ['leader'])
    assert events.index(('resolve',)) < next(i for i, e in enumerate(events) if e[0] == 'gather')
    assert sum(e[0] == 'record_layer_ready' for e in events) == (3 if group else 1)
    assert leader._swap_staged and all(f._swap_staged == group for f in followers)
