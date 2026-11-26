# SPDX-License-Identifier: Apache-2.0
"""Construction tracking must not retain owners outside its load scope."""

from concurrent.futures import ThreadPoolExecutor

from vllm_ascend.models.host_offload import register_host_offload_owner, track_host_offload_owners


def test_nested_loads_restore_outer_scope():
    outer_owner, inner_owner = object(), object()
    with track_host_offload_owners() as outer:
        register_host_offload_owner(outer_owner)
        try:
            with track_host_offload_owners() as inner:
                register_host_offload_owner(inner_owner)
                raise ValueError("load failed")
        except ValueError:
            pass
        register_host_offload_owner(outer_owner)
    register_host_offload_owner(inner_owner)
    assert outer == [outer_owner, outer_owner]
    assert inner == [inner_owner]


def test_worker_thread_does_not_register_in_another_load():
    with track_host_offload_owners() as owners, ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(register_host_offload_owner, object()).result()
    assert owners == []
