# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import Mock


def make_session(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = runtime.lease
    session.planner.remove_seed.return_value = True
    session.transfer_backend = Mock()
    session.transfer_backend.register_memory_region.return_value = True
    session.transfer_backend.read_weights_from_seed.return_value = True
    session.transfer_backend.unregister_memory_region.return_value = True
    session.transfer_backend.finalize_transfer_engine.return_value = True
    session.transfer_backend.weight_formats = {}
    assert session.register_destination(object(), True)
    assert session.acquire_seed()
    return session


def run_and_join(session):
    # Exercise release after failed transfers without resetting transfer state.
    with session._lock:
        assert session._release_seed_locked() is False
        worker = session.lease_release_thread
    assert worker is not None
    worker.join(2)
    assert not worker.is_alive()
