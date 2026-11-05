# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import time
from unittest.mock import Mock

import torch


def dummy_model() -> torch.nn.Module:
    # CPU modules yield an empty transferable tensor set, which keeps the
    # structural digest well defined without NPU tensors.
    return torch.nn.Module()


def make_session(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = runtime.lease
    session.planner.remove_seed.return_value = True
    session.transfer_backend = Mock()
    session.transfer_backend.register_memory_region.return_value = True
    session.transfer_backend.registered_structural_digest = "registered-digest"
    session.transfer_backend.read_weights_from_seed.return_value = True
    session.transfer_backend.unregister_memory_region.return_value = True
    session.transfer_backend.finalize_transfer_engine.return_value = True
    session.transfer_backend.weight_formats = {}
    assert session.register_receiver_weights(dummy_model())
    assert session.acquire_seed_lease()
    return session


def run_and_join(session):
    # Exercise release after failed transfers without resetting transfer state.
    with session._lock:
        assert session._release_seed_locked() is False
        worker = session._seed_worker_thread
    assert worker is not None
    worker.join(2)
    assert not worker.is_alive()


def join_seed_worker(session, timeout=5):
    with session._lock:
        worker = session._seed_worker_thread
    if worker is not None:
        worker.join(timeout)
        assert not worker.is_alive()


def wait_for_lease_release(session):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with session._lock:
            if session.seed_lease is None:
                return
        time.sleep(0.005)
    raise AssertionError("seed lease was not released")
