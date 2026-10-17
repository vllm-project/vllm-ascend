# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .rfork_test_support import _load_module


def _configure_real_session(runtime, monkeypatch, model, processed_layout):
    prefix = "vllm_ascend.model_loader.rfork"
    layout = sys.modules[f"{prefix}.tensor_layout"]
    transfer = _load_module(monkeypatch, f"{prefix}.transfer_backend", "transfer_backend.py")
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", transfer.RForkTransferBackend)
    monkeypatch.setattr(
        runtime.client,
        "build_seed_key",
        lambda **kwargs: f"model-key:{kwargs['structural_digest']}",
    )
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    monkeypatch.setattr(layout, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(sys.modules[f"{prefix}.manifest"], "read_npu_format", lambda tensor: 0)

    def memory_snapshot():
        blocks = [
            {
                "address": tensor.data_ptr(),
                "size": tensor.numel() * tensor.element_size(),
                "state": "active_allocated",
            }
            for _, tensor in layout.collect_transferable_tensors(model, processed_layout)
        ]
        return [{"blocks": blocks}]

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=memory_snapshot)),
        raising=False,
    )

    success = SimpleNamespace(is_error=lambda: False, to_string=lambda: "success")
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    backend = session.transfer_backend
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda registrations: success,
        batch_unregister_memory=lambda addresses: success,
        batch_transfer_sync_read=lambda *args: success,
        finalize=lambda: success,
    )
    backend._memory_registration_cls = lambda *fields: fields
    backend.transfer_session_id = "receiver-session"
    backend._is_initialized = True

    # Stub external scheduling hooks while exercising real registration and lifecycle logic.
    monkeypatch.setattr(session, "_ensure_lease_release_retry_locked", lambda: None)
    monkeypatch.setattr(session, "_run_seed_heartbeat", lambda *args: None)
    monkeypatch.setattr(session.planner, "report_seed_once", lambda *args, **kwargs: True)
    monkeypatch.setattr(session.planner, "remove_seed", lambda *args, **kwargs: True)
    server = Mock(is_alive=True, port=1234, stop=Mock(return_value=True))
    start_server = Mock(return_value=server)
    monkeypatch.setattr(runtime.session, "start_rfork_server", start_server)
    return session, backend, layout, start_server


def _transfer_model(runtime, monkeypatch, session, backend, model, processed_layout):
    seed_info = runtime.types.SeedTransferInfo(
        "seed-session", dict(backend.weight_manifest), formats=dict(backend.weight_formats)
    )
    monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: seed_info)
    session.state = runtime.types.RForkLifecycleState.LEASED
    session.seed_lease = runtime.lease
    assert session.transfer_from_seed(model, processed_layout)


def test_checkpoint_post_load_rebinds_manifest_without_changing_seed_identity(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, backend, layout, start_server = _configure_real_session(runtime, monkeypatch, model, False)

    try:
        assert session.register_destination(model, False)
        original_digest = session.planner.structural_digest
        original_key = session.planner.seed_key
        _transfer_model(runtime, monkeypatch, session, backend, model, False)
        session.seed_lease = None

        model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        new_weight = model.weight
        post_load_digest = session.log_transferred_model_layout(model, False)
        assert post_load_digest != original_digest

        compute_digest = Mock(wraps=runtime.session._compute_structural_digest)
        verify_digest = Mock(wraps=session.planner.verify_structural_digest)
        monkeypatch.setattr(runtime.session, "_compute_structural_digest", compute_digest)
        monkeypatch.setattr(session.planner, "verify_structural_digest", verify_digest)

        result = session.start_seed_service(model, False)

        assert result is runtime.types.RForkSeedServiceStartResult.STARTED
        assert session.planner.structural_digest == original_digest
        assert session.planner.seed_key == original_key
        assert compute_digest.call_count == 0
        assert verify_digest.call_count == 0
        assert backend.weight_manifest["weight"][0] == new_weight.data_ptr()
        assert backend.weight_manifest["weight"][3] == (3, 2)
        assert backend.registered_structural_digest == layout.build_structural_digest([("weight", new_weight)])
        assert start_server.call_count == 1
        advertised_info = start_server.call_args.args[1]
        assert start_server.call_args.args[0] == original_key
        assert advertised_info.weights["weight"][0] == new_weight.data_ptr()
        assert advertised_info.weights["weight"][3] == (3, 2)
    finally:
        session.seed_lease = None
        assert session.shutdown()


def test_checkpoint_deferred_promotion_keeps_original_seed_identity_and_new_manifest(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, backend, layout, start_server = _configure_real_session(runtime, monkeypatch, model, False)

    try:
        assert session.register_destination(model, False)
        original_digest = session.planner.structural_digest
        original_key = session.planner.seed_key
        _transfer_model(runtime, monkeypatch, session, backend, model, False)

        model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        new_weight = model.weight
        assert session.log_transferred_model_layout(model, False) != original_digest
        compute_digest = Mock(wraps=runtime.session._compute_structural_digest)
        verify_digest = Mock(wraps=session.planner.verify_structural_digest)
        monkeypatch.setattr(runtime.session, "_compute_structural_digest", compute_digest)
        monkeypatch.setattr(session.planner, "verify_structural_digest", verify_digest)

        result = session.start_seed_service(model, False)

        assert result is runtime.types.RForkSeedServiceStartResult.DEFERRED
        assert session.planner.seed_key == original_key
        assert session.planner.structural_digest == original_digest
        assert compute_digest.call_count == 0
        assert verify_digest.call_count == 0
        session.seed_lease = None
        session._promote_deferred_seed()

        assert session.state is runtime.types.RForkLifecycleState.SERVING
        assert backend.weight_manifest["weight"][0] == new_weight.data_ptr()
        assert backend.weight_manifest["weight"][3] == (3, 2)
        assert start_server.call_count == 1
        advertised_info = start_server.call_args.args[1]
        assert start_server.call_args.args[0] == original_key
        assert advertised_info.weights["weight"][0] == new_weight.data_ptr()
        assert advertised_info.weights["weight"][3] == (3, 2)
        assert compute_digest.call_count == 0
        assert verify_digest.call_count == 0
    finally:
        session.seed_lease = None
        assert session.shutdown()


def test_checkpoint_fallback_reuses_bound_seed_identity_for_new_registration(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, backend, layout, start_server = _configure_real_session(runtime, monkeypatch, model, False)

    try:
        assert session.register_destination(model, False)
        original_digest = session.planner.structural_digest
        original_key = session.planner.seed_key
        assert session.prepare_for_fallback().can_schedule_seed

        model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        new_weight = model.weight
        compute_digest = Mock(wraps=runtime.session._compute_structural_digest)
        verify_digest = Mock(wraps=session.planner.verify_structural_digest)
        monkeypatch.setattr(runtime.session, "_compute_structural_digest", compute_digest)
        monkeypatch.setattr(session.planner, "verify_structural_digest", verify_digest)

        result = session.start_seed_service(model, False)

        assert result is runtime.types.RForkSeedServiceStartResult.STARTED
        assert session.planner.seed_key == original_key
        assert session.planner.structural_digest == original_digest
        assert backend.weight_manifest["weight"][0] == new_weight.data_ptr()
        assert backend.weight_manifest["weight"][3] == (3, 2)
        assert compute_digest.call_count == 0
        assert verify_digest.call_count == 0
        assert start_server.call_count == 1
        assert start_server.call_args.args[0] == original_key
    finally:
        session.seed_lease = None
        assert session.shutdown()


def test_checkpoint_fallback_without_bound_identity_binds_successful_registration(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, backend, layout, start_server = _configure_real_session(runtime, monkeypatch, model, False)

    try:
        assert session.planner.structural_digest is None
        result = session.start_seed_service(model, False)

        assert result is runtime.types.RForkSeedServiceStartResult.STARTED
        assert session.planner.structural_digest == backend.registered_structural_digest
        assert session.planner.seed_key == f"model-key:{backend.registered_structural_digest}"
        assert start_server.call_count == 1
        assert start_server.call_args.args[0] == session.planner.seed_key
    finally:
        session.seed_lease = None
        assert session.shutdown()


@pytest.mark.parametrize("deferred", [False, True])
def test_processed_layout_live_shape_drift_rejects_seed_publication(runtime, monkeypatch, deferred):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, backend, layout, start_server = _configure_real_session(runtime, monkeypatch, model, True)

    try:
        assert session.register_destination(model, True)
        original_digest = session.planner.structural_digest
        _transfer_model(runtime, monkeypatch, session, backend, model, True)
        if not deferred:
            session.seed_lease = None

        verify_digest = Mock(wraps=session.planner.verify_structural_digest)
        monkeypatch.setattr(session.planner, "verify_structural_digest", verify_digest)
        if deferred:
            result = session.start_seed_service(model, True)
            assert result is runtime.types.RForkSeedServiceStartResult.DEFERRED
            model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2))
            live_digest = layout.build_structural_digest([("weight", model.weight)])
            session.seed_lease = None
            session._promote_deferred_seed()
        else:
            model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2))
            live_digest = layout.build_structural_digest([("weight", model.weight)])
            result = session.start_seed_service(model, True)
            assert result is runtime.types.RForkSeedServiceStartResult.FAILED

        assert verify_digest.call_count == 1
        verify_digest.assert_called_once_with(live_digest)
        assert start_server.call_count == 0
        assert session.state is runtime.types.RForkLifecycleState.INITIALIZED
        assert session.planner.structural_digest == original_digest
    finally:
        session.seed_lease = None
        assert session.shutdown()
