# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Guard the LoadConfig-held sessions against reuse across different model identities."""

from types import SimpleNamespace

import pytest
import torch


def _patch_identity_inputs(monkeypatch, module, *, tp_rank=0, fingerprint="draft-fingerprint"):
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 11)
    monkeypatch.setattr(module, "_get_pp_rank", lambda config: None)
    monkeypatch.setattr(module, "_get_ep_rank", lambda config: None)
    monkeypatch.setattr(module, "build_compatibility_fingerprint", lambda *args, **kwargs: fingerprint)


def _misjudged_draft_configs():
    # No runner_type, no MTP naming: _is_draft_model returns False for a real draft.
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(),
    )
    return vllm_config, model_config


def test_misdetected_draft_is_rejected_by_fingerprint(rfork_helpers, monkeypatch):
    module = rfork_helpers.loader
    # The target model's main session, already stored under rfork_session.
    target_identity = module.RForkIdentity(
        tp_rank=0,
        global_rank=11,
        is_draft_model=False,
        compatibility_fingerprint="target-fingerprint",
    )
    load_config = SimpleNamespace(
        model_loader_extra_config={},
        rfork_session=SimpleNamespace(identity=target_identity),
    )
    loader = module.RForkModelLoader(load_config)
    # The misjudged draft computes the same wrong is_draft_model=False and the same
    # ranks; only the compatibility fingerprint separates it from the target.
    _patch_identity_inputs(monkeypatch, module, tp_rank=0, fingerprint="draft-fingerprint")
    created = []
    monkeypatch.setattr(module, "RForkSession", lambda *args, **kwargs: created.append(args))

    vllm_config, model_config = _misjudged_draft_configs()

    with pytest.raises(RuntimeError, match="different model identity") as excinfo:
        loader._ensure_rfork_session(vllm_config, model_config)

    assert "differing fields: compatibility_fingerprint" in str(excinfo.value)
    assert created == []


def test_matching_identity_reuses_existing_session(rfork_helpers, monkeypatch):
    module = rfork_helpers.loader
    stored = SimpleNamespace(
        identity=module.RForkIdentity(
            tp_rank=0,
            global_rank=11,
            is_draft_model=False,
            compatibility_fingerprint="shared-fingerprint",
        )
    )
    load_config = SimpleNamespace(model_loader_extra_config={}, rfork_session=stored)
    loader = module.RForkModelLoader(load_config)
    _patch_identity_inputs(monkeypatch, module, tp_rank=0, fingerprint="shared-fingerprint")
    created = []
    monkeypatch.setattr(module, "RForkSession", lambda *args, **kwargs: created.append(args))

    vllm_config, model_config = _misjudged_draft_configs()

    assert loader._ensure_rfork_session(vllm_config, model_config) is stored
    assert created == []
    assert load_config.rfork_session is stored


def test_foreign_session_fallback_does_not_touch_stored_session(rfork_helpers, monkeypatch):
    module = rfork_helpers.loader
    calls = []
    stored_session = SimpleNamespace(
        identity=module.RForkIdentity(
            tp_rank=0,
            global_rank=11,
            is_draft_model=False,
            compatibility_fingerprint="target-fingerprint",
        ),
        register_destination=lambda *args, **kwargs: calls.append("register_destination"),
        prepare_for_fallback=lambda: calls.append("prepare_for_fallback"),
        start_seed_service=lambda *args, **kwargs: calls.append("start_seed_service"),
    )
    load_config = SimpleNamespace(device=None, model_loader_extra_config={}, rfork_session=stored_session)
    loader = module.RForkModelLoader(load_config)
    _patch_identity_inputs(monkeypatch, module, tp_rank=0, fingerprint="draft-fingerprint")
    monkeypatch.setattr(
        module,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    fallback_model = SimpleNamespace()

    def default_loader(*args, **kwargs):
        calls.append("default_loader")
        return fallback_model

    monkeypatch.setattr(module, "_load_with_default_loader", default_loader)

    draft_model_config = SimpleNamespace(dtype=torch.float32)
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        model_config=draft_model_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(),
    )

    result = loader.load_model(vllm_config, draft_model_config)

    assert result is fallback_model
    # The stored target session was never touched: no fallback cleanup ran on it.
    assert calls == ["default_loader"]
    assert load_config.rfork_session is stored_session
