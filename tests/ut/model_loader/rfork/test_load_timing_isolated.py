# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import logging
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import regex as re
import torch


@pytest.mark.parametrize("source", ["transfer", "fallback"])
def test_seed_publication_registers_only_local_models(rfork_helpers, source):
    session = SimpleNamespace(
        identity=SimpleNamespace(tp_rank=0, is_draft_model=False),
        register_local_weights_for_seed=Mock(return_value=True),
        schedule_seed_service=Mock(return_value=True),
    )
    model = object()

    assert rfork_helpers.loader._schedule_seed_publication(session, model, load_source=source)
    if source == "fallback":
        session.register_local_weights_for_seed.assert_called_once_with(model)
    else:
        session.register_local_weights_for_seed.assert_not_called()
    session.schedule_seed_service.assert_called_once_with()


def test_seed_publication_stops_when_local_registration_fails(rfork_helpers):
    session = SimpleNamespace(
        identity=SimpleNamespace(tp_rank=0, is_draft_model=False),
        register_local_weights_for_seed=Mock(return_value=False),
        schedule_seed_service=Mock(),
    )
    assert not rfork_helpers.loader._schedule_seed_publication(session, object(), load_source="fallback")
    session.schedule_seed_service.assert_not_called()


@pytest.mark.parametrize("path", ["transfer", "coordinated", "local", "fallback", "layout_error", "bypass", "error"])
def test_load_timing_accounts_for_all_paths_and_context_exit(rfork_helpers, monkeypatch, caplog, path):
    module = rfork_helpers.loader
    clock = SimpleNamespace(ns=0)
    events = []
    published_models = []

    def work(name, ns=100):
        clock.ns += ns * 10_000
        events.append(name)

    @contextmanager
    def dtype_context(_dtype):
        work("dtype_enter")
        try:
            yield
        finally:
            work("dtype_exit", 10_000)

    def register(*args):
        work("register")
        if path == "fallback":
            raise RuntimeError("registration failed")
        if path == "error":
            raise TypeError("unexpected registration failure")
        return True

    def result(name, value):
        work(name)
        return value

    def validate_layout(*args):
        work("layout_summary")
        if path == "layout_error":
            raise RuntimeError("final layout mismatch")

    session = SimpleNamespace(
        loading=nullcontext,
        identity=SimpleNamespace(tp_rank=1, global_rank=7, is_draft_model=False),
        register_receiver_weights=register,
        acquire_seed_lease=lambda: result("acquire", path != "local"),
        read_weights_from_seed=lambda *args: result("transfer", True),
        validate_transferred_model_layout=validate_layout,
        prepare_for_fallback=lambda: result("cleanup", True),
    )
    model = torch.nn.Module()
    default_model = torch.nn.Module()
    loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    loader._get_session = lambda *args: result("session", session)
    monkeypatch.setattr(
        module, "time", SimpleNamespace(perf_counter_ns=lambda: clock.ns, perf_counter=lambda: clock.ns / 1e9)
    )
    monkeypatch.setattr(
        sys.modules["vllm_ascend.model_loader.rfork.timing"], "time", SimpleNamespace(perf_counter_ns=lambda: clock.ns)
    )
    monkeypatch.setattr(module, "set_default_torch_dtype", dtype_context)
    monkeypatch.setattr(module, "mutable_weights_bypass_reason", lambda *args: "test" if path == "bypass" else None)
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(eplb_config=SimpleNamespace()))
    monkeypatch.setattr(module, "_is_dynamic_eplb_enabled", lambda *args: False)
    monkeypatch.setattr(module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "_reset_process_global_model_state", lambda *args: work("reset"))
    monkeypatch.setattr(module, "_prepare_rfork_model_state", lambda *args: work("prepare"))
    monkeypatch.setattr(module, "initialize_model", lambda **kwargs: result("construct", model))
    monkeypatch.setattr(module, "load_local_weights", lambda *args: work("host_load"))
    monkeypatch.setattr(module, "release_local_weights", lambda *args: work("host_release"))
    monkeypatch.setattr(module, "record_loaded_local_weights", lambda *args: work("record_local"))
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: work("layout_processing"))
    monkeypatch.setattr(module, "_rfork_pre_transfer_weight_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "_refresh_rfork_flatquant_state", lambda *args: work("finalize"))
    monkeypatch.setattr(module, "_validate_rfork_shared_experts", lambda *args: work("validate_shared"))

    def schedule_seed(session, seed_model, *args, **kwargs):
        published_models.append(seed_model)
        work("seed_service")

    monkeypatch.setattr(module, "_schedule_seed_publication", schedule_seed)
    monkeypatch.setattr(module, "_load_with_default_loader", lambda *args: result("default_load", default_model))
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: work("collect")))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: work("synchronize"), empty_cache=lambda: work("empty_cache")),
        raising=False,
    )
    model_config = SimpleNamespace(dtype=torch.float32)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=model_config)
    if path == "coordinated":
        vllm_config.engram_config = object()
        monkeypatch.setattr(
            sys.modules["vllm.distributed.parallel_state"],
            "get_world_group",
            lambda: SimpleNamespace(world_size=2, cpu_group=object()),
        )
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: work("all_reduce", 10_000))

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        if path == "error":
            with pytest.raises(TypeError, match="unexpected registration failure"):
                loader.load_model(vllm_config, model_config)
        else:
            loaded = loader.load_model(vllm_config, model_config)
            assert loaded is (default_model if path in ("local", "fallback", "layout_error", "bypass") else model)

    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("RFork load timing:")]
    assert len(lines) == 1
    assert "_ns" not in lines[0]
    match = re.search(r"source=(\w+) outcome=(\w+) total_s=(\d+\.\d{6}) stages_s=(\{.*\})$", lines[0])
    assert match, lines[0]
    source, outcome, total_s, encoded_stages_s = match.groups()
    stages_s = json.loads(encoded_stages_s)
    assert source == ("transfer" if path == "coordinated" else "fallback" if path == "layout_error" else path)
    assert outcome == ("error" if path == "error" else "success")
    assert float(total_s) == round(clock.ns / 1e9, 6)
    assert sum(stages_s.values()) == pytest.approx(float(total_s), abs=len(stages_s) * 1e-6)
    stages = {name: round(seconds * 100_000) for name, seconds in stages_s.items()}
    assert stages["context_exit"] >= 10_000
    if path != "bypass":
        assert "global_rank=7" in lines[0]
    if path in ("transfer", "coordinated"):
        assert stages["destination_registration"] >= 100
        assert stages["seed_acquisition"] >= 100
        assert stages["weight_transfer"] >= 100
        assert stages["seed_service"] >= 100
        if path == "coordinated":
            assert events.count("all_reduce") == 6
            assert stages["coordination_and_overhead"] == events.count("all_reduce") * 10_000
            assert stages["transfer_layout_preparation"] == 200
    elif path in ("local", "fallback", "layout_error"):
        assert stages["fallback_cleanup"] >= 100
        assert stages["local_model_loading"] >= 100
        if path == "layout_error":
            assert events.index("layout_summary") < events.index("cleanup") < events.index("default_load")
            assert published_models == [default_model]
    elif path == "bypass":
        assert stages["bypass_local_loading"] >= 100
        assert "destination_registration" not in stages
