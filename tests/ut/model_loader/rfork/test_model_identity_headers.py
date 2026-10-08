# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Tests for RFork's opt-in model identity request headers."""

import pytest

from .rfork_test_support import _load_module


def _config(monkeypatch, extra_config=None):
    config = _load_module(monkeypatch, "rfork_test_model_identity_config", "config.py")
    raw_config = {
        "model_url": "model",
        "model_deploy_strategy_name": "strategy",
        "rfork_scheduler_url": "http://planner",
    }
    if extra_config is not None:
        raw_config.update(extra_config)
    return config.RForkConfig.from_extra_config(raw_config)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("false", False),
        ("true", True),
        ("FALSE", False),
        ("True", True),
        (" 0 ", False),
        (" 1 ", True),
        (" false ", False),
        (" TRUE ", True),
    ],
    ids=[
        "unset",
        "zero",
        "one",
        "false",
        "true",
        "uppercase-false",
        "mixed-case-true",
        "spaced-zero",
        "spaced-one",
        "spaced-false",
        "spaced-true",
    ],
)
def test_model_identity_header_switch_reads_rfork_environment(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("RFORK_MODEL_IDENTITY_HEADERS", raising=False)
    else:
        monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", env_value)

    cfg = _config(monkeypatch)

    assert cfg.send_model_identity_headers is expected


@pytest.mark.parametrize("env_value", ["", "   ", "2", "yes", "no", "true!"])
def test_model_identity_header_switch_rejects_invalid_values(monkeypatch, env_value):
    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", env_value)

    with pytest.raises(
        ValueError,
        match="RFORK_MODEL_IDENTITY_HEADERS must be '0', '1', 'true' or 'false'",
    ):
        _config(monkeypatch)


@pytest.mark.parametrize("json_value, expected", [(True, True), (False, False)])
def test_model_identity_header_switch_accepts_json_boolean(monkeypatch, json_value, expected):
    monkeypatch.delenv("RFORK_MODEL_IDENTITY_HEADERS", raising=False)

    cfg = _config(monkeypatch, {"rfork_model_identity_headers": json_value})

    assert cfg.send_model_identity_headers is expected


@pytest.mark.parametrize(
    "json_value, env_value, expected",
    [
        (True, "0", True),
        (True, "false", True),
        (False, "1", False),
        (False, "true", False),
    ],
)
def test_json_switch_takes_precedence_over_environment(monkeypatch, json_value, env_value, expected):
    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", env_value)

    cfg = _config(monkeypatch, {"rfork_model_identity_headers": json_value})

    assert cfg.send_model_identity_headers is expected


@pytest.mark.parametrize("json_value", [None, 0, 1, "true", "", [], {}])
def test_model_identity_header_switch_rejects_non_boolean_json(monkeypatch, json_value):
    monkeypatch.delenv("RFORK_MODEL_IDENTITY_HEADERS", raising=False)

    with pytest.raises(ValueError, match="rfork_model_identity_headers must be a JSON boolean"):
        _config(monkeypatch, {"rfork_model_identity_headers": json_value})


@pytest.mark.parametrize("json_value, env_value", [(True, ""), (False, "invalid")])
def test_valid_json_switch_overrides_invalid_environment(monkeypatch, json_value, env_value):
    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", env_value)

    cfg = _config(monkeypatch, {"rfork_model_identity_headers": json_value})

    assert cfg.send_model_identity_headers is json_value


def test_existing_config_does_not_change_when_environment_changes(monkeypatch):
    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", "0")
    cfg = _config(monkeypatch)

    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", "1")

    assert cfg.send_model_identity_headers is False


@pytest.mark.parametrize("field", ["model_url", "model_deploy_strategy_name"])
@pytest.mark.parametrize("switch_source", ["json", "environment"])
def test_identity_headers_reject_non_ascii_metadata_at_initialization(monkeypatch, field, switch_source):
    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", "1" if switch_source == "environment" else "0")
    extra_config: dict[str, str | bool] = {field: "model/部署A"}
    if switch_source == "json":
        extra_config["rfork_model_identity_headers"] = True

    with pytest.raises(ValueError, match=f"{field} must contain only ASCII characters"):
        _config(monkeypatch, extra_config)


@pytest.mark.parametrize("field", ["model_url", "model_deploy_strategy_name"])
def test_disabled_identity_headers_allow_non_ascii_metadata(monkeypatch, field):
    monkeypatch.setenv("RFORK_MODEL_IDENTITY_HEADERS", "1")

    cfg = _config(monkeypatch, {field: "model/部署A", "rfork_model_identity_headers": False})

    assert getattr(cfg, field) == "model/部署A"


def test_enabled_identity_headers_accept_ascii_metadata(monkeypatch):
    cfg = _config(
        monkeypatch,
        {
            "model_url": "/models/Qwen-3",
            "model_deploy_strategy_name": "deployment_tp8",
            "rfork_model_identity_headers": True,
        },
    )

    assert cfg.model_url == "/models/Qwen-3"
    assert cfg.model_deploy_strategy_name == "deployment_tp8"
