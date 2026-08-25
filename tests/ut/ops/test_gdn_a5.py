# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.ops.gdn_a5 import (
    GDNBackendMode,
    GDNOperator,
    parse_gdn_backend_config,
)


def test_parse_gdn_backend_config_defaults_to_auto_without_overrides():
    config = parse_gdn_backend_config("auto", "")

    assert config.mode is GDNBackendMode.AUTO
    assert config.overrides == {}


def test_parse_gdn_backend_config_accepts_explicit_global_modes():
    assert parse_gdn_backend_config("native", "").mode is GDNBackendMode.NATIVE
    assert parse_gdn_backend_config("fla_npu", "").mode is GDNBackendMode.FLA_NPU


def test_parse_gdn_backend_config_accepts_per_operator_overrides():
    config = parse_gdn_backend_config(
        "auto",
        "causal_conv1d=fla_npu,chunk_fwd_o=native",
    )

    assert config.overrides == {
        GDNOperator.CAUSAL_CONV1D: GDNBackendMode.FLA_NPU,
        GDNOperator.CHUNK_FWD_O: GDNBackendMode.NATIVE,
    }


@pytest.mark.parametrize("mode", ["", "invalid", "FLA"])
def test_parse_gdn_backend_config_rejects_invalid_global_mode(mode):
    with pytest.raises(ValueError, match="GDN backend mode"):
        parse_gdn_backend_config(mode, "")


@pytest.mark.parametrize(
    "overrides",
    [
        "unknown=native",
        "causal_conv1d=invalid",
        "causal_conv1d=auto",
        "causal_conv1d=native,causal_conv1d=fla_npu",
        "causal_conv1d",
        "=native",
    ],
)
def test_parse_gdn_backend_config_rejects_invalid_overrides(overrides):
    with pytest.raises(ValueError, match="GDN operator backend override"):
        parse_gdn_backend_config("auto", overrides)
