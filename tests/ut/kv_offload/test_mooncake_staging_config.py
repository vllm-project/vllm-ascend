# SPDX-License-Identifier: Apache-2.0
"""Typed startup settings select staging capacity and bank execution policy."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ascend_config import AscendConfig, MooncakeStagingConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import staging
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.staging import MooncakeSFAStagingReader


def test_staging_defaults_and_nested_config_conversion():
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    assert config.mooncake_staging_config == MooncakeStagingConfig(window_chunks=16, bank_count=4)
    configured = AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        mooncake_staging_config={"window_chunks": 8, "bank_count": 1},
    )
    assert configured.mooncake_staging_config == MooncakeStagingConfig(window_chunks=8, bank_count=1)
    assert config.mooncake_staging_config.bank_count == 4


@pytest.mark.parametrize("field", ["window_chunks", "bank_count"])
@pytest.mark.parametrize("value", [True, False, 2.5, "4", None])
def test_invalid_staging_count_types_fail_at_config_time(field, value):
    with pytest.raises(ValueError):
        MooncakeStagingConfig(**{field: value})


@pytest.mark.parametrize("field", ["window_chunks", "bank_count"])
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_counts_retain_existing_clamping(field, value):
    assert getattr(MooncakeStagingConfig(**{field: value}), field) == 1


@pytest.mark.parametrize("field", ["pipeline_enabled", "copy_streams_per_bank", "read_probe", "unknown"])
def test_removed_or_unknown_staging_settings_are_rejected(field):
    with pytest.raises(ValueError):
        MooncakeStagingConfig(**{field: 1})


@pytest.mark.parametrize("window,banks,effective", [(16, 4, 4), (16, 1, 1), (3, 7, 3)])
def test_reader_receives_startup_config_and_allocates_one_stream_per_effective_bank(
    monkeypatch, window, banks, effective
):
    stream = MagicMock()
    monkeypatch.setattr(staging.torch.npu, "Stream", stream)
    monkeypatch.setattr(MooncakeSFAStagingReader, "_ensure_dsa_pd_mooncake_staging_tensor", MagicMock())
    config = MooncakeStagingConfig(window_chunks=window, bank_count=banks)
    reader = MooncakeSFAStagingReader(
        object(), SimpleNamespace(registry=object(), transfer_caches={}), [[128]], staging_config=config
    )
    assert reader._staging_config is config
    assert reader._get_dsa_pd_mooncake_staging_window() == window
    assert stream.call_count == effective
