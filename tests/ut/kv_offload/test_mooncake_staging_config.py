# SPDX-License-Identifier: Apache-2.0
"""Staging uses an internal policy instead of user-facing tuning options."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ascend_config import AscendConfig, ExpertOffloadConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import staging
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.staging import MooncakeSFAStagingReader


@pytest.mark.parametrize(
    "settings", [{}, {"window_chunks": 16, "bank_count": 4}, {"window_chunks": 8, "bank_count": 1}]
)
def test_user_staging_settings_are_rejected(settings):
    with pytest.raises(ValueError, match="mooncake_staging_config"):
        AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), mooncake_staging_config=settings)


def test_expert_offload_config_parses_without_staging_settings():
    config = AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        expert_offload_config={"enabled": True, "num_device_experts": 4},
    )
    assert config.expert_offload_config == ExpertOffloadConfig(enabled=True, num_device_experts=4)
    assert not hasattr(config, "mooncake_staging_config")


def test_reader_uses_sixteen_chunks_and_four_streams_without_user_config(monkeypatch):
    stream = MagicMock()
    monkeypatch.setattr(staging.torch.npu, "Stream", stream)
    monkeypatch.setattr(MooncakeSFAStagingReader, "_ensure_dsa_pd_mooncake_staging_tensor", MagicMock())
    reader = MooncakeSFAStagingReader(object(), SimpleNamespace(registry=object(), transfer_caches={}), [[128]])
    assert reader._get_dsa_pd_mooncake_staging_window() == 16
    assert reader._staging_bank_count == 4
    assert stream.call_count == len(reader._landing_streams) == 4
    assert reader._get_dsa_pd_mooncake_staging_chunk_bytes() * reader._get_dsa_pd_mooncake_staging_window() == 32 << 20


@pytest.mark.parametrize(
    "block_lens,expected_mib",
    [([], 2), ([[128]], 2), ([[2 << 20]], 2), ([[(2 << 20) + 1]], 4), ([[3 << 20], [(8 << 20) + 1]], 10)],
)
def test_chunk_size_still_follows_kv_layout(block_lens, expected_mib):
    reader = MooncakeSFAStagingReader.__new__(MooncakeSFAStagingReader)
    reader.block_len_per_addr = block_lens
    assert reader._get_dsa_pd_mooncake_staging_chunk_bytes() == expected_mib << 20
