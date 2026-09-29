# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.spec_decode.config_utils import (
    disable_profiling_chunk_for_draft,
)


def _config(additional_config, pp_size=2):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        additional_config=additional_config,
    )


@pytest.mark.parametrize("enabled", [True, 1, "true", "yes", "on"])
@pytest.mark.parametrize("legacy", [False, True])
def test_disable_profiling_chunk_for_draft_accepts_pydantic_true_values(enabled, legacy):
    profiling_chunk = {"enabled": enabled, "min_chunk": 128}
    if legacy:
        additional_config = {"profiling_chunk_config": profiling_chunk, "enable_cpu_binding": True}
    else:
        additional_config = {
            "scheduler_config": {"profiling_chunk_config": profiling_chunk},
            "enable_cpu_binding": True,
        }
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        draft_additional_config = config.additional_config
        draft_profiling_chunk = (
            draft_additional_config["profiling_chunk_config"]
            if legacy
            else draft_additional_config["scheduler_config"]["profiling_chunk_config"]
        )
        assert draft_profiling_chunk == {"enabled": False, "min_chunk": 128}
        assert draft_additional_config is not additional_config
        assert draft_profiling_chunk is not profiling_chunk

    assert config.additional_config is additional_config
    assert profiling_chunk["enabled"] == enabled


@pytest.mark.parametrize("enabled", [False, 0, "false", "no", "off"])
def test_disable_profiling_chunk_for_draft_leaves_effectively_disabled_config_unchanged(enabled):
    additional_config = {"profiling_chunk_config": {"enabled": enabled}}
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        assert config.additional_config is additional_config


@pytest.mark.parametrize(
    ("nested_enabled", "legacy_enabled", "expect_rewrite"),
    [(False, True, False), (True, False, True)],
)
def test_disable_profiling_chunk_for_draft_uses_nested_precedence(
    nested_enabled, legacy_enabled, expect_rewrite
):
    additional_config = {
        "scheduler_config": {"profiling_chunk_config": {"enabled": nested_enabled}},
        "profiling_chunk_config": {"enabled": legacy_enabled},
    }
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        if expect_rewrite:
            assert config.additional_config is not additional_config
            assert config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
            assert config.additional_config["profiling_chunk_config"]["enabled"] is False
        else:
            assert config.additional_config is additional_config


@pytest.mark.parametrize("pp_size", [1, 2])
def test_disable_profiling_chunk_for_draft_restores_after_failure(pp_size):
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": True}}}
    config = _config(additional_config, pp_size=pp_size)
    expected_context = pytest.raises(RuntimeError, match="draft failed")

    with expected_context:
        with disable_profiling_chunk_for_draft(config):
            if pp_size > 1:
                assert config.additional_config is not additional_config
            else:
                assert config.additional_config is additional_config
            raise RuntimeError("draft failed")

    assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_rejects_invalid_boolean():
    config = _config({"profiling_chunk_config": {"enabled": "sometimes"}})

    with pytest.raises(ValueError, match="additional_config.profiling_chunk_config.enabled must be a boolean"):
        with disable_profiling_chunk_for_draft(config):
            pass
