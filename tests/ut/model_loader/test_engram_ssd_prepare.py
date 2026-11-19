# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ut.model_loader.test_engram_ssd_artifact import checkpoint
from vllm_ascend.model_loader.engram_ssd_prepare import prepare_ssd_config


def config(tmp_path, **options):
    model = checkpoint(tmp_path / "model")
    return SimpleNamespace(
        additional_config={"engram_ssd": {"path": str(tmp_path / "ssd"), **options}},
        model_config=SimpleNamespace(
            model=str(model),
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(
                engram_layer_ids=[1, 14],
                engram_num_embeddings=[3, 5],
                engram_head_dim=256,
                engram_max_ngram_size=3,
                engram_n_heads=8,
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            data_parallel_size=2,
            data_parallel_size_local=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_eplb=False,
        ),
        engram_config=SimpleNamespace(
            cpu_offload=True, dp_shared_memory=None, embedding_across_dp=False, use_thp=False
        ),
        load_config=SimpleNamespace(load_format="auto"),
    )


def test_path_only_resolves_loader_and_persistent_artifact(tmp_path):
    cfg = config(tmp_path)
    prepare_ssd_config(cfg)
    ssd = cfg.additional_config["engram_ssd"]
    assert cfg.load_config.load_format == "ascend_engram_ssd"
    assert cfg.engram_config.dp_shared_memory is True
    assert ssd["enabled"] is True
    assert ssd["cache_payload_bytes"] == 5_000_000_000
    assert ssd["total_host_budget_bytes"] == 8_000_000_000
    table = Path(ssd["table_paths"][0])
    before = table.stat().st_mtime_ns
    prepare_ssd_config(cfg)
    assert table.stat().st_mtime_ns == before


def test_explicit_legacy_config_unchanged():
    raw = {
        "enabled": True,
        "manifest": "/existing.json",
        "artifact_version": "version",
        "table_paths": ["/existing.bin"],
    }
    cfg = SimpleNamespace(additional_config={"engram_ssd": raw})
    prepare_ssd_config(cfg)
    assert cfg.additional_config["engram_ssd"] is raw


@pytest.mark.parametrize("change", ["loader", "pcp", "cpu_offload", "manual_manifest", "disabled"])
def test_conflicts_fail_before_generating(tmp_path, change):
    cfg = config(tmp_path)
    if change == "loader":
        cfg.load_config.load_format = "netloader"
    elif change == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    elif change == "cpu_offload":
        cfg.engram_config.cpu_offload = False
    elif change == "manual_manifest":
        cfg.additional_config["engram_ssd"]["manifest"] = "/manual.json"
    else:
        cfg.additional_config["engram_ssd"]["enabled"] = False
    with pytest.raises(ValueError):
        prepare_ssd_config(cfg)
    assert not list(tmp_path.glob("ssd/**/*.bin"))


def test_total_budget_and_low_budget_rejection(tmp_path):
    cfg = config(tmp_path, host_memory_bytes=16_000_000_000)
    prepare_ssd_config(cfg)
    ssd = cfg.additional_config["engram_ssd"]
    assert ssd["total_host_budget_bytes"] == 4_000_000_000
    assert 0 < ssd["cache_payload_bytes"] < 4_000_000_000
    cfg.additional_config["engram_ssd"] = {"path": str(tmp_path / "small"), "host_memory_bytes": 1}
    with pytest.raises(ValueError):
        prepare_ssd_config(cfg)
    assert not (tmp_path / "small").exists()


@pytest.mark.parametrize("change", ["one_layer", "hash_sharding", "budget_override"])
def test_unsupported_runtime_layout_fails_before_artifact(tmp_path, change):
    cfg = config(tmp_path)
    if change == "one_layer":
        cfg.model_config.hf_text_config.engram_layer_ids = [1]
        cfg.model_config.hf_text_config.engram_num_embeddings = [3]
    elif change == "hash_sharding":
        cfg.model_config.hf_text_config.engram_n_heads = 3
    else:
        cfg.additional_config["engram_ssd"]["host_memory_bytes"] = 1_000_000_000
        cfg.additional_config["engram_ssd"]["total_host_budget_bytes"] = 8_000_000_000
    with pytest.raises(ValueError):
        prepare_ssd_config(cfg)
    assert not (tmp_path / "ssd").exists()
