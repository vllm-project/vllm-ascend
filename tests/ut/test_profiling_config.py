from unittest.mock import patch

import yaml

from vllm_ascend import profiling_config


def test_service_profiling_symbols_yaml():
    entries = yaml.safe_load(profiling_config.SERVICE_PROFILING_SYMBOLS_YAML)
    symbols = {entry["symbol"] for entry in entries}

    assert len(entries) == 76
    assert len(symbols) == 75
    assert {
        "vllm.entrypoints.openai.completion.serving:OpenAIServingCompletion.create_completion",
        "vllm.entrypoints.openai.chat_completion.serving:OpenAIServingChat.create_chat_completion",
        "vllm.v1.engine.async_llm:AsyncLLM.generate",
        "vllm_ascend.core.recompute_scheduler:RecomputeScheduler.schedule",
        "vllm_ascend.worker.model_runner_v1:NPUModelRunner._model_forward",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector:MooncakeConnector.start_load_kv",
    }.issubset(symbols)
    assert all(
        isinstance(entry[version_key], str)
        for entry in entries
        for version_key in ("min_version", "max_version")
        if version_key in entry
    )


def test_get_config_dir_uses_home(monkeypatch, tmp_path):
    monkeypatch.setattr(
        profiling_config.Path,
        "home",
        classmethod(lambda cls: tmp_path),
    )

    assert profiling_config.get_config_dir() == tmp_path / ".config" / "vllm_ascend"


def test_generate_service_profiling_config_creates_default(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(profiling_config, "get_config_dir", lambda: config_dir)

    config_file = profiling_config.generate_service_profiling_config()

    assert config_file == config_dir / profiling_config.CONFIG_FILENAME
    assert config_file.read_text(encoding="utf-8") == profiling_config.SERVICE_PROFILING_SYMBOLS_YAML
    assert list(config_dir.glob("*.tmp")) == []


def test_generate_service_profiling_config_preserves_existing_file(
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / profiling_config.CONFIG_FILENAME
    config_file.write_text("custom config\n", encoding="utf-8")
    monkeypatch.setattr(profiling_config, "get_config_dir", lambda: config_dir)

    result = profiling_config.generate_service_profiling_config()

    assert result == config_file
    assert config_file.read_text(encoding="utf-8") == "custom config\n"


def test_generate_service_profiling_config_handles_mkdir_failure(
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(profiling_config, "get_config_dir", lambda: config_dir)

    with (
        patch.object(profiling_config.Path, "mkdir", side_effect=PermissionError("denied")),
        patch.object(profiling_config.logger, "exception") as mock_exception,
    ):
        result = profiling_config.generate_service_profiling_config()

    assert result is None
    mock_exception.assert_called_once()


def test_generate_service_profiling_config_cleans_up_after_replace_failure(
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "config"
    monkeypatch.setattr(profiling_config, "get_config_dir", lambda: config_dir)

    with (
        patch.object(profiling_config.Path, "replace", side_effect=OSError("replace failed")),
        patch.object(profiling_config.logger, "exception") as mock_exception,
    ):
        result = profiling_config.generate_service_profiling_config()

    assert result is None
    assert not (config_dir / profiling_config.CONFIG_FILENAME).exists()
    assert list(config_dir.glob("*.tmp")) == []
    mock_exception.assert_called_once()
