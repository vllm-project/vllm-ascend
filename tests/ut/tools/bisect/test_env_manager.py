from pathlib import Path

import pytest

from tools.bisect import env_manager
from tools.bisect.env_manager import EnvironmentManager, EnvSwitchError
from tools.bisect.env_table import RuntimeEnv


@pytest.mark.parametrize("value", ["", "N/A", "n/a", "unknown", "None"])
def test_known_rejects_status_table_placeholders(value: str):
    assert env_manager._known(value) is False


def test_read_cann_version_from_install_info(tmp_path: Path):
    info = tmp_path / "ascend_toolkit_install.info"
    info.write_text('other=value\nversion="9.0.1"\n', encoding="utf-8")

    assert env_manager._read_cann_version_from_info(info) == "9.0.1"


def test_ensure_applies_components_in_dependency_order(monkeypatch: pytest.MonkeyPatch):
    manager = EnvironmentManager(vllm_repo_dir="unused")
    calls = []
    monkeypatch.setattr(manager, "_ensure_cann", lambda version: calls.append(("cann", version)) or False)
    monkeypatch.setattr(
        manager,
        "_ensure_torch_npu",
        lambda version, log: calls.append(("torch-npu", version)) or True,
    )
    monkeypatch.setattr(manager, "_ensure_vllm", lambda ref, log: calls.append(("vllm", ref)) or False)

    changed = manager.ensure(RuntimeEnv(vllm_ref="vllm-sha", cann_version="9.0.1", torch_npu_version="2.6.0"))

    assert changed is True
    assert calls == [("cann", "9.0.1"), ("torch-npu", "2.6.0"), ("vllm", "vllm-sha")]


def test_ensure_empty_target_is_noop(monkeypatch: pytest.MonkeyPatch):
    manager = EnvironmentManager(vllm_repo_dir="unused")
    monkeypatch.setattr(manager, "_ensure_cann", pytest.fail)

    assert manager.ensure(None) is False
    assert manager.ensure(RuntimeEnv()) is False


def test_ensure_cann_reports_unavailable_runtime(monkeypatch: pytest.MonkeyPatch):
    manager = EnvironmentManager(vllm_repo_dir="unused")
    monkeypatch.setattr(env_manager, "_installed_cann_version", lambda: "8.0.0")
    monkeypatch.setattr(env_manager, "_cann_source_candidates", lambda version: [Path("missing")])

    with pytest.raises(EnvSwitchError, match="CANN 9.0.1 is not available"):
        manager._ensure_cann("9.0.1")


def test_ensure_torch_npu_installs_only_when_version_differs(monkeypatch: pytest.MonkeyPatch):
    manager = EnvironmentManager(vllm_repo_dir="unused")
    commands = []
    monkeypatch.setattr(env_manager, "_installed_package_version", lambda *names: "2.5.0")
    monkeypatch.setattr(env_manager, "_run", lambda cmd, log, label: commands.append((cmd, label)))

    assert manager._ensure_torch_npu("2.6.0", None) is True
    assert commands[0][0][3:] == [
        "install",
        "torch-npu==2.6.0",
        "--force-reinstall",
        "--no-input",
        "--disable-pip-version-check",
    ]
    assert commands[0][1] == "install torch-npu"
