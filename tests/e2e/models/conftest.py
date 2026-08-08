# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest
import yaml


def pytest_addoption(parser):
    parser.addoption(
        "--config-list-file",
        action="store",
        default=None,
        help="Path to the file listing model config YAMLs (one per line)",
    )
    parser.addoption(
        "--tp-size",
        action="store",
        default="1",
        help="Tensor parallel size to use for evaluation",
    )
    parser.addoption(
        "--config",
        action="store",
        default="./tests/e2e/models/configs/Qwen3-8B.yaml",
        help="Path to the model config YAML file",
    )
    parser.addoption(
        "--report-dir",
        action="store",
        default="./benchmarks/accuracy",
        help="Directory to store report files",
    )


@pytest.fixture(scope="session")
def config_list_file(pytestconfig, config_dir):
    rel_path = pytestconfig.getoption("--config-list-file")
    return config_dir / rel_path


@pytest.fixture(scope="session")
def tp_size(pytestconfig):
    return pytestconfig.getoption("--tp-size")


@pytest.fixture(scope="session")
def config(pytestconfig):
    return pytestconfig.getoption("--config")


@pytest.fixture(scope="session")
def report_dir(pytestconfig):
    return pytestconfig.getoption("report_dir")


def pytest_generate_tests(metafunc):
    if "config_filename" in metafunc.fixturenames:
        if metafunc.config.getoption("--config-list-file"):
            rel_path = metafunc.config.getoption("--config-list-file")
            config_list_file = Path(rel_path).resolve()
            config_dir = config_list_file.parent
            with open(config_list_file, encoding="utf-8") as f:
                configs = [config_dir / line.strip() for line in f if line.strip() and not line.startswith("#")]
            metafunc.parametrize("config_filename", configs)
        else:
            single_config = metafunc.config.getoption("--config")
            config_path = Path(single_config).resolve()
            metafunc.parametrize("config_filename", [config_path])


def _report_path_for_config(config_path: Path, report_dir: str) -> Path:
    return Path(report_dir) / f"{config_path.stem}.md"


def _write_missing_report_stub(config_path: Path, report_dir: str) -> None:
    report_path = _report_path_for_config(config_path, report_dir)
    if report_path.exists():
        return

    try:
        eval_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_name = eval_config.get("model_name", config_path.stem)
    except Exception:
        model_name = config_path.stem

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# {model_name}\n\n"
        "- **Status**: accuracy report was not generated\n"
        "- **Note**: The test may have failed during collection or setup.\n",
        encoding="utf-8",
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--config-list-file"):
        return

    config_path = Path(session.config.getoption("--config")).resolve()
    report_dir = session.config.getoption("report_dir")
    _write_missing_report_stub(config_path, report_dir)
