# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest


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
        default="./tests/e2e/models/configs/Qwen3-8B-Base.yaml",
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
                configs = [
                    config_dir / line.strip() for line in f
                    if line.strip() and not line.startswith("#")
                ]
            metafunc.parametrize("config_filename", configs)
        else:
            single_config = metafunc.config.getoption("--config")
            config_path = Path(single_config).resolve()
            metafunc.parametrize("config_filename", [config_path])
