# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tp-size",
        action="store",
        default="1",
        help="Tensor parallel size to use for evaluation",
    )
    parser.addoption(
        "--config",
        action="store",
        default="./tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml",
        help="Path to the model config YAML file",
    )
    parser.addoption(
        "--report-dir",
        action="store",
        default="./benchmarks/accuracy",
        help="Directory to store report files",
    )


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
        single_config = metafunc.config.getoption("--config")
        config_path = Path(single_config).resolve()
        metafunc.parametrize("config_filename", [config_path])
