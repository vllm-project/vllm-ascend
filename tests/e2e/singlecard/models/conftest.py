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
        default="./tests/e2e/singlecard/models/configs/Qwen3-8B-Base.yaml",
        help="Path to the model config YAML file",
    )
    parser.addoption(
        "--report_output",
        action="store",
        default="./benchmarks/accuracy/Qwen3-8B-Base.md",
        help="Path to the report output file",
    )
    parser.addoption(
        "--report-dir",
        action="store",
        default="./benchmarks/accuracy",
        help="Directory to store report files when using config list",
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


@pytest.fixture(scope="function")
def report_output(pytestconfig, config_filename):
    if pytestconfig.getoption("--config-list-file"):
        report_dir = pytestconfig.getoption("--report-dir")
        model_name = Path(config_filename).stem
        report_path = Path(report_dir) / f"{model_name}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        return report_path

    output_path = pytestconfig.getoption("--report_output")
    if output_path:
        output_path = Path(output_path)
        return output_path


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
            return
        single_config = metafunc.config.getoption("--config")
        config_path = Path(single_config).resolve()

        metafunc.parametrize("config_filename", [config_path])
