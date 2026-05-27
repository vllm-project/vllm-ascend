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


def _patch_nvlm_config_cls(cls):
    """Monkey-patch NVLM_D_Config.__init__ to not raise on empty llm_config."""
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        try:
            original_init(self, *args, **kwargs)
        except ValueError:
            from transformers import Qwen2Config

            llm_config = kwargs.get("llm_config") or {}
            self.llm_config = Qwen2Config(**llm_config)
            self.use_backbone_lora = kwargs.get("use_backbone_lora", 0)
            self.use_llm_lora = kwargs.get("use_llm_lora", 0)
            self.select_layer = kwargs.get("select_layer", -1)
            self.force_image_size = kwargs.get("force_image_size")
            self.downsample_ratio = kwargs.get("downsample_ratio", 0.5)
            self.template = kwargs.get("template")
            self.dynamic_image_size = kwargs.get("dynamic_image_size", False)
            self.use_thumbnail = kwargs.get("use_thumbnail", False)
            self.ps_version = kwargs.get("ps_version", "v1")
            self.min_dynamic_patch = kwargs.get("min_dynamic_patch", 1)
            self.max_dynamic_patch = kwargs.get("max_dynamic_patch", 6)

    cls.__init__ = patched_init


def _patch_nvlm_config():
    """Patch NVLM_D_Config to not raise ValueError on empty llm_config.

    modelscope patches AutoConfig.from_pretrained and uses its own
    code path that bypasses transformers' get_class_from_dynamic_module.
    We patch PretrainedConfig.to_diff_dict directly, which is called
    during from_dict -> __repr__ -> to_json_string -> to_diff_dict,
    and is the exact point where NVLM_D_Config() default-construction
    raises ValueError.
    """
    from transformers import PretrainedConfig

    _original_to_diff_dict = PretrainedConfig.to_diff_dict

    def _patched_to_diff_dict(self):
        if type(self).__name__ == "NVLM_D_Config" and not hasattr(self, "llm_config"):
            # NVLM_D_Config() without args raises ValueError because
            # llm_config defaults to {} with no 'architectures' key.
            # Catch it here so to_diff_dict returns an empty diff.
            return {}
        return _original_to_diff_dict(self)

    PretrainedConfig.to_diff_dict = _patched_to_diff_dict


def pytest_configure(config):
    _patch_nvlm_config()
