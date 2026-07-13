# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest
import regex as re


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


def _patch_nvlm_config():
    from transformers import PretrainedConfig

    original_to_diff_dict = PretrainedConfig.to_diff_dict

    def patched_to_diff_dict(self):
        try:
            return original_to_diff_dict(self)
        except ValueError:
            if type(self).__name__ == "NVLM_D_Config":
                return {}
            raise

    PretrainedConfig.to_diff_dict = patched_to_diff_dict


def _patch_nvlm_chat_template():
    from lm_eval.models.vllm_vlms import VLLM_VLM

    original_apply_chat_template = VLLM_VLM.apply_chat_template

    def patched_apply_chat_template(self, chat_history, add_generation_prompt=True):
        if self.model_args.get("model") == "AI-ModelScope/NVLM-D-72B":
            self.chat_applied = True
            for message in chat_history:
                content = message.get("content")
                if message.get("role") == "user" and isinstance(content, str):
                    message["content"] = re.sub(r"<image>[ \t]*(?:\r?\n)?", "<image>\n", content)
            return self.processor.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        return original_apply_chat_template(self, chat_history, add_generation_prompt)

    VLLM_VLM.apply_chat_template = patched_apply_chat_template


def _patch_nvlm_multimodal_encode():
    from lm_eval.models.vllm_vlms import VLLM_VLM

    original_tok_batch_multimodal_encode = VLLM_VLM.tok_batch_multimodal_encode

    def patched_tok_batch_multimodal_encode(self, strings, images, *args, **kwargs):
        if self.model_args.get("model") == "AI-ModelScope/NVLM-D-72B":
            strings = [
                re.sub(r"<image>[ \t]*(?:\r?\n)?", "<image>\n", text) if isinstance(text, str) else text
                for text in strings
            ]
        return original_tok_batch_multimodal_encode(self, strings, images, *args, **kwargs)

    VLLM_VLM.tok_batch_multimodal_encode = patched_tok_batch_multimodal_encode


def _patch_nvlm_hf_prompt_update():
    from vllm.model_executor.models.nvlm_d import NVLMMultiModalProcessor
    from vllm.multimodal.processing.processor import BaseMultiModalProcessor

    original_apply_hf_processor_main = BaseMultiModalProcessor._apply_hf_processor_main

    def patched_apply_hf_processor_main(
        self,
        prompt,
        mm_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
        *,
        enable_hf_prompt_update,
    ):
        if isinstance(self, NVLMMultiModalProcessor):
            enable_hf_prompt_update = False
        return original_apply_hf_processor_main(
            self,
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            enable_hf_prompt_update=enable_hf_prompt_update,
        )

    BaseMultiModalProcessor._apply_hf_processor_main = patched_apply_hf_processor_main


def pytest_configure(config):
    _patch_nvlm_config()
    _patch_nvlm_chat_template()
    _patch_nvlm_multimodal_encode()
    _patch_nvlm_hf_prompt_update()
