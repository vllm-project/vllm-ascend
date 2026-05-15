# isort: off
import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoConfig
from vllm_ascend.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

AutoConfig.register("deepseek_v4", DeepseekV4Config)

import vllm  # noqa: E402
from vllm.transformers_utils.config import _CONFIG_REGISTRY  # noqa: E402
from vllm.transformers_utils.configs import _CLASS_TO_MODULE, __all__  # noqa: E402
from vllm.transformers_utils.config import _maybe_update_auto_config_kwargs, _maybe_remap_hf_config_attrs  # noqa: E402
from pathlib import Path  # noqa: E402
from vllm.transformers_utils.config_parser_base import ConfigParserBase  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402
import huggingface_hub  # noqa: E402
from vllm.transformers_utils.repo_utils import _get_hf_token  # noqa: E402

from vllm import envs  # noqa: E402
if envs.VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig  # noqa: E402
else:
    from transformers import AutoConfig  # noqa: E402


def __getattr__(name: str):  # noqa: F811
    if "DeepseekV4Config" not in _CLASS_TO_MODULE:
        _CLASS_TO_MODULE.update({
            "DeepseekV4Config":
            "vllm_ascend.transformers_utils.configs.deepseek_v4",
        })
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    if "DeepseekV4Config" not in __all__:
        __all__.append("DeepseekV4Config")
    return sorted(list(__all__))


class HFConfigParser(ConfigParserBase):

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            token=_get_hf_token(),
            **kwargs,
        )
        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        if model_type is None:
            model_type = ("speculators"
                          if config_dict.get("speculators_config") is not None
                          else model_type)
        # Allow hf_overrides to override model_type before checking _CONFIG_REGISTRY
        if (hf_overrides := kwargs.pop("hf_overrides", None)) is not None:
            model_type = hf_overrides.get("model_type", model_type)

        if "deepseek_v4" not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY.update(deepseek_v4="DeepseekV4Config")

        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(
                model,
                revision=revision,
                code_revision=code_revision,
                token=_get_hf_token(),
                **kwargs,
            )
        else:
            try:
                kwargs = _maybe_update_auto_config_kwargs(
                    kwargs, model_type=model_type)
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    token=_get_hf_token(),
                    **kwargs,
                )
            except ValueError as e:
                if (not trust_remote_code
                        and "requires you to execute the configuration file"
                        in str(e)):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI.")
                    raise RuntimeError(err_msg) from e
                else:
                    raise e
        config = _maybe_remap_hf_config_attrs(config)
        return config_dict, config


vllm.transformers_utils.configs.__getattr__ = __getattr__
vllm.transformers_utils.configs.__dir__ = __dir__
vllm.transformers_utils.config.HFConfigParser = HFConfigParser
