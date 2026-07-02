#!/usr/bin/env python3
"""Start vLLM server with get_config hf_overrides fix for Gemma4 MTP.

This script MUST run before any other vllm imports so the patch takes effect
before the plugin system loads.
"""

import sys

# ---------------------------------------------------------------------------
# PATCH: Fix vLLM 0.23.0 get_config() — hf_overrides_fn is consumed by
# config_parser.parse() and never applied to the loaded config.
# ---------------------------------------------------------------------------
import vllm.transformers_utils.config as _vllm_cfg
_orig_get_config = _vllm_cfg.get_config


def _patched_get_config(
    model,
    trust_remote_code=False,
    revision=None,
    code_revision=None,
    config_format="auto",
    hf_overrides_kw=None,
    hf_overrides_fn=None,
    **kwargs,
):
    # Do NOT pass hf_overrides_fn to the original — it gets consumed
    # by config_parser.parse()'s dummy-model-type test and is never
    # seen again.  Apply the override ourselves, using INLINE logic
    # (calling the override function directly does not work here
    # because the plugin system may have wrapped/replaced it).
    hf_config = _orig_get_config(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=code_revision,
        config_format=config_format,
        hf_overrides_kw=hf_overrides_kw,
        hf_overrides_fn=None,  # skip internal (buggy) application
        **kwargs,
    )
    if hf_overrides_fn is not None:
        # Inline application — equivalent to SpeculativeConfig.hf_config_override
        # but not affected by plugin system wrapping of the function reference.
        if hf_config.model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
            hf_config.model_type = "gemma4_mtp"
            text_config = getattr(hf_config, "text_config", hf_config)
            if hasattr(text_config, "num_kv_shared_layers"):
                text_config.num_kv_shared_layers = 0
            hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})
        elif hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({"n_predict": n_predict, "architectures": ["HYV3MTPModel"]})
    if hf_overrides_kw:
        hf_config.update(hf_overrides_kw)
    return hf_config


_vllm_cfg.get_config = _patched_get_config

# Also patch ModelConfig.__post_init__ as a safety net
from vllm.config.model import ModelConfig as _ModelConfig
_orig_post_init = _ModelConfig.__post_init__


def _patched_post_init(self, *args, **kwargs):
    result = _orig_post_init(self, *args, **kwargs)
    if hasattr(self, 'hf_overrides') and callable(self.hf_overrides):
        self.hf_config = self.hf_overrides(self.hf_config)
    return result


_ModelConfig.__post_init__ = _patched_post_init

# ---------------------------------------------------------------------------
# PATCH: Disable AOT autograd cache bundling — incompatible with torch_npu
# npugraph backend. npugraph returns _CompiledFxGraph (not OutputCode), so
# AOTAutogradCache.make_entry's unwrap_output_code asserts and crashes FDO
# graph capture. The npugraph backend has its own cache, so this is safe.
# ---------------------------------------------------------------------------
import torch._dynamo.config as _dynamo_config
_dynamo_config.caching_precompile = False
try:
    import torch._functorch.config as _functorch_config
    _functorch_config.bundled_autograd_cache = False
except (AttributeError, ImportError):
    pass

# ---------------------------------------------------------------------------
# Now start the server normally
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Remove this script from argv and pass through to vllm
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
