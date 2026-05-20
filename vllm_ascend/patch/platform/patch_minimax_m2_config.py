# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch target: vllm/config/model.py
# - MiniMax-M2 fp8 checkpoint on NPU: disable fp8 quantization (load bf16
#   dequantized weights in worker patch) instead of failing validation.
# - For ACL graph capture, set HCCL_OP_EXPANSION_MODE=AIV if user didn't set it.
#

import os

from vllm.config.model import ModelConfig
from vllm.logger import logger
from vllm.platforms import current_platform

_original_verify_quantization = getattr(ModelConfig, "_verify_quantization", None)
_original_verify_cuda_graph = getattr(ModelConfig, "_verify_cuda_graph", None)

_DISABLE_FP8_LOG = (
    "Detected fp8 MiniMax-M2 checkpoint on NPU. "
    "Disabling fp8 quantization and loading dequantized bf16 "
    "weights instead."
)


def _get_model_type(cfg: ModelConfig) -> str | None:
    # vLLM config fields have changed across versions; try multiple sources.
    model_arch_cfg = getattr(cfg, "model_arch_config", None)
    if model_arch_cfg is not None:
        mt = getattr(model_arch_cfg, "model_type", None)
        if mt:
            return mt

    hf_text_cfg = getattr(cfg, "hf_text_config", None)
    if hf_text_cfg is not None:
        mt = getattr(hf_text_cfg, "model_type", None)
        if mt:
            return mt

    hf_cfg = getattr(cfg, "hf_config", None)
    if hf_cfg is not None:
        mt = getattr(hf_cfg, "model_type", None)
        if mt:
            return mt

    return getattr(cfg, "model_type", None)


def _should_disable_fp8(cfg: ModelConfig, quant_method: str | None) -> bool:
    return current_platform.device_name == "npu" and _get_model_type(cfg) == "minimax_m2" and quant_method == "fp8"


def _disable_fp8(cfg: ModelConfig, *, log: bool) -> bool:
    if not _should_disable_fp8(cfg, getattr(cfg, "quantization", None)):
        return False
    if log:
        logger.warning(_DISABLE_FP8_LOG)
    cfg.quantization = None
    return True


def _patched_verify_quantization(self: ModelConfig) -> None:
    """Inject mid-function behavior for ModelConfig._verify_quantization.

    Upstream validates quantization inside this method via:
        current_platform.verify_quantization(self.quantization)

    We emulate a mid-function patch without copying upstream code by temporarily
    overriding current_platform.verify_quantization while the original verifier
    executes.
    """
    assert _original_verify_quantization is not None

    orig_platform_verify = getattr(current_platform, "verify_quantization", None)

    def _platform_verify_hook(quant_method: str | None) -> None:
        if _should_disable_fp8(self, quant_method):
            # This is the effective "middle of _verify_quantization" interception.
            _disable_fp8(self, log=True)
            return
        assert orig_platform_verify is not None
        return orig_platform_verify(quant_method)

    # Some versions may read self.quantization before calling platform verifier.
    _disable_fp8(self, log=True)

    try:
        if orig_platform_verify is not None:
            current_platform.verify_quantization = _platform_verify_hook
        return _original_verify_quantization(self)
    finally:
        if orig_platform_verify is not None:
            current_platform.verify_quantization = orig_platform_verify
        # Ensure fp8 isn't restored by upstream logic.
        _disable_fp8(self, log=False)


def _patched_verify_cuda_graph(self: ModelConfig) -> None:
    assert _original_verify_cuda_graph is not None

    if (
        current_platform.device_name == "npu"
        and _get_model_type(self) == "minimax_m2"
        and not getattr(self, "enforce_eager", True)
    ):
        expansion_mode = os.environ.get("HCCL_OP_EXPANSION_MODE")
        if expansion_mode is None:
            os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
            logger.info("Set HCCL_OP_EXPANSION_MODE=AIV for MiniMax-M2 ACL graph capture on NPU.")
        elif expansion_mode != "AIV":
            logger.warning(
                "HCCL_OP_EXPANSION_MODE=%s may reduce ACL graph shape "
                "coverage for MiniMax-M2 on NPU. Recommended value: AIV.",
                expansion_mode,
            )

    return _original_verify_cuda_graph(self)


if _original_verify_quantization is not None:
    ModelConfig._verify_quantization = _patched_verify_quantization

if _original_verify_cuda_graph is not None:
    ModelConfig._verify_cuda_graph = _patched_verify_cuda_graph


# ---------------------------------------------------------------------------
# Speculative decoding (Eagle3): allow MiniMax targets and registry alias.
# ---------------------------------------------------------------------------
def _patch_speculative_minimax_whitelist() -> None:
    """Allow MiniMax target models for eagle3/extract_hidden_states checks.

    Upstream vLLM validates that the target model_type is in a whitelist for
    methods that rely on auxiliary hidden states. Older upstream versions may
    not include MiniMax yet.
    """
    try:
        from vllm.config.speculative import SpeculativeConfig  # type: ignore
    except Exception:
        logger.warning(
            "SpeculativeConfig is not found, skip patching eagle3/extract_hidden_states checks for MiniMax-M2 on NPU."
        )
        return

    original_verify_args = getattr(SpeculativeConfig, "_verify_args", None)
    if original_verify_args is None:
        logger.warning(
            "SpeculativeConfig._verify_args is not found, skip patching "
            "eagle3/extract_hidden_states checks for MiniMax-M2 on NPU."
        )
        return
    if getattr(original_verify_args, "_vllm_ascend_minimax_eagle3_patched", False):
        logger.warning("eagle3/extract_hidden_states checks for MiniMax-M2 on NPU have already been patched.")
        return

    # Pydantic dataclass validation invokes `model_validators["_verify_args"].func`, not
    # necessarily the current `SpeculativeConfig._verify_args` attribute.
    decorators = getattr(SpeculativeConfig, "__pydantic_decorators__", None)
    mv = None
    if decorators is not None:
        model_validators = getattr(decorators, "model_validators", None)
        if isinstance(model_validators, dict):
            mv = model_validators.get("_verify_args")
    inner_verify = mv.func if mv is not None and getattr(mv, "func", None) is not None else original_verify_args

    def _patched_verify_args(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return inner_verify(self, *args, **kwargs)
        except ValueError as e:
            method = getattr(self, "method", None)
            if method not in ("eagle3", "extract_hidden_states"):
                raise

            target_cfg = getattr(self, "target_model_config", None)
            model_type = getattr(getattr(target_cfg, "hf_text_config", None), "model_type", "")
            if "minimax" not in str(model_type).lower():
                logger.warning(
                    "Model type %s is not a MiniMax-M2 model, skip eagle3/extract_hidden_states checks.",
                    model_type,
                )
                raise

            msg = str(e).lower()
            if "only supported for" in msg and "models" in msg:
                # Upstream `_verify_args` calls `verify_equal_vocab_size_if_draft_model` after
                # the aux-hidden allowlist; returning here would skip it.
                verify_vocab = getattr(self, "verify_equal_vocab_size_if_draft_model", None)
                if callable(verify_vocab):
                    verify_vocab()
                return self
            raise

    _patched_verify_args._vllm_ascend_minimax_eagle3_patched = True  # type: ignore[attr-defined]
    SpeculativeConfig._verify_args = _patched_verify_args  # type: ignore[assignment]

    if mv is not None:
        try:
            mv.func = _patched_verify_args  # type: ignore[misc]
        except (TypeError, AttributeError):
            object.__setattr__(mv, "func", _patched_verify_args)
    else:
        logger.warning(
            "Could not find SpeculativeConfig.__pydantic_decorators__.model_validators["
            "'_verify_args']; eagle3 whitelist patch may not run at init validation."
        )

    try:
        from pydantic.dataclasses import rebuild_dataclass  # type: ignore
    except Exception as e:
        logger.warning(
            "Cannot import rebuild_dataclass (%s); SpeculativeConfig eagle3 whitelist "
            "patch may not apply at instance construction time.",
            e,
        )
    else:
        try:
            rebuild_dataclass(SpeculativeConfig, force=True)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(
                "rebuild_dataclass(SpeculativeConfig) failed (%s); eagle3 whitelist patch may not apply.",
                e,
            )
        # If `VllmConfig` was imported before this patch ran, its pydantic-core schema
        # for the nested `speculative_config` field may still embed the *pre-patch*
        # SpeculativeConfig validators. `create_speculative_config()` calls
        # `SpeculativeConfig(...)` directly (uses updated class validator), but
        # `VllmConfig(..., speculative_config=...)` validates via the parent's cached
        # nested schema and can still raise the whitelist error unless we rebuild.
        try:
            from vllm.config.vllm import VllmConfig  # type: ignore
        except Exception:
            pass
        else:
            try:
                rebuild_dataclass(VllmConfig, force=True)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning(
                    "rebuild_dataclass(VllmConfig) failed (%s); VllmConfig(...) may "
                    "still use stale nested SpeculativeConfig validation.",
                    e,
                )


def _patch_eagle3_registry_alias() -> None:
    """Register Eagle3MiniMaxM2ForCausalLM architecture alias if missing."""
    try:
        import vllm.model_executor.models.registry as registry  # type: ignore
    except Exception:
        return

    # Prefer patching the underlying dicts when available.
    if hasattr(registry, "_SPECULATIVE_DECODING_MODELS"):
        models = registry._SPECULATIVE_DECODING_MODELS
        if isinstance(models, dict):
            models.setdefault("Eagle3MiniMaxM2ForCausalLM", ("llama_eagle3", "Eagle3LlamaForCausalLM"))

    # Fallback: patch resolved registry instance if present.
    model_registry = getattr(registry, "ModelRegistry", None)
    if model_registry is not None and hasattr(model_registry, "models"):
        try:
            model_registry.models.setdefault(  # type: ignore[attr-defined]
                "Eagle3MiniMaxM2ForCausalLM",
                ("llama_eagle3", "Eagle3LlamaForCausalLM"),
            )
        except Exception:
            return


_patch_speculative_minimax_whitelist()
_patch_eagle3_registry_alias()
