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
# Patch target: vllm/config/speculative.py, vllm/config/vllm.py,
#               vllm/v1/engine/input_processor.py
# - UNO (arxiv 2510.xxxxx) drafts with the *target* model plus a gated LoRA
#   adapter instead of a separate draft checkpoint, so it needs a speculative
#   method name ("uno") and a drafter-free configuration path. Upstream's
#   ``SpeculativeConfig.method`` is typed as a closed ``Literal`` and
#   ``__post_init__`` rejects any member it does not model, so neither
#   ``--spec-method uno`` nor a programmatic ``SpeculativeConfig(method="uno")``
#   is reachable without touching the upstream tree.
#
# Why a platform patch:
#   UNO ships inside vllm-ascend first (upstream RFC is still open, see
#   ``Related PR`` below). Two independent gates must be opened:
#
#     Gate 1 -- pydantic field validation. ``SpeculativeConfig`` is a pydantic
#       dataclass (``@config`` -> ``extra="forbid"``), so the Literal is
#       compiled into the core schema when the class is created. Rebinding the
#       module-level ``SpeculativeMethod`` afterwards has no effect; the schema
#       must be rebuilt.
#     Gate 2 -- ``__post_init__``. Even with Gate 1 open, the drafter-free
#       branch raises ``ValueError("num_speculative_tokens was provided but
#       without speculative model.")`` and the trailing dispatch raises
#       ``NotImplementedError("Unsupported speculative method: 'uno'")``.
#       pydantic compiles ``__post_init__`` into ``__init__`` and re-raises
#       whatever it throws as a ``ValidationError`` with ``type=value_error``,
#       so both gates surface as ``pydantic.ValidationError`` and can only be
#       told apart by the inner error type -- verified on the CI pin, see the
#       A0 gate in the PR description.
#
#   Because ``--spec-method`` argparse choices are derived from the upstream
#   Literal while ``AsyncEngineArgs.add_cli_args`` runs, *before*
#   ``current_platform.pre_register_and_update`` applies this module, the CLI
#   choices are widened separately in ``platform.py`` (same pattern as
#   ``--kv-cache-dtype``).
#
# This file is deliberately standalone: it does not modify
# ``patch_speculative_config.py`` so that the two patches cannot conflict.
#
# Related PR (if no, explain why):
#   https://github.com/vllm-project/vllm/pull/55947 -- UNO core (open).
# Future Plan:
#   Delete this patch once #55947 lands and the CI pin contains "uno" in
#   ``SpeculativeMethod`` and the UNO branch in ``SpeculativeConfig`` /
#   ``VllmConfig.num_lookahead_tokens``.

import typing

import pydantic.dataclasses as _pdc
from vllm.config import speculative as _spec_mod
from vllm.config import vllm as _vllm_mod
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import logger

UNO_METHOD = "uno"

# Derive from the live Literal so upstream additions stay accepted verbatim.
# The starred-subscript Literal is only analyzable at runtime, hence the type
# ignore -- same pattern as patch_kv_cache_dtype.py / patch_indexer_kv_dtype.py.
_SPECULATIVE_METHOD_WITH_UNO = typing.Literal[*_spec_mod.SpeculativeMethod.__args__ + (UNO_METHOD,)]  # type: ignore[valid-type]


def _widen_speculative_method() -> None:
    """Gate 1: let pydantic accept ``method="uno"``.

    A pydantic dataclass caches its core schema / validator at class-build time
    from the field annotations. Mutating one place is not enough; three places
    hold the old Literal:

      1. ``vllm.config.speculative.SpeculativeMethod`` -- the module-level
         symbol other modules import directly.
      2. ``SpeculativeConfig.__annotations__["method"]`` -- the inline
         annotation on the class.
      3. ``SpeculativeConfig.__dataclass_fields__["method"].type`` -- the
         stdlib dataclass field type. pydantic's ``collect_dataclass_fields``
         reads *this* when rebuilding (not the ``__pydantic_fields__``
         FieldInfo, which it overwrites). ``arg_utils._compute_kwargs`` also
         reads it through ``dataclasses.fields(cls)`` when deriving argparse
         choices.

    ``rebuild_dataclass(force=True)`` then regenerates the validator and core
    schema. It is idempotent and safe across plugin entry points.
    """
    if UNO_METHOD in getattr(_spec_mod.SpeculativeMethod, "__args__", ()):
        return

    _spec_mod.SpeculativeMethod = _SPECULATIVE_METHOD_WITH_UNO
    SpeculativeConfig.__annotations__["method"] = _SPECULATIVE_METHOD_WITH_UNO | None

    method_field = SpeculativeConfig.__dataclass_fields__.get("method")
    if method_field is None:
        raise RuntimeError(
            "SpeculativeConfig has no dataclass field 'method'; the upstream "
            "config layout changed and patch_uno_config.py must be updated."
        )
    method_field.type = _SPECULATIVE_METHOD_WITH_UNO | None

    # ``_parent_namespace_depth=1`` resolves types against this module's
    # namespace, where the widened Literal lives.
    _pdc.rebuild_dataclass(
        SpeculativeConfig,
        force=True,
        raise_errors=True,
        _parent_namespace_depth=1,
    )
    logger.info("Patched SpeculativeConfig.method to accept '%s'.", UNO_METHOD)


def _patch_uno_post_init() -> None:
    """Gate 2: let ``__post_init__`` accept UNO and populate its fields.

    The UNO branch returns early instead of running upstream's drafter setup,
    which assumes a draft checkpoint and would reject or mis-parse ``"uno"``.
    The previous implementation is captured at call time and chained, so this
    patch composes with ``patch_speculative_config.py`` regardless of import
    order.
    """
    current = SpeculativeConfig.__post_init__
    if getattr(current, "_uno_patched", False):
        return

    def _uno_post_init(self) -> None:
        if getattr(self, "method", None) == UNO_METHOD:
            # UNO reuses the target model and weights through a gated LoRA
            # adapter, so there is no draft checkpoint, draft parallel config
            # or prompt lookup. ``draft_sample_method`` must be probabilistic:
            # UNO verifies the gated draft distribution by rejection sampling.
            self.model = UNO_METHOD
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
            self.parallel_drafting = True
            self.draft_sample_method = "probabilistic"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            return
        current(self)

    _uno_post_init._uno_patched = True  # type: ignore[attr-defined]
    SpeculativeConfig.__post_init__ = _uno_post_init
    logger.info("Patched SpeculativeConfig.__post_init__ to accept '%s'.", UNO_METHOD)


def _patch_num_lookahead_tokens() -> None:
    """Reserve the draft query slots UNO writes past the target's range.

    Upstream derives the margin from ``use_eagle()`` and
    ``uses_draft_model()``; UNO is neither, so it would return 0 and the
    scheduler would never allocate the ``K`` KV positions the draft pass
    writes. Without this the draft forward reads unallocated blocks.
    """
    original = _vllm_mod.VllmConfig.num_lookahead_tokens
    if getattr(original, "_uno_patched", False):
        return

    def _uno_num_lookahead_tokens(self) -> int:
        speculative_config = self.speculative_config
        if speculative_config is not None and getattr(speculative_config, "method", None) == UNO_METHOD:
            return self.num_speculative_tokens
        return original.fget(self)

    _uno_num_lookahead_tokens._uno_patched = True  # type: ignore[attr-defined]
    _vllm_mod.VllmConfig.num_lookahead_tokens = property(_uno_num_lookahead_tokens)
    logger.info("Patched VllmConfig.num_lookahead_tokens for '%s'.", UNO_METHOD)


def _patch_v1_model_runner_refusal() -> None:
    """Refuse UNO on Model Runner V1.

    Only the V2 speculator implements UNO. Ascend replaces
    ``VllmConfig.use_v2_model_runner`` (``mrv2_utils.py``) so V2 is the default
    and its validation is a no-op, but an explicit ``VLLM_USE_V2_MODEL_RUNNER=0``
    would otherwise fall through to the V1 runner and fail deep inside the
    drafter setup with a misleading message.
    """
    original = _vllm_mod.VllmConfig._get_v1_model_runner_unsupported_features
    if getattr(original, "_uno_patched", False):
        return

    def _uno_v1_unsupported_features(self) -> list[str]:
        unsupported = original(self)
        speculative_config = self.speculative_config
        if speculative_config is not None and getattr(speculative_config, "method", None) == UNO_METHOD:
            unsupported.append("uno speculative decoding")
        return unsupported

    _uno_v1_unsupported_features._uno_patched = True  # type: ignore[attr-defined]
    _vllm_mod.VllmConfig._get_v1_model_runner_unsupported_features = _uno_v1_unsupported_features
    logger.info("Patched VllmConfig to refuse '%s' on Model Runner V1.", UNO_METHOD)


def _patch_lora_request_refusal() -> None:
    """Reject per-request LoRA while UNO owns the LoRA slot.

    UNO registers its gated adapter as the single active adapter for the whole
    engine. A request-scoped adapter would take over that slot and silently
    corrupt the draft distribution, so refuse it early with an actionable
    message instead of failing inside the worker.
    """
    from vllm.v1.engine.input_processor import InputProcessor

    original = InputProcessor._validate_lora
    if getattr(original, "_uno_patched", False):
        return

    def _uno_validate_lora(self, lora_request) -> None:
        original(self, lora_request)
        if lora_request is None:
            return
        speculative_config = self.speculative_config
        if speculative_config is not None and getattr(speculative_config, "method", None) == UNO_METHOD:
            raise ValueError(
                "UNO speculative decoding reserves the engine-wide LoRA slot for "
                f"its draft adapter (requested adapter: {lora_request.lora_name}). "
                "Per-request LoRA is not supported together with method='uno'."
            )

    _uno_validate_lora._uno_patched = True  # type: ignore[attr-defined]
    InputProcessor._validate_lora = _uno_validate_lora
    logger.info("Patched InputProcessor to refuse per-request LoRA under '%s'.", UNO_METHOD)


def apply() -> None:
    """Apply every UNO config patch. Safe to call repeatedly."""
    _widen_speculative_method()
    _patch_uno_post_init()
    _patch_num_lookahead_tokens()
    _patch_v1_model_runner_refusal()
    _patch_lora_request_refusal()


apply()
