#
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
# Patch targets:
#   - vllm/transformers_utils/config.py  : teach the model_type lookup about
#     the SDAR checkpoints that UNO bundles ship.
#   - vllm/config/speculative.py         : admit ``method="uno"``.
#   - vllm/v1/core/sched/scheduler.py    : reserve UNO's lookahead KV slots.
#
# UNO ("one model") speculative decoding has no draft model at all: both
# passes of a decode cycle run the *target* transformer, the draft pass with a
# gated LoRA applied to its noise rows.  vLLM's ``SpeculativeMethod`` literal
# does not know that method, and because ``SpeculativeConfig`` is a pydantic
# dataclass the rejection happens during *field validation*, before
# ``__post_init__`` runs.  Chaining ``__post_init__`` -- the technique
# patch_speculative_config.py uses for DSpark -- is therefore not enough on
# its own; the field schema has to be rebuilt as well.

from typing import Literal

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig
from vllm.config import speculative as speculative_module
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import logger
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.models.sdar import SDARConfig

UNO_METHOD = "uno"

# ---------------------------------------------------------------------------
# 1. SDAR config registration
# ---------------------------------------------------------------------------
# `_CONFIG_REGISTRY` is a LazyConfigDict whose values may be either a module
# attribute name or the class itself, so the class can be inserted directly.
# Without this, `AutoConfig.from_pretrained` on a UNO bundle raises
# "contains custom code which must be executed" before vLLM's ModelRegistry is
# ever consulted -- and `trust_remote_code=True` is not a usable escape hatch,
# because the bundled `SDARForCausalLM.__init__` reads a `config.noise` key
# that the released config.json does not contain.
_CONFIG_REGISTRY["sdar"] = SDARConfig


# ---------------------------------------------------------------------------
# 2. Admit method="uno"
# ---------------------------------------------------------------------------
# Pydantic caches a dataclass's compiled schema on the class. `rebuild_dataclass`
# deletes these before regenerating them, so anything that rebuilds a schema has
# to be able to put the previous ones back: a class with no
# `__pydantic_validator__` cannot be constructed at all.
_PYDANTIC_SCHEMA_ATTRS = (
    "__pydantic_core_schema__",
    "__pydantic_validator__",
    "__pydantic_serializer__",
    "__pydantic_fields__",
    "__pydantic_complete__",
)

# Captured before anything is rewritten, so a failed rebuild can undo the
# annotation change as well as the schema deletion.
_original_method_literal = speculative_module.SpeculativeMethod
_original_method_annotation = SpeculativeConfig.__annotations__.get("method")


def _restore_pydantic_schema(cls, snapshot: dict) -> None:
    for attr in _PYDANTIC_SCHEMA_ATTRS:
        if attr in cls.__dict__ and attr not in snapshot:
            delattr(cls, attr)
    for attr, value in snapshot.items():
        setattr(cls, attr, value)


def _extend_speculative_method_literal() -> bool:
    """Add ``"uno"`` to the validated ``SpeculativeConfig.method`` literal.

    Returns True when the schema was rebuilt successfully.  Everything here is
    idempotent, so a double import (or a second patch of the same class) is
    harmless.
    """
    method_field = SpeculativeConfig.__dataclass_fields__.get("method")
    if method_field is None:  # pragma: no cover - upstream renamed the field
        logger.warning("SpeculativeConfig has no 'method' field; UNO cannot be registered.")
        return False

    extended = Literal[speculative_module.SpeculativeMethod, UNO_METHOD]  # type: ignore[valid-type]
    annotation = extended | None

    # Keep the module-level alias in sync so anything that reads
    # `get_args(SpeculativeMethod)` (help text, downstream validation) sees UNO.
    speculative_module.SpeculativeMethod = extended  # type: ignore[misc]
    SpeculativeConfig.__annotations__["method"] = annotation
    method_field.type = annotation

    # Drop the cached pydantic artefacts so `rebuild_dataclass` regenerates the
    # core schema from the new annotation.  `__pydantic_decorators__` is left
    # alone, which is what keeps `_verify_args` and `_parse_attention_backend`
    # attached to the rebuilt validator.
    #
    # Snapshot first: without the cached validator SpeculativeConfig cannot be
    # constructed at all, so a failed rebuild has to put back what it found or
    # it breaks *every* speculative method rather than just UNO.
    snapshot = {
        attr: SpeculativeConfig.__dict__[attr] for attr in _PYDANTIC_SCHEMA_ATTRS if attr in SpeculativeConfig.__dict__
    }
    for attr in _PYDANTIC_SCHEMA_ATTRS:
        if attr in SpeculativeConfig.__dict__:
            delattr(SpeculativeConfig, attr)
    SpeculativeConfig.__pydantic_complete__ = False

    try:
        rebuilt = rebuild_dataclass(SpeculativeConfig, force=True)
    except Exception as exc:  # pragma: no cover - depends on the pydantic build
        rebuilt = False
        logger.warning("Rebuilding SpeculativeConfig's schema raised %s.", exc)
    if rebuilt is False:
        _restore_pydantic_schema(SpeculativeConfig, snapshot)
        speculative_module.SpeculativeMethod = _original_method_literal
        SpeculativeConfig.__annotations__["method"] = _original_method_annotation
        method_field.type = _original_method_annotation
        logger.warning(
            "Could not rebuild SpeculativeConfig's schema; restored the "
            'original one, so --speculative-config \'{"method": "uno"}\' will '
            "be rejected but every other method keeps working."
        )
        return False

    # An outer pydantic dataclass caches the inner schema it was built with, so
    # VllmConfig still holds the pre-UNO literal. That is harmless on every real
    # path -- arg_utils builds a SpeculativeConfig *instance* and pydantic does
    # not revalidate dataclass instances -- but rebuilding VllmConfig too closes
    # the dict-shaped hole (VllmConfig(speculative_config={"method": "uno"})).
    #
    # `rebuild_dataclass` deletes the cached artefacts *before* it regenerates
    # them, so a failure half way through would leave VllmConfig with no
    # validator at all -- every VllmConfig(...) in the process would then raise.
    # Snapshot them first and put them back if the rebuild does not complete.
    _rebuild_vllm_config_schema()
    return True


def _rebuild_vllm_config_schema() -> None:
    """Best effort: regenerate VllmConfig's schema, restoring it on failure."""
    snapshot = {attr: VllmConfig.__dict__[attr] for attr in _PYDANTIC_SCHEMA_ATTRS if attr in VllmConfig.__dict__}
    try:
        # `rebuild_dataclass` can also report failure by *returning* False --
        # pydantic installs mock validators in that case rather than raising --
        # so the return value has to be inspected, not just the exception.
        rebuilt = rebuild_dataclass(VllmConfig, force=True)
        if rebuilt is False:
            raise RuntimeError("rebuild_dataclass returned False")
    except Exception as exc:  # pragma: no cover - depends on the pydantic build
        _restore_pydantic_schema(VllmConfig, snapshot)
        logger.warning(
            "Could not rebuild VllmConfig's schema after registering UNO (%s); "
            "restored the previous schema. Constructing VllmConfig from a raw "
            "speculative_config dict will still reject method='uno'.",
            exc,
        )


_extend_speculative_method_literal()


# ---------------------------------------------------------------------------
# 3. UNO's __post_init__ branch
# ---------------------------------------------------------------------------
_prev_post_init = SpeculativeConfig.__post_init__


def _uno_post_init(self):
    if self.method != UNO_METHOD:
        return _prev_post_init(self)

    # UNO reuses the target transformer for both passes, so there is no draft
    # model to resolve; `model` carries the path (or HF/ModelScope id) of the
    # gated draft LoRA instead.  Point the draft configs at the target, the
    # same way the ngram / suffix / custom_class branches upstream do, so that
    # everything downstream which reads `draft_model_config` for vocab size or
    # KV-cache shape keeps working.
    if self.model is None:
        raise ValueError(
            "UNO speculative decoding requires the draft LoRA path in "
            "speculative_config['model'], e.g. "
            '--speculative-config \'{"method": "uno", "model": '
            '"s-sahoo/uno-qwen3-8B/adapter", "num_speculative_tokens": 8}\''
        )
    if self.target_model_config is None:
        raise ValueError("target_model_config must be present for uno")

    self.prompt_lookup_max = 0
    self.prompt_lookup_min = 0
    self.draft_model_config = self.target_model_config
    self.draft_parallel_config = self.target_parallel_config

    # Sampling the draft rows is part of UNO, not a tuning knob. The clean token
    # is only accepted with certainty when it was *drawn* from the distribution
    # the verify forward scores it against; with the upstream default ("greedy")
    # a temperature>0 request proposes an argmax against an unknown q, which the
    # rejection sampler accepts with probability p(x) -- often low enough that a
    # cycle emits fewer than two tokens for its two forwards. The cost is the
    # [num_draft_tokens, vocab_size] probability tensor.
    if self.draft_sample_method != "probabilistic":
        logger.info(
            "UNO speculative decoding: setting draft_sample_method='probabilistic' "
            "(was %s). UNO's clean token is only accepted with certainty when the "
            "draft rows are sampled rather than argmaxed.",
            self.draft_sample_method,
        )
        self.draft_sample_method = "probabilistic"

    # UNO drafts a whole block in one forward, like DFlash and DSpark.  This
    # flag is load-bearing in two places: it sizes the per-request slot budget
    # (`max_num_new_slots_for_drafting`), and it makes the Ascend attention
    # metadata builder read the *device* `seq_lens` tensor rather than a CPU
    # mirror -- which is the only one the UNO draft forward rewrites.
    self.parallel_drafting = True

    return self


SpeculativeConfig.__post_init__ = _uno_post_init


# ---------------------------------------------------------------------------
# 4. Lookahead KV slots
# ---------------------------------------------------------------------------
# `Scheduler.__init__` picks `num_lookahead_tokens` from a hard-coded list of
# methods.  UNO's draft forward writes KV for F rows starting at the new
# frontier C', and C' can be as far as C + F + 1 after a fully accepted step,
# so it needs exactly `num_speculative_tokens` (= F) reserved positions beyond
# the verify window -- the same budget EAGLE uses.  Without this the draft
# forward's slot mapping is derived from an unallocated block-table entry and
# silently corrupts KV block 0.
_orig_scheduler_init = Scheduler.__init__


def _uno_scheduler_init(self, *args, **kwargs):
    _orig_scheduler_init(self, *args, **kwargs)
    speculative_config = self.vllm_config.speculative_config
    if speculative_config is not None and speculative_config.method == UNO_METHOD:
        self.num_lookahead_tokens = self.num_spec_tokens


Scheduler.__init__ = _uno_scheduler_init


# ---------------------------------------------------------------------------
# 5. Keep the captured graphs base-only
# ---------------------------------------------------------------------------
# UNO synthesises a `LoRAConfig` for its own use, which normally makes the
# dispatcher capture a graph per LoRA case. That is pure waste here -- no
# request can ever select an adapter -- and it is also unsafe: whether the LoRA
# ops appear in a captured graph is decided by Python branches (`no_lora`, and
# the dense route's `_dense_lora_slot`) evaluated at *capture* time, so a graph
# captured with an adapter live would keep applying it when replayed for the
# verify forward, which must run on base weights.
#
# Capturing only the no-LoRA case makes the verify forward's graph provably
# adapter-free; UNO's draft forward is the only adapted forward and it runs
# eager. `_dummy_run` has a matching change: it must not force the dummy
# adapters on for a capture whose descriptor says "no LoRA".
_orig_get_lora_cases = CudagraphDispatcher._get_lora_cases


def _uno_get_lora_cases(self) -> list[int]:
    speculative_config = self.vllm_config.speculative_config
    if speculative_config is not None and speculative_config.method == UNO_METHOD:
        return [0]
    return _orig_get_lora_cases(self)


CudagraphDispatcher._get_lora_cases = _uno_get_lora_cases


# ---------------------------------------------------------------------------
# 6. Get past vLLM's own drafter dispatch
# ---------------------------------------------------------------------------
# `GPUModelRunner.__init__` carries a second, independent drafter dispatch that
# ends in `raise ValueError("Unknown speculative decoding method: ...")`. It
# runs from `NPUModelRunner.__init__`'s `super().__init__()`, i.e. *before*
# vllm-ascend's own `_set_up_drafter`, so an out-of-tree method dies there --
# which is exactly what UNO did on hardware: "Unknown speculative decoding
# method: uno", raised while constructing the runner.
#
# Every method vllm-ascend supports today is also dispatchable by that parent
# chain (dspark, for instance, lands in the `use_eagle()` branch and gets its
# throwaway EagleProposer replaced immediately afterwards), so the house pattern
# is "let the parent build something, then overwrite it".
#
# Inside that constructor `method` is read *only* by this dispatch: everything
# else keys on `num_speculative_tokens`, `draft_model_config` or
# `use_ngram_gpu()`, none of which the swap below touches. "medusa" is the
# cheapest branch that exists -- `MedusaProposer.__init__` is pure attribute
# assignment, no allocation and no weight loading -- and it is also the closest
# in shape to UNO, being the only other method with no separate draft model.
_orig_gpu_runner_init = GPUModelRunner.__init__


def _uno_gpu_runner_init(self, vllm_config, *args, **kwargs):
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None or speculative_config.method != UNO_METHOD:
        return _orig_gpu_runner_init(self, vllm_config, *args, **kwargs)

    speculative_config.method = "medusa"
    try:
        return _orig_gpu_runner_init(self, vllm_config, *args, **kwargs)
    finally:
        # Restored on the shared config object, so `self.speculative_config`
        # (the same instance) reads "uno" again for the Ascend runner's own
        # `_set_up_drafter`, which replaces the throwaway proposer.
        speculative_config.method = UNO_METHOD


GPUModelRunner.__init__ = _uno_gpu_runner_init
