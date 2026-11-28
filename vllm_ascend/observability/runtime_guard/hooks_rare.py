#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""rare_chars sample-path helpers used by ``hooks.py``.

Keeps logits stash, batch-sharded all-to-all wrap, and async feed attach out of
the generic runtime_guard decorator surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Stashed pre-sample logits for Guard-side rare_chars (top1, chosen) collect.
# Detection-only: never attached to the OpenAI / SamplingParams.logprobs path.
_PRE_SAMPLE_LOGITS_ATTR = "_rg_pre_sample_logits"


def stash_pre_sample_logits(runner: Any, logits: Any) -> None:
    """Snapshot this step's logits for ``rare_chars`` (cleared after use).

    Clone tensors so a later in-place grammar bitmask cannot zero out the
    (top1, chosen) signal before collect runs.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().clone()
    setattr(runner, _PRE_SAMPLE_LOGITS_ATTR, logits)


def take_pre_sample_logits(runner: Any) -> Any:
    """Pop stashed pre-sample logits (or ``None``)."""
    logits = getattr(runner, _PRE_SAMPLE_LOGITS_ATTR, None)
    if hasattr(runner, _PRE_SAMPLE_LOGITS_ATTR):
        try:
            delattr(runner, _PRE_SAMPLE_LOGITS_ATTR)
        except Exception:
            setattr(runner, _PRE_SAMPLE_LOGITS_ATTR, None)
    return logits


def rare_chars_active(guard: Any) -> bool:
    """Config enable and not latched off after warmup/native failure."""
    detectors = getattr(guard, "detectors", None)
    if detectors is not None and hasattr(detectors, "rare_chars_enabled"):
        return bool(detectors.rare_chars_enabled())
    cfg = getattr(guard, "runtime_config", None)
    if cfg is None:
        return False
    return bool(cfg.detector_get("rare_chars", "enabled", False))


# Back-compat alias for tests that import the private name from hooks.
_rare_chars_active = rare_chars_active


def trim_vocab_logits(runner: Any, logits: Any) -> Any:
    """Match ``GPUModelRunner.sample`` truncating all-to-all width to ``vocab_size``."""
    if not isinstance(logits, torch.Tensor):
        return logits
    vocab = getattr(runner, "vocab_size", None)
    if vocab is None or logits.size(-1) <= int(vocab):
        return logits
    return logits[:, : int(vocab)]


def _restore_bound_override(obj: Any, name: str, orig: Any, *, had_instance_attr: bool) -> None:
    """Undo a temporary instance override installed by a wrap contextmanager."""
    if had_instance_attr:
        setattr(obj, name, orig)
    elif name in getattr(obj, "__dict__", {}):
        delattr(obj, name)


@contextmanager
def wrap_compute_logits_for_pre_sample(runner: Any, input_batch: Any) -> Iterator[None]:
    """Run ``check_before_sample`` once on full pre-grammar logits.

    Covers both sample paths:
    - Normal: wrap ``model.compute_logits`` (parent has no mid-hook before grammar).
    - Batch-sharded: patch ``all_to_all_vocab_shards`` (or legacy
      ``all_to_all_logits``) in ``GPUModelRunner.sample``'s globals so we see
      request-mapped full-vocab logits (wrapping ``compute_logits_local`` alone
      would stash a vocab shard).

    Fires once per context (v2 may call ``compute_logits`` twice per step).
    Restored in ``finally``.
    """
    from vllm_ascend.observability.runtime_guard.hooks import check_before_sample_from_batch

    model = runner.model
    # Prefer deleting the instance override so the class method is restored.
    had_instance_attr = "compute_logits" in getattr(model, "__dict__", {})
    orig = model.compute_logits
    guard = runner.runtime_guard
    fired = False

    # Patch the name ``GPUModelRunner.sample`` closes over (import-bound global).
    # Pinned vLLM uses ``all_to_all_vocab_shards``; older trees used ``all_to_all_logits``.
    sample_globals = getattr(GPUModelRunner.sample, "__globals__", None)
    all_to_all_names = ("all_to_all_vocab_shards", "all_to_all_logits")
    patch_targets: list[tuple[str, Any]] = []
    if isinstance(sample_globals, dict):
        for name in all_to_all_names:
            if name in sample_globals and callable(sample_globals[name]):
                patch_targets.append((name, sample_globals[name]))

    def _fire(logits: Any) -> None:
        nonlocal fired
        if fired:
            return
        fired = True
        batch = input_batch if input_batch is not None else getattr(runner, "input_batch", None)
        check_before_sample_from_batch(guard, trim_vocab_logits(runner, logits), batch)

    def wrapped(hidden_states, *args, **kwargs):
        logits = orig(hidden_states, *args, **kwargs)
        _fire(logits)
        return logits

    def _make_wrapped_all_to_all(orig_fn: Any):
        def wrapped_all_to_all(*args: Any, **kwargs: Any):
            logits = orig_fn(*args, **kwargs)
            _fire(logits)
            return logits

        return wrapped_all_to_all

    model.compute_logits = wrapped
    for name, orig_fn in patch_targets:
        sample_globals[name] = _make_wrapped_all_to_all(orig_fn)
    try:
        yield
    finally:
        for name, orig_fn in patch_targets:
            sample_globals[name] = orig_fn
        _restore_bound_override(model, "compute_logits", orig, had_instance_attr=had_instance_attr)


def take_rare_batch_for_async(runner: Any) -> Any | None:
    """Pop this step's rare batch for :class:`AscendAsyncOutput` ownership."""
    guard = getattr(runner, "runtime_guard", None)
    take = getattr(guard, "take_step_rare_batch", None) if guard is not None else None
    if callable(take):
        return take()
    return None


def feed_rare_batch_safe(runner: Any, output: Any, batch: Any) -> None:
    """Feed rare batch from async ``get_output``; never raise into the engine."""
    guard = getattr(runner, "runtime_guard", None)
    if guard is None:
        return
    try:
        guard.feed_rare_logprobs(output, batch=batch)
    except Exception:
        logger.exception("[runtime_guard soft-fail] async feed_rare_logprobs raised")


def maybe_feed_rare_sync_output(guard: Any, output: Any) -> None:
    """Non-AsyncOutput path: feed this step's rare batch immediately."""
    from vllm_ascend.observability.runtime_guard.hooks import AscendAsyncOutput

    take = getattr(guard, "take_step_rare_batch", None)
    feed = getattr(guard, "feed_rare_logprobs", None)
    if callable(take) and callable(feed) and not isinstance(output, AscendAsyncOutput):
        batch = take()
        if batch is not None:
            feed(output, batch=batch)
