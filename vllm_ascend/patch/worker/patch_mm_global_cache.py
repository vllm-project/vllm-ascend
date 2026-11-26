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

"""Process-shared multimodal embedding cache for the MRV2 encoder runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.mm_global_cache import GlobalMMEmbeddingCache

logger = init_logger(__name__)

_CACHEABLE_MODALITIES = frozenset({"image", "video"})
_CACHE_UNSET = object()


def _model_namespace(model: Any) -> str:
    parts = [type(model).__name__]
    model_config = getattr(model, "model_config", None)
    if model_config is not None:
        parts.extend(str(getattr(model_config, field, "")) for field in ("model", "revision", "dtype", "quantization"))
        multimodal_config = getattr(model_config, "multimodal_config", None)
        if multimodal_config is not None:
            parts.append(str(getattr(multimodal_config, "mm_encoder_tp_mode", "")))

    hf_config = getattr(model, "config", None)
    if hf_config is not None:
        try:
            parts.append(json.dumps(hf_config.to_dict(), sort_keys=True, default=str))
        except (AttributeError, TypeError, ValueError):
            parts.append(repr(hf_config))

    with contextlib.suppress(Exception):
        parts.append(f"tp{get_tp_group().world_size}")

    with contextlib.suppress(AttributeError, StopIteration):
        parts.append(str(next(model.parameters()).dtype))

    material = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def _is_encoder_decoder_model(model: Any) -> bool:
    model_config = getattr(model, "model_config", None)
    if model_config is not None:
        return bool(getattr(model_config, "is_encoder_decoder", False))
    return bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))


def _build_cache(model: Any) -> GlobalMMEmbeddingCache | None:
    cache_config = get_ascend_config().mm_global_cache_config
    if not cache_config.cache_dir:
        return None
    if _is_encoder_decoder_model(model):
        logger.warning_once(
            "Global multimodal embedding cache is not supported for encoder-decoder models; the cache is disabled."
        )
        return None
    root = Path(os.path.expanduser(cache_config.cache_dir)) / _model_namespace(model)
    cache = GlobalMMEmbeddingCache(str(root), cache_config.capacity_gb * 1024**3)
    logger.info(
        "Enabled global multimodal embedding cache at %s (%d GiB)",
        root,
        cache_config.capacity_gb,
    )
    return cache


def _get_cache(runner: Any) -> GlobalMMEmbeddingCache | None:
    cache = getattr(runner, "_mm_global_cache", _CACHE_UNSET)
    if cache is _CACHE_UNSET:
        try:
            cache = _build_cache(runner.model)
        except RuntimeError:
            logger.warning_once("Global multimodal embedding cache disabled: Ascend config is not initialized.")
            cache = None
        runner._mm_global_cache = cache
    return cache


def _agree_hits(device: torch.device, local_hits: list[bool]) -> list[bool]:
    if not local_hits:
        return local_hits
    try:
        tp_group = get_tp_group()
    except Exception:
        return local_hits
    if tp_group.world_size == 1:
        return local_hits

    flags = torch.tensor([1.0 if hit else 0.0 for hit in local_hits], dtype=torch.float32, device=device)
    reduced = tp_group.all_reduce(flags)
    return (reduced == tp_group.world_size).tolist()


def _is_encoder_write_rank() -> bool:
    try:
        return get_tp_group().rank_in_group == 0
    except Exception:
        return True


def _patch_encoder_runner() -> None:
    try:
        from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
    except Exception:
        return

    if getattr(EncoderRunner, "_mm_global_cache_patched", False):
        return

    original_prepare = EncoderRunner.prepare_mm_inputs
    original_execute = EncoderRunner.execute_mm_encoder

    def prepare_mm_inputs(self, scheduled_encoder_inputs):
        mm_hashes, mm_kwargs = original_prepare(self, scheduled_encoder_inputs)
        self._mm_global_pending = None
        cache = _get_cache(self)
        if cache is None or not mm_hashes:
            return mm_hashes, mm_kwargs

        local_hits: list[bool] = []
        cached_values: list[torch.Tensor | None] = []
        for mm_hash, (modality, _) in zip(mm_hashes, mm_kwargs):
            value = cache.get(mm_hash) if modality in _CACHEABLE_MODALITIES else None
            cached_values.append(value)
            local_hits.append(value is not None)

        hits = _agree_hits(self.device, local_hits)

        kept_hashes: list[str] = []
        kept_kwargs: list[tuple[str, Any]] = []
        pending: list[tuple[str, str]] = []
        for mm_hash, (modality, mm_kwarg), hit, value in zip(mm_hashes, mm_kwargs, hits, cached_values):
            if hit:
                assert value is not None
                self.encoder_cache.encoder_outputs[mm_hash] = value.to(self.device, non_blocking=True)
                logger.debug("Global MM cache hit for %s", mm_hash)
                continue
            kept_hashes.append(mm_hash)
            kept_kwargs.append((modality, mm_kwarg))
            pending.append((mm_hash, modality))

        self._mm_global_pending = pending or None
        return kept_hashes, kept_kwargs

    def execute_mm_encoder(self, mm_kwargs):
        outputs = original_execute(self, mm_kwargs)
        pending = getattr(self, "_mm_global_pending", None)
        self._mm_global_pending = None
        cache = _get_cache(self)
        if cache is None or pending is None or len(pending) != len(outputs):
            return outputs
        if not _is_encoder_write_rank():
            return outputs
        for (mm_hash, modality), output in zip(pending, outputs):
            if modality in _CACHEABLE_MODALITIES and not mm_hash.startswith("tmp_"):
                cache.put(mm_hash, output)
        return outputs

    EncoderRunner.prepare_mm_inputs = prepare_mm_inputs  # type: ignore[method-assign]
    EncoderRunner.execute_mm_encoder = execute_mm_encoder  # type: ignore[method-assign]
    EncoderRunner._mm_global_cache_patched = True  # type: ignore[attr-defined]


_patch_encoder_runner()
