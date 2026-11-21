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

from typing import Any

from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)

_CACHE: Any = None
_CACHE_INITIALIZED = False


def _get_cache():
    global _CACHE, _CACHE_INITIALIZED
    if _CACHE_INITIALIZED:
        return _CACHE
    _CACHE_INITIALIZED = True
    cache_dir = envs.VLLM_ASCEND_MM_GLOBAL_CACHE_DIR
    if not cache_dir:
        return None

    from vllm_ascend.worker.mm_global_cache import GlobalMMEmbeddingCache

    _CACHE = GlobalMMEmbeddingCache(
        cache_dir,
        envs.VLLM_ASCEND_MM_GLOBAL_CACHE_GB * 1024**3,
    )
    logger.info(
        "Enabled global multimodal embedding cache at %s (%.1f GiB)",
        cache_dir,
        envs.VLLM_ASCEND_MM_GLOBAL_CACHE_GB,
    )
    return _CACHE


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
        cache = _get_cache()
        self._mm_global_pending_hashes = None
        if cache is None or not mm_hashes:
            return mm_hashes, mm_kwargs

        kept_hashes: list[str] = []
        kept_kwargs: list[tuple[str, Any]] = []
        for mm_hash, (modality, mm_kwarg) in zip(mm_hashes, mm_kwargs):
            if modality in ("image", "video"):
                cached = cache.get(mm_hash)
                if cached is not None:
                    self.encoder_cache.encoder_outputs[mm_hash] = cached.to(
                        self.device, non_blocking=True
                    )
                    logger.debug("Global MM cache hit for %s", mm_hash)
                    continue
            kept_hashes.append(mm_hash)
            kept_kwargs.append((modality, mm_kwarg))

        self._mm_global_pending_hashes = kept_hashes or None
        return kept_hashes, kept_kwargs

    def execute_mm_encoder(self, mm_kwargs):
        outputs = original_execute(self, mm_kwargs)
        cache = _get_cache()
        pending_hashes = getattr(self, "_mm_global_pending_hashes", None)
        self._mm_global_pending_hashes = None
        if (
            cache is not None
            and pending_hashes is not None
            and len(pending_hashes) == len(outputs)
        ):
            for mm_hash, output in zip(pending_hashes, outputs):
                if not mm_hash.startswith("tmp_"):
                    cache.put(mm_hash, output)
        return outputs

    EncoderRunner.prepare_mm_inputs = prepare_mm_inputs  # type: ignore[method-assign]
    EncoderRunner.execute_mm_encoder = execute_mm_encoder  # type: ignore[method-assign]
    EncoderRunner._mm_global_cache_patched = True  # type: ignore[attr-defined]


_patch_encoder_runner()
