# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

from vllm_ascend.ec_manager.score_encoder_cache import ScoreEncoderCache

if TYPE_CHECKING:
    from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
    from vllm.multimodal.inputs import MultiModalFeatureSpec


class AscendScoreEncoderRunner(EncoderRunner):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.storage = ScoreEncoderCache(self.encoder_cache, self.device)

    def cache_encoder_outputs(
        self,
        mm_hashes: list[str],
        outputs: list[torch.Tensor],
    ) -> None:
        for mm_hash, output in zip(mm_hashes, outputs):
            self._cache_encoder_output(mm_hash, output)

    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: torch.Tensor,
    ) -> None:
        self.storage.store(mm_hash, output)

    def _has_cached_encoder_output(self, mm_hash: str) -> bool:
        return self.storage.has(mm_hash)

    def _get_encoder_output_from_cache(
        self, mm_hash: str
    ) -> torch.Tensor | None:
        return self.storage.get(mm_hash)

    def process_scheduler_metadata(
        self,
        metadata: "EncoderCacheManagerMetadata",
    ) -> None:
        self.storage.apply_metadata(metadata)

    def track_request(
        self,
        req_id: str,
        mm_features: "list[MultiModalFeatureSpec]",
    ) -> None:
        self.storage.track_request(req_id, mm_features)

    def on_request_removed(self, req_id: str) -> None:
        self.storage.on_request_removed(req_id)

    def reset_encoder_cache(self) -> None:
        self.storage.reset()
