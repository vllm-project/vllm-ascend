# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Configuration and limits for perceptual similarity caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_ascend.distributed.ec_transfer.ssim import DEFAULT_DATA_RANGE

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_SUPPORTED_MATCHERS = frozenset({"phash_ssim"})
_PARALLEL_SIMILARITY_COMPUTE_THRESHOLD = 3
_MAX_SIMILARITY_COMPUTE_WORKERS = 8
# Configured persistent host-buffer bounds, in bytes.
MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE = 4 * 1024
MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE = 512 * 1024 * 1024
DEFAULT_RESIZED_IMAGE_RECV_BUFFER_SIZE = MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE
# A single resized tensor above this size is not loaded for optional matching.
MAX_RESIZED_IMAGE_TRANSFER_SIZE = 1024 * 1024 * 1024


@dataclass(frozen=True)
class SimilarityCacheConfig:
    """Validated ``ec_connector_extra_config.similarity_cache`` settings."""

    enabled: bool = False
    matcher: str = "phash_ssim"
    max_hamming: int = 5
    ssim_threshold: float = 0.99
    ssim_data_range: float = DEFAULT_DATA_RANGE
    resized_image_recv_buffer_size: int = DEFAULT_RESIZED_IMAGE_RECV_BUFFER_SIZE

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "SimilarityCacheConfig":
        ec_config = vllm_config.ec_transfer_config
        if ec_config is None:
            raise ValueError("ECMemcacheConnector requires ec_transfer_config")
        raw = ec_config.get_from_extra_config("similarity_cache", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("ec_connector_extra_config.similarity_cache must be a mapping")

        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("similarity_cache.enabled must be a boolean")

        matcher = raw.get("matcher", "phash_ssim")
        if not isinstance(matcher, str) or matcher not in _SUPPORTED_MATCHERS:
            raise ValueError(
                "similarity_cache.matcher must be one of "
                f"{sorted(_SUPPORTED_MATCHERS)}, got {matcher!r}"
            )

        max_hamming = raw.get("max_hamming", 5)
        if isinstance(max_hamming, bool) or not isinstance(max_hamming, int):
            raise ValueError("similarity_cache.max_hamming must be an integer")
        if not 0 <= max_hamming < 8:
            raise ValueError("similarity_cache.max_hamming must be between 0 and 7")

        ssim_threshold = raw.get("ssim_threshold", 0.99)
        if isinstance(ssim_threshold, bool) or not isinstance(ssim_threshold, (int, float)):
            raise ValueError("similarity_cache.ssim_threshold must be a number")
        ssim_threshold = float(ssim_threshold)
        if not 0.0 <= ssim_threshold <= 1.0:
            raise ValueError("similarity_cache.ssim_threshold must be between 0 and 1")

        ssim_data_range = raw.get("ssim_data_range", DEFAULT_DATA_RANGE)
        if isinstance(ssim_data_range, bool) or not isinstance(ssim_data_range, (int, float)):
            raise ValueError("similarity_cache.ssim_data_range must be a number")
        ssim_data_range = float(ssim_data_range)
        if ssim_data_range <= 0:
            raise ValueError("similarity_cache.ssim_data_range must be greater than 0")

        resized_image_recv_buffer_size = raw.get(
            "resized_image_recv_buffer_size",
            DEFAULT_RESIZED_IMAGE_RECV_BUFFER_SIZE,
        )
        if isinstance(resized_image_recv_buffer_size, bool) or not isinstance(
            resized_image_recv_buffer_size, int
        ):
            raise ValueError(
                "similarity_cache.resized_image_recv_buffer_size must be an integer"
            )
        if not (
            MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE
            <= resized_image_recv_buffer_size
            <= MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE
        ):
            raise ValueError(
                "similarity_cache.resized_image_recv_buffer_size must be between "
                f"{MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE} and "
                f"{MAX_RESIZED_IMAGE_RECV_BUFFER_SIZE} bytes, got "
                f"{resized_image_recv_buffer_size}"
            )

        return cls(
            enabled=enabled,
            matcher=matcher,
            max_hamming=max_hamming,
            ssim_threshold=ssim_threshold,
            ssim_data_range=ssim_data_range,
            resized_image_recv_buffer_size=resized_image_recv_buffer_size,
        )

