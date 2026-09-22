# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING

from pydantic import Field
from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig


def _default_cpu_offload() -> bool:
    """Honor the legacy environment variable only when the field is omitted."""
    value = os.getenv("VLLM_PLE_CPU_OFFLOAD")
    if value is not None:
        logger.warning_once(
            "VLLM_PLE_CPU_OFFLOAD is a legacy setting and may be removed in a "
            "future release. On vLLM v0.29, use "
            '--additional-config \'{"engram_config": {"cpu_offload": true}}\' '
            "instead."
        )
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = Field(default_factory=_default_cpu_offload)
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Defaults to False, or VLLM_PLE_CPU_OFFLOAD when set for compatibility.
    An explicit value takes precedence over the legacy environment variable."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without supported embeddings."""
        text_config = None if model_config is None else model_config.hf_text_config
        if not getattr(text_config, "engram_layer_ids", None):
            raise ValueError("EngramConfig requires non-empty engram_layer_ids.")
        if self.embedding_across_dp:
            raise ValueError("Ascend Engram does not support DP-sharded embeddings.")

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1 and parallel_config.enable_elastic_ep:
            raise ValueError("Engram embedding_across_dp is not supported with elastic EP yet.")

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
