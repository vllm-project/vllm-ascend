# SPDX-License-Identifier: Apache-2.0
"""Qwen PLE embedding using an independently owned BM table."""

import re

import torch
from vllm.config import get_current_vllm_config
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

from .ple_bm_reader import open_bm_reader


def shared_ple_config(vllm_config):
    config = (vllm_config.additional_config or {}).get("shared_ple")
    if config is None:
        return None
    if not isinstance(config, dict) or config.get("backend") not in ("bm", "rdma"):
        raise ValueError("shared_ple requires backend='bm' or backend='rdma'")
    endpoint_key = "root" if config["backend"] == "bm" else "endpoint"
    if not all(isinstance(config.get(key), str) and config[key] for key in (endpoint_key, "checkpoint_prefix")):
        raise ValueError(f"shared_ple requires {endpoint_key} and checkpoint_prefix")
    engram = vllm_config.engram_config
    if engram is None or not engram.cpu_offload:
        raise ValueError("shared_ple requires engram_config.cpu_offload=true")
    if config["backend"] == "rdma" and getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError("Shared PLE RDMA does not support speculative decoding yet")
    parallel = getattr(vllm_config, "parallel_config", None)
    if config["backend"] == "rdma" and parallel is not None:
        if (
            getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
        ):
            raise ValueError("Shared PLE RDMA currently requires PCP=DCP=1")
        if getattr(parallel, "enable_dbo", False) or getattr(parallel, "enable_elastic_ep", False):
            raise ValueError("Shared PLE RDMA does not support DBO or elastic EP")
    return config


def validate_table_prefix(model_prefix: str, checkpoint_prefix: str) -> None:
    pattern = r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding$"
    model_layer = re.search(pattern, model_prefix)
    checkpoint_layer = re.search(pattern, checkpoint_prefix)
    if model_layer is None or checkpoint_layer is None or model_layer.group(1) != checkpoint_layer.group(1):
        raise ValueError("Shared PLE checkpoint prefix does not match the model layer")


class AscendSharedPLEEmbedding(Qwen4ExpPLEEmbedding):
    """Retain Qwen ETP semantics while reading a globally indexed BM table."""

    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        config = get_current_vllm_config()
        shared = shared_ple_config(config)
        if shared is None:
            raise ValueError("Shared PLE embedding requires shared_ple configuration")
        validate_table_prefix(kwargs.get("prefix", ""), shared["checkpoint_prefix"])
        super().__init__(*args, **kwargs)
        checkpoint_config = config.model_config.hf_text_config
        # Reuse #19528's per-process AICPU binding before importing remote BM.
        from vllm_ascend.models.deepseek_v41.engram.common import warmup_shared_engram_aicpu

        warmup_shared_engram_aicpu()
        self.reader = open_bm_reader(
            config.model_config.model,
            shared["checkpoint_prefix"],
            checkpoint_config.split_ngram_parts,
            self.embedding_dim,
            shared["root"],
            torch.device("npu", torch.npu.current_device()),
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )
        # The owner loaded this immutable parameter. Keep metadata for Qwen's
        # dtype checks without making it a missing model parameter.
        try:
            weight = self._parameters.pop("weight")
            self.register_buffer("weight", weight.detach(), persistent=False)
        except BaseException:
            self.reader.close_host_offload()
            raise

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
        if dtype != torch.bfloat16:
            raise ValueError("Shared PLE currently supports BF16 tables only")
        return torch.empty((num_embeddings, embedding_dim), dtype=dtype, device="meta")

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        raise NotImplementedError("Shared PLE asynchronous prefetch is not supported")

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise NotImplementedError("Shared PLE asynchronous prefetch is not supported")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered = self._gather_dp_ids(ngram_ids, slot_size).contiguous()
        embeddings = self.reader(gathered)
        if self.reduce_results and self.tp_size > 1:
            assert self.parallel_group is not None
            embeddings = self.parallel_group.all_reduce(embeddings)
        return self._select_embeddings(embeddings, ngram_ids.shape[0], slot_offset)
