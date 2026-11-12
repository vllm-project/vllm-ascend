# SPDX-License-Identifier: Apache-2.0
"""Qwen PLE embedding backed by rows fetched before model graph execution."""

import torch
from vllm.config import get_current_vllm_config
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

from .ple_bm_embedding import shared_ple_config, validate_table_prefix
from .ple_rdma import MAX_ROWS, PLERemoteReader


class AscendRemotePLEEmbedding(Qwen4ExpPLEEmbedding):
    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        config = get_current_vllm_config()
        shared = shared_ple_config(config)
        if shared is None or shared["backend"] != "rdma":
            raise ValueError("Remote PLE embedding requires shared_ple RDMA configuration")
        validate_table_prefix(kwargs.get("prefix", ""), shared["checkpoint_prefix"])
        tokens, heads = kwargs.get("max_total_tokens", 0), kwargs.get("num_ngram_heads", 0)
        if tokens <= 0 or heads <= 0 or tokens * heads > MAX_ROWS:
            raise ValueError("Remote PLE token/head capacity is invalid")
        super().__init__(*args, **kwargs)
        if self.etp_data_parallel_size != 1:
            raise ValueError("Remote PLE currently requires ETP=TP")
        checkpoint_config = config.model_config.hf_text_config
        self._shared_pool_client: PLERemoteReader | None = PLERemoteReader(
            config.model_config.model,
            shared["checkpoint_prefix"],
            checkpoint_config.split_ngram_parts,
            self.embedding_dim,
            shared["endpoint"],
            capacity=tokens * heads,
            library_path=shared.get("library_path"),
        )
        try:
            assert self._shared_pool_client is not None
            self.register_buffer(
                "prepared_rows",
                torch.zeros((tokens, heads, self.embedding_dim), dtype=torch.bfloat16, device="npu"),
                persistent=False,
            )
            weight = self._parameters.pop("weight")
            self.register_buffer("weight", weight.detach(), persistent=False)
            self.ple_external_checkpoint_names = self._shared_pool_client.ple_external_checkpoint_names
        except BaseException:
            self._shared_pool_client.close()
            raise
        self._preparation_bound = False
        self._closing = False

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
        if dtype != torch.bfloat16:
            raise ValueError("Remote PLE currently supports BF16 tables only")
        return torch.empty((num_embeddings, embedding_dim), dtype=dtype, device="meta")

    def bind_preparation(self) -> None:
        self._preparation_bound = True

    def prepare_rows(self, ids: torch.Tensor) -> None:
        if not self._preparation_bound or self._closing:
            raise RuntimeError("Remote PLE preparation is unavailable")
        if self._shared_pool_client is None:
            raise RuntimeError("Remote PLE reader is closed")
        if ids.ndim != 2 or ids.shape[1] != self.prepared_rows.shape[1] or ids.shape[0] > self.prepared_rows.shape[0]:
            raise ValueError("Remote PLE IDs exceed the prepared graph buffer")
        # Previous graph replay may still read this buffer on another stream.
        # This conservative fence is required until a measured per-step event exists.
        torch.npu.synchronize(self.prepared_rows.device)
        self._shared_pool_client.fetch_into(
            ids,
            self.prepared_rows[: ids.shape[0]],
            vocab_start=self.shard_indices.org_vocab_start_index,
            vocab_end=self.shard_indices.org_vocab_end_index,
        )

    def clear_prepared_rows(self) -> None:
        torch.npu.synchronize(self.prepared_rows.device)
        self.prepared_rows.zero_()

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        raise NotImplementedError("Remote PLE asynchronous prefetch is not supported")

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise NotImplementedError("Remote PLE asynchronous prefetch is not supported")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        if not self._preparation_bound or self._closing:
            raise RuntimeError("Remote PLE requires model-state preparation")
        if (
            ngram_ids.ndim != 2
            or ngram_ids.shape[0] > self.prepared_rows.shape[0]
            or ngram_ids.shape[1] != self.prepared_rows.shape[1]
        ):
            raise ValueError("Remote PLE graph input exceeds its prepared buffer")
        embeddings = self.prepared_rows[: ngram_ids.shape[0]]
        if self.reduce_results and self.tp_size > 1:
            assert self.parallel_group is not None
            embeddings = self.parallel_group.all_reduce(embeddings)
        return embeddings

    def close_host_offload(self) -> None:
        if self._shared_pool_client is None:
            return
        self._closing = True
        self._shared_pool_client.close()
        self._shared_pool_client = None
