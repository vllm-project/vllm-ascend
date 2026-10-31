# SPDX-License-Identifier: Apache-2.0
"""BF16 PLE tables in CANN-registered host memory."""

import torch
from vllm.logger import logger
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS, HostUvaBuffer


class AscendPLEHostEmbedding(Qwen4ExpPLEEmbedding):
    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        self._host_table = None
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            self.close_host_offload()
            raise

    def close_host_offload(self) -> None:
        """Release after graph teardown; failed unregister remains retryable."""
        table = self._host_table
        if table is None:
            return
        table.close()
        # Parameters alias the allocation: never expose an unmapped tensor.
        if hasattr(self, "_parameters"):
            self.register_parameter("weight", None)
        self._host_table = None
        logger.info("PLE registered host table released")

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
        if dtype != torch.bfloat16:
            raise ValueError("Ascend PLE offload requires BF16 weights")
        self._host_table = HostUvaBuffer(
            (num_embeddings, embedding_dim), dtype, torch.device("npu", torch.npu.current_device())
        )
        self._host_table.tensor.zero_()
        logger.info(
            "PLE host offload: %d rows x %d, %.3f GiB per rank, BF16",
            num_embeddings,
            embedding_dim,
            self._host_table.tensor.numel() * self._host_table.tensor.element_size() / 1024**3,
        )
        return self._host_table.tensor

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        raise NotImplementedError("PLE asynchronous prefetch is not enabled")

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise NotImplementedError("PLE asynchronous prefetch is not enabled")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        from vllm_ascend.ops.triton.qwen4_exp.ple_lookup import lookup

        if self._host_table is None:
            raise RuntimeError("PLE host table has been closed")
        if self.etp_data_parallel_size != 1:
            raise ValueError("PLE host lookup requires ETP=TP")
        output = torch.empty((*ngram_ids.shape, self.embedding_dim), dtype=torch.bfloat16, device=ngram_ids.device)
        lookup(
            self._host_table.ptrs,
            ngram_ids,
            output,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            CHUNK_ROWS,
        )
        if self.reduce_results and self.tp_size > 1:
            output = self.parallel_group.all_reduce(output)
        return output
