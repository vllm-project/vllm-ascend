# SPDX-License-Identifier: Apache-2.0
"""BF16 PLE tables in CANN-registered host memory."""

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_engram_dp_group, get_etp_group, get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import logger
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS, HostUvaBuffer, SharedUvaBuffer


class AscendPLEHostEmbedding(Qwen4ExpPLEEmbedding):
    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        self._host_table: HostUvaBuffer | SharedUvaBuffer | None = None
        self._shared_group = None
        config = get_current_vllm_config().engram_config
        if config is not None and config.dp_shared_memory:
            group = get_engram_dp_group()
            if group is None or group.world_size <= 1:
                raise ValueError("PLE shared tables require a local DP group with more than one rank")
            if not all(in_the_same_node_as(group.cpu_group)):
                raise ValueError("PLE shared tables require one node and shared IPC namespace")
            if get_etp_group().world_size != get_tp_group().world_size:
                raise ValueError("PLE shared tables require ETP=TP")
            self._shared_group = group
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
        device = torch.device("npu", torch.npu.current_device())
        if self._shared_group is None:
            self._host_table = HostUvaBuffer((num_embeddings, embedding_dim), dtype, device)
            tensor = self._host_table.tensor
            tensor.zero_()
        else:
            self._host_table = SharedUvaBuffer((num_embeddings, embedding_dim), dtype, device, self._shared_group)
            tensor = self._host_table.tensor
            if self._shared_group.rank_in_group == 0:
                tensor.zero_()
            dist.barrier(group=self._shared_group.cpu_group)
        logger.info(
            "PLE host offload: %d rows x %d, %.3f GiB mapped per rank, BF16, shared=%s",
            num_embeddings,
            embedding_dim,
            tensor.numel() * tensor.element_size() / 1024**3,
            self._shared_group is not None,
        )
        return tensor

    def weight_loader(self, param, loaded_weight, checkpoint_start=None) -> None:
        group = self._shared_group
        if group is None:
            super().weight_loader(param, loaded_weight, checkpoint_start)
            return
        error = None
        if group.rank_in_group == 0:
            try:
                super().weight_loader(param, loaded_weight, checkpoint_start)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
        if any(error is not None for error in errors):
            raise RuntimeError(f"PLE shared weight loading failed: {errors}")

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
