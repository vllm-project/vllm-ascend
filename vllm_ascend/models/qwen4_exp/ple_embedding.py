# SPDX-License-Identifier: Apache-2.0
"""BF16, INT8 and MXFP8 PLE tables in CANN-registered host memory."""

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_engram_dp_group, get_etp_group, get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import logger
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEPinnedHostEmbedding

from vllm_ascend.models.deepseek_v41.engram.npu import (
    CHUNK_ROWS,
    HostUvaBuffer,
    HostUvaBufferCleanupError,
    SharedUvaBuffer,
)
from vllm_ascend.models.host_offload import register_host_offload_owner
from vllm_ascend.models.qwen4_exp.ple_quant import AscendPLEQuantizedEmbeddingMethod
from vllm_ascend.ops.triton.qwen4_exp import ple_lookup, ple_lookup_quantized


@Qwen4ExpPLEPinnedHostEmbedding.register_oot
class AscendPLEHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        self._host_table: HostUvaBuffer | SharedUvaBuffer | None = None
        self._host_scale: HostUvaBuffer | SharedUvaBuffer | None = None
        method = kwargs.get("embedding_method")
        self._quantized = isinstance(method, AscendPLEQuantizedEmbeddingMethod)
        self._mxfp8 = isinstance(method, AscendPLEQuantizedEmbeddingMethod) and method.mxfp8
        self._scale_dtype = method.scale_dtype if isinstance(method, AscendPLEQuantizedEmbeddingMethod) else None
        self._weight_dtype = (
            method.weight_dtype if isinstance(method, AscendPLEQuantizedEmbeddingMethod) else torch.bfloat16
        )
        if self._quantized:
            if get_etp_group().world_size != get_tp_group().world_size:
                raise ValueError("Quantized PLE host lookup requires ETP=TP")
            if kwargs.get("params_dtype") != torch.bfloat16:
                raise ValueError("Quantized PLE host lookup requires BF16 activations")
        self._shared_group = None
        register_host_offload_owner(self)
        config = get_current_vllm_config().engram_config
        if config is not None and config.dp_shared_memory:
            group = get_engram_dp_group()
            if group is not None and group.world_size > 1:
                if not all(in_the_same_node_as(group.cpu_group)):
                    logger.info(
                        "PLE dp_shared_memory requested but DP replicas span multiple nodes; "
                        "falling back to private host tables."
                    )
                else:
                    self._shared_group = group
            else:
                logger.info(
                    "PLE dp_shared_memory requested but no local DP peers; falling back to private host tables."
                )
        try:
            # Bypass the CUDA pinned-host initializer while retaining the public
            # embedding contract and checkpoint/ETP sharding implementation.
            Qwen4ExpPLEEmbedding.__init__(self, *args, **kwargs)
        except HostUvaBufferCleanupError as exc:
            # The buffer constructor could not finish assigning _host_table.
            # Preserve its owner without hiding the original cleanup failure.
            if exc.host_buffer is not self._host_scale:
                self._host_table = exc.host_buffer
            raise
        except Exception as initialization_error:
            try:
                self.close_host_offload()
            except Exception as cleanup_error:
                raise HostUvaBufferCleanupError(
                    self._host_table, initialization_error, cleanup_error
                ) from cleanup_error
            raise

    def close_host_offload(self) -> None:
        """Release after graph teardown; failed unregister remains retryable."""
        if self._host_scale is not None:
            # Drop aliases before closing shared storage, retaining failed owners.
            if hasattr(self, "_parameters"):
                self.register_parameter("weight_scale", None)
            self._host_scale.close()
            self._host_scale = None
        table = self._host_table
        if table is None:
            return
        # Drop the parameter alias before closing shared-memory storage.
        # Retain the owner on failure so unregister/free can be retried.
        if hasattr(self, "_parameters"):
            self.register_parameter("weight", None)
        table.close()
        self._host_table = None
        logger.info("PLE registered host table released")

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
        allowed = (self._weight_dtype, self._scale_dtype) if self._quantized else (torch.bfloat16,)
        if dtype not in allowed:
            raise ValueError(f"Unsupported PLE host storage dtype: {dtype}")
        device = torch.device("npu", torch.npu.current_device())
        buffer_name = "_host_scale" if dtype == self._scale_dtype else "_host_table"
        table: HostUvaBuffer | SharedUvaBuffer
        try:
            if self._shared_group is None:
                table = HostUvaBuffer((num_embeddings, embedding_dim), dtype, device, buffer_name, self)
            else:
                table = SharedUvaBuffer(
                    (num_embeddings, embedding_dim), dtype, device, self._shared_group, buffer_name, self
                )
        except HostUvaBufferCleanupError as exc:
            if dtype == self._scale_dtype:
                self._host_scale = exc.host_buffer
            else:
                self._host_table = exc.host_buffer
            raise
        if dtype == self._scale_dtype:
            self._host_scale = table
        else:
            self._host_table = table
        tensor = table.tensor
        assert tensor is not None
        if self._shared_group is None or self._shared_group.rank_in_group == 0:
            tensor.view(torch.uint8).zero_()
        if self._shared_group is not None:
            dist.barrier(group=self._shared_group.cpu_group)
        logger.info(
            "PLE host storage: %d rows x %d, dtype=%s, %.3f GiB per rank",
            num_embeddings,
            embedding_dim,
            dtype,
            tensor.numel() * tensor.element_size() / 1024**3,
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
        return self.sync_lookup(ngram_ids)

    def _lookup(self, input_ids: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        if self._host_table is None or self.weight is None:
            raise RuntimeError("PLE host table has been closed")
        if self._quantized and (self._host_scale is None or self.weight_scale is None):
            raise RuntimeError("Quantized PLE host storage has been closed")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(expected_shape, dtype=torch.bfloat16, device=input_ids.device)
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != torch.bfloat16
            or output.device != input_ids.device
            or (self._quantized and not output.is_contiguous())
        ):
            raise ValueError("PLE prefetch output must match the input shape, lookup dtype, and input device")
        if self._quantized:
            assert self._host_scale is not None
            ple_lookup_quantized.lookup_quantized(
                self._host_table.ptrs,
                self._host_scale.ptrs,
                input_ids,
                output,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                CHUNK_ROWS,
                is_mxfp8=self._mxfp8,
            )
        else:
            ple_lookup.lookup(
                self._host_table.ptrs,
                input_ids,
                output,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                CHUNK_ROWS,
            )
        return output
