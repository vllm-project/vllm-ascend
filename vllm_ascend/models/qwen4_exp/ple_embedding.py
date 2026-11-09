# SPDX-License-Identifier: Apache-2.0
"""BF16 PLE tables backed by independent MemFabric GVA pools."""

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_engram_dp_group, get_etp_group, get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.distributed.ple_gva import PLEGvaTable


def ple_gva_lookup(ids: torch.Tensor, layer_name: str, width: int) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    if layer._host_table is None:
        raise RuntimeError("PLE GVA table has been closed")
    return layer._host_table.lookup(
        ids, layer.shard_indices.org_vocab_start_index, layer.shard_indices.org_vocab_end_index
    )


def ple_gva_lookup_fake(ids: torch.Tensor, layer_name: str, width: int) -> torch.Tensor:
    return torch.empty((*ids.shape, width), dtype=torch.bfloat16, device=ids.device)


direct_register_custom_op(
    op_name="ple_gva_lookup",
    op_func=ple_gva_lookup,
    fake_impl=ple_gva_lookup_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


class AscendPLEHostEmbedding(Qwen4ExpPLEEmbedding):
    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        self._host_table: PLEGvaTable | None = None
        self._shared_group = None
        self._prefix = kwargs.get("prefix", "")
        config = get_current_vllm_config().engram_config
        if config is not None and config.dp_shared_memory:
            group = get_engram_dp_group()
            if group is not None and group.world_size > 1:
                if not all(in_the_same_node_as(group.cpu_group)):
                    logger.info(
                        "PLE dp_shared_memory requested but DP replicas span multiple nodes; "
                        "falling back to private host tables."
                    )
                elif get_etp_group().world_size != get_tp_group().world_size:
                    raise ValueError("PLE shared tables require ETP=TP")
                else:
                    self._shared_group = group
            else:
                logger.info(
                    "PLE dp_shared_memory requested but no local DP peers; falling back to private host tables."
                )
        try:
            super().__init__(*args, **kwargs)
            context = get_current_vllm_config().compilation_config.static_forward_context
            self._context_key = f"{self._prefix}.ascend_ple_gva"
            if self._context_key in context:
                raise ValueError(f"Duplicate PLE GVA layer: {self._context_key}")
            context[self._context_key] = self
        except Exception:
            self.close_host_offload()
            raise

    def close_host_offload(self) -> None:
        """Release this instance after graph teardown; failed close is retryable."""
        table = self._host_table
        if table is None:
            return
        # Detach the parameter before unmapping its CPU storage.
        if hasattr(self, "_parameters"):
            self.register_parameter("weight", None)
        table.close()
        self._host_table = None
        logger.info("PLE GVA table released")

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
        if dtype != torch.bfloat16:
            raise ValueError("Ascend PLE offload requires BF16 weights")
        device = torch.device("npu", torch.npu.current_device())
        gva_config = (get_current_vllm_config().additional_config or {}).get("ple_gva", {})
        store_urls = gva_config.get("store_urls", {})
        if not isinstance(store_urls, dict) or any(not isinstance(url, str) for url in store_urls.values()):
            raise ValueError("ple_gva.store_urls must map table/shard names to tcp:// endpoints")
        if len(set(store_urls.values())) != len(store_urls):
            raise ValueError("PLE GVA shared pools require distinct store URLs")
        binding = f"{self._prefix}/etp{get_etp_group().rank_in_group}"
        store_url = store_urls.get(binding)
        if self._shared_group is not None and store_url is None:
            raise ValueError(f"Missing dedicated PLE GVA endpoint: ple_gva.store_urls[{binding!r}]")
        self._host_table = PLEGvaTable(
            (num_embeddings, embedding_dim), device, group=self._shared_group, store_url=store_url
        )
        tensor = self._host_table.tensor
        assert tensor is not None
        logger.info(
            "PLE GVA offload: %d rows x %d, %.3f GiB per table, BF16, shared=%s",
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
        if self._host_table is None:
            raise RuntimeError("PLE host table has been closed")
        if self.etp_data_parallel_size != 1:
            raise ValueError("PLE host lookup requires ETP=TP")
        # Keep native addresses and the MemFabric Python API outside Dynamo.
        # The op records its device-only metadata and copy onto the current
        # stream during ACLGraph capture.
        output = torch.ops.vllm.ple_gva_lookup(ngram_ids, self._context_key, self.embedding_dim)
        if self.reduce_results and self.tp_size > 1:
            output = self.parallel_group.all_reduce(output)
        return output
