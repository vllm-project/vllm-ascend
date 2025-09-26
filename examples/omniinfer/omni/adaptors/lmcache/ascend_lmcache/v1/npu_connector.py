# Standard
from typing import List, Tuple

# Third Party
import torch
import os

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.gpu_connector import GPUConnectorInterface
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class VLLMPagedMemNPUConnectorV2(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        block_size: int = 128,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_lora_rank_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.rope_dim_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.page_buffer_size = 0
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        self.use_nz = self.use_mla
        self.block_size = block_size

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]):
        if len(kv_caches) == 0:
            return None, None, 0, 0
        if isinstance(kv_caches[0], tuple):  # for NZ
            # kv tuple (lora: num block, block size, num head, head size) (rope: num block, block size, num head, head siz
            if not self.use_mla or len(kv_caches[0]) != 2 or kv_caches[0][0].dim() != 4:
                logger.error("not supported")
                return None, None, 0, 0
            self.kv_lora_rank_pointers.numpy()[:] = [t[0].data_ptr() for t in kv_caches]
            self.rope_dim_pointers.numpy()[:] = [t[1].data_ptr() for t in kv_caches]

            self.page_buffer_size = kv_caches[0][0].shape[0] * kv_caches[0][0].shape[1]

            lora_rank = kv_caches[0][0].shape[2] * kv_caches[0][0].shape[3]
            rope_dim = kv_caches[0][1].shape[2] * kv_caches[0][1].shape[3]
            return self.kv_lora_rank_pointers, self.rope_dim_pointers, lora_rank, rope_dim
        else:
            self.kv_lora_rank_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
            if self.use_mla:
                # kv  (num block, block size, num head, head size)
                self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[1]
            else:
                # kv  (2, num block, block size, num head, head size)
                self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]
            return self.kv_lora_rank_pointers, None, 0, 0

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        if memory_obj.tensor is None:
            raise ValueError("memory_obj.tensor should not None.")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if len(kvcaches) == 0:
            return

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnectorV3"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemNPUConnectorV3"
                )

        kv_lora_rank_pointers, rope_dim_pointers, lora_rank, rope_dim = self._initialize_pointers(kvcaches)

        if kv_lora_rank_pointers is None:
            return
        if rope_dim_pointers is None:
            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,
                kv_lora_rank_pointers,
                slot_mapping[start:end],
                kvcaches[0].device,
                self.page_buffer_size,
                False,
                self.use_mla,
            )
        else:
            lmc_ops.multi_layer_block_kv_transfer(
                memory_obj.tensor,
                kv_lora_rank_pointers,
                rope_dim_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                lora_rank,
                rope_dim,
                False
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        if memory_obj.tensor is None:
            raise ValueError("memory_obj.tensor should not None.")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_lora_rank_pointers, rope_dim_pointers, lora_rank, rope_dim = self._initialize_pointers(kvcaches)

        if kv_lora_rank_pointers is None:
            return
        if rope_dim_pointers is None:
            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,
                kv_lora_rank_pointers,
                slot_mapping[start:end],
                kvcaches[0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
        else:
            lmc_ops.multi_layer_block_kv_transfer(
                memory_obj.tensor,
                kv_lora_rank_pointers,
                rope_dim_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                lora_rank,
                rope_dim,
                True
            )
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if memory_obj is None:
                continue
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_block_size(self) -> int:
        return self.block_size

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla and self.use_nz:
            return torch.Size([self.num_layers, num_tokens, self.block_size, self.hidden_dim_size])
        else:
            kv_size = 1 if self.use_mla else 2
            return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


