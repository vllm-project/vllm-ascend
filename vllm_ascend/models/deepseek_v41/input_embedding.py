# SPDX-License-Identifier: Apache-2.0
"""Rank-local Host VMM for DeepSeek V4.1 input embeddings."""

import tempfile
from pathlib import Path

import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding

from vllm_ascend.models.deepseek_v41.engram.local_shared_vmm import (
    CHUNK_ROWS,
    MappedHostBuffer,
    load_vmm_library,
    mapped_table_view,
    retry_pending_rollbacks,
)
from vllm_ascend.models.host_offload import register_host_offload_owner
from vllm_ascend.ops.triton.qwen4_exp import ple_lookup

_ROWS_PER_COPY = 8192


@torch.library.custom_op("vllm::host_vmm_input_embedding", mutates_args=())
def _host_vmm_lookup(ids: torch.Tensor, ptrs: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    output = ids.new_empty((*ids.shape, width), dtype=torch.bfloat16)
    ple_lookup.lookup(ptrs, ids, output, 0, rows, CHUNK_ROWS)
    return output


@_host_vmm_lookup.register_fake
def _host_vmm_lookup_fake(ids: torch.Tensor, ptrs: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    return ids.new_empty((*ids.shape, width), dtype=torch.bfloat16)


class HostVmmEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Read local BF16 vocabulary rows from a mapping owned by the runner."""

    def __init__(self, weight: nn.Parameter):
        self.rows, self.width = weight.shape
        self.mapping: MappedHostBuffer | None = None
        self.directory: tempfile.TemporaryDirectory[str] | None = None
        self.ptrs: torch.Tensor | None = None
        self.lib = load_vmm_library()
        register_host_offload_owner(self)
        self.directory = tempfile.TemporaryDirectory(prefix="input-embedding-vmm-", dir="/dev/shm")
        self.mapping = MappedHostBuffer(
            Path(self.directory.name) / "table",
            self.rows,
            self.width,
            weight.dtype,
            owner=True,
            device=weight.device,
            lib=self.lib,
        )
        for start in range(0, self.rows, _ROWS_PER_COPY):
            self.mapping.publish(start, weight[start : start + _ROWS_PER_COPY].detach().cpu())
        self.metadata, self.ptrs = mapped_table_view(self.mapping, weight.shape, weight.dtype, weight.device)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if self.ptrs is None:
            raise RuntimeError("Host VMM input embedding has been closed")
        return _host_vmm_lookup(input_, self.ptrs, self.rows, self.width)

    def close_host_offload(self) -> None:
        if self.directory is None:
            return
        if self.mapping is None:
            retry_pending_rollbacks(lib=self.lib)
        else:
            self.mapping.close()
        self.directory.cleanup()
        self.directory = None
        self.mapping = None
        self.ptrs = None


def _language_model(model: nn.Module) -> nn.Module:
    get_language_model = getattr(model, "get_language_model", None)
    return get_language_model() if get_language_model is not None else model


def _weight_storage(weight: nn.Parameter) -> tuple[torch.device, int]:
    # Distinct Parameters can share an allocation.
    return weight.device, weight.untyped_storage().data_ptr()


def _input_layers(target: nn.Module, draft: nn.Module | None) -> list[VocabParallelEmbedding]:
    models = [_language_model(target)]
    if draft is not None:
        models.append(_language_model(draft))
    head_storage = {
        _weight_storage(head.weight)
        for model in models
        if (head := getattr(model, "lm_head", None)) is not None and getattr(head, "weight", None) is not None
    }
    layers = []
    seen = set()
    for layer in dict.fromkeys(model.model.embed_tokens for model in models):
        weight = getattr(layer, "weight", None)
        if (
            not isinstance(layer, VocabParallelEmbedding)
            or type(layer.quant_method) is not UnquantizedEmbeddingMethod
            or weight is None
            or weight.device.type != "npu"
            or weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or not weight.numel()
            or not weight.is_contiguous()
        ):
            raise ValueError("Host VMM requires contiguous unquantized BF16 input embeddings on the NPU")
        storage = _weight_storage(weight)
        if storage in head_storage:
            continue
        if storage in seen:
            raise ValueError("Host VMM cannot migrate distinct input layers sharing one weight allocation")
        layers.append(layer)
        seen.add(storage)
    if not layers:
        raise ValueError("Host VMM requires at least one input embedding independent of the LM head")
    return layers


@torch.inference_mode()
def offload_input_embeddings(target: nn.Module, draft: nn.Module | None) -> list[HostVmmEmbeddingMethod]:
    """Prepare every mapping before replacing any input weight."""
    layers = _input_layers(target, draft)
    prepared = []
    for layer in layers:
        weight = layer.weight
        method = HostVmmEmbeddingMethod(weight)
        rows = weight.shape[0]
        ids = torch.tensor([0, rows // 2, rows - 1], dtype=torch.long, device=weight.device)
        expected = layer.quant_method.embedding(layer, ids).contiguous().view(torch.int16)
        actual = method.embedding(layer, ids).contiguous().view(torch.int16)
        if not torch.equal(expected, actual):
            raise RuntimeError("Host VMM input embedding differs from the loaded weight")
        prepared.append((layer, method, nn.Parameter(method.metadata, requires_grad=False)))
    for layer, method, weight in prepared:
        layer.weight = weight
        layer.quant_method = method
    return [method for _, method, _ in prepared]
