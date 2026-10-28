# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import numpy as np
import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.worker.gpu import structured_outputs
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

GRAMMAR_BITMASK_BLOCK_SIZE = 8192


class AscendStructuredOutputsWorker(StructuredOutputsWorker):
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
        num_bonus_tokens: int = 1,
    ):
        kwargs = {}
        if "num_bonus_tokens" in inspect.signature(super().__init__).parameters:
            kwargs["num_bonus_tokens"] = num_bonus_tokens
        super().__init__(max_num_logits, vocab_size, device, mask_stride, **kwargs)
        self._max_num_logits = max_num_logits
        self._vocab_size = vocab_size
        self._host_logits_indices: torch.Tensor | None = None
        self._host_logits_indices_np = None
        self._host_grammar_bitmask: torch.Tensor | None = None
        self._copy_event = torch.cuda.Event()
        self._copies_pending = False
        self._mapping_key: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def _ensure_host_buffers(self) -> None:
        if self._host_grammar_bitmask is not None:
            return
        # Allocated lazily: workloads without grammar requests never pin
        # host huge pages (required on 310P unified-memory hosts).
        self._host_logits_indices = torch.empty(
            self._max_num_logits,
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        self._host_logits_indices_np = self._host_logits_indices.numpy()
        self._host_grammar_bitmask = torch.empty(
            (self._max_num_logits, cdiv(self._vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

    def _wait_for_host_buffers(self) -> None:
        if self._copies_pending:
            self._copy_event.synchronize()
            self._copies_pending = False

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        num_masks = grammar_bitmask.shape[0]
        mapping_key = None
        if self.mask_stride == 1:
            mapping_key = (tuple(input_batch.req_ids), tuple(grammar_req_ids))
        reuse_mapping = mapping_key is not None and mapping_key == self._mapping_key

        mapping = None
        if not reuse_mapping:
            req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
            if self.mask_stride == 1:
                mapping = [req_id_to_idx[req_id] for req_id in grammar_req_ids]
            elif hasattr(structured_outputs, "_build_grammar_mapping"):
                mapping = structured_outputs._build_grammar_mapping(
                    input_batch.req_ids,
                    grammar_req_ids,
                    input_batch.cu_num_logits_np,
                    input_batch.num_draft_tokens_per_req,
                    self.num_bonus_tokens,
                    self.mask_stride,
                )
            else:
                mapping = [
                    req_id_to_idx[req_id] * self.mask_stride + position
                    for req_id in grammar_req_ids
                    for position in range(self.mask_stride)
                ]
            assert mapping is not None
            assert num_masks == len(mapping)

        self._wait_for_host_buffers()
        self._ensure_host_buffers()
        assert self._host_grammar_bitmask is not None
        assert self._host_logits_indices is not None
        host_bitmask = self._host_grammar_bitmask[:num_masks]
        host_bitmask.copy_(torch.from_numpy(grammar_bitmask))

        with torch.cuda.stream(self.copy_stream):
            bitmask = self.grammar_bitmask[:num_masks].copy_(
                host_bitmask,
                non_blocking=True,
            )
            if reuse_mapping:
                logits_indices = self.logits_indices[:num_masks]
            else:
                assert mapping is not None
                assert self._host_logits_indices_np is not None
                self._host_logits_indices_np[:num_masks] = mapping
                logits_indices = self.logits_indices[:num_masks].copy_(
                    self._host_logits_indices[:num_masks],
                    non_blocking=True,
                )
            self._copy_event.record(self.copy_stream)
            self._copies_pending = True

        self._mapping_key = mapping_key
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        vocab_size = logits.shape[-1]
        block_size = GRAMMAR_BITMASK_BLOCK_SIZE
        grid = (num_masks, triton.cdiv(vocab_size, block_size))
        structured_outputs._apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            input_batch.cu_num_logits,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            MASK_STRIDE=self.mask_stride,
            BLOCK_SIZE=block_size,
        )

        self.copy_stream.wait_stream(current_stream)
