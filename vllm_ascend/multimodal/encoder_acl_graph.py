#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""NPU-specific encoder ACL graph manager (budget NPUGraph + FIA task updates)."""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager

from vllm_ascend.multimodal.encoder_forward_context import (
    encoder_graph_capture_scope,
    encoder_graph_replay_scope,
)
from vllm_ascend.multimodal.encoder_graph_params import set_encoder_graph_params
from vllm_ascend.multimodal.encoder_graph_update import update_encoder_full_graph_params
from vllm_ascend.utils import weak_ref_tensors

logger = init_logger(__name__)


def _cu_prefix_to_host_endpoints(cu: torch.Tensor | None) -> list[int] | None:
    if cu is None:
        return None
    flat = cu.detach().cpu().view(-1).tolist()
    return flat[1:] if flat else flat


def _per_seq_lengths_to_fia_endpoints(per_seq: list[int]) -> list[int]:
    acc = 0
    ends: list[int] = []
    for raw in per_seq:
        L = int(raw)
        if L == 0:
            continue
        acc += L
        ends.append(acc)
    return ends


class EncoderAclGraphManager(EncoderCudaGraphManager):
    """Hooks encoder capture/replay into Ascend FIA graph-task infrastructure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update_stream: torch.npu.Stream | None = None

    def capture(self):
        set_encoder_graph_params(self.token_budgets)
        super().capture()
        weak_ref_encoder_graph_workspaces()

    def _capture_budget_graph(self, token_budget: int):
        logger.debug(
            "Capturing encoder aclgraph for budget=%d, max_batch_size=%d, "
            "max_frames_per_batch=%d",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            self.device,
            self.dtype,
        )

        mm_kwargs = capture_inputs.mm_kwargs
        buffers = capture_inputs.buffers

        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
            output_buffer = torch.empty_like(output)

        graph = torch.cuda.CUDAGraph()
        with encoder_graph_capture_scope(token_budget):
            with torch.inference_mode(), torch.cuda.graph(graph):
                output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
                output_buffer.copy_(output)

        input_key = self.config.input_key_by_modality["image"]
        self.budget_graphs[token_budget] = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffer=mm_kwargs[input_key],
            metadata_buffers=buffers,
            output_buffer=output_buffer,
        )

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        replay_buffers: dict[str, torch.Tensor | None],
    ) -> torch.Tensor | None:
        num_items = self.model.get_encoder_cudagraph_num_items(mm_kwargs)
        if token_budget not in self.budget_graphs:
            self.graph_misses += num_items
            return None

        graph_meta = self.budget_graphs[token_budget]

        input_key = self.config.input_key_by_modality[self.model.get_input_modality(mm_kwargs)]
        src = mm_kwargs[input_key]
        n = src.shape[0]
        graph_meta.input_buffer[:n].copy_(src)

        for key in self.config.buffer_keys:
            src_buf = replay_buffers.get(key)
            if src_buf is None:
                continue
            buf = graph_meta.metadata_buffers[key]
            if src_buf.ndim == 0:
                buf.copy_(src_buf)
            else:
                slice_n = src_buf.shape[0]
                buf.zero_()
                buf[:slice_n].copy_(src_buf)

        meta = graph_meta.metadata_buffers
        host_full = _cu_prefix_to_host_endpoints(meta.get("cu_seqlens"))
        host_win = _cu_prefix_to_host_endpoints(meta.get("cu_window_seqlens"))
        seq_lens_tensor = meta.get("sequence_lengths")
        host_seq_lens = None
        if isinstance(seq_lens_tensor, torch.Tensor):
            host_seq_lens = _per_seq_lengths_to_fia_endpoints(seq_lens_tensor.detach().cpu().view(-1).tolist())

        update_stream = self.update_stream
        if update_stream is None:
            update_stream = torch.npu.Stream()

        visual = getattr(self.model, "visual", None)
        fa_raw = getattr(visual, "fullatt_block_indexes", None) if visual is not None else None
        fullatt = frozenset(fa_raw) if fa_raw is not None else None

        with encoder_graph_replay_scope(
            token_budget,
            host_cu_seqlens_ends=host_full,
            host_cu_window_seqlens_ends=host_win,
            host_sequence_lengths=host_seq_lens,
        ):
            update_encoder_full_graph_params(
                update_stream, token_budget, fullatt_block_indexes=fullatt
            )

        torch.npu.current_stream().wait_stream(update_stream)
        graph_meta.graph.replay()

        self.graph_hits += num_items
        return graph_meta.output_buffer


def weak_ref_encoder_graph_workspaces():
    from vllm_ascend.multimodal.encoder_graph_params import get_encoder_graph_params

    params = get_encoder_graph_params()
    if params is None:
        return
    for budget, ws in list(params.workspaces.items()):
        if ws is None:
            continue
        params.workspaces[budget] = weak_ref_tensors(ws)
