# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Local experiment for rebinding the original FIA graph kernel's KV lengths."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.compilation.updatable_graph import GraphUpdateTask

try:
    from vllm_ascend import _experimental_fia_replay as _native
except ImportError:
    _native = None

CAPTURE_FIA_REPLAY: ContextVar[bool] = ContextVar("capture_fia_replay", default=False)


def available() -> bool:
    return _native is not None


def captured_model(stream: torch.npu.Stream) -> int:
    if _native is None or not CAPTURE_FIA_REPLAY.get():
        return 0
    return _native.capture_model(stream.npu_stream)


@dataclass(slots=True)
class FIAReplay:
    layer_name: str
    block_table: torch.Tensor
    lengths: torch.Tensor
    launch_args: torch.Tensor

    @classmethod
    def bind(cls, model: int, tasks: list["GraphUpdateTask"]) -> "FIAReplay | None":
        if _native is None or not tasks:
            return None
        first = tasks[0].kwargs
        table = first.get("block_table")
        lengths = first.get("actual_seq_lengths_kv")
        if table is None or not lengths or table.shape[0] != len(lengths):
            return None

        query_outputs = []
        for task in tasks:
            kwargs = task.kwargs
            if (
                task.operation != torch_npu.npu_fused_infer_attention_score.out
                or kwargs["input_layout"] != "TND"
                or kwargs["sparse_mode"] != 0
                or kwargs["query"].dtype != torch.bfloat16
                or kwargs["query"].shape[0] != len(lengths)
                or kwargs["actual_seq_lengths_kv"] != lengths
                or kwargs["block_table"] is None
                or kwargs["block_table"].data_ptr() != table.data_ptr()
            ):
                return None
            query_outputs.append((kwargs["query"].data_ptr(), kwargs["out"][0].data_ptr()))

        shared_lengths = torch.empty(len(lengths), dtype=torch.int64, device=table.device)
        launch_args = torch.empty((len(tasks), _native.ARGS_CAPACITY), dtype=torch.uint8, device=table.device)
        if not _native.bind_lengths(
            model, query_outputs, table.data_ptr(), lengths, shared_lengths.data_ptr(), launch_args.data_ptr()
        ):
            logger.debug("FIA capture did not match the experimental ACL Graph ABI.")
            return None
        logger.info("Bound %d captured FIA kernels to one device KV length buffer.", len(tasks))
        return cls(tasks[0].provider.layer_name, table, shared_lengths, launch_args)

    def prepare(self, attn_metadata) -> None:
        metadata = attn_metadata[self.layer_name]
        seq_lens = metadata.seq_lens_gpu
        if (
            metadata.max_query_len != 1
            or metadata.block_tables.data_ptr() != self.block_table.data_ptr()
            or seq_lens.shape != self.lengths.shape
        ):
            raise RuntimeError("Shared FIA graph replay requires the captured single-token batch shape.")
        self.lengths.copy_(seq_lens, non_blocking=True)
