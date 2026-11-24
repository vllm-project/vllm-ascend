from __future__ import annotations

from collections.abc import Callable

import torch

from vllm_ascend.attention.kimi_c8_dcp import KimiC8DCPQueryPlan as KimiC8GraphQueryPlan
from vllm_ascend.attention.kimi_c8_dcp import local_cache_seqlens
from vllm_ascend.attention.kimi_c8_metadata import build_quant_mla_metadata_kwargs


def build_c8_graph_query_plan(
    query_start_loc: torch.Tensor,
    global_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    interleave_size: int,
) -> KimiC8GraphQueryPlan:
    """Use persistent request-level inputs during capture and every replay."""
    lengths = global_seq_lens.to(torch.int32)
    used = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32)
    return KimiC8GraphQueryPlan(
        cache_seqlens=local_cache_seqlens(lengths, world_size, rank, interleave_size),
        cu_seqlens_q=query_start_loc,
        seqused_q=torch.where(lengths > 0, used, 0),
        block_table=block_table,
        global_seqlens=lengths,
        cp_world=world_size,
        cp_rank=rank,
        interleave_size=interleave_size,
    )


def prepare_c8_graph_metadata(plan: KimiC8GraphQueryPlan, metadata_op: Callable, *, num_heads_q: int) -> torch.Tensor:
    """Record metadata generation in the graph; share its result across layers.

    The caller caches this plan/result only within the current model step and
    geometry. No value-based Python cache may skip the recorded device op.
    """
    return metadata_op(
        **build_quant_mla_metadata_kwargs(
            plan.cache_seqlens,
            plan.cu_seqlens_q,
            plan.seqused_q,
            num_heads_q=num_heads_q,
            max_seqlen_q=-1,
            max_seqlen_kv=plan.max_seqlen_kv,
            global_seqlens=plan.global_seqlens,
            enable_dcp=plan.cp_world > 1,
            cp_world=plan.cp_world,
            cp_rank=plan.cp_rank,
            cp_kv_cache_interleave_size=plan.interleave_size,
        )
    )


class C8MetadataEvents:
    """Load-time stream/events reused by eager steps and ACL Graph capture.

    The first MLA layer starts the producer; the step's shared metadata cache
    keeps both its output and this producer alive for every consuming layer.
    The CPU path is synchronous and implements the identical start/wait API.
    """

    @classmethod
    def for_model(cls, vllm_config, device: torch.device | str, *, npu_runtime=None):
        """Share the model's producer stream while retaining private events."""
        producer = cls(device, npu_runtime=npu_runtime, stream=getattr(vllm_config, "_kimi_c8_metadata_stream", None))
        vllm_config._kimi_c8_metadata_stream = producer.stream
        return producer

    def __init__(self, device: torch.device | str, *, npu_runtime=None, stream=None):
        self.device = torch.device(device)
        self._metadata: torch.Tensor | None = None
        self._waited_streams: set[int] = set()
        self._runtime = npu_runtime
        if self._runtime is None and self.device.type in ("npu", "privateuseone"):
            self._runtime = torch.npu
        if self._runtime is None:
            self.stream = self.input_ready = self.output_ready = None
        else:
            self.stream = stream if stream is not None else self._runtime.Stream(device=self.device)
            self.input_ready = self._runtime.Event()
            self.output_ready = self._runtime.Event()

    def start(self, plan: KimiC8GraphQueryPlan, metadata_op: Callable, *, num_heads_q: int) -> torch.Tensor:
        self._metadata = None
        self._waited_streams.clear()
        if self._runtime is None:
            self._metadata = prepare_c8_graph_metadata(plan, metadata_op, num_heads_q=num_heads_q)
            return self._metadata
        current = self._runtime.current_stream(device=self.device)
        self.input_ready.record(current)
        self.stream.wait_event(self.input_ready)
        for tensor in (plan.cache_seqlens, plan.cu_seqlens_q, plan.seqused_q, plan.global_seqlens):
            tensor.record_stream(self.stream)
        with self._runtime.stream(self.stream):
            self._metadata = prepare_c8_graph_metadata(plan, metadata_op, num_heads_q=num_heads_q)
            self._metadata.record_stream(self.stream)
            self.output_ready.record(self.stream)
        return self._metadata

    def wait(self, expected_metadata: torch.Tensor | None = None) -> torch.Tensor:
        metadata = self._metadata if expected_metadata is None else expected_metadata
        if metadata is None:
            raise RuntimeError("C8 metadata producer has no successful start")
        if self._runtime is not None:
            current = self._runtime.current_stream(device=self.device)
            # A newer start may have advanced the event; waiting for it also
            # covers earlier work on the same stream, but must still return A.
            stream_id = current.npu_stream
            if stream_id not in self._waited_streams:
                current.wait_event(self.output_ready)
                self._waited_streams.add(stream_id)
            metadata.record_stream(current)
        return metadata
