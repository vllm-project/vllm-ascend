# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned scheduling of the accepted GDN transition, after sampling.

Cache pages stay in their original pools. Only request metadata is staged for
graph replay; different groups keep their own page indices and destinations.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.logger import logger
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.worker.gpu.model_states.recoverssm import (
    RecoverSSMState,
    _postprocess_recoverssm_align_kernel,
)

from vllm_ascend.attention.gdn_recoverssm_metadata import (
    AscendGDNRecoverSSMMetadata,
    AscendGDNRecoverSSMMetadataBuilder,
)
from vllm_ascend.attention.recoverssm_gdn import (
    GDNRecoverSSMCommitContext,
    _create_conv_cache_address_stride_table,
    _create_state_commit_cache_address_stride_table,
    _get_shared_conv_dim_stride,
    _inner_stride_signature,
    _validate_layer_caches,
)
from vllm_ascend.ops.triton.mamba.stage_gdn_commit import stage_gdn_commit


@dataclass(frozen=True)
class GDNCommitJob:
    context: GDNRecoverSSMCommitContext
    indices: torch.Tensor
    offsets: torch.Tensor
    mapping: torch.Tensor | None
    table: torch.Tensor | None
    computed: torch.Tensor | None
    block_size: int | None

    def run(self, accepted: torch.Tensor) -> None:
        self.context.commit(
            accepted, self.indices, self.offsets, self.mapping, self.table, self.computed, self.block_size
        )

    def tensors(self):
        return tuple(t for t in (self.indices, self.offsets, self.mapping, self.table, self.computed) if t is not None)


def _tensor_identity(tensor):
    if tensor is None:
        return None
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device


def _same_inputs(left: GDNCommitJob, right: GDNCommitJob) -> bool:
    # Equal geometry is insufficient: distinct cache groups can have different
    # physical page IDs. Never compare device values through a CPU sync.
    return (
        left.block_size == right.block_size
        and left.context.commit_plan.capacity == right.context.commit_plan.capacity
        and all(
            _tensor_identity(a) == _tensor_identity(b)
            for a, b in zip(
                (left.indices, left.offsets, left.mapping, left.table, left.computed),
                (right.indices, right.offsets, right.mapping, right.table, right.computed),
            )
        )
    )


def commit_bucket(batch: int, capacity: int) -> int:
    if not 0 < batch <= capacity:
        raise ValueError("GDN commit batch must fit the configured sequence capacity")
    return min(1 << (batch - 1).bit_length(), capacity)


def commit_buckets(capacity: int) -> tuple[int, ...]:
    """Prepare every dispatch bucket, largest first, before serving requests."""
    return tuple(sorted({commit_bucket(n, capacity) for n in range(1, capacity + 1)}, reverse=True))


def _context_layout(context):
    cache = context.layer_caches[0]
    return (
        context.spec_query_len,
        context.conv_history_len,
        cache.num_key_heads,
        tuple((tensor.shape, tensor.dtype, tensor.device) for tensor in cache.all_pages()),
        _inner_stride_signature(cache.checkpoint),
        _inner_stride_signature(cache.replay),
    )


def startup_gdn_commit_jobs(runner) -> tuple[GDNCommitJob, ...]:
    """Bind the actual target groups, keeping each physical page table separate."""
    jobs = []
    capacity = runner.max_num_reqs
    align = runner.cache_config.mamba_cache_mode == "align"
    seen = set()
    for group_index, groups in enumerate(runner.attn_groups):
        for group in groups:
            builder = group.get_metadata_builder(0)
            if not isinstance(builder, AscendGDNRecoverSSMMetadataBuilder) or id(builder) in seen:
                continue
            seen.add(id(builder))
            reference = getattr(builder, "_checkpoint_indices", builder.spec_state_indices_tensor[:, 0])
            indices = torch.empty_strided(
                (capacity,), reference.stride(), dtype=reference.dtype, device=runner.device
            ).zero_()
            table = runner.block_tables.block_tables[group_index].gpu
            jobs.append(
                GDNCommitJob(
                    builder._get_context(),
                    indices,
                    torch.zeros(capacity + 1, dtype=torch.int32, device=runner.device),
                    None,
                    torch.zeros_like(table[:capacity]) if align else None,
                    torch.zeros(capacity, dtype=torch.int32, device=runner.device) if align else None,
                    builder.kv_cache_spec.block_size if align else None,
                )
            )
    return tuple(jobs)


class GDNCommitGraph:
    """One replay launch for all groups, with graph-stable metadata buffers."""

    def __init__(self, jobs: tuple[GDNCommitJob, ...], accepted: torch.Tensor, bucket: int):
        self.capacity = min(job.context.commit_plan.capacity for job in jobs)
        self.accepted = torch.zeros(self.capacity, dtype=accepted.dtype, device=accepted.device)
        staged = []
        for job in jobs:

            def zeros(shape, reference):
                return torch.zeros(shape, dtype=reference.dtype, device=reference.device)

            staged.append(
                replace(
                    job,
                    indices=zeros((bucket,), job.indices),
                    offsets=zeros((bucket + 1,), job.offsets),
                    mapping=zeros((bucket,), job.indices),
                    table=(zeros((self.capacity, job.table.shape[1]), job.table) if job.table is not None else None),
                    computed=(zeros((self.capacity,), job.computed) if job.computed is not None else None),
                )
            )
        self.jobs = tuple(staged)
        self.graph: Any = None

    def run(self) -> None:
        for job in self.jobs:
            job.run(self.accepted)

    def capture(self, stream) -> None:
        # Warm/JIT and capture use NULL pages and zero query lengths. Running
        # either against real accepted records would apply the transition twice.
        self.run()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.run()

    def update(self, jobs: tuple[GDNCommitJob, ...], accepted: torch.Tensor) -> None:
        if accepted.ndim != 1 or accepted.numel() > self.capacity:
            raise ValueError("GDN sampled counts exceed graph capacity")
        for group_index, (source, target) in enumerate(zip(jobs, self.jobs)):
            batch = source.indices.numel()
            if source.indices.ndim != 1 or source.offsets.shape != (batch + 1,):
                raise ValueError("GDN group indices and query offsets have incompatible shapes")
            if source.mapping is None and accepted.numel() < batch:
                raise ValueError("GDN sampled counts do not cover the group")
            if source.mapping is not None and (source.mapping.ndim != 1 or source.mapping.numel() < batch):
                raise ValueError("GDN sampler mapping does not cover the group")
            if source.indices.numel() > target.indices.numel():
                raise ValueError("GDN group exceeds graph bucket")
            if source.table is not None and (
                source.computed is None
                or source.table.shape[0] < accepted.numel()
                or source.computed.numel() < accepted.numel()
            ):
                raise ValueError("GDN align metadata does not cover the sampler rows")
            stage_gdn_commit(source, target, accepted, self.accepted, copy_accepted=group_index == 0)

    def replay(self) -> None:
        self.graph.replay()


class GDNCommitExecutor:
    """One worker-owned side stream; no process-global Tensor or graph cache."""

    def __init__(self, config, capacity: int):
        self.config = config
        self.capacity = capacity
        self.stream: Any = None
        self.ready: Any = None
        self.done: Any = None
        self.pending = False
        self.overlap_safe = False
        self.graphs: dict[tuple, GDNCommitGraph] = {}
        self.merged: dict[tuple[int, ...], GDNRecoverSSMCommitContext] = {}

    def configure_draft(self, model: Any) -> None:
        if not self.config.async_commit:
            return
        if model is None or any(
            isinstance(layer, MambaBase) or bool(getattr(layer, "use_gdn_recoverssm", False))
            for layer in model.modules()
        ):
            raise ValueError("Async GDN recovery requires a draft model without recurrent GDN state")
        self.overlap_safe = True

    def _merge_jobs(self, jobs: tuple[GDNCommitJob, ...]) -> tuple[GDNCommitJob, ...]:
        if not self.config.merge_groups:
            return jobs
        groups: list[list[GDNCommitJob]] = []
        for job in jobs:
            for group in groups:
                if (
                    _same_inputs(group[0], job)
                    and getattr(group[0].context, "_backend_ops", None) is getattr(job.context, "_backend_ops", None)
                    and _context_layout(group[0].context) == _context_layout(job.context)
                ):
                    group.append(job)
                    break
            else:
                groups.append([job])
        result = []
        for group in groups:
            first = group[0]
            if len(group) == 1:
                result.append(first)
                continue
            key = tuple(id(job.context) for job in group)
            if key not in self.merged:
                caches = tuple(cache for job in group for cache in job.context.layer_caches)
                # Full layout validation runs once, never per layer/iteration.
                _validate_layer_caches(caches, first.context.spec_query_len)
                if len({cache.checkpoint_base_addr for cache in caches}) != len(caches):
                    raise ValueError("GDN commit groups contain duplicate checkpoint pages")
                self.merged[key] = replace(
                    first.context,
                    layer_caches=caches,
                    conv_shared_dim_stride=_get_shared_conv_dim_stride(caches),
                    _conv_cache_address_stride_table=_create_conv_cache_address_stride_table(caches),
                    _state_commit_cache_address_stride_table=_create_state_commit_cache_address_stride_table(caches),
                )
            result.append(replace(first, context=self.merged[key]))
        return tuple(result)

    def _runtime(self) -> None:
        if self.stream is None:
            self.stream = torch.npu.Stream()
            self.ready = torch.npu.Event()
            self.done = torch.npu.Event()

    def _run(self, jobs: tuple[GDNCommitJob, ...], accepted: torch.Tensor) -> None:
        if not self.config.capture_commit:
            for job in jobs:
                job.run(accepted)
            return
        graph = self.graphs.get(self._graph_key(jobs, accepted))
        if graph is None:
            # A different context/subset or metadata layout must never start a
            # device-wide synchronize/capture in a live request. Keep the same
            # provider and stream, and let startup diagnostics expose the miss.
            logger.warning_once("GDN recovery graph was not prepared at startup; using eager recovery")
            for job in jobs:
                job.run(accepted)
            return
        graph.update(jobs, accepted)
        graph.replay()

    def _graph_key(self, jobs: tuple[GDNCommitJob, ...], accepted: torch.Tensor, bucket: int | None = None):
        capacity = min(self.capacity, *(job.context.commit_plan.capacity for job in jobs))
        if bucket is None:
            bucket = commit_bucket(max(job.indices.numel() for job in jobs), capacity)
        # Contexts own immutable cache pointers. Values and source buffer
        # addresses vary per step, and are copied into stable graph inputs.
        return (
            tuple(id(job.context) for job in jobs),
            bucket,
            accepted.dtype,
            tuple(
                (
                    job.indices.dtype,
                    job.offsets.dtype,
                    None if job.table is None else job.table.shape[1],
                    None if job.table is None else job.table.dtype,
                    job.block_size,
                )
                for job in jobs
            ),
        )

    def prepare(self, jobs: tuple[GDNCommitJob, ...]) -> None:
        """Startup only: compile staging and capture with NULL/zero metadata."""
        if not self.config.capture_commit or not jobs:
            return
        if self.pending:
            raise RuntimeError("Cannot prepare recovery graphs with an outstanding transition")
        jobs = self._merge_jobs(jobs)
        self._runtime()
        capacity = min(self.capacity, *(job.context.commit_plan.capacity for job in jobs))
        device = jobs[0].indices.device
        with torch.npu.stream(self.stream):
            for dtype in (torch.int32, torch.int64):
                accepted = torch.zeros(capacity, dtype=dtype, device=device)
                for bucket in commit_buckets(capacity):
                    key = self._graph_key(jobs, accepted, bucket)
                    if key in self.graphs:
                        continue
                    graph = GDNCommitGraph(jobs, accepted, bucket)
                    # The sources are graph-owned NULL metadata, never live
                    # cache indices. Warm both sampler-mapping launch variants.
                    graph.update(graph.jobs, accepted)
                    graph.update(tuple(replace(job, mapping=None) for job in graph.jobs), accepted)
                    # Match source strides without reading or staging any
                    # caller-owned page IDs or offsets during warmup.
                    bound = tuple(
                        replace(
                            target,
                            indices=torch.empty_strided(
                                (bucket,), source.indices.stride(), dtype=source.indices.dtype, device=device
                            ).zero_(),
                            offsets=torch.empty_strided(
                                (bucket + 1,), source.offsets.stride(), dtype=source.offsets.dtype, device=device
                            ).zero_(),
                            mapping=None,
                        )
                        for source, target in zip(jobs, graph.jobs)
                    )
                    graph.update(bound, accepted)
                    graph.update(
                        tuple(replace(job, mapping=graph_job.mapping) for job, graph_job in zip(bound, graph.jobs)),
                        accepted,
                    )
                    graph.capture(self.stream)
                    self.graphs[key] = graph
        self.stream.synchronize()
        logger.info("Prepared %d GDN recovery graphs before serving requests", len(self.graphs))

    def submit(self, jobs: tuple[GDNCommitJob, ...], accepted: torch.Tensor) -> None:
        if not jobs:
            return
        if self.pending:
            raise RuntimeError("Previous GDN recovery must be joined before submitting another step")
        jobs = self._merge_jobs(jobs)
        self._runtime()
        main = torch.npu.current_stream()
        self.ready.record(main)
        # record_stream protects allocator reuse. The runner's entry fence
        # additionally protects metadata and cache pages from in-place reuse.
        seen = set()
        for tensor in (accepted, *(tensor for job in jobs for tensor in job.tensors())):
            if id(tensor) not in seen:
                tensor.record_stream(self.stream)
                seen.add(id(tensor))
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self.ready)
            try:
                self._run(jobs, accepted)
            finally:
                self.done.record(self.stream)
                self.pending = True
        overlap = self.config.async_commit and self.overlap_safe and all(job.table is None for job in jobs)
        if not overlap:
            if self.config.async_commit and any(job.table is not None for job in jobs):
                logger.warning_once("GDN align postprocessing is serialized with recovery to protect boundary copies")
            self.wait()

    def wait(self) -> None:
        if self.pending:
            torch.npu.current_stream().wait_event(self.done)
            self.pending = False

    def close(self) -> None:
        if self.stream is not None:
            # Shutdown only: graph-private workspaces and page owners must not
            # be released while recovery kernels are still in flight.
            self.stream.synchronize()
        self.graphs.clear()
        self.merged.clear()
        self.pending = False


class AscendGDNRecoverSSMState(RecoverSSMState):
    """Keep upstream recording, but submit all GDN groups as one transition."""

    def __init__(self, executor: GDNCommitExecutor):
        super().__init__()
        self.executor = executor

    def commit_step(self, num_sampled, idx_mapping, *, state_indices, num_accepted_tokens):
        step = self._step
        if (
            isinstance(num_sampled, int)
            or not step
            or not all(isinstance(meta, AscendGDNRecoverSSMMetadata) for meta in step)
        ):
            return super().commit_step(
                num_sampled, idx_mapping, state_indices=state_indices, num_accepted_tokens=num_accepted_tokens
            )
        self._step = None
        jobs = []
        for meta in step:
            commit = meta.recoverssm_commit
            if commit is None or not meta.num_spec_decodes:
                continue
            assert meta.recoverssm_context is not None
            jobs.append(
                GDNCommitJob(
                    meta.recoverssm_context,
                    commit.state_indices[: meta.num_spec_decodes, 0],
                    commit.query_start_loc[: meta.num_spec_decodes + 1],
                    commit.request_indices,
                    commit.block_table,
                    commit.num_computed_tokens,
                    commit.block_size,
                )
            )
        self.executor.submit(tuple(jobs), num_sampled)
        # Running-column bookkeeping stays on the main stream. In align mode
        # submit() joins first, preserving the inherited boundary-copy ordering.
        for meta in step:
            commit = meta.recoverssm_commit
            if commit is None or commit.block_table is None:
                continue
            assert state_indices is not None
            _postprocess_recoverssm_align_kernel[(meta.num_spec_decodes,)](
                idx_mapping,
                num_sampled,
                commit.request_indices,
                commit.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=commit.block_size,
                BLOCK_TABLE_WIDTH=commit.block_table.shape[1],
            )
