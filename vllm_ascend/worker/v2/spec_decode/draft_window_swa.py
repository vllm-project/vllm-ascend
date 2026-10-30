# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Rebase a fresh native-SWA metadata result using its existing exact pair."""

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_read import (
    NativeSWASource,
    materialize_native_swa_intersection,
)


class NativeSWABuilder:
    """Proposal-local adapter; the concrete baseline builder still runs first.

    Its CPU length consumer already establishes exact at this preparation.
    Reuse that result instead of copying device lengths again. Only a typed
    ordinary GQA SWA route can select this adapter. Non-causal DFlash2
    additionally requires the entire projected interval to remain resident.
    """

    def __init__(self, builder: Any, runtime: Any, owner: Any, gid: int, native_window: int, *, causal: bool = True):
        self.builder = builder
        self.runtime = runtime
        self.owner = owner
        self.gid = gid
        self.native_window = native_window
        if type(causal) is not bool:
            raise ValueError("Native SWA requires an explicit causal declaration")
        if not causal:
            span = runtime.speculator.block_tables.kernel_block_sizes[gid]
            width = runtime.speculator.num_query_per_req
            if runtime.window_size + span - 1 + width > native_window:
                raise NotImplementedError("Non-causal native Window must fit its certified resident interval")
        self.causal = causal
        self.built = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.builder, name)

    def build_for_cudagraph_capture(self, *args: Any, **kwargs: Any):
        raise RuntimeError("Native SWA extra Window requires eager Draft metadata")

    def build(self, common_prefix_len: int, common_attn_metadata: Any, **kwargs: Any):
        runtime, owner = self.runtime, self.owner
        if runtime._owner is not owner or self.built:
            raise RuntimeError("Native SWA builder belongs to a stale or repeated proposal")
        self.built = True
        common = common_attn_metadata
        batch = runtime._input_batch
        n = batch.num_reqs
        if common.num_reqs != n:
            raise ValueError("Native SWA projection requires the active request extent")
        mapping = (tuple(batch.req_ids), tuple(int(row) for row in batch.idx_mapping_np[:n]))
        metadata = self.builder.build(common_prefix_len, common, **kwargs)
        if runtime._owner is not owner or mapping != (
            tuple(batch.req_ids),
            tuple(int(row) for row in batch.idx_mapping_np[:n]),
        ):
            raise RuntimeError("Native SWA preparation mapping changed during build")
        # The concrete parallel-Draft builder obtains its native list from
        # current device E. seq_lens_cpu can still be that device tensor;
        # reuse the completed list, rather than copying it to CPU again.
        values = metadata.seq_lens_list
        if (
            metadata.seq_lens is not common.seq_lens
            or metadata.seq_lens_gpu is not common.seq_lens
            or metadata.block_tables is not common.block_table_tensor
            or common.seq_lens.dtype != torch.int32
            or common.seq_lens.shape != (n,)
            or not isinstance(values, list)
            or len(values) != n
            or any(type(value) is not int for value in values)
            or metadata.causal is not self.causal
        ):
            raise ValueError("Native SWA requires its concrete builder's fresh unprojected exact pair")
        exact = torch.tensor(values, dtype=torch.int32)
        widths = common.query_start_loc_cpu[: n + 1].diff()
        if not bool((widths == runtime.speculator.num_query_per_req).all()):
            raise ValueError("Native SWA query width differs from the DFlash preparation")
        from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows

        continuing = continuing_prefill_rows(batch, n)
        if continuing.all():
            return self._finish_build(metadata)
        source = NativeSWASource.from_block_tables(
            runtime.speculator.block_tables,
            self.gid,
            np.asarray(mapping[1]),
            self.native_window,
        )
        starts = torch.clamp(exact[:n] - runtime.window_size, min=0)
        starts = starts // source.block_span * source.block_span
        starts = torch.where(torch.from_numpy(continuing), 0, starts)
        # torch_npu synchronizes pageable host-to-device copies even with
        # non_blocking=True. Retain the actual pinned int64 DMA source; the
        # shared projector's dtype conversion must not unpin it again.
        starts = starts.to(torch.int64)
        if source.table.device.type == "npu":
            starts = starts.pin_memory()
        retain = lambda *items: runtime.pool.retain(owner, *items)
        # Retain both original exact sources and final native result before
        # any derived table copy can submit asynchronous device work.
        retain(metadata, common, exact, source)
        view = materialize_native_swa_intersection(
            source,
            exact_cpu=exact[:n],
            exact_device=common.seq_lens,
            upper_cpu=common.seq_lens_cpu_upper_bound[:n],
            starts_cpu=starts,
            padded_rows=common.num_reqs,
            query_width=runtime.speculator.num_query_per_req,
            row_indices=batch.idx_mapping[:n],
            retain=retain,
        )
        owned = replace(
            metadata,
            block_tables=view.block_table,
            seq_lens=view.seq_lens_cpu,
            seq_lens_cpu=view.seq_lens_cpu,
            seq_lens_gpu=view.seq_lens,
            seq_lens_list=view.seq_lens_cpu.tolist(),
        )
        retain(view, owned)
        return self._finish_build(owned)

    def _finish_build(self, metadata: Any):
        # A native-only drafter has no Full-group callback to certify its
        # preparation. Count it only after every native group built fresh
        # metadata; mixed Full/SWA routes keep their existing Full authority.
        runtime = self.runtime
        if not runtime.gids:
            runtime._native_swa_built.add(self.gid)
            if runtime._native_swa_built == runtime.native_swa_intersections.keys():
                runtime._build_count = 1
        return metadata
