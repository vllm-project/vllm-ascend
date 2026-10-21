# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Eager proposal integration for resident direct-prefix Draft KV reads.

An enabled runtime is installed only after the actual Draft consumers have
been checked. Target graph selection and Full write/query operands are owned
by the existing runner. Window projections and built Draft metadata are owned
by a proposal until its final consumer event completes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerPool
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan, freeze_eagle_window_plan
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import DraftWindowPolicyDecision
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource, materialize_direct_prefix


class _NPUCompletion:
    def __init__(self) -> None:
        # Construction and recording both execute under the owner pool's
        # failure guard. No fallback token can substitute for either failure.
        self.event = torch.npu.Event()
        self.event.record(torch.npu.current_stream())

    def is_complete(self) -> bool:
        return bool(self.event.query())

    def wait(self) -> None:
        self.event.synchronize()


def continuing_prefill_rows(input_batch: Any, num_reqs: int) -> np.ndarray:
    """Prompt progress excludes speculative placeholders in scheduled tokens.

    This uses the same new-token count as MRV2 gather_batch_req_state. Decode
    rows already have computed_prefill >= prefill_len. A terminal chunk may
    therefore use a Window even while the input row is marked is_prefilling.
    """
    computed = input_batch.num_computed_prefill_tokens_np[:num_reqs]
    prompt = input_batch.prefill_len_np[:num_reqs]
    scheduled = input_batch.num_scheduled_tokens[:num_reqs]
    drafts = input_batch.num_draft_tokens_per_req
    if drafts is not None:
        scheduled = scheduled - drafts[:num_reqs]
    if any(array.shape != (num_reqs,) for array in (computed, prompt, scheduled)):
        raise ValueError("Draft Window phase state does not cover the active requests")
    if np.any(scheduled < 0):
        raise ValueError("Speculative placeholders exceed scheduled tokens")
    return computed + scheduled < prompt


class DraftWindowRuntime:
    """One speculator's bounded ownership and fresh metadata projection seam."""

    def __init__(self, speculator: Any, policy: DraftWindowPolicyDecision) -> None:
        if not policy.enabled:
            raise ValueError("An OFF Draft Window policy must not allocate a runtime")
        self.method = policy.normalized_method
        self.static_k = speculator.num_speculative_steps if self.method == "eagle3" else None
        if self.static_k is not None and policy.effective_window_size < 2 * self.static_k + 1:
            raise ValueError("Eagle3 draft_window_size must cover first-pass query width plus the proposal horizon")
        if self.method == "eagle3" and speculator.use_fused_multi_step_decode:
            raise NotImplementedError("Eagle3 Draft Window requires per-step metadata preparation")
        if not speculator.draft_model_config.enforce_eager:
            raise ValueError("Draft Window requires an eager Draft model; Target graph mode is independent")
        self.speculator = speculator
        self.window_size = policy.effective_window_size
        self.mla_layer_names: tuple[str, ...] = ()
        self.sfa_layer_names: tuple[str, ...] = ()
        self.dsa_layer_names: tuple[str, ...] = ()
        self.dsa_native_windows: dict[int, int] = {}
        self._dsa_builders = None
        self.gids = self._validate_route(speculator)
        if self.dsa_native_windows:
            from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import DSABuilderOwners

            self._dsa_builders = DSABuilderOwners()
        self.anchor_block_span = int(speculator.block_tables.kernel_block_sizes[self.gids[0]])
        self.pool = DraftWindowOwnerPool(int(speculator.vllm_config.max_concurrent_batches))
        self._owner = None
        self._input_batch = None
        self._bypass = False
        self._build_count = 0
        self._eagle_plans = None
        self._next_step = 0
        self._proposal_mapping = None

    @property
    def reads_active(self) -> bool:
        return self._owner is not None and not self._bypass

    def _validate_route(self, speculator: Any) -> tuple[int, ...]:
        # Imported lazily: provider inventory depends on the installed NPU
        # backends, while phase/owner tests need no device import boundary.
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
            DraftWindowProviderStatus,
            require_direct_prefix_routes,
        )

        config = speculator.attn_vllm_config
        if config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Draft Window pipeline parallelism requires a distributed reader provider")
        transfer = config.kv_transfer_config
        if (
            (transfer is not None and transfer.kv_connector is not None)
            or config.cache_config.kv_offloading_size
            or get_ascend_config().sparse_kv_offload_config.enabled
        ):
            raise NotImplementedError("Draft Window offloaded/transferred KV requires a resident-set provider")
        # For this FullAttentionSpec route the scheduler retains every active
        # request block through execution. No circular/native eviction,
        # offload, transfer or CP projection is admitted by this provider.
        # Standard GQA builders launch attention and read-derived preparation
        # on the current stream. Ordinary TP collectives consume hidden states,
        # not Window metadata. Ordinary Flashcomm/SP keeps the baseline query
        # padding and Full write coordinates; actual CP builders are excluded
        # by the provider inventory rather than a target-global CP switch.
        routes = require_direct_prefix_routes(speculator)
        self.mla_layer_names = tuple(
            name for route in routes if route.status is DraftWindowProviderStatus.DIRECT_MLA for name in route.layers
        )
        self.sfa_layer_names = tuple(
            name
            for route in routes
            if route.status in (DraftWindowProviderStatus.DIRECT_SFA, DraftWindowProviderStatus.DIRECT_SFA_INDEXER)
            for name in route.layers
        )
        dsa_routes = tuple(
            route
            for route in routes
            if route.status in (DraftWindowProviderStatus.DIRECT_DSA, DraftWindowProviderStatus.DIRECT_DSA_CP)
        )
        self.dsa_layer_names = tuple(name for route in dsa_routes for name in route.layers)
        for route in dsa_routes:
            # The typed inventory checks every consumer's native spec against
            # the builder authority; no target model name selects this path.
            builder = next(
                group.get_metadata_builder(0)
                for group in speculator.attn_groups[route.gid]
                if tuple(group.layer_names) == route.layers
            )
            self.dsa_native_windows[route.gid] = int(builder.model_config.hf_config.sliding_window)
        return tuple(dict.fromkeys(route.gid for route in routes))

    @contextmanager
    def proposal_context(
        self, speculator: Any, input_batch: Any, dummy_run: bool = False, is_profile: bool = False
    ) -> Iterator[None]:
        if speculator is not self.speculator:
            raise RuntimeError("Draft Window runtime belongs to another speculator")
        if self._owner is not None or self._bypass:
            raise RuntimeError("Draft Window proposal scope is already active")
        if dummy_run or is_profile or getattr(input_batch, "is_dummy", False):
            self._bypass = True
            try:
                yield
            finally:
                self._bypass = False
            return
        with self.pool.operation(_NPUCompletion) as owner:
            self._owner = owner
            self._input_batch = input_batch
            self._build_count = 0
            self._eagle_plans = None
            self._next_step = 0
            self._proposal_mapping = None
            try:
                # Preparation and builders enqueue copies/kernels before model
                # forward. Even a later host error must retain this proposal.
                self.pool.begin_native(owner)
                yield
                if not self._build_count:
                    raise RuntimeError("Draft Window proposal did not consume a read projection")
                if self.method == "eagle3" and self._next_step != self.static_k:
                    raise RuntimeError("Eagle3 proposal did not consume every configured draft step")
            finally:
                self._owner = None
                self._input_batch = None
                self._eagle_plans = None
                self._proposal_mapping = None

    def metadata_callback(self, step: int, num_reqs: int):
        if self._bypass:
            return None
        owner = self._owner
        if owner is None:
            raise RuntimeError("Draft Window metadata has no active proposal")
        speculator = self.speculator
        if self.method != "eagle3" and step != speculator.num_query_per_req:
            raise ValueError("Parallel Draft metadata step must match its actual query width")
        if self.method == "eagle3" and (step != self._next_step or step >= self.static_k):
            raise ValueError("Eagle3 metadata must advance exactly one configured step")

        # All direct-prefix groups use the same logical exact lengths and S.
        # The callback runs once per fresh common object, before each builder.
        exact = starts = rows = full_exact_input = None
        request_ids = None
        seen: set[int] = set()

        def apply(gid: int, common_metadata: Any) -> None:
            nonlocal exact, starts, rows, request_ids, full_exact_input
            if gid not in self.gids:
                return
            if self._owner is not owner:
                raise RuntimeError("Draft Window metadata callback belongs to a stale proposal")
            if gid in seen:
                raise RuntimeError("Draft Window common metadata was projected twice for one group")
            batch = self._input_batch
            if batch.num_reqs != num_reqs or len(batch.req_ids) != num_reqs:
                raise ValueError("Draft Window metadata and request counts differ")
            if exact is None:
                request_ids = tuple(batch.req_ids)
                rows = batch.idx_mapping_np[:num_reqs].copy()
                # One exact D2H per metadata build, shared across groups. A CPU
                # upper cannot select S after rejection. This synchronization
                # remains a performance cost until preparation exposes exact CPU
                # state; it must not be mislabeled as a no-sync implementation.
                full_exact_input = common_metadata.seq_lens
                exact = full_exact_input[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
                continuing = continuing_prefill_rows(batch, num_reqs)
                mapping = (request_ids, tuple(int(row) for row in rows))
                if self._proposal_mapping is not None and self._proposal_mapping != mapping:
                    raise RuntimeError("Draft Window request mapping changed within a proposal")
                self._proposal_mapping = mapping
                if self.method == "eagle3":
                    if step == 0:
                        self._eagle_plans = tuple(
                            freeze_eagle_window_plan(
                                proposal_exact_end=int(end),
                                static_k=self.static_k,
                                max_rejected_tokens=self.static_k,
                                window_size=self.window_size,
                                block_span_tokens=self.anchor_block_span,
                            )
                            for end in exact
                        )
                    if self._eagle_plans is None:
                        raise RuntimeError("Eagle3 decode metadata has no frozen first-pass plan")
                    starts_list = [
                        0
                        if continuing[row] or (step == 0 and batch.is_prefilling_np[row])
                        else plan.at_exact_end(int(exact[row])).window_start
                        for row, plan in enumerate(self._eagle_plans)
                    ]
                else:
                    starts_list = [
                        0
                        if continuing[row]
                        else compute_draft_window_plan(
                            full_exact_end=int(exact[row]),
                            window_size=self.window_size,
                            block_span_tokens=self.anchor_block_span,
                        ).window_start
                        for row in range(num_reqs)
                    ]
                starts = torch.tensor(starts_list, dtype=torch.int32)
            if tuple(batch.req_ids) != request_ids or not np.array_equal(batch.idx_mapping_np[:num_reqs], rows):
                raise RuntimeError("Draft Window request mapping changed during metadata preparation")
            if common_metadata.seq_lens.data_ptr() != full_exact_input.data_ptr():
                raise ValueError("Direct-prefix groups must share the prepared exact-length input")
            source = DirectPrefixSource.from_block_tables(speculator.block_tables, gid, rows)
            if gid in self.dsa_native_windows:
                from vllm_ascend.attention.utils import split_decodes_and_prefills
                from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import validate_dsa_read_subset

                _, num_prefills, _, _ = split_decodes_and_prefills(
                    common_metadata,
                    decode_threshold=speculator.num_speculative_steps + 1,
                    treat_short_extends_as_decodes=False,
                )
                queries = common_metadata.query_start_loc_cpu[: num_reqs + 1].diff().tolist()
                # Noncausal sparse indices are the backend's decode path.
                # Prefill retains its existing causal native mask semantics.
                certificate = validate_dsa_read_subset(
                    window_starts=starts.tolist(),
                    exact_ends=exact.tolist(),
                    query_lengths=queries,
                    kernel_block_span=source.block_span,
                    native_window=self.dsa_native_windows[gid],
                    causal=bool(common_metadata.causal or num_prefills),
                )
                self.pool.retain(owner, certificate)
            view = materialize_direct_prefix(
                source,
                exact_cpu=exact,
                exact_device=common_metadata.seq_lens,
                upper_cpu=common_metadata.seq_lens_cpu_upper_bound[:num_reqs],
                starts_cpu=starts,
                padded_rows=common_metadata.num_reqs,
            )
            # Retain before a builder or native reader can receive the view.
            self.pool.retain(owner, view, common_metadata)
            view.apply(common_metadata)
            seen.add(gid)
            if len(seen) == len(self.gids):
                self._build_count += 1
                if self.method == "eagle3":
                    self._next_step += 1

        return apply

    def metadata_builder_callback(self, step: int):
        """Select a complete private DSA builder before any derived-state write."""
        if self._bypass or self._dsa_builders is None:
            return None
        owner = self._owner
        if owner is None:
            raise RuntimeError("Draft Window builder has no active proposal")

        def select(gid, builder, layer_names):
            if gid not in self.dsa_native_windows or not set(layer_names).issubset(self.dsa_layer_names):
                return builder
            if self._owner is not owner:
                raise RuntimeError("Draft Window builder callback belongs to a stale proposal")
            owned = self._dsa_builders.select(builder, layer_names=layer_names, lane=owner.lane, step=step)
            if self.method != "eagle3" and hasattr(owned, "set_dspark_num_query_per_req"):
                owned.set_dspark_num_query_per_req(self.speculator.num_query_per_req)
            self.pool.retain(owner, owned)
            return owned

        return select

    def retain_metadata(self, metadata: Any) -> None:
        """Keep the final builder result, including newly allocated derived state."""
        if self._bypass:
            return
        if self._owner is None:
            raise RuntimeError("Draft Window metadata has no active proposal")
        if self.mla_layer_names:
            from vllm_ascend.worker.v2.spec_decode.draft_window_mla import freeze_mla_read_metadata

            rotary_reads = freeze_mla_read_metadata(metadata, self.mla_layer_names)
            self.pool.retain(self._owner, *rotary_reads)
        if self.sfa_layer_names:
            from vllm_ascend.worker.v2.spec_decode.draft_window_sfa import freeze_sfa_read_metadata

            self.pool.retain(self._owner, *freeze_sfa_read_metadata(metadata, self.sfa_layer_names))
        if self.dsa_layer_names:
            from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import freeze_dsa_rotary_reads

            self.pool.retain(self._owner, *freeze_dsa_rotary_reads(metadata, self.dsa_layer_names))
        self.pool.retain(self._owner, metadata)
