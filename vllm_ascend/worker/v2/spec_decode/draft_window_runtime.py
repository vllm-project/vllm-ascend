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
from vllm.logger import logger

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (
    DraftWindowOwner,
    DraftWindowOwnerPool,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    DraftWindowPlan,
    compute_draft_window_plan,
    freeze_eagle_window_plan,
)
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
        if not policy.enabled or policy.effective_window_size is None:
            raise ValueError("An OFF Draft Window policy must not allocate a runtime")
        self.method = policy.normalized_method
        self.static_k = speculator.num_speculative_steps if self.method == "eagle3" else None
        if self.static_k is not None and policy.effective_window_size < 2 * self.static_k + 1:
            raise ValueError("Eagle3 draft_window_size must cover first-pass query width plus the proposal horizon")
        if self.method == "eagle3":
            # Main moved the capability to DraftModelSpeculator and renamed
            # it; the release lane still uses the fused-decode flag.
            updates_metadata_in_place = (
                speculator.supports_decode_metadata_update
                if hasattr(speculator, "supports_decode_metadata_update")
                else speculator.use_fused_multi_step_decode
            )
            if updates_metadata_in_place:
                raise NotImplementedError("Eagle3 Draft Window requires per-step metadata preparation")
        # The model config inherits the Target flag. All three Ascend
        # speculators also force their graph managers to NONE for the explicit
        # speculative override, allowing Target Graph with Draft eager.
        if not (speculator.draft_model_config.enforce_eager or speculator.speculative_config.enforce_eager):
            raise ValueError("Draft Window requires an eager Draft model; Target graph mode is independent")
        self.speculator = speculator
        self.window_size = policy.effective_window_size
        self.mla_layer_names: tuple[str, ...] = ()
        self.sfa_layer_names: tuple[str, ...] = ()
        self.dsa_layer_names: tuple[str, ...] = ()
        self.dsa_native_windows: dict[int, int] = {}
        self.native_swa_intersections: dict[int, int] = {}
        self.native_swa_causal: dict[int, bool] = {}
        self._dsa_builders = None
        self._dense_device_reads = False
        self.gids = self._validate_route(speculator)
        if self.dsa_native_windows:
            from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import DSABuilderOwners

            self._dsa_builders = DSABuilderOwners()
        read_groups = self.gids or tuple(self.native_swa_intersections)
        if not read_groups:
            raise NotImplementedError("Draft Window requires at least one certified read group")
        self.anchor_block_span = int(speculator.block_tables.kernel_block_sizes[read_groups[0]])
        self._native_swa_built: set[int] = set()
        self.pool = DraftWindowOwnerPool(int(speculator.vllm_config.max_concurrent_batches))
        self._owner: DraftWindowOwner | None = None
        self._input_batch: Any = None
        self._bypass = False
        self._build_count = 0
        self._eagle_plans: tuple[DraftWindowPlan, ...] | None = None
        self._eagle_frozen_starts: torch.Tensor | None = None
        self._eagle_first_pass_pair: torch.Tensor | None = None
        self._eagle_full_prefill_base: tuple[Any, torch.Tensor] | None = None
        self._eagle_prompt_exact_base: tuple[Any, torch.Tensor] | None = None
        self._next_step = 0
        self._proposal_mapping: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        narrowed_native_windows = tuple(
            sorted(
                {
                    native_window
                    for native_window in self.native_swa_intersections.values()
                    if self.window_size < native_window
                }
            )
        )
        if narrowed_native_windows:
            logger.warning_once(
                "The Draft model contains native SWA layers with windows %s. "
                "The additional draft_window_size=%s narrows those layers and may reduce "
                "acceptance or throughput. Prefer the native window by default and "
                "enable an additional window only after measuring your workload.",
                narrowed_native_windows,
                self.window_size,
                scope="global",
            )

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
        routes = require_direct_prefix_routes(speculator, window_size=self.window_size)
        self.native_swa_intersections = {}
        for route in routes:
            if route.status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION:
                if route.native_window is None:
                    raise ValueError("A native SWA provider must identify its native window")
                self.native_swa_intersections[route.gid] = route.native_window
        if self.native_swa_intersections:
            if self.window_size < speculator.num_query_per_req:
                raise ValueError("DFlash draft_window_size must cover its query width")
            declared = getattr(speculator, "_group_causal", True)
            self.native_swa_causal = {
                gid: declared[gid] if isinstance(declared, dict) else declared for gid in self.native_swa_intersections
            }
            routes = tuple(route for route in routes if route.gid not in self.native_swa_intersections)
        if (
            routes
            and self.method in ("eagle3", "dflash", "dspark")
            and all(route.status is DraftWindowProviderStatus.DIRECT_DENSE for route in routes)
        ):
            from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

            specs = [speculator.kv_cache_config.kv_cache_groups[route.gid].kv_cache_spec for route in routes]
            specs = [item for spec in specs for item in getattr(spec, "kv_cache_specs", {"": spec}).values()]
            context = config.compilation_config.static_forward_context
            implementations = [getattr(context.get(name), "impl", None) for route in routes for name in route.layers]
            self._dense_device_reads = all(spec.dtype in (torch.float16, torch.bfloat16) for spec in specs) and all(
                type(impl) is AscendAttentionBackendImpl
                and not impl.use_bnsd_kv_cache
                and impl.sinks is None
                and impl.sliding_window is None
                for impl in implementations
            )
        self.mla_layer_names = tuple(
            name for route in routes if route.status is DraftWindowProviderStatus.DIRECT_MLA for name in route.layers
        )
        self.sfa_layer_names = tuple(
            name
            for route in routes
            if route.status in (DraftWindowProviderStatus.DIRECT_SFA, DraftWindowProviderStatus.DIRECT_SFA_INDEXER)
            for name in route.layers
        )
        dsa_routes = tuple(route for route in routes if route.status is DraftWindowProviderStatus.DIRECT_DSA)
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
        self,
        speculator: Any,
        input_batch: Any,
        dummy_run: bool = False,
        is_profile: bool = False,
        *,
        context_kv_only: bool = False,
    ) -> Iterator[None]:
        if speculator is not self.speculator:
            raise RuntimeError("Draft Window runtime belongs to another speculator")
        if self._owner is not None or self._bypass:
            raise RuntimeError("Draft Window proposal scope is already active")
        if context_kv_only and self.method not in ("dflash", "dspark"):
            raise ValueError("Context-only Draft KV updates require a parallel drafter")
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
            self._eagle_frozen_starts = None
            self._eagle_first_pass_pair = None
            self._eagle_full_prefill_base = None
            self._eagle_prompt_exact_base = None
            self._next_step = 0
            self._proposal_mapping = None
            try:
                # Preparation and builders enqueue copies/kernels before model
                # forward. Even a later host error must retain this proposal.
                if self.native_swa_intersections:
                    self._native_swa_built.clear()
                self.pool.begin_native(owner)
                yield
                # DFlash/DSpark keep context KV current when the scheduler
                # requests zero drafts. That upstream branch writes KV but
                # never executes draft attention. Retain completion protection
                # for its writes without demanding a nonexistent read view.
                if context_kv_only and self._build_count:
                    raise RuntimeError("Context-only Draft KV update unexpectedly consumed a read projection")
                if not context_kv_only and not self._build_count:
                    raise RuntimeError("Draft Window proposal did not consume a read projection")
                if self.method == "eagle3" and self._next_step != self.static_k:
                    raise RuntimeError("Eagle3 proposal did not consume every configured draft step")
            finally:
                self._owner = None
                self._input_batch = None
                self._eagle_plans = None
                self._eagle_frozen_starts = None
                self._eagle_first_pass_pair = None
                self._eagle_full_prefill_base = None
                self._eagle_prompt_exact_base = None
                self._proposal_mapping = None

    def metadata_callback(
        self,
        step: int,
        num_reqs: int,
        *,
        prepared_exact_cpu: torch.Tensor | None = None,
        prepared_exact_device: torch.Tensor | None = None,
    ):
        legacy = self._host_metadata_callback(
            step,
            num_reqs,
            prepared_exact_cpu=prepared_exact_cpu,
            prepared_exact_device=prepared_exact_device,
        )
        if self.method == "eagle3":
            if (
                legacy is not None
                and self._dense_device_reads
                and continuing_prefill_rows(self._input_batch, num_reqs).all()
                and (
                    self._input_batch.num_draft_tokens_per_req is None
                    or not self._input_batch.num_draft_tokens_per_req[:num_reqs].any()
                )
                and getattr(self.speculator, "advance_draft_positions", False)
                and not getattr(self.speculator, "replicated_pcp", False)
            ):
                # Every unfinished prompt already requires the original Full
                # read at every AR step. Re-projecting S=0 adds an exact D2H
                # snapshot and private table without changing that read set.
                # With no draft placeholders, scheduled prompt tokens have
                # no rejection uncertainty. Use that current prompt producer,
                # never the optimistic CPU seq_lens mirror, for native lengths.
                return self._full_prefill_metadata_callback(step, num_reqs)
            # Mixed terminal/continuing prompt-only rows retain their
            # existing certified producer and native Full/window handling.
            # Mixed decode/prompt rows require the device-exact Full-capacity
            # path because rejection can make a CPU mirror optimistic.
            if continuing_prefill_rows(self._input_batch, num_reqs).any() and (
                self._input_batch.is_prefilling_np[:num_reqs].all()
                or not getattr(self.speculator, "advance_draft_positions", False)
                or getattr(self.speculator, "replicated_pcp", False)
            ):
                return legacy
            if legacy is not None and self._dense_device_reads:
                if step > 0 and self._eagle_frozen_starts is None:
                    raise RuntimeError("Eagle3 AR read has no current proposal anchor")
                return self._dense_metadata_callback(num_reqs, legacy, eagle_step=step)
            return legacy
        if legacy is not None and getattr(self, "_dense_device_reads", False):
            prompt_exact = self._parallel_prefill_exact_lengths(num_reqs)
            if prompt_exact is not None:
                # Preserve the original Full native reader for unfinished
                # prompts, without gathering their entire KV into FP32 or
                # copying E back from device. This producer is certified
                # separately from the DSpark MLA preparation's exact copy.
                return self._host_metadata_callback(
                    step,
                    num_reqs,
                    prepared_exact_cpu=prompt_exact,
                    prepared_exact_device=self.speculator.input_buffers.seq_lens[:num_reqs],
                )
            return self._dense_metadata_callback(num_reqs, legacy)
        return legacy

    def _parallel_prefill_exact_lengths(self, num_reqs: int) -> torch.Tensor | None:
        """Certify current unsharded, unfinished prompt rows before parallel KV.

        The shared parallel preparation places its query block immediately
        after the last valid target position. With no speculative placeholders
        and all prompts unfinished, the sampler rejects zero target tokens.
        Current prompt progress + actual scheduled prompt width + this method's
        real query width therefore gives its clamped native E. No CPU upper or
        accepted-count estimate is used. Mixed, terminal and replicated-PCP
        rows retain their existing device-exact route.
        """
        batch = self._input_batch
        if (
            self.method not in ("dflash", "dspark")
            or not self._dense_device_reads
            or getattr(self.speculator, "replicated_pcp", False)
            or batch.num_reqs != num_reqs
            or not batch.is_prefilling_np[:num_reqs].all()
        ):
            return None
        drafts = batch.num_draft_tokens_per_req
        boundaries = getattr(batch, "query_start_loc_np", None)
        if (drafts is not None and drafts[:num_reqs].any()) or boundaries is None:
            return None
        progress = batch.num_computed_prefill_tokens_np[:num_reqs]
        scheduled = batch.num_scheduled_tokens[:num_reqs]
        ends = progress + scheduled
        if (
            len(boundaries) < num_reqs + 1
            or not np.array_equal(np.diff(boundaries[: num_reqs + 1]), scheduled)
            or np.any(progress < 0)
            or np.any(scheduled <= 0)
            or np.any(ends >= batch.prefill_len_np[:num_reqs])
            or np.any(ends > self.speculator.max_model_len)
        ):
            return None
        return torch.tensor(
            np.minimum(ends + self.speculator.num_query_per_req, self.speculator.max_model_len),
            dtype=torch.int32,
            device="cpu",
        )

    def _full_prefill_metadata_callback(self, step: int, num_reqs: int):
        """Retain Full addressing and current prompt-exact native lengths.

        Request-state prompt progress and this batch's scheduled prompt count
        are exact while all prompts continue without speculative placeholders.
        The rejection sampler sets rejected=0 before prompt completion. Decode,
        mixed, terminal, placeholder and replicated-PCP batches keep their
        existing device-exact planning. Later AR steps use this proposal's
        retained base, advancing and clamping exactly as prepare_decode_inputs.
        """
        owner, batch = self._owner, self._input_batch
        if owner is None:
            raise RuntimeError("Full prefill metadata has no active Draft Window proposal")
        mapping = (tuple(batch.req_ids), tuple(int(row) for row in batch.idx_mapping_np[:num_reqs]))
        progress = tuple(int(value) for value in batch.num_computed_prefill_tokens_np[:num_reqs])
        scheduled = tuple(int(value) for value in batch.num_scheduled_tokens[:num_reqs])
        if any(computed < 0 or count <= 0 for computed, count in zip(progress, scheduled)):
            raise ValueError("Full prefill requires valid current scheduled prompt progress")
        phase_identity = (mapping, progress, scheduled)
        if step == 0:
            base = torch.tensor([computed + count for computed, count in zip(progress, scheduled)], dtype=torch.int32)
            self._eagle_full_prefill_base = phase_identity, base
            self.pool.retain(owner, base)
        if self._eagle_full_prefill_base is None or self._eagle_full_prefill_base[0] != phase_identity:
            raise RuntimeError("Full prefill prompt preparation changed within a proposal")
        base = self._eagle_full_prefill_base[1]
        seen = set()
        full_exact = None

        def apply(gid: int, common: Any) -> None:
            nonlocal full_exact
            if gid not in self.gids:
                return
            if self._owner is not owner or gid in seen:
                raise RuntimeError("Full prefill callback is stale or its group was already prepared")
            current = (tuple(batch.req_ids), tuple(int(row) for row in batch.idx_mapping_np[:num_reqs]))
            if batch.num_reqs != num_reqs or len(current[0]) != num_reqs or current != mapping:
                raise RuntimeError("Full prefill request mapping changed during metadata preparation")
            if common.num_reqs < num_reqs or common.seq_lens.ndim != 1 or common.seq_lens.numel() < num_reqs:
                raise ValueError("Full prefill metadata does not cover the active requests")
            if self._proposal_mapping is not None and self._proposal_mapping != mapping:
                raise RuntimeError("Full prefill request mapping changed within a proposal")
            if full_exact is None:
                full_exact = common.seq_lens
            assert full_exact is not None
            if common.seq_lens.data_ptr() != full_exact.data_ptr():
                raise ValueError("Full prefill groups must share current preparation lengths")
            if hasattr(common, "_draft_window_read") or hasattr(common, "_draft_window_device_read"):
                raise ValueError("Full prefill requires fresh, unprojected common metadata")
            exact_cpu = torch.zeros(common.num_reqs, dtype=torch.int32)
            exact_cpu[:num_reqs] = torch.clamp(base + step, max=self.speculator.max_model_len)
            common.seq_lens_cpu = exact_cpu
            common._seq_lens_cpu = exact_cpu
            self._proposal_mapping = mapping
            self.pool.retain(owner, common, exact_cpu, common.seq_lens, common.block_table_tensor)
            seen.add(gid)
            if len(seen) == len(self.gids):
                self._build_count += 1
                self._next_step += 1

        return apply

    def _dense_metadata_callback(self, num_reqs: int, legacy, *, eagle_step: int | None = None):
        from vllm_ascend.worker.v2.spec_decode.draft_window_dense import prepare_dense_draft_read

        owner = self._owner
        if owner is None:
            raise RuntimeError("Dense metadata has no active Draft Window proposal")
        seen = set()
        mapping = None
        full_exact: torch.Tensor | None = None
        use_device = None
        sources: dict[int, DirectPrefixSource] | None = None
        full_rows_cpu = None
        packed_query_cpu = packed_query_device = None
        query_width = 1 if eagle_step is not None else self.speculator.num_query_per_req

        def apply(gid: int, common: Any) -> None:
            nonlocal mapping, full_exact, use_device, sources, query_width
            nonlocal full_rows_cpu, packed_query_cpu, packed_query_device
            if gid not in self.gids:
                return
            if self._owner is not owner or gid in seen:
                raise RuntimeError("Dense Draft callback is stale or its group was already projected")
            batch = self._input_batch
            current = (tuple(batch.req_ids), tuple(int(row) for row in batch.idx_mapping_np[:num_reqs]))
            if batch.num_reqs != num_reqs or len(current[0]) != num_reqs:
                raise ValueError("Dense Draft request mapping does not match its preparation")
            if use_device is None:
                mapping = current
                full_exact = common.seq_lens
                rows = batch.idx_mapping_np[:num_reqs]
                sources = {
                    group: DirectPrefixSource.from_block_tables(self.speculator.block_tables, group, rows)
                    for group in self.gids
                }
                upper = common.seq_lens_cpu_upper_bound[:num_reqs]
                widths = common.query_start_loc_cpu[: num_reqs + 1].diff()
                if eagle_step == 0:
                    query_width = int(widths.max())
                continuing = continuing_prefill_rows(batch, num_reqs)
                full_rows = continuing | (batch.is_prefilling_np[:num_reqs] if eagle_step == 0 else False)
                full_rows_cpu = torch.tensor(full_rows, dtype=torch.bool, device="cpu")
                # Keep the existing exact path when scheduling bounds do not
                # prove allocated address capacity. Parallel methods preserve
                # their fixed native query width; Eagle may right-align its
                # current ragged queries in private attention scratch.
                use_device = (
                    bool((widths > 0).all())
                    and (eagle_step == 0 or bool((widths == query_width).all()))
                    and all(
                        all(
                            0 < int(bound) <= count * source.block_span
                            for bound, count in zip(upper, source.valid_columns)
                        )
                        for source in sources.values()
                    )
                )
                if eagle_step == 0:
                    assert self.static_k is not None
                    prefilling = batch.is_prefilling_np[:num_reqs]
                    if prefilling.any():
                        # Entire prompt-only batches already have the cheaper
                        # certified CPU producer. Only mixed decode/prompt
                        # batches need a Full-capacity device read here.
                        use_device = (
                            use_device
                            and not prefilling.all()
                            and bool(
                                getattr(self.speculator, "advance_draft_positions", False)
                                and not getattr(self.speculator, "replicated_pcp", False)
                            )
                        )
                    else:
                        use_device = use_device and query_width <= self.static_k + 1
                    if use_device:
                        self._eagle_first_pass_device_lengths(common.seq_lens, num_reqs)
                if use_device and not bool((widths == query_width).all()):
                    packed_query_cpu = common.query_start_loc_cpu[: num_reqs + 1].clone()
                    packed_query_device = common.query_start_loc[: num_reqs + 1]
            if not use_device:
                legacy(gid, common)
                return
            if current != mapping or (self._proposal_mapping is not None and self._proposal_mapping != mapping):
                raise RuntimeError("Dense Draft request mapping changed within a proposal")
            assert full_exact is not None and sources is not None
            if common.seq_lens.data_ptr() != full_exact.data_ptr():
                raise ValueError("Dense Draft groups must share current preparation lengths")
            if packed_query_cpu is not None:
                assert packed_query_device is not None
                if (
                    not torch.equal(common.query_start_loc_cpu[: num_reqs + 1], packed_query_cpu)
                    or common.query_start_loc.data_ptr() != packed_query_device.data_ptr()
                ):
                    raise ValueError("Dense Draft groups must share current packed query boundaries")
            self._proposal_mapping = mapping
            retain = lambda *objects: self.pool.retain(owner, *objects)
            first_pass = self._eagle_first_pass_pair if eagle_step == 0 else None
            read = prepare_dense_draft_read(
                sources[gid],
                exact_device=common.seq_lens if first_pass is None else first_pass[1],
                row_indices=batch.idx_mapping[:num_reqs],
                upper_cpu=common.seq_lens_cpu_upper_bound[:num_reqs],
                window_size=self.window_size,
                anchor_span=self.anchor_block_span,
                query_width=query_width,
                causal=common.causal,
                retain=retain,
                frozen_starts=self._eagle_frozen_starts if eagle_step is not None else None,
                query_ends=None if first_pass is None else first_pass[0],
                full_rows_cpu=full_rows_cpu,
                query_start_loc_cpu=packed_query_cpu,
                query_start_loc_device=packed_query_device,
            )
            retain(common)
            read.apply(common)
            seen.add(gid)
            if len(seen) == len(self.gids):
                self._build_count += 1
                if eagle_step is not None:
                    self._next_step += 1

        return apply

    def _eagle_first_pass_device_lengths(self, native_extent: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """Freeze current live/native lengths without a Host exact consumer.

        MRV2 keeps rejected query padding in the first-pass native layout.
        Its last valid query index, rather than that padded extent, identifies
        the logical proposal end. Native lengths keep the original causal
        alignment; only the frozen Window anchor uses the live end.
        """
        positions = self.speculator.input_buffers.positions
        indices = self.speculator.last_token_indices[:num_reqs]
        if (
            positions.ndim != 1
            or indices.shape != (num_reqs,)
            or positions.device != native_extent.device
            or indices.device != native_extent.device
            or positions.dtype not in (torch.int32, torch.int64)
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Eagle3 Window requires current one-dimensional prepared positions and last query indices")
        owner = self._owner
        if owner is None:
            raise RuntimeError("Eagle3 length preparation has no active Draft Window proposal")
        indices = indices.to(dtype=torch.int64)
        self.pool.retain(owner, indices)
        last_positions = positions.index_select(0, indices)
        self.pool.retain(owner, last_positions)
        live_end = last_positions.to(dtype=torch.int32)
        self.pool.retain(owner, live_end)
        live_end.add_(1)
        if self._dense_device_reads:
            # Freeze from this preparation's last valid query. Later AR
            # inputs advance within this horizon;
            # their device exact lengths must not move this proposal anchor.
            self._eagle_frozen_starts = (
                torch.div(
                    torch.clamp(live_end + self.static_k - self.window_size, min=0),
                    self.anchor_block_span,
                    rounding_mode="floor",
                )
                * self.anchor_block_span
            )
            self.pool.retain(owner, self._eagle_frozen_starts)
        pair = torch.stack((native_extent[:num_reqs], live_end))
        self.pool.retain(owner, pair)
        self._eagle_first_pass_pair = pair
        return pair

    def _eagle_first_pass_lengths(self, native_extent: torch.Tensor, num_reqs: int):
        """Certify first-pass lengths for a legacy consumer or later fallback."""
        pair = self._eagle_first_pass_pair
        if pair is None:
            pair = self._eagle_first_pass_device_lengths(native_extent, num_reqs)
        owner = self._owner
        if owner is None:
            raise RuntimeError("Eagle3 length snapshot has no active Draft Window proposal")
        snapshot = pair.to(device="cpu", dtype=torch.int32, copy=True)
        self.pool.retain(owner, snapshot)
        extent_cpu, live_cpu = snapshot[0], snapshot[1]
        if bool(((live_cpu <= 0) | (live_cpu > extent_cpu) | (extent_cpu - live_cpu > self.static_k)).any()):
            raise ValueError("Eagle3 prepared live queries disagree with the current native padding extent")
        return extent_cpu, live_cpu

    def _prompt_only_exact_lengths(self, step: int, num_reqs: int):
        """Experimental producer reuse for prompt-only, non-PCP Eagle rows.

        This certifies current prompt progress and actual scheduled prompt
        widths, not the optimistic seq_lens mirror. Terminal rows have one
        target logit and no speculative candidates, so rejection is zero;
        continuing rows have rejected=0 in the existing sampler contract.
        Reader selection and per-row Full/window rules remain unchanged.
        """
        batch = self._input_batch
        owner = self._owner
        if owner is None:
            raise RuntimeError("Prompt exact lengths have no active Draft Window proposal")
        if not self._dense_device_reads or not getattr(self.speculator, "advance_draft_positions", False):
            return None
        if getattr(self.speculator, "replicated_pcp", False):
            return None
        if not batch.is_prefilling_np[:num_reqs].all():
            return None
        drafts = batch.num_draft_tokens_per_req
        if drafts is not None and drafts[:num_reqs].any():
            return None
        boundaries = getattr(batch, "query_start_loc_np", None)
        logits = getattr(batch, "cu_num_logits_np", None)
        if boundaries is None or logits is None:
            return None
        progress = batch.num_computed_prefill_tokens_np[:num_reqs]
        scheduled = batch.num_scheduled_tokens[:num_reqs]
        prompt = batch.prefill_len_np[:num_reqs]
        ends = progress + scheduled
        if (
            len(boundaries) < num_reqs + 1
            or len(logits) < num_reqs + 1
            or not np.array_equal(np.diff(boundaries[: num_reqs + 1]), scheduled)
            or np.any(progress < 0)
            or np.any(scheduled <= 0)
            or np.any(ends > prompt)
            or np.any(ends > self.speculator.max_model_len)
            or np.any(np.diff(logits[: num_reqs + 1])[ends == prompt] != 1)
        ):
            return None
        identity = (
            tuple(batch.req_ids),
            tuple(batch.idx_mapping_np[:num_reqs]),
            tuple(progress),
            tuple(scheduled),
            tuple(prompt),
        )
        if self._eagle_prompt_exact_base is None:
            if step != 0:
                return None
            base = torch.tensor(ends, dtype=torch.int32, device="cpu")
            self._eagle_prompt_exact_base = identity, base
            self.pool.retain(owner, base)
        if self._eagle_prompt_exact_base[0] != identity:
            raise RuntimeError("Prompt-only exact preparation changed within a proposal")
        exact = torch.clamp(self._eagle_prompt_exact_base[1] + step, max=self.speculator.max_model_len)
        self.pool.retain(owner, exact)
        return exact

    def _host_metadata_callback(
        self,
        step: int,
        num_reqs: int,
        *,
        prepared_exact_cpu: torch.Tensor | None = None,
        prepared_exact_device: torch.Tensor | None = None,
    ):
        """Project current prepared lengths; CPU reuse requires an explicit pair.

        The caller owns a fresh exact snapshot in this active proposal and
        request order: either a completed copy from the device input or the
        separately certified unfinished-parallel-prompt producer. It must not
        modify that snapshot afterward. A common CPU field alone is not an
        exactness certificate. The proposal retains the supplied snapshot.
        """
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

        prepared_mapping = None
        if (prepared_exact_cpu is None) != (prepared_exact_device is None):
            raise ValueError("Prepared exact CPU lengths require their device input")
        if prepared_exact_cpu is not None:
            if (
                not isinstance(prepared_exact_cpu, torch.Tensor)
                or prepared_exact_cpu.device.type != "cpu"
                or prepared_exact_cpu.dtype != torch.int32
                or prepared_exact_cpu.shape != (num_reqs,)
                or not prepared_exact_cpu.is_contiguous()
            ):
                raise ValueError("Prepared exact CPU lengths must be one contiguous int32 value per request")
            if (
                not isinstance(prepared_exact_device, torch.Tensor)
                or prepared_exact_device.dtype != torch.int32
                or prepared_exact_device.ndim != 1
                or prepared_exact_device.numel() < num_reqs
                or not prepared_exact_device.is_contiguous()
            ):
                raise ValueError("Prepared exact device input must cover the active int32 request prefix")
            batch = self._input_batch
            prepared_mapping = (tuple(batch.req_ids), tuple(int(row) for row in batch.idx_mapping_np[:num_reqs]))
            self.pool.retain(owner, prepared_exact_cpu)

        # All direct-prefix groups use the same logical exact lengths and S.
        # The callback runs once per fresh common object, before each builder.
        exact = starts = rows = full_exact_input = proposal_exact = None
        request_ids = None
        seen: set[int] = set()

        def apply(gid: int, common_metadata: Any) -> None:
            nonlocal exact, starts, rows, request_ids, full_exact_input, proposal_exact
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
                full_exact_input = common_metadata.seq_lens
                mapping = (request_ids, tuple(int(row) for row in rows))
                prompt_exact = (
                    self._prompt_only_exact_lengths(step, num_reqs)
                    if self.method == "eagle3" and prepared_exact_cpu is None
                    else None
                )
                if self.method == "eagle3" and step == 0 and prompt_exact is not None:
                    # Retain the real device native/live pair and frozen starts
                    # for a subsequent dense AR reader, without copying it.
                    self._eagle_first_pass_device_lengths(full_exact_input, num_reqs)
                    exact = proposal_exact = prompt_exact
                elif self.method == "eagle3" and step == 0:
                    exact, proposal_exact = self._eagle_first_pass_lengths(full_exact_input, num_reqs)
                elif prompt_exact is not None:
                    exact = prompt_exact
                elif prepared_exact_cpu is not None:
                    assert prepared_exact_device is not None
                    if mapping != prepared_mapping:
                        raise RuntimeError("Prepared exact lengths belong to another request mapping")
                    if (
                        full_exact_input.data_ptr() != prepared_exact_device.data_ptr()
                        or full_exact_input.device != prepared_exact_device.device
                        or full_exact_input.dtype != prepared_exact_device.dtype
                        or full_exact_input.ndim != 1
                        or full_exact_input.numel() < num_reqs
                        or not full_exact_input.is_contiguous()
                    ):
                        raise ValueError("Prepared exact CPU lengths belong to another device input")
                    exact = prepared_exact_cpu
                else:
                    # One exact D2H per build, shared across groups. The explicit
                    # DSpark MLA snapshot avoids this duplicate copy; other
                    # routes still synchronize here. An upper is never exact.
                    exact = full_exact_input[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
                continuing = continuing_prefill_rows(batch, num_reqs)
                if self._proposal_mapping is not None and self._proposal_mapping != mapping:
                    raise RuntimeError("Draft Window request mapping changed within a proposal")
                self._proposal_mapping = mapping
                if self.method == "eagle3":
                    if self._eagle_plans is None:
                        if step > 0:
                            if self._eagle_first_pass_pair is None:
                                raise RuntimeError("Eagle3 fallback has no current first-pass snapshot")
                            # A device first pass retained this generation's
                            # native/live pair. A later legacy consumer may
                            # copy that pair, never infer the anchor from E-W.
                            _, proposal_exact = self._eagle_first_pass_lengths(self._eagle_first_pass_pair[0], num_reqs)
                        assert proposal_exact is not None
                        self._eagle_plans = tuple(
                            freeze_eagle_window_plan(
                                proposal_exact_end=int(end),
                                static_k=self.static_k,
                                max_rejected_tokens=self.static_k,
                                window_size=self.window_size,
                                block_span_tokens=self.anchor_block_span,
                            )
                            for end in proposal_exact
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
                row_indices=(
                    self._input_batch.idx_mapping[:num_reqs] if hasattr(self._input_batch, "idx_mapping") else None
                ),
                retain=lambda *objects: self.pool.retain(owner, *objects),
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
        """Select a proposal-owned builder at the existing metadata extension point."""
        if self._bypass or (self._dsa_builders is None and not self.native_swa_intersections):
            return None
        owner = self._owner
        if owner is None:
            raise RuntimeError("Draft Window builder has no active proposal")

        def select(gid, builder, layer_names):
            if gid in self.native_swa_intersections:
                from vllm_ascend.worker.v2.spec_decode.draft_window_swa import NativeSWABuilder

                if self._owner is not owner:
                    raise RuntimeError("Native SWA builder selection belongs to a stale proposal")
                owned = NativeSWABuilder(
                    builder,
                    self,
                    owner,
                    gid,
                    self.native_swa_intersections[gid],
                    causal=self.native_swa_causal.get(gid, True),
                )
                self.pool.retain(owner, owned)
                return owned
            if gid not in self.dsa_native_windows or not set(layer_names).issubset(self.dsa_layer_names):
                return builder
            if self._owner is not owner:
                raise RuntimeError("Draft Window builder callback belongs to a stale proposal")
            assert self._dsa_builders is not None
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

            owner = self._owner
            rotary_reads = freeze_mla_read_metadata(
                metadata, self.mla_layer_names, retain=lambda *objects: self.pool.retain(owner, *objects)
            )
            self.pool.retain(owner, *rotary_reads)
        if self.sfa_layer_names:
            from vllm_ascend.worker.v2.spec_decode.draft_window_sfa import freeze_sfa_read_metadata

            self.pool.retain(self._owner, *freeze_sfa_read_metadata(metadata, self.sfa_layer_names))
        if self.dsa_layer_names:
            from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import freeze_dsa_rotary_reads

            self.pool.retain(self._owner, *freeze_dsa_rotary_reads(metadata, self.dsa_layer_names))
        self.pool.retain(self._owner, metadata)
