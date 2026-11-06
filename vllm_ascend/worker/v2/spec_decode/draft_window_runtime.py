# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Eager proposal integration for resident direct-prefix Draft KV reads.

An enabled runtime is installed only after the actual Draft consumers have
been checked. Target graph selection and Full write/query operands are owned
by the existing runner. Window projections and built Draft metadata remain owned
until the next existing blocking exact copy completes earlier stream readers.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    compute_draft_window_start,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import (
    materialize_draft_read,
)


def continuing_prefill_rows(input_batch: Any, num_reqs: int) -> np.ndarray:
    """Prompt progress excludes speculative placeholders in scheduled tokens.

    This uses the same new-token count as MRV2 gather_batch_req_state. Decode
    rows already have computed_prefill >= prefill_len. A terminal chunk may
    therefore use a Window even while the input row is marked is_prefilling.
    """
    # Fixed MRV2 gathers these integer arrays together in active batch order.
    # Recheck the changing placeholder count, not the framework's array schema.
    computed = input_batch.num_computed_prefill_tokens_np[:num_reqs]
    prompt = input_batch.prefill_len_np[:num_reqs]
    scheduled = input_batch.num_scheduled_tokens[:num_reqs]
    drafts = input_batch.num_draft_tokens_per_req
    if drafts is not None:
        drafts = drafts[:num_reqs]
        if np.any(drafts < 0):
            raise ValueError("Draft placeholders must be nonnegative")
        scheduled = scheduled - drafts
    if np.any(scheduled < 0):
        raise ValueError("Speculative placeholders exceed scheduled tokens")
    if np.any(computed < 0) or np.any(prompt < 0):
        raise ValueError("Draft Window phase counters must be nonnegative")
    return computed + scheduled < prompt


class DraftWindowRuntime:
    """One speculator's bounded ownership and fresh metadata projection seam."""

    def __init__(self, speculator: Any, policy: int | None) -> None:
        if policy is None:
            raise ValueError("An OFF Draft Window policy must not allocate a runtime")
        self.speculator = speculator
        if speculator.method == "eagle3" and policy < 2 * speculator.num_speculative_steps + 1:
            raise ValueError("Eagle3 draft_window_size must cover first-pass query width plus the proposal horizon")
        if self.speculator.method == "eagle3" and speculator.supports_decode_metadata_update:
            raise NotImplementedError("Eagle3 Draft Window requires per-step metadata preparation")
        # The model config inherits the Target flag. All three Ascend
        # speculators also force their graph managers to NONE for the explicit
        # speculative override, allowing Target Graph with Draft eager.
        if not (speculator.draft_model_config.enforce_eager or speculator.speculative_config.enforce_eager):
            raise ValueError("Draft Window requires an eager Draft model; Target graph mode is independent")
        self.window_size = policy
        self.native_swa_causal: dict[int, bool] = {}
        self._dense_reads = False
        self.gids = self._validate_route(speculator)
        read_groups = tuple(gid for gid in self.gids if gid not in self.native_swa_causal) or self.gids
        self.anchor_block_span = int(speculator.block_tables.kernel_block_sizes[read_groups[0]])
        self._pending: list[object] = []
        self._poisoned = False
        self._owner: list[object] | None = None
        self._continuing: np.ndarray | None = None
        self._bypass = False
        self._build_count = 0
        self._eagle_window: tuple[tuple[int, int], ...] | None = None

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

        config = speculator.vllm_config
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
        all_gids = tuple(dict.fromkeys(route.gid for route in routes))
        native_gids = {route.gid for route in routes if route.status is DraftWindowProviderStatus.NATIVE_SWA}
        if native_gids:
            if self.window_size < speculator.num_query_per_req:
                raise ValueError("DFlash draft_window_size must cover its query width")
            declared = getattr(speculator, "_group_causal", True)
            self.native_swa_causal = {
                gid: declared[gid] if isinstance(declared, dict) else declared for gid in native_gids
            }
            routes = tuple(route for route in routes if route.gid not in self.native_swa_causal)
        self._dense_reads = bool(routes) and all(route.dense_read for route in routes)
        return all_gids

    @contextmanager
    def proposal_context(
        self,
        input_batch: Any,
        dummy_run: bool = False,
        is_profile: bool = False,
        *,
        context_kv_only: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> Iterator[None]:
        if self._owner is not None or self._bypass:
            raise RuntimeError("Draft Window proposal scope is already active")
        if context_kv_only and self.speculator.method not in ("dflash", "dspark"):
            raise ValueError("Context-only Draft KV updates require a parallel drafter")
        expected_eagle_steps = self.speculator.num_speculative_steps
        if num_speculative_tokens is not None:
            if (
                self.speculator.method != "eagle3"
                or type(num_speculative_tokens) is not int
                or not 0 <= num_speculative_tokens <= self.speculator.num_speculative_steps
            ):
                raise ValueError("Eagle3 scheduled draft count must stay within its configured static K")
            # A zero-draft AR proposal still runs prefill attention to update
            # KV, unlike the parallel drafter's context-only write branch.
            # Keep the configured K for window geometry; only completion's
            # expected number of consumed steps follows this real invocation.
            expected_eagle_steps = max(1, num_speculative_tokens)
        if dummy_run or is_profile or getattr(input_batch, "is_dummy", False):
            self._bypass = True
            try:
                yield
            finally:
                self._bypass = False
            return
        if self._poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        owner: list[object] = []
        self._owner = owner
        self._build_count = 0
        self._eagle_window = None
        try:
            # The worker preserves Host mapping and prompt progress throughout
            # propose; validate them once before native preparation.
            num_reqs = input_batch.num_reqs
            rows = input_batch.idx_mapping_np[:num_reqs]
            counts = self.speculator.block_tables.num_blocks.np
            if rows.shape != (num_reqs,) or np.any(rows < 0) or np.any(rows >= counts.shape[1]):
                raise ValueError("Draft allocation counts require active persistent request slots")
            self._continuing = continuing_prefill_rows(input_batch, num_reqs)
            # Preparation and builders enqueue copies/kernels before model
            # forward. Even a later host error must retain this proposal.
            yield
            # DFlash/DSpark keep context KV current when the scheduler
            # requests zero drafts. That upstream branch writes KV but
            # never executes draft attention or allocates Window operands.
            # Its context buffers remain owned by the original speculator.
            if context_kv_only and self._build_count:
                raise RuntimeError("Context-only Draft KV update unexpectedly consumed a read projection")
            if not context_kv_only and not self._build_count:
                raise RuntimeError("Draft Window proposal did not consume a read projection")
            if self.speculator.method == "eagle3" and self._build_count != expected_eagle_steps:
                raise RuntimeError("Eagle3 proposal did not consume every configured draft step")
        except BaseException:
            self._pending.extend(owner)
            self._poisoned = True
            raise
        else:
            if context_kv_only:
                self._pending.extend(owner)
            else:
                self._pending = owner
        finally:
            self._owner = None
            self._continuing = None
            self._eagle_window = None

    def metadata_callback(
        self,
        step: int,
        num_reqs: int,
        *,
        prepared_exact_cpu: torch.Tensor | None = None,
    ):
        """Own one metadata build; each group only chooses its read layout.

        Builders receive fresh common metadata in this proposal's batch order.
        Validate that boundary once, before either native Full, projected Host
        exact, or device-exact reads can mutate it. All routes retain operands until
        the next completed exact-copy boundary.
        """
        if self._bypass:
            return None
        owner = self._owner
        if owner is None:
            raise RuntimeError("Draft Window metadata has no active proposal")
        batch = self.speculator.input_batch
        if self.speculator.method != "eagle3" and step != self.speculator.num_query_per_req:
            raise ValueError("Parallel Draft metadata step must match its actual query width")
        if self.speculator.method == "eagle3" and (
            step != self._build_count or step >= self.speculator.num_speculative_steps
        ):
            raise ValueError("Eagle3 metadata must advance exactly one configured step")
        if batch.num_reqs != num_reqs:
            raise RuntimeError("Draft Window request mapping does not cover the active preparation")
        continuing = self._continuing
        assert continuing is not None
        counts = self.speculator.block_tables.num_blocks.np
        rows = batch.idx_mapping_np[:num_reqs]
        eagle = self.speculator.method == "eagle3"
        first = eagle and step == 0
        dense, native_full = self._dense_reads, False
        if dense:
            prefilling = batch.is_prefilling_np[:num_reqs]
            replicated = getattr(self.speculator, "replicated_pcp", False)
            drafts = batch.num_draft_tokens_per_req
            full_prompt = continuing.all() and not replicated and (drafts is None or not drafts[:num_reqs].any())
            if eagle:
                native_full = bool(continuing.any() and full_prompt)
                dense = not native_full and not (
                    (continuing.any() or first and prefilling.any()) and (prefilling.all() or replicated)
                )
            elif full_prompt:
                # A local prompt-only DP rank can still execute drafts when
                # another rank decodes. Keep large Full queries native, but
                # not adaptive preparations whose query extent is only upper.
                scheduled = batch.num_scheduled_tokens[:num_reqs]
                native_full = bool(
                    (batch.num_computed_prefill_tokens_np[:num_reqs] + scheduled <= self.speculator.max_model_len).all()
                    and np.array_equal(np.diff(batch.query_start_loc_np[: num_reqs + 1]), scheduled)
                )
                dense = not native_full

        def retain(*objects: object) -> None:
            if self._owner is not owner:
                raise RuntimeError("Draft Window callback belongs to a stale proposal")
            owner.extend(objects)

        exact = proposal_exact = starts = None

        def apply(gid: int, common: Any) -> None:
            nonlocal exact, proposal_exact, starts, dense
            if gid not in self.gids:
                return
            retain(common)
            # The real builder enumerates each KV group once and constructs a
            # fresh Common from this same preparation before invoking us.
            if exact is None:
                if common.num_reqs < num_reqs or common.seq_lens.ndim != 1 or common.seq_lens.numel() < num_reqs:
                    raise ValueError("Draft Window metadata does not cover the active requests")
                full_exact = common.seq_lens
                retain(full_exact)
                if dense and first and not batch.is_prefilling_np[:num_reqs].any():
                    dense = (
                        int(common.query_start_loc_cpu[: num_reqs + 1].diff().max())
                        <= self.speculator.num_speculative_steps + 1
                    )
                # Without Draft placeholders, continuing Full queries cannot
                # reject tokens in the native sampler: E is already the live end.
                if first and not native_full:
                    exact, proposal_exact = self._eagle_first_pass_lengths(full_exact, num_reqs)
                elif prepared_exact_cpu is not None:
                    # The actual DSpark MLA helper completed this exact copy
                    # immediately before constructing the group Common inputs.
                    exact = prepared_exact_cpu
                else:
                    # One exact D2H per build, shared across groups. The explicit
                    # DSpark MLA snapshot avoids this duplicate copy; other
                    # routes still synchronize here. An upper is never exact.
                    exact = full_exact[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
                retain(exact)
                # The worker prepares every ordinary Window consumer on the
                # worker execution stream. Its existing blocking exact copy
                # completes all earlier proposal readers on that same stream.
                # MLA supplies the copy completed immediately before this call.
                self._pending.clear()
                if not native_full:
                    if eagle and self._eagle_window is None:
                        if step > 0:
                            raise RuntimeError("Eagle3 AR metadata has no frozen first-pass plan")
                        assert proposal_exact is not None
                        self._eagle_window = tuple(
                            (
                                compute_draft_window_start(
                                    full_exact_end=int(end),
                                    window_size=self.window_size,
                                    block_span_tokens=self.anchor_block_span,
                                    future_offset=self.speculator.num_speculative_steps,
                                ),
                                int(end) + self.speculator.num_speculative_steps,
                            )
                            for end in proposal_exact
                        )
                    starts_list = []
                    for row, end in enumerate(exact):
                        if continuing[row] or (first and batch.is_prefilling_np[row]):
                            start = 0
                        elif self._eagle_window is not None:
                            start, horizon = self._eagle_window[row]
                            if not start <= int(end) <= horizon:
                                raise ValueError("Eagle3 exact end is outside its frozen window or proposal horizon")
                        else:
                            start = compute_draft_window_start(
                                full_exact_end=int(end),
                                window_size=self.window_size,
                                block_span_tokens=self.anchor_block_span,
                            )
                        starts_list.append(start)
                    starts = torch.tensor(starts_list, dtype=torch.int32, device="cpu")
                self._build_count += 1
            assert exact is not None
            if native_full:
                if not eagle:
                    columns = counts[gid, rows]
                    span = int(self.speculator.block_tables.kernel_block_sizes[gid])
                    if np.any((columns < 0) | (columns > common.block_table_tensor.shape[1])):
                        raise ValueError("Invalid initialized compact column extent")
                    if bool(((exact < 0) | (exact > common.seq_lens_cpu_upper_bound[:num_reqs])).any()) or any(
                        int(end) > int(count) * span for end, count in zip(exact, columns)
                    ):
                        raise ValueError("Invalid exact/upper ordering or unreadable canonical extent")
                exact_cpu = torch.zeros(common.num_reqs, dtype=torch.int32, device="cpu")
                exact_cpu[:num_reqs] = exact
                common.seq_lens_cpu = common._seq_lens_cpu = exact_cpu
                if not eagle:
                    # Parallel builders otherwise select device lengths again.
                    common._draft_window_exact_cpu = exact_cpu
            else:
                assert starts is not None
                table = common.block_table_tensor
                # The framework gathered this batch order before Target forward.
                # Canonical allocation counts remain indexed by persistent slots.
                columns = counts[gid, rows].tolist()
                block_span = int(self.speculator.block_tables.kernel_block_sizes[gid])
                read_starts = starts
                if gid in self.native_swa_causal:
                    if (
                        common.num_reqs != num_reqs
                        or common.seq_lens.dtype != torch.int32
                        or common.seq_lens.shape != (num_reqs,)
                        or common.causal is not self.native_swa_causal[gid]
                        or not bool(
                            (
                                common.query_start_loc_cpu[: num_reqs + 1].diff() == self.speculator.num_query_per_req
                            ).all()
                        )
                    ):
                        raise ValueError("Native SWA Window requires its declared causal mask and actual query width")
                    if continuing.all():
                        # Keep the native band/table, supplying the same exact CPU
                        # snapshot so its builder need not read device E again.
                        common.seq_lens_cpu = common._seq_lens_cpu = exact
                        common._draft_window_exact_cpu = exact
                        return
                    read_starts = torch.clamp(exact - self.window_size, min=0).to(torch.int64)
                    read_starts = read_starts // block_span * block_span
                    read_starts[torch.from_numpy(continuing)] = 0
                    if bool((exact - read_starts < self.speculator.num_query_per_req).any()):
                        raise ValueError("Extra Window must cover every native query")
                elif dense:
                    from vllm_ascend.worker.v2.spec_decode.draft_window_dense import prepare_dense_draft_read

                    boundaries = common.query_start_loc_cpu[: num_reqs + 1]
                    widths = boundaries.diff()
                    query_width = int(widths.max())
                    ragged = not bool((widths == query_width).all())
                    common._draft_window_device_read = prepare_dense_draft_read(
                        table,
                        columns,
                        block_span,
                        exact_cpu=proposal_exact if first else exact,
                        query_ends_cpu=exact,
                        upper_cpu=common.seq_lens_cpu_upper_bound[:num_reqs],
                        query_width=query_width,
                        causal=common.causal,
                        frozen_starts=starts,
                        full_rows_cpu=torch.tensor(
                            continuing | (batch.is_prefilling_np[:num_reqs] if first else False), device="cpu"
                        ),
                        query_start_loc_cpu=boundaries if ragged else None,
                        query_start_loc_device=common.query_start_loc[: num_reqs + 1] if ragged else None,
                        retain=retain,
                    )
                    return
                materialize_draft_read(
                    common,
                    columns,
                    block_span,
                    exact_cpu=exact,
                    starts_cpu=read_starts,
                    retain=retain,
                )

        return apply

    def _eagle_first_pass_lengths(self, native_extent: torch.Tensor, num_reqs: int):
        """Snapshot native Q extent and initialized live KV end together.

        Rejected first-pass query padding belongs to Q's causal alignment,
        not the readable KV interval. The frozen CPU plan uses the last valid
        query position, while the native extent retains its query coordinates.
        """
        positions = self.speculator.input_buffers.positions
        indices = self.speculator.last_token_indices[:num_reqs]
        # Fixed InputBuffers and AR allocate both same-device int64 vectors.
        owner = self._owner
        if owner is None:
            raise RuntimeError("Eagle3 length snapshot has no active Draft Window proposal")
        last_positions = positions.index_select(0, indices)
        owner.append(last_positions)
        live_end = last_positions.to(dtype=torch.int32)
        owner.append(live_end)
        live_end.add_(1)
        pair = torch.stack((native_extent[:num_reqs], live_end))
        owner.append(pair)
        snapshot = pair.to(device="cpu", dtype=torch.int32, copy=True)
        owner.append(snapshot)
        extent_cpu, live_cpu = snapshot[0], snapshot[1]
        if bool(
            (
                (live_cpu <= 0)
                | (live_cpu > extent_cpu)
                | (extent_cpu - live_cpu > self.speculator.num_speculative_steps)
            ).any()
        ):
            raise ValueError("Eagle3 prepared live queries disagree with the current native padding extent")
        return extent_cpu, live_cpu

    def retain_metadata(self, metadata: Any) -> None:
        """Keep the final builder result, including newly allocated derived state."""
        if self._bypass:
            return
        if self._owner is None:
            raise RuntimeError("Draft Window metadata has no active proposal")
        self._owner.append(metadata)
