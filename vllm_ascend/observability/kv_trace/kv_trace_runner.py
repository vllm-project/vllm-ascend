# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-specific KV Trace installation, batch ordering and graph boundaries."""

from __future__ import annotations

import functools
from typing import Any, cast

from vllm_ascend.observability.kv_trace.kv_trace_runtime import KVTraceRuntime, get_trace


def batch_values(batch: Any) -> dict[str, Any]:
    """Keep native coordinates; request state indices are not request row order."""
    if any(getattr(batch, name, None) is None for name in ("positions", "query_start_loc", "seq_lens")):
        return {}
    rows, tokens = batch.num_reqs, batch.num_tokens
    prefilling = getattr(batch, "is_prefilling_np", None)
    return dict(
        request_ids=list(batch.req_ids[:rows]),
        num_tokens=tokens,
        num_tokens_after_padding=batch.num_tokens_after_padding,
        query_start_loc=batch.query_start_loc[: rows + 1],
        positions=batch.positions[:tokens],
        seq_lens=batch.seq_lens[:rows],
        is_prefilling=None if prefilling is None else prefilling[:rows].copy(),
    )


class _V2RunnerAdapter:
    """Only V2 batch ordering, metadata timing and capture need adaptation."""

    def __init__(self, runner: Any, runtime: KVTraceRuntime):
        self.runner = runner
        self.runtime = runtime
        self.request_ids: list[str] = []

    def configure(self) -> None:
        prepare = self.runner.model_state.prepare_attn
        if getattr(prepare, "_ascend_kv_trace", False):
            return

        @functools.wraps(prepare)
        def observed(input_batch, cudagraph_mode, *args, **kwargs):
            metadata = prepare(input_batch, cudagraph_mode, *args, **kwargs)
            if self.runtime._phase is not None:
                self.runtime._finish_pending()
                self.request_ids = list(input_batch.req_ids[: input_batch.num_reqs])
                self.runtime._batch = batch_values(input_batch)
                context = getattr(self.runner.model_state, "pcp_context", None)
                self.runtime._parallel = (
                    {}
                    if context is None
                    else dict(
                        batch=batch_values(context.global_batch),
                        block_tables=context.global_block_tables,
                        slot_mappings=context.global_slot_mappings,
                        **{
                            name: getattr(context, name, None)
                            for name in (
                                "local_token_indices",
                                "local_to_global_req_indices",
                                "padded_gather_idx",
                                "gathered_kv_write_mask",
                            )
                        },
                    )
                )
                self.runtime._full_graph = "FULL" in str(cudagraph_mode)
                self.runtime.observe_metadata(metadata)
                scheduled = self.runtime._scheduled
                if self.runtime._phase == "forward" and scheduled is not None:
                    pcp = getattr(self.runtime.config.parallel_config, "prefill_context_parallel_size", 1) > 1
                    ids = self.runtime._parallel["batch"]["request_ids"] if self.runtime._parallel else self.request_ids
                    dynamic = self.runtime.config.speculative_config is not None
                    mismatch = (
                        not set(ids).issubset(scheduled)
                        if dynamic
                        else (len(ids) != len(set(ids)) or set(ids) != set(scheduled))
                    )
                    if (not pcp or self.runtime._parallel) and mismatch:
                        self.runtime.report(
                            "kv_trace.violation", operation="bind", reason="v2_batch_request_mismatch", request_ids=ids
                        )
                self.runtime._check_failure()
            return metadata

        cast(Any, observed)._ascend_kv_trace = True
        self.runner.model_state.prepare_attn = observed

    def wrap_execute(self, execute: Any):
        @functools.wraps(execute)
        def observed(scheduler_output, *args, **kwargs):
            # Forward positional and keyword arguments unchanged across vLLM lanes.
            flags = dict(zip(("intermediate_tensors", "dummy_run", "skip_attn_for_dummy_run", "is_profile"), args))
            flags.update(kwargs)
            dummy = bool(flags.get("dummy_run", False))
            profile = bool(flags.get("is_profile", False))
            phase = "warmup" if profile else "dummy" if dummy else "forward"
            required = bool(scheduler_output.total_num_scheduled_tokens) and not (
                profile and flags.get("skip_attn_for_dummy_run")
            )
            if dummy or profile:
                self.runtime._finish_pending()
            self.request_ids = []

            # Synthetic dummy/profile outputs have no scheduler binding, but
            # their write plans still obey capacity, PAD and native null-block rules.
            with (
                self.runtime.execute_scope(scheduler_output, bind=not (dummy or profile)),
                self.runtime.forward_scope(self, phase=phase, required=required),
            ):
                result = execute(scheduler_output, *args, **kwargs)
            return self.runtime._guard_output(result) if result is not None else result

        return observed

    def wrap_update(self, update: Any):
        @functools.wraps(update)
        def observed(scheduler_output, *args, **kwargs):
            result = update(scheduler_output, *args, **kwargs)
            # Record the native update return, without asserting KV content readiness.
            zeroed = getattr(scheduler_output, "new_block_ids_to_zero", None)
            if zeroed:
                self.runtime.zero_blocks(zeroed)
            copies = getattr(scheduler_output, "kv_cache_block_copies", None)
            if copies:
                self.runtime.copy_blocks(copies)
            return result

        return observed

    def wrap_capture(self, capture: Any):
        @functools.wraps(capture)
        def observed(*args, **kwargs):
            self.runtime._check_failure()
            self.runtime._finish_pending()
            # Do not copy metadata inside graph capture. Real steps are audited
            # by wrap_execute using their current metadata.
            return capture(*args, **kwargs)

        return observed


def install_trace(runner: Any) -> KVTraceRuntime | None:
    """Install the Model Runner V2 observer."""
    runtime = get_trace(runner.vllm_config)
    if runtime is None:
        return None
    adapter = _V2RunnerAdapter(runner, runtime)
    runtime._configure_runner = adapter.configure
    cast(Any, runner).execute_model = adapter.wrap_execute(runner.execute_model)
    cast(Any, runner).update_requests = adapter.wrap_update(runner.update_requests)
    cast(Any, runner).capture_model = adapter.wrap_capture(runner.capture_model)
    if hasattr(runner, "sample_tokens"):
        cast(Any, runner).sample_tokens = runtime.wrap_sample(runner.sample_tokens)
    return runtime
