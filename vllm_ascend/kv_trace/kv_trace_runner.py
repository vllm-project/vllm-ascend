# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-specific KV Trace installation, batch ordering and graph boundaries."""

from __future__ import annotations

import functools
from typing import Any, cast

from vllm_ascend.kv_trace.kv_trace_runtime import KVTraceRuntime, get_trace


class _V2RunnerAdapter:
    """Only V2 batch ordering, metadata timing and capture need adaptation."""

    def __init__(self, runner: Any, runtime: KVTraceRuntime):
        self.runner = runner
        self.runtime = runtime
        self.request_ids: list[str] = []

    def configure(self) -> None:
        tables = self.runner.block_tables
        sizes = [group.kv_cache_spec.block_size for group in self.runner.kv_cache_config.kv_cache_groups]
        if list(tables.block_sizes) != sizes or list(tables.kernel_block_sizes) != sizes:
            raise ValueError("KV Trace: unadapted MRV2 allocator/kernel block coordinates")
        prepare = self.runner.model_state.prepare_attn
        if getattr(prepare, "_ascend_kv_trace", False):
            return

        @functools.wraps(prepare)
        def observed(input_batch, cudagraph_mode, *args, **kwargs):
            metadata = prepare(input_batch, cudagraph_mode, *args, **kwargs)
            if self.runtime._phase is not None:
                self.runtime._finish_pending()
                self.request_ids = list(input_batch.req_ids[: input_batch.num_reqs])
                self.runtime._full_graph = "FULL" in str(cudagraph_mode)
                self.runtime.observe_metadata(metadata)
                scheduled = self.runtime._scheduled
                if self.runtime._phase == "forward" and scheduled is not None:
                    ids = self.request_ids
                    if len(ids) != len(set(ids)) or set(ids) != set(scheduled):
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
    """Install the MRV2 observer; MRV1 is left unchanged."""
    if not getattr(runner.vllm_config, "use_v2_model_runner", False):
        return None
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
