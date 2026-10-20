# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 batch adapter for the shared KV address checker.

Batch IDs come from prepare_attn after sorting, not RequestState row order.
Metadata is retained by reference and copied only after forward/graph updates.
"""

import functools
from types import SimpleNamespace
from typing import Any, cast

from vllm_ascend.debug.kv_guard_runtime import GuardRuntime, get_guard


class KVGuardV2:
    def __init__(self, runner: Any, runtime: GuardRuntime):
        self.runner = runner
        self.runtime = runtime
        self.input_batch = SimpleNamespace(req_ids=[])
        self._prepared = False

    def configure(self) -> None:
        tables = self.runner.block_tables
        sizes = [group.kv_cache_spec.block_size for group in self.runner.kv_cache_config.kv_cache_groups]
        if list(tables.block_sizes) != sizes or list(tables.kernel_block_sizes) != sizes:
            raise ValueError("KV guard: unadapted MRV2 allocator/kernel block coordinates")
        self.runtime.configure(self.runner)
        prepare = self.runner.model_state.prepare_attn
        if getattr(prepare, "_ascend_kv_guard", False):
            return

        @functools.wraps(prepare)
        def observed(input_batch, cudagraph_mode, *args, **kwargs):
            metadata = prepare(input_batch, cudagraph_mode, *args, **kwargs)
            if self.runtime._phase is not None:
                self.input_batch = SimpleNamespace(req_ids=list(input_batch.req_ids[: input_batch.num_reqs]))
                self._prepared = True
                self.runtime._full_graph = "FULL" in str(cudagraph_mode)
                self.runtime.observe_metadata(metadata)
                scheduled = self.runtime._scheduled
                if self.runtime._phase == "forward" and scheduled is not None:
                    ids = self.input_batch.req_ids
                    if len(ids) != len(set(ids)) or set(ids) != set(scheduled):
                        self.runtime.gap("v2_batch_request_mismatch")
                        self.runtime._missing_groups.update(self.runtime.layer_groups.values())
                self.runtime._check_failure()
            return metadata

        cast(Any, observed)._ascend_kv_guard = True
        self.runner.model_state.prepare_attn = observed

    def wrap_execute(self, execute: Any):
        def forward(output, *args, **kwargs):
            # Preserve positional and keyword calls across the supported vLLM
            # lanes; forward every argument unchanged to the original method.
            flags = dict(zip(("intermediate_tensors", "dummy_run", "skip_attn_for_dummy_run", "is_profile"), args))
            flags.update(kwargs)
            dummy = bool(flags.get("dummy_run", False))
            profile = bool(flags.get("is_profile", False))
            phase = "warmup" if profile else "dummy" if dummy else "forward"
            required = bool(output.total_num_scheduled_tokens) and not (
                profile and flags.get("skip_attn_for_dummy_run")
            )
            self.input_batch = SimpleNamespace(req_ids=[])
            self._prepared = False

            with self.runtime.forward_scope(self, phase=phase, required=required):
                result = execute(output, *args, **kwargs)
                if phase == "forward" and required and not self._prepared:
                    self.runtime.gap("v2_attention_metadata_missing")
                    self.runtime._missing_groups.update(self.runtime.layer_groups.values())
                return result

        execute_bound = self.runtime.wrap_execute(forward)

        @functools.wraps(execute)
        def observed(scheduler_output, *args, **kwargs):
            flags = dict(zip(("intermediate_tensors", "dummy_run", "skip_attn_for_dummy_run", "is_profile"), args))
            flags.update(kwargs)
            # Dummy/profile outputs are synthesized without a scheduler binding.
            # They must still audit writes against any existing live KV state.
            if flags.get("dummy_run", False) or flags.get("is_profile", False):
                self.runtime._scheduled = None
                return forward(scheduler_output, *args, **kwargs)
            return execute_bound(scheduler_output, *args, **kwargs)

        return observed

    def wrap_update(self, update: Any):
        @functools.wraps(update)
        def observed(scheduler_output, *args, **kwargs):
            result = update(scheduler_output, *args, **kwargs)
            # The parent has completed the physical operations at this point.
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
            with self.runtime.state._lock:
                if (
                    self.runtime.state.requests
                    or self.runtime.state.pending_loads
                    or any(block.valid_end != 0 for block in self.runtime.state.blocks.values())
                ):
                    raise ValueError(
                        "KV guard: capture requires empty KV state; live/cache-preserving recapture is unsupported"
                    )
            # Startup capture has no request KV to protect. Do not synchronize
            # metadata to the CPU inside a device graph capture. Real replays
            # are audited by wrap_execute using their current metadata.
            return capture(*args, **kwargs)

        return observed


def install_guard(runner: Any) -> KVGuardV2 | None:
    runtime = get_guard(runner.vllm_config)
    if runtime is None:
        return None
    adapter = KVGuardV2(runner, runtime)
    cast(Any, runner).execute_model = adapter.wrap_execute(runner.execute_model)
    cast(Any, runner).update_requests = adapter.wrap_update(runner.update_requests)
    cast(Any, runner).capture_model = adapter.wrap_capture(runner.capture_model)
    return adapter
