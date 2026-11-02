"""Worker interface contract registry for RFC #18920 (Phase 0).

RFC: https://github.com/vllm-project/vllm-ascend/issues/18920

This module is the machine-readable half of the Phase 0 deliverable. It records,
for every method that :class:`vllm_ascend.worker.worker.NPUWorker` (and its two
subclasses) defines, whether that method is meant to disappear in favour of the
upstream ``vllm.v1.worker.gpu_worker.Worker`` (``GPUWorker``), or whether it is a
permanent Ascend specialization.

``tests/ut/worker/test_worker_contract.py`` enforces three properties:

1. **Coverage** - every override is registered here. A new override cannot be
   added without classifying it, which is the "forbidden override" CI gate from
   the RFC.
2. **Resolution** - an ``INHERIT`` entry must actually resolve to the parent, and
   a quarantined entry must still be overridden. This is the list that grows
   phase by phase; when Phase 2 removes a thin override, its entry moves to
   ``INHERIT`` in the same pull request.
3. **Contract compatibility** - a quarantined override must still accept every
   call the parent accepts (signature) and must not lose its parent decorators.
   Known gaps are enumerated in ``KNOWN_SIGNATURE_GAPS`` /
   ``KNOWN_DECORATOR_GAPS`` and must match reality exactly, so fixing a gap and
   forgetting to update the list fails the test as loudly as introducing a new
   one.

How to update the registry
--------------------------
* Phase 2/3/4/5 removes an override -> change its disposition to ``INHERIT`` and
  keep the phase that removed it. The resolution test then proves that the
  method really comes from the parent.
* Phase 0.5 fixes a known signature or decorator gap -> delete the entry from
  ``KNOWN_SIGNATURE_GAPS`` / ``KNOWN_DECORATOR_GAPS``.
* A new Ascend-only helper -> add an ``ASCEND_SPECIFIC`` entry with the concrete
  NPU behaviour it exists for. "Historical reason" is not a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: RFC phases that can settle an override. ``keep`` means the override is a
#: permanent Ascend specialization.
PHASES = frozenset({"0.5a", "0.5b", "0.5c", "1a", "1b", "2", "3", "4", "5", "keep"})

#: Phases that remove an override.
REMOVAL_PHASES = frozenset({"2", "3", "4", "5"})


class Disposition(str, Enum):
    """What is supposed to happen to an override."""

    #: Already resolved from the parent Worker; the override is gone.
    INHERIT = "inherit"
    #: Redundant copy of the parent; delete it in the recorded phase.
    PENDING_REMOVAL = "pending-removal"
    #: Behaviour has diverged; reconcile with the parent contract first.
    ALIGN_FIRST = "align-first"
    #: Permanent Ascend specialization; explain the concrete NPU behaviour.
    ASCEND_SPECIFIC = "ascend-specific"


@dataclass(frozen=True, slots=True)
class WorkerInterface:
    name: str
    disposition: Disposition
    phase: str
    reason: str


def _i(name: str, disposition: Disposition, phase: str, reason: str) -> WorkerInterface:
    return WorkerInterface(name=name, disposition=disposition, phase=phase, reason=reason)


#: ``NPUWorker`` overrides. Phase 0 records the current state, so no entry is
#: ``INHERIT`` yet: every quarantined override is still defined downstream.
NPU_WORKER_INTERFACES: tuple[WorkerInterface, ...] = (
    _i(
        "__init__",
        Disposition.ASCEND_SPECIFIC,
        "1a",
        "Ascend bootstrap (ascend config, custom ops, CaMem and sleep managers, static-kernel "
        "signal handlers) runs before the parent constructor. Phase 1a must chain into "
        "GPUWorker.__init__ instead of WorkerBase.__init__, and review the state the parent adds "
        "(elastic EP executor, fault-tolerance sentinel, profiler validation, sleep state).",
    ),
    _i(
        "uninstall_static_kernel",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "npugraph_ex static-kernel uninstall driven by ASCEND_HOME_PATH; no upstream counterpart.",
    ),
    _i(
        "_save_sleep_parameters",
        Disposition.ALIGN_FIRST,
        "3",
        "Parent also retains draft-model parameters and derives them through _get_sleep_parameters; "
        "align before inheriting level-2 sleep.",
    ),
    _i(
        "_restore_sleep_parameters",
        Disposition.ALIGN_FIRST,
        "3",
        "Parent restores through _get_sleep_parameters so retained draft parameters are covered.",
    ),
    _i(
        "sleep",
        Disposition.ALIGN_FIRST,
        "3",
        "Uses CaMemAllocator.sleep instead of the sleep_mode_backend, and skips KV-cache "
        "unregistration plus device-comm suspension.",
    ),
    _i(
        "wake_up",
        Disposition.ALIGN_FIRST,
        "3",
        "Adds the FRACTAL_NZ guard and CaMem wake_up; the parent also restores draft buffers and "
        "synchronizes the device.",
    ),
    _i(
        "_check_weight_transfer_engine",
        Disposition.PENDING_REMOVAL,
        "3",
        "Identical to the parent implementation.",
    ),
    _i(
        "init_weight_transfer_engine",
        Disposition.PENDING_REMOVAL,
        "3",
        "Identical to the parent implementation.",
    ),
    _i(
        "_check_nz_disabled",
        Disposition.ASCEND_SPECIFIC,
        "3",
        "Ascend-only FRACTAL_NZ guard; Phase 3 keeps it as the narrow pre-update hook.",
    ),
    _i(
        "start_weight_update",
        Disposition.ALIGN_FIRST,
        "3",
        "Narrower state machine: no current-config context, no draft target, no cleanup on failure.",
    ),
    _i(
        "update_weights",
        Disposition.ALIGN_FIRST,
        "3",
        "No DP-indexed payload selection and no cleanup after a failed update.",
    ),
    _i(
        "finish_weight_update",
        Disposition.ALIGN_FIRST,
        "3",
        "Does not reset LoRA state or restore quantized draft heads after the session.",
    ),
    _i(
        "shutdown",
        Disposition.ALIGN_FIRST,
        "5",
        "Does not release parent-owned resources (elastic EP executor, model runner, profiler) or "
        "follow the parent pool-teardown order.",
    ),
    _i(
        "initialize_cache",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "V0-era cache_config setter kept for compatibility; the upstream Worker has no counterpart.",
    ),
    _i(
        "_init_device",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "NPU device selection, triton device properties, memory snapshot and HCCL bootstrap.",
    ),
    _i(
        "init_device",
        Disposition.ASCEND_SPECIFIC,
        "5",
        "The parent implementation raises for non-CUDA devices. Phase 5 should keep the NPU device "
        "path but reuse _init_workspace_and_model_runner and _make_model_runner.",
    ),
    _i(
        "_apply_kv_offload_decode_memory_constraints",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Sparse KV offload decode memory policy.",
    ),
    _i(
        "_apply_kvpp_memory_budget",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "KVPP memory budget accounting.",
    ),
    _i(
        "determine_available_memory",
        Disposition.ALIGN_FIRST,
        "5",
        "Accounts memory from total_memory * gpu_memory_utilization instead of the parent "
        "request_memory and startup-plan contract.",
    ),
    _i(
        "_scale_kv_cache_memory_for_multi_group",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "KV cache scaling across multiple NPU memory groups.",
    ),
    _i(
        "log_memory_stats",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "NPU memory logging used by execute_model and the dummy-batch path.",
    ),
    _i(
        "execute_model",
        Disposition.ALIGN_FIRST,
        "0.5a",
        "Misses inference_mode and with_gpu_sync_check, keeps every PP send handle instead of "
        "handles[1:], and has no annotate_profile context. Align in 0.5a, inherit in Phase 4 once "
        "SP, KV-connector and msMonitor differences have hooks.",
    ),
    _i(
        "sample_tokens",
        Disposition.ALIGN_FIRST,
        "0.5a",
        "Keeps @torch.inference_mode but not the @with_gpu_sync_check layer, and attaches profiling-chunk "
        "statistics. Align decorators in 0.5a, inherit once the statistics have an extension point.",
    ),
    _i(
        "load_model",
        Disposition.ALIGN_FIRST,
        "0.5c",
        "Signature lacks the keyword-only load_dummy_weights argument, and the body uses the CaMem "
        "weights pool. Widen the signature in 0.5c, converge in Phase 5.",
    ),
    _i(
        "compile_or_warm_up_model",
        Disposition.ALIGN_FIRST,
        "5",
        "Full downstream copy of the warmup flow; the parent now splits _compile_or_warm_up_model "
        "and wraps it for MRV2 eplb suppression.",
    ),
    _i(
        "_warm_up_atb",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Warms up ATB kernels for the Ascend graph path.",
    ),
    _i(
        "get_model",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to model_runner.get_model, identical to the parent.",
    ),
    _i(
        "profile_prefill_latency",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Prefill latency probing used by the Ascend scheduler.",
    ),
    _i(
        "get_kv_connector_handshake_metadata",
        Disposition.ALIGN_FIRST,
        "5",
        "Adds PCP-aware handshake metadata on top of the parent contract.",
    ),
    _i(
        "get_simple_cpu_offload_num_cpu_blocks",
        Disposition.ALIGN_FIRST,
        "5",
        "Keeps a vLLM 0.32.0 compatibility lane alongside the parent implementation.",
    ),
    _i(
        "_get_layerwise_kv_cache_memory_info",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Layerwise KV reuse memory accounting.",
    ),
    _i(
        "get_kv_cache_spec",
        Disposition.ALIGN_FIRST,
        "5",
        "Adds KVPP and layerwise-reuse sizing on top of the parent delegation.",
    ),
    _i(
        "update_max_model_len",
        Disposition.PENDING_REMOVAL,
        "2",
        "Identical to the parent implementation.",
    ),
    _i(
        "_maybe_get_memory_pool_context",
        Disposition.ALIGN_FIRST,
        "3",
        "Wraps the CaMem allocator instead of the parent cumem/xpu/cpu branches.",
    ),
    _i(
        "initialize_from_config",
        Disposition.ALIGN_FIRST,
        "5",
        "Does not apply the parent cache_config updates (num_gpu_blocks, KV layout, hash_block_size, "
        "cache_hit_alignment_tokens) before allocating the KV cache.",
    ),
    _i(
        "profile",
        Disposition.ASCEND_SPECIFIC,
        "0.5c",
        "NPU profiler wrapper (TorchNPUProfilerWrapper). Widen the signature with "
        "delay_iterations/max_iterations in 0.5c so it stays assignable to the parent.",
    ),
    _i(
        "add_lora",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "remove_lora",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "list_loras",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "pin_lora",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "reset_encoder_cache",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "synchronize_device",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Synchronizes through torch.npu; the parent Worker has no counterpart (WorkerBase does).",
    ),
    _i(
        "get_encoder_cudagraph_stats",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "NPU encoder graph diagnostics RPC.",
    ),
    _i(
        "execute_dummy_batch",
        Disposition.ALIGN_FIRST,
        "0.5c",
        "Passes skip_gdn_state_update and has no randomize_inputs contract; align the signature and "
        "the _dummy_run contract in 0.5c.",
    ),
    _i(
        "_init_worker_distributed_environment",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "HCCL distributed init plus Ascend model-parallel groups.",
    ),
    _i(
        "get_supported_pooling_tasks",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Worker-level pooling task report consumed by the Ascend runner; the parent exposes only "
        "get_supported_tasks, so there is nothing to inherit.",
    ),
    _i(
        "get_supported_tasks",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "take_draft_token_ids",
        Disposition.PENDING_REMOVAL,
        "2",
        "Pure delegation to the model runner, identical to the parent.",
    ),
    _i(
        "update_config",
        Disposition.ALIGN_FIRST,
        "0.5b",
        "Forwards to the model runner only; the parent replaces the config object on vllm_config, "
        "the worker and the model runner.",
    ),
    _i(
        "reload_weights",
        Disposition.ALIGN_FIRST,
        "0.5a",
        "Calls the model runner without set_current_vllm_config; align in 0.5a, inherit in Phase 2.",
    ),
    _i(
        "check_health",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Probes device health through npu-smi.",
    ),
)

#: ``NPUWorker310`` overrides. Its parent is ``NPUWorker``, so only coverage and
#: documentation are enforced here.
NPU_WORKER_310_INTERFACES: tuple[WorkerInterface, ...] = (
    _i(
        "_scale_kv_cache_memory_for_multi_group",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "310P overrides the multi-group scaling for its RC memory policy.",
    ),
    _i(
        "_create_model_runner",
        Disposition.ALIGN_FIRST,
        "5",
        "Local runner factory; the parent exposes _make_model_runner and Phase 5 should converge.",
    ),
    _i(
        "init_device",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "310P device bootstrap on top of the NPU implementation.",
    ),
    _i(
        "save_sharded_state",
        Disposition.ALIGN_FIRST,
        "5",
        "310P sharded-state variant; verify against the parent contract before inheriting.",
    ),
    _i(
        "determine_available_memory",
        Disposition.ALIGN_FIRST,
        "5",
        "310P RC memory policy on top of the NPU accounting.",
    ),
    _i(
        "_warm_up_atb",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "310P ATB kernel warmup variant.",
    ),
    _i(
        "_init_device",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "310P device selection.",
    ),
)

#: ``XliteWorker`` overrides.
XLITE_WORKER_INTERFACES: tuple[WorkerInterface, ...] = (
    _i(
        "init_device",
        Disposition.ASCEND_SPECIFIC,
        "keep",
        "Xlite device bootstrap on top of the NPU implementation.",
    ),
    _i(
        "compile_or_warm_up_model",
        Disposition.ALIGN_FIRST,
        "5",
        "Xlite warmup variant; converge on the parent flow in Phase 5.",
    ),
)

#: Overrides whose local signature does not yet accept every call the parent
#: accepts. Phase 0.5c removes them one by one.
KNOWN_SIGNATURE_GAPS = frozenset({"load_model", "profile"})

#: Overrides that lost (or never adopted) a parent decorator. Phase 0.5a fixes
#: them. ``sample_tokens`` is added at test time only where
#: ``with_gpu_sync_check`` actually wraps, since it is a no-op off CUDA.
KNOWN_DECORATOR_GAPS = frozenset({"execute_model"})

#: Decorated methods the guard inspects. Extending this set is how later phases
#: pull more decorators under CI.
DECORATOR_GUARDED_METHODS = frozenset({"execute_model", "sample_tokens"})
