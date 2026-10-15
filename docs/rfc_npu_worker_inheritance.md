# [RFC]: Make NPUWorker Inherit from the vLLM GPU Worker

## Summary

`NPUWorker` currently inherits directly from `WorkerBase` and maintains its own
copies of many vLLM GPU Worker control flows. This RFC proposes inheriting from
`vllm.v1.worker.gpu_worker.Worker` (called `GPUWorker` below), then removing
duplicated overrides in stages grouped by interface type. Ascend-specific
device setup, memory management, communication, graph capture, and profiling
remain in vllm-ascend.

The intended result is that a fix or feature implemented entirely in an
inherited upstream Worker method becomes available after vllm-ascend upgrades
to that vLLM version, without a downstream copy of the change. Inheritance
alone does not make new ModelRunner contracts or CUDA-specific behavior work
on NPU.

## Baseline and Motivation

This proposal compares the official `main` branches as of October 15, 2026:

| Repository | Commit | Relevant source |
| --- | --- | --- |
| vllm-ascend | [`f73001e0`](https://github.com/vllm-project/vllm-ascend/tree/f73001e0b9b82419d34eb23919a5eef8e6018682) | [`NPUWorker`](https://github.com/vllm-project/vllm-ascend/blob/f73001e0b9b82419d34eb23919a5eef8e6018682/vllm_ascend/worker/worker.py#L126) |
| vLLM | [`7020600b`](https://github.com/vllm-project/vllm/tree/7020600b1db32f239611b037f07e0dc5268e5b4f) | [`gpu_worker.py::Worker`](https://github.com/vllm-project/vllm/blob/7020600b1db32f239611b037f07e0dc5268e5b4f/vllm/v1/worker/gpu_worker.py#L192) |

Both classes inherit from `WorkerBase`. Their implementations overlap in LoRA
forwarding, weight reload, KV-cache setup, model execution, pipeline-parallel
communication, and lifecycle management. The copies have already diverged:

- Upstream `update_config` updates the shared config, Worker, and ModelRunner;
  Ascend currently forwards only to the ModelRunner.
- Upstream `reload_weights` sets the current vLLM config context; Ascend
  currently forwards directly.
- Upstream weight updates support DP-indexed payloads, draft-model updates,
  and cleanup after failure; Ascend maintains a narrower state machine.
- Upstream `execute_model` retains only the device handles from a
  pipeline-parallel send for the next step; Ascend retains all returned
  handles and separately handles V1 KV-connector output.

These differences motivate reuse, but they also prevent a one-line base-class
change. Upstream [`init_device`](https://github.com/vllm-project/vllm/blob/7020600b1db32f239611b037f07e0dc5268e5b4f/vllm/v1/worker/gpu_worker.py#L442)
currently accepts only CUDA. Its default sleep backend uses a CuMem-based
allocator, while Ascend uses NPU initialization, HCCL, CaMemAllocator, NPU
Graph, and specialized KV-cache allocation.

The [vLLM main commit verified by the Ascend baseline](https://github.com/vllm-project/vllm-ascend/blob/f73001e0b9b82419d34eb23919a5eef8e6018682/.github/vllm-main-verified.commit)
is `c4df37df`, not the vLLM `main` commit analyzed here. Implementation must
recheck every interface against the actual supported dependency and coordinate
any dependency update separately.

## Goals

1. Reuse upstream Worker orchestration where its contract is independent of
   the accelerator.
2. Remove downstream overrides once their Ascend behavior is represented by
   a smaller device-specific hook, backend, or ModelRunner method.
3. Receive upstream Worker-only fixes and features through normal dependency
   upgrades, with no downstream copy of those methods.
4. Make the remaining overrides explainable by concrete Ascend behavior.

## Non-goals

- Rework the full ModelRunner inheritance hierarchy.
- Promise that every new GPU Worker feature works on NPU automatically.
- Remove Ascend-specific KVPP, layer reuse, sparse KV offload, 310P memory
  policy, ATB warmup, NPU profiling, or device health checks.

## Design Principles

1. **Inherit orchestration; own device mechanisms.** The parent Worker should
   own common sequencing and state transitions. Ascend should own device
   selection, memory allocation, graph capture, HCCL integration, and policy
   that is specific to NPU.
2. **Measure reuse by removed overrides.** Calling `super()` from a copied
   downstream method does not ensure future upstream changes are inherited.
3. **Verify contracts before deleting overrides.** Check parameters, return
   values, initialized attributes, ModelRunner methods, and device assumptions
   for every inherited method.
4. **Prefer narrow upstream hooks.** When a common flow has one
   device-specific step, contribute a hook upstream and override that step
   rather than keep a downstream copy of the whole flow.
5. **State the inheritance boundary precisely.** Automatic adoption applies
   only to methods actually resolved from the parent, for a compatible vLLM
   version and ModelRunner contract.

## Interface Groups and Target State

| Group | Representative methods | Target state |
| --- | --- | --- |
| Simple forwarding and control plane | `get_model`, `get_supported_tasks`, `take_draft_token_ids`, LoRA methods, `reset_encoder_cache`, `update_max_model_len`, `reload_weights`, `update_config` | Remove equivalent overrides one by one. Verify NPU ModelRunner config-update behavior before inheriting `update_config`. |
| RL and dynamic weights | `sleep`, `wake_up`, `discard`, `reload_weights`, `init_weight_transfer_engine`, `start_weight_update`, `update_weights`, `finish_weight_update`, draft weight updates | Reuse the upstream state machine and dispatch. Keep CaMem, NZ validation, and extra cleanup behind a backend or narrow hook. |
| Basic execution | `execute_model`, `sample_tokens`, `execute_dummy_batch` | Inherit common execution and PP communication when SP, V1 KV-connector, monitoring, and sampling-statistics differences have extension points. |
| Initialization and model loading | `__init__`, `init_device`, `load_model`, ModelRunner construction | Inherit common Worker state. Keep NPU device and distributed initialization. Use upstream `_make_model_runner` where possible. |
| Memory and KV cache | `determine_available_memory`, `get_kv_cache_spec`, `initialize_from_config`, memory-pool methods | Retain Ascend overrides initially; extract shared steps before removing any of them. |
| Compilation, observability, and shutdown | `compile_or_warm_up_model`, `profile`, `shutdown`, `check_health` | Keep NPU Graph, ATB, NPU profiler, health checks, and device cleanup while identifying reusable lifecycle steps. |

## Implementation Plan

Each phase is independently reviewable. An override is removed only after its
replacement preserves the applicable V1, V2, 310P, and Xlite behavior.

### Phase 0: Pin Contracts and Establish a Regression Baseline

For the vLLM commit actually supported by the implementation, inventory every
NPUWorker override and its upstream counterpart. Record its signature, parent
calls, required attributes, ModelRunner dependencies, and device assumptions.
Include the subclasses [`NPUWorker310`](https://github.com/vllm-project/vllm-ascend/blob/f73001e0b9b82419d34eb23919a5eef8e6018682/vllm_ascend/_310p/worker_310p.py#L32)
and [`XliteWorker`](https://github.com/vllm-project/vllm-ascend/blob/f73001e0b9b82419d34eb23919a5eef8e6018682/vllm_ascend/xlite/xlite_worker.py#L22).

Add hardware-independent contract tests for method resolution, initialized
state, important ModelRunner calls, and failure paths. In particular, verify
the V1 and V2 contracts for `load_model`, `profile_run`, `_dummy_run`,
`reload_weights`, and KV-cache initialization. Set the target vLLM version and
dependency-upgrade order.

**Exit criterion:** Every override proposed for deletion has a documented
parent contract and regression coverage. Version-dependent differences are
listed before changing the class hierarchy.

### Phase 1: Change the Base Class Without Migrating Behavior

Make the class derive from the upstream Worker, using an explicit alias:

```python
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class NPUWorker(GPUWorker):
    ...
```

Keep existing NPU method overrides initially. Call the upstream Worker
constructor so that inherited methods can rely on its state. Review its
profiler validation, fault-tolerance sentinel, elastic EP executor, sleep
state, and weight-update state on Ascend. Calling `WorkerBase.__init__`
directly would bypass that state and is not an acceptable shortcut.

Keep Ascend `init_device`: calling the current parent implementation would
raise for NPU. Keep the existing initialization paths of `NPUWorker310` and
`XliteWorker` until their contracts are migrated.

**Exit criterion:** The hierarchy changes without changing the existing
startup and basic inference behavior, and every new parent-owned field has
been reviewed.

### Phase 2: Migrate Forwarding and Configuration Interfaces

Remove equivalent thin overrides first: LoRA methods, `get_model`,
`get_supported_tasks`, `take_draft_token_ids`, `reset_encoder_cache`, and
`update_max_model_len`. Then inherit `reload_weights` so the upstream config
context is used.

Handle `update_config` separately. Its current upstream and Ascend semantics
differ; verify that both NPU ModelRunners behave correctly after the upstream
method replaces config objects on the shared config, Worker, and ModelRunner.
Expose newly inherited upstream utility methods only after their ModelRunner
contract is verified.

**Exit criterion:** Migrated methods resolve to the parent in the method
resolution order. Config updates and weight reload preserve consistent state.

### Phase 3: Migrate RL and Dynamic-Weight Interfaces

Unify weight-transfer state transitions, including connection shutdown,
cleanup after failure, selection of DP-indexed update data, and optional
draft-model updates. Move Ascend `weight_nz_mode` validation into a narrow
pre-update hook instead of retaining full downstream copies of
`start_weight_update`, `update_weights`, and `finish_weight_update`. Enable
draft updates only when the Ascend transfer engine and ModelRunner support
their upstream contract.

Adapt sleep separately. Register or implement a CaMem-compatible sleep
backend, then verify levels 1 and 2, selective tags, parameter and buffer
restore, KV-connector registration, HCCL communication, and the ordering of
`SleepWakeupManager` extra cleanup. Retain NPU `sleep` and `wake_up` until
these semantics match. The [upstream sleep backend abstraction](https://github.com/vllm-project/vllm/blob/7020600b1db32f239611b037f07e0dc5268e5b4f/vllm/device_allocator/sleep_mode_backend.py)
provides a possible integration point, but its default backend is not an
Ascend implementation.

**Exit criterion:** Ordinary RL weight updates use the upstream state machine.
Sleep overrides are removed only after each level and tag behavior is covered
by contract tests.

### Phase 4: Migrate the Basic Execution Flow

The common part of `execute_model` waits for previous PP sends, receives
intermediate tensors, invokes the ModelRunner, sends intermediate tensors, and
handles pooling. Ascend additionally selects `all_gather_group=None` for SP,
propagates V1 KV-connector output, advances msMonitor, and logs NPU memory.
Introduce acceptable upstream extension points for these differences before
inheriting the common flow.

Keep the Ascend `sample_tokens` override until profiling-chunk statistics can
be attached without copying the parent method. Keep `execute_dummy_batch`
until its runtime guard and `skip_gdn_state_update` requirement have an
equivalent extension point. Do not claim this phase complete by copying a
newer upstream `execute_model` into the subclass.

**Exit criterion:** Ordinary execution resolves to the parent Worker. Tests
cover PP, SP, pooling, V1/V2, KV connectors, and empty batches.

### Phase 5: Converge Initialization, Memory, KV Cache, and Lifecycle

Use the existing upstream `_make_model_runner` extension point for Ascend
V1/V2, 310P, and Xlite ModelRunner construction. Keep NPU device selection and
HCCL initialization in `init_device`; selectively reuse workspace and usage
reporting steps without entering the upstream CUDA branch.

Evaluate memory profiling, KV-cache specifications and allocation, NPU Graph
warmup, profiling, and shutdown method by method. KVPP, layer reuse, sparse
KV offload, 310P RC memory policy, ATB warmup, and the NPU profiler are
device-specific. Where common steps justify reuse, split out upstream hooks;
otherwise retain documented overrides.

**Exit criterion:** Every remaining override identifies a concrete Ascend
difference. No large upstream control flow remains copied solely for
historical reasons.

## Validation and Rollout

Submit each phase separately for review. Focus regression coverage on method
resolution, state transitions, error cleanup, and V1/V2/310P/Xlite interface
compatibility. Use local unit tests and normal CI. Run remote NPU validation
only when explicitly requested, following the repository's local validation
rules.

Maintain a short CI-enforced list of methods expected to resolve from the
parent Worker. If a future change reintroduces an Ascend override for one of
those methods, require a review of the reason. Each vLLM upgrade must still
inspect newly introduced device assumptions, but fixes confined to inherited
Worker methods should no longer need manual downstream ports.

## Risks

| Risk | Mitigation |
| --- | --- |
| A parent method gains a CUDA-specific assumption | Pin the dependency, run contract tests, and add a narrow upstream hook where needed. |
| Same-named ModelRunner methods have different signatures or side effects | Check each contract before deleting an override. |
| Inheritance exposes new interfaces such as `discard`, draft updates, state export, or elastic EP | Mark each as supported, conditionally supported, or unsupported after explicit review. |
| Changes to common state or startup affect 310P or Xlite | Include both subclasses in every relevant phase's regression checks. |

The intended inheritance benefit is limited to Worker logic proven to be
device-independent and actually inherited by NPUWorker.
