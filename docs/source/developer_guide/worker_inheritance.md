# Worker inheritance contracts

RFC: <https://github.com/vllm-project/vllm-ascend/issues/18920>.

## Dependency order

The migration targets the verified vLLM main lane, based on `24420b7a` plus
<https://github.com/vllm-project/vllm/pull/65565>. The verification pin records
that exact companion commit so the parent contracts remain reproducible.
Merge the vLLM companion first, update the pin to the merged commit, then merge
Ascend phases 0.5a, 0.5b, 0.5c, 1a, 1b, 2, 3, 4 and 5 in order.
The older v0.32.0 lane needs the same Worker-hook backport before this migration
can be released on that lane. Missing hooks fail before the parent constructor;
there is no fallback that bypasses the parent constructor.

## Parent-owned constructor state (phase 1a)

| State | Ascend contract |
| --- | --- |
| Shared config and ranks | Initialized by the real GPU Worker constructor |
| Float32 matmul precision | Adopt upstream `VLLM_FLOAT32_MATMUL_PRECISION` |
| Elastic EP executor | Explicit unsupported executor, no CUDA imports or thread pool |
| Fault-tolerance sentinel | Reject enabled fault tolerance before parent construction |
| Sleep buffer, parameter and draft dictionaries | Created by parent; NPU sleep retains its own policy |
| Weight transfer engine and active/draft flags | Created by parent; transitions migrate in phase 3 |
| Profiler config and wrapper | Validate NPU torch profiler through a constructor hook; create lazily |
| PP send handles | Created by parent; device handles only |
| Sleep backend cache | Created by parent; CaMem integration follows in phase 3 |

V1/V2, 310P and Xlite retain NPU device initialization and their existing runner
construction paths in this phase. Parent `init_device` is never called.
Runtime startup and serving qualification remain required before merging this
stack; portable contract tests are not a substitute for those checks.

## Newly visible interfaces (phase 1b)

Every method newly reachable at phase 1a is listed below. Private helpers are
listed too; a helper kept for a future migration is not an advertised RPC.

| Interface | Classification | Contract |
| --- | --- | --- |
| `_after_pp_send` | Deferred to phase 4 | GPU default is unused until Ascend supplies connector output handling |
| `_before_execute_model` | Deferred to phase 4 | Default empty extension point, not an RPC capability |
| `_compile_or_warm_up_model` | Unsupported | GPU compilation and warmup raises explicitly |
| `_get_cudagraph_capture_context` | Unsupported | CUDA graph capture profiling raises explicitly |
| `_get_pp_all_gather_group` | Deferred to phase 4 | Default TP transport; NPU execution still overrides it |
| `_get_sleep_parameters` | Conditional | CPU parameter inventory; NPU sleep uses its own retention policy |
| `_init_workspace_and_model_runner` | Unsupported | GPU workspace and model-runner initialization raises explicitly |
| `_make_model_runner` | Unsupported | GPU model-runner construction raises explicitly |
| `_maybe_activate_jit_monitor` | Unsupported | GPU JIT monitoring raises explicitly |
| `_scoped_allocator_max_split` | Unsupported | CUDA allocator split tuning raises explicitly |
| `_set_draft_weight_update_target` | Conditional | Requires verified runner and engine draft support |
| `_start_weight_update` | Deferred to phase 3 | Internal helper only; public NPU update overrides remain in place |
| `_weight_checksum_prefix` | Supported | Device-independent runner forwarding or tensor/rank bookkeeping |
| `annotate_profile` | Conditional | Profiler wrapper annotations; phase 4 retains NPU iteration behavior |
| `checkpoint_prepare` | Unsupported | CUDA distributed checkpoint preparation raises explicitly |
| `checkpoint_restore` | Unsupported | CUDA distributed checkpoint restore raises explicitly |
| `compute_weight_checksums` | Supported | Device-independent runner forwarding or tensor/rank bookkeeping |
| `discard` | Unsupported until phase 3 | Fails via the explicit sleep backend boundary |
| `elastic_ep_execute` | Unsupported | Executor raises without CUDA imports |
| `get_compilation_match_table` | Supported | Device-independent runner forwarding or tensor/rank bookkeeping |
| `get_draft_model` | Conditional | V1/V2 runner returns the draft module or None |
| `get_encoder_timing_stats` | Supported | Device-independent runner forwarding or tensor/rank bookkeeping |
| `handle_ft_command` | Unsupported | GPU fault-tolerance commands raises explicitly |
| `randomize_dummy_inputs` | Supported | DP configuration and upstream randomization setting |
| `reset_mm_cache` | Supported | Both runners expose the upstream cache reset contract |
| `reset_weights` | Supported | Device-independent runner forwarding or tensor/rank bookkeeping |
| `save_sharded_state` | Conditional | Generic CPU state export; 310P retains its custom serializer |
| `save_tensorized_model` | Unsupported | CUDA tensorized model export raises explicitly |
| `shutdown_weight_transfer_engine` | Supported | Close transfer connection through engine shutdown |
| `sleep_mode_backend` | Unsupported until phase 3 | Never select CuMem on NPU |
| `start_draft_weight_update` | Unsupported | draft weight updates before transfer contract migration raises explicitly |
| `supports_draft_weight_updates` | Unsupported until phase 3 | Returns False rather than advertising unverified draft updates |

Shutdown order is: stop Ascend EPLB, drain PP device sends, unfreeze GC, stop KV
and encoder connectors, profiler, transfer engine, elastic executor, model
runner, then release existing CaMem pools. An EPLB close failure prevents
releasing resources it may still be using. No CaMem instance is created during
shutdown. GPU sentinel and executor resources are never allocated on NPU.

## Forwarding migration (phase 2)

LoRA add/remove/list/pin, get_model, get_supported_tasks, take_draft_token_ids,
reset_encoder_cache, update_max_model_len, reload_weights and update_config now
resolve directly to GPUWorker. The V1/V2 runner methods already implement those
contracts; 310P and Xlite inherit the same forwarding contracts. The Worker,
shared config and runner receive the same replacement config objects. Reload
executes inside the parent current-config context.

`INHERITED_WORKER_METHODS` in `worker_interfaces.py` is immutable. Runtime tests
assert actual method identity for all three Worker classes, and portable tests
reject reintroduced overrides in any of those classes.

## Dynamic weights and sleep (phase 3)

Ordinary start/update/finish, initialization and connection shutdown now resolve
to GPUWorker. Only `_start_weight_update` adds NZ validation before delegating.
DP-indexed updates select `data_parallel_index * world_size + rank`; failed
start/chunk loads reset the transfer target through the parent state machine.
At this pinned parent version, a failed finish leaves the session active for a
retry; successful finish resets the target, active flag and LoRA state.
Draft updates remain explicitly unsupported and advertise False until the HCCL
engine and every runner variant are qualified together.

`CaMemSleepBackend` implements suspend/resume/discard without CuMem. The default
`cumem` selection is mapped to CaMem on NPU; other backend names raise. It
preserves communicator and compiled-artifact identity. Worker-level HCCL extra
cleanup remains ordered around allocator suspension, saved parameter/buffer
restore and graph recapture. Generic NCCL communicator suspension is rejected.

NPU sleep/wake overrides are deliberately retained: they include memory
reporting, persistent buffer policy, Mooncake registration, selective restoration
and SleepWakeupManager ordering. Level 2 now also retains private draft buffers
and the inherited parameter retention rules; a KV-only wake does not consume
saved weights. Runtime CaMem/HCCL qualification remains a merge requirement.

## Execution migration (phase 4)

`execute_model` resolves to GPUWorker, including its inference-mode/sync-check
wrappers, previous-send waits, receive, runner invocation, pooling and send.
Ascend owns only the before-iteration, PP-group, profiler and post-send hooks.
V2 SP selects no all-gather group; V1 connector output is passed back without
mutating the shared empty result. V2 never accesses V1 connector fields.

The NPU sample_tokens override is retained for profiling-chunk execution-time
statistics. execute_dummy_batch is retained for idle runtime-guard wave sync,
current-config scope and GDN-state suppression. Neither is claimed as migrated.

## Initialization, memory and lifecycle (phase 5)

All ordinary NPU, 310P and Xlite runner construction now goes through
`_make_model_runner`. The common Ascend workspace hook runs only after NPU device
selection; it keeps one workspace rather than entering the GPU DBO policy.
The 310P compile-mode setup and Xlite runner identity/warmup context are retained.
Common usage reporting is performed after runner construction on rank zero.

Shutdown delegates common teardown to GPUWorker, bracketed only by Ascend EPLB /
PP draining and CaMem pool release. Duplicate initialization of profiler, sleep,
weight-transfer, runner-version and PP-send state has been removed from the NPU
constructor. The simple CPU-offload block-count RPC now resolves to the parent.

Memory policy is deliberately unchanged: requested memory remains device total
memory times gpu_memory_utilization, measured after NPU/HCCL initialization.
Upstream request_memory and its external-weight accounting are not assumed to
model CaMem, KVPP, sparse offload or layer-reuse allocations. Explicit KV budgets
and all existing Ascend adjustments remain in the NPU profiler path. Adopting a
new memory policy needs separate allocation evidence rather than an inheritance
refactor. NPU Graph/ATB capture and TorchNPUProfilerWrapper remain device owned.

### Remaining override rationale

| Method | Concrete Ascend difference or explicit unsupported boundary |
| --- | --- |
| `__init__` | Register Ascend ops/patches/config, cache dtype, NPU graph state and sleep cleanup manager; shared state comes only from parent |
| `_validate_profiler_config` | Validate torch NPU profiler instead of CUDA activities |
| `_make_elastic_ep_executor` | Prevent CUDA/flashinfer import and unsupported elastic EP thread pool |
| `checkpoint_prepare` | Reject CUDA distributed checkpoint API |
| `checkpoint_restore` | Reject CUDA distributed checkpoint API |
| `handle_ft_command` | Reject GPU fault-tolerance sentinel commands |
| `save_tensorized_model` | Reject unqualified CUDA tensorizer export |
| `_compile_or_warm_up_model` | Reject parent GPU compiler; use Ascend compile_or_warm_up_model |
| `_get_cudagraph_capture_context` | Reject CUDA profiler graph capture; NPU graph capture remains local |
| `_maybe_activate_jit_monitor` | Reject GPU-only JIT monitor |
| `_scoped_allocator_max_split` | Reject CUDA allocator tuning |
| `_init_workspace_and_model_runner` | Select one NPU workspace and construct via the common runner factory hook |
| `_make_model_runner` | Select NPU V1/V2 or the encoder runner under torch_cuda_wrapper |
| `start_draft_weight_update` | Reject draft transfer pending HCCL and runner qualification |
| `supports_draft_weight_updates` | Do not advertise unqualified draft transfer |
| `sleep_mode_backend` | Resolve CaMem rather than CuMem, reject foreign backend selection |
| `_start_weight_update` | Check NZ policy then delegate ordinary state transitions |
| `uninstall_static_kernel` | Uninstall CANN static kernels under the process lock |
| `sleep` | NPU memory stats, saved buffers, connector registration and HCCL/graph extra-cleanup ordering |
| `wake_up` | NZ policy, selective parameter/buffer restore and HCCL/graph extra-cleanup ordering |
| `_check_nz_disabled` | Reject FRACTAL_NZ in RL weight updates |
| `shutdown` | Stop EPLB and drain PP before parent teardown, release CaMem pools afterwards |
| `initialize_cache` | Ascend legacy block-count RPC compatibility |
| `_init_device` | NPU selection, DP isolation, HCCL, Triton device properties and NPU memory snapshot |
| `init_device` | NPU device and FLA registration before workspace/runner construction |
| `_apply_kv_offload_decode_memory_constraints` | Ascend sparse decode KV-offload memory plan |
| `_apply_kvpp_memory_budget` | Ascend KVPP physical cache allocation budget |
| `determine_available_memory` | NPU profiling plus KVPP, layer reuse and sparse-offload budget adjustments |
| `_scale_kv_cache_memory_for_multi_group` | Ascend allocation and compressed/shared backing layouts |
| `log_memory_stats` | NPU allocator statistics |
| `_before_execute_model` | NPU memory logging and msMonitor iteration step |
| `annotate_profile` | NPU profiler iteration step without CUDA annotations |
| `_get_pp_all_gather_group` | V2 Ascend SP already owns the gathered layout |
| `_after_pp_send` | V1 KV connector result propagation; V2 excludes V1-only fields |
| `sample_tokens` | Ascend profiling-chunk execution-time statistics |
| `load_model` | CaMem weights pool, NPU empty-cache protection, NPU transfer-engine setup and elastic-EP rejection |
| `compile_or_warm_up_model` | NPU Graph capture, ATB warmup, CPU binding and NPU memory measurement |
| `_warm_up_atb` | ATB extension initialization and warmup |
| `profile_prefill_latency` | Ascend scheduler prefill latency profiling |
| `get_kv_connector_handshake_metadata` | Ascend PCP tuple layout and connector metadata |
| `_get_layerwise_kv_cache_memory_info` | Ascend layer-reuse physical allocation topology |
| `get_kv_cache_spec` | KVPP draft registration, layer reuse and sparse offload bookkeeping |
| `_maybe_get_memory_pool_context` | CaMem allocation pools |
| `initialize_from_config` | Ascend KV allocation contexts, block sharing and zeroing metadata |
| `profile` | TorchNPUProfilerWrapper construction and trace lifecycle |
| `synchronize_device` | NPU synchronization primitive |
| `get_encoder_cudagraph_stats` | Serializable NPU encoder graph diagnostics |
| `execute_dummy_batch` | Idle runtime-guard wave sync, NPU GDN suppression and config context |
| `_init_worker_distributed_environment` | HCCL and Ascend PCP/Engram group topology |
| `get_supported_pooling_tasks` | Ascend pooling-task compatibility RPC |
| `check_health` | npu-smi card health check |

310P additionally retains RC-device memory accounting, 310P shared-cache sizing
and quantized shard export. Xlite retains its device-init path and wrapper bypass
during warmup. Each now supplies the same runner factory contract.

This table records implementation boundaries, not a hardware qualification claim.
Before merge, verify startup, serving, PP/SP, pooling, connectors and sleep on the
applicable V1/V2/310P/Xlite runtime matrix. No remote NPU validation was performed
for this stack.
