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
