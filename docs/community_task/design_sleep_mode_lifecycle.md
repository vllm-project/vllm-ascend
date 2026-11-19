# Sleep Mode Unified Lifecycle Design

Task: 【vllm-ascend】[Sleep Mode] HCCL 通信域轻量化操作 + ACL Graph 原地恢复 — 总体规划

## 1. Background

Current `enable_sleep_mode_extra_cleanup` path:

1. ACL Graph: clear workspaces + invalidate captured graphs + reset runner graph manager
2. HCCL: destroy process groups / communicators
3. Wake: recreate HCCL + `capture_model()` again

This frees memory but pays high wake latency and risk of rebuild failure. The task requires a **unified lifecycle** with:

- HCCL **lightweight** suspend/resume (`HcclCommSuspend` / `HcclCommResume`)
- ACL Graph **in-place** restore (keep captured graph entries; free workspaces only)
- Explicit ownership, order, rollback, compatibility matrix, and measurable gates

## 2. Resource ownership

| Resource | Owner in sleep path | Sleep action | Wake action |
|----------|---------------------|--------------|-------------|
| Weights | `CaMemAllocator` (tag `weights`) | offload (L1) / discard (L2) | restore allocator memory + MoE transpose fixup |
| KV Cache | `CaMemAllocator` (tag `kv_cache`) | discard | restore allocator memory |
| ACL Graph entries | `ACLGraphWrapper.concrete_aclgraph_entries` | **inplace**: keep; **recapture**: clear | inplace keep / else `capture_model` |
| ACL Graph workspaces | `acl_graph.*_graph_params.workspaces` | set to `None` (free HBM) | lazy rebuild / recapture |
| ACL Graph events/handles | graph params list fields | inplace keep / recapture clear | unchanged / recapture |
| HCCL communicator | `ProcessGroupHCCL` / raw `HcclComm` | **lightweight** suspend / **destroy** release | resume / restore_hccl |
| HCCL device communicator | `NPUCommunicator` | destroy only on destroy policy | recreate on destroy policy |
| PP send works / stream | worker `_pp_send_work` + current stream | wait + `torch.npu.synchronize` before HCCL | n/a |
| Host CPU backup | sleep buffers for L2 | save named_buffers | copy back |

## 3. Release / restore order

### Sleep (extra cleanup)

```text
IDLE
  -> SLEEPING
       1) ACL Graph policy (inplace|workspace_only|recapture|skip)
       2) HCCL policy (lightweight|destroy|skip) after PP drain + NPU sync
  -> ASLEEP
```

Allocator sleep (weights/KV) remains in `Worker.sleep()` around this manager, unchanged.

### Wake

```text
ASLEEP
  -> WAKING
       1) HCCL resume/restore + MoE HCCL metadata refresh
       2) ACL Graph inplace restore or recapture (only if tags is None or contains kv_cache)
  -> IDLE
```

The synchronization rule is strict: drain PP work and complete any
pre-suspend barriers before the first communicator is suspended. No barrier is
issued after a suspend. On wake, all required ProcessGroup/raw-communicator
resumes must succeed before the corresponding group or global barrier runs.
Raw fallback resumes use the exact handle recorded during sleep; an unrecognized
current handle is left pending rather than being resumed speculatively.
If a lightweight wake cannot complete, ACL Graph recovery and anchor release
are deferred so a later wake can retry the pending handles safely.

## 4. Unified state machine

States: `idle | sleeping | asleep | waking | failed`

Illegal transitions raise. On sleep failure: rollback HCCL/ACL then return to `idle`. On wake failure: cold rebuild (`destroy+restore` / `force_recapture`) then `idle`.

Concurrent / repeated sleep-wake:

- Sleep while `asleep`: rejected by FSM (caller should no-op at API layer if already sleeping)
- Wake while `idle`: idempotent no-op
- Nested sleep during `sleeping/waking`: illegal

## 5. Policies (`additional_config`)

| Key | Values | Default |
|-----|--------|---------|
| `sleep_hccl_policy` | `lightweight` / `destroy` / `skip` | `lightweight` |
| `sleep_acl_graph_policy` | `inplace` / `workspace_only` / `recapture` / `skip` | `inplace` |
| `sleep_allow_fallback` | bool | `true` |
| `enable_sleep_mode_extra_cleanup` | bool | `false` (master switch; unchanged) |

Fallback:

- Lightweight suspend unavailable → destroy (if allowed)
- In-place entries missing → recapture (if allowed)

## 6. Compatibility matrix

| Component | Min version | Notes |
|-----------|-------------|-------|
| Hardware | Atlas A2 910B2 / A3 | Multi-card required for HCCL E2E |
| CANN | 9.0.0+ with `HcclCommSuspend/Resume` | Verified symbols in `libhccl.so` |
| torch_npu | 2.10.x | Uses `get_hccl_comm` / optional `resume_hccl_comm` |
| vLLM | 0.23.x | Sleep tags `weights` / `kv_cache` |
| vllm-ascend | 0.23.0 + this change | |

Single-card: HCCL lightweight may suspend 0 communicators; still valid for ACL graph paths.

## 7. Test plan

### UT

- FSM illegal transitions
- ACL inplace does not call `capture_model`
- HCCL lightweight suspend/resume path
- Sleep rollback on HCCL failure
- Legacy destroy/recapture still covered

### E2E (multi-card)

Matrix:

| Path | Config |
|------|--------|
| Lightweight HCCL | `sleep_hccl_policy=lightweight` |
| Destroy HCCL | `sleep_hccl_policy=destroy` |
| Inplace ACL graph | `sleep_acl_graph_policy=inplace` |
| Workspace-only | `sleep_acl_graph_policy=workspace_only` |
| Recapture fallback | `sleep_acl_graph_policy=recapture` |
| Combined | lightweight + inplace |
| Multi-round | N sleep/wake loops |

Metrics: sleep/wake latency, first-token after wake, HBM free delta, optional throughput.

## 8. Risks / default enable / limits

**Benefits**

- Lower wake latency vs full HCCL recreate + graph recapture
- Keep communication domain identity (lightweight)
- Preserve graph capture investment (inplace)

**Risks**

- Suspended HCCL may return `HCCL_E_SUSPENDING` if collectives race; must drain/sync first
- In-place graphs may still need workspace re-init on first replay depending on runner path
- Older CANN without suspend symbols must fallback

**Default**

- Extra cleanup still off by default
- When enabled, prefer lightweight + inplace with fallback on

**Known limits**

- Custom ops build may be skipped on some CANN 9.0 hosts (`COMPILE_CUSTOM_KERNELS=0`); Sleep Mode Python path still works
- TP sizes should be power-of-two for clean parallel plans; 3-NPU hosts can run TP=2

## Multi-NPU note (aligned with #9585 / #3172)

On multi-rank HCCL groups, `sleep_hccl_policy=lightweight` uses destroy/restore
of ProcessGroupHCCL (same as upstream residual-memory cleanup). Single-rank
keeps CANN `HcclCommSuspend`/`Resume`. Combining Suspend then Destroy before
Restore was observed to leave TP generate hung after wake on Atlas A2 + CANN
9.0.0, so the multi-rank path never suspends before destroy. The lifecycle
anchor and MoE teardown/refresh hooks remain active on this destroy/restore
fallback.

## Multi-NPU HCCL note (910B2)

| Path | Sleep | Wake | Generate after wake |
|------|-------|------|---------------------|
| TP=1 lightweight | `HcclCommSuspend` | `ProcessGroupHCCL.resume_hccl_comm` / `reinit_process_group(rebuild_link=False)` | OK |
| TP≥2 lightweight | destroy ProcessGroupHCCL (#9585/#3172) | restore_hccl | OK |
| TP≥2 Suspend→Resume | Suspend OK | PG resume returns but TP allreduce verify fails | N/A (auto-fallback) |
| TP≥2 Suspend→destroy/restore | Suspend then destroy | restore | generate can hang — **avoided** |

Root cause: raw `HcclCommResume` / `resume_hccl_comm` after multi-rank Suspend does not fully rebind vLLM TP device communicators on current CANN 9.0 + torch_npu. Lightweight policy therefore uses true Suspend/Resume on single-rank and destroy/restore on multi-rank under the same `sleep_hccl_policy=lightweight` knob.

## 9. Teacher review gate (PR #20481 / yjyang62)

### 9.1 Propositions under test

| ID | Proposition |
|----|-------------|
| P1 | Sleep uses HCCL **Suspend**, not Destroy |
| P2 | Sleep still yields a measurable memory-release benefit |
| P3 | Wake does **not** recapture ACL Graphs (`capture_model` not called) |

Completion rule from review: prove whether **P1∧P2∧P3** hold together (TP=1/TP=2, multi-round, first generate after wake, with numbers + recapture evidence). If not achievable, give a layered conclusion — that also completes the task.

### 9.2 Evidence instrumentation

Each completed sleep→wake cycle emits a structured `SleepCycleReport` (also logged as `Sleep cycle report: {...}`):

- `hccl_policy_requested` vs `hccl_action_actual` (`suspend` / `destroy` / `fallback_destroy` / `skip`)
- `acl_policy_requested` vs `acl_action_actual` (`inplace_keep` / `recapture` / `fallback_recapture` / `skip`)
- `hbm_freed_bytes`, `capture_model_called`, `wake_completed`, `tp_size`, `round`, `notes`

Long-term rule: assert / document on **actual actions**, never on policy names alone. Example: `sleep_hccl_policy=lightweight` on multi-rank records `hccl_action_actual=fallback_destroy`.

UT coverage: `tests/ut/device_allocator/test_sleep_mem_optimized.py` (report fields, suspend vs fallback_destroy, inplace vs fallback_recapture).  
NPU e2e (TP=1 joint path): `tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake_lightweight_inplace.py`.

### 9.3 Layered empirical conclusion

Measured on a fresh 4×910B2C cloud host (2026-11-19), model `Qwen3-0.6B`,
`rl_config={enabled, sleep_mode_extra_cleanup, sleep_hccl_policy=lightweight,
sleep_acl_graph_policy=inplace}`, CANN 9.2.1 / torch_npu 2.10.
Logs: `/data/sleep_mode_evidence/tp1_lightweight_inplace.log`,
`tp2_lightweight_inplace.log`.

| Layer | Question | TP=1 | TP=2 |
|-------|----------|------|------|
| L0 CANN | Does Suspend / destroy reclaim device memory with workspace clear? | `HcclCommSuspend` works (`Suspended 7 HCCL communicators`). Extra-cleanup-only HBM delta in `SleepCycleReport` ≈ **0 GiB** (`hbm_freed_bytes=0`; log `released ... -0.029/0.000 GiB`). **Overall** level-1 sleep (CaMem weights + cleanup) frees **~42.79 / 42.82 GiB** across 2 rounds | Default action is **destroy/restore** (`Lightweight multi-NPU sleep via destroy/restore (destroyed_pgs=7)`). Extra-cleanup frees **~0.39 GiB**; overall level-1 sleep frees **~42.54 GiB** |
| L1 torch_npu PG | After sleep action, can TP collectives resume? | Yes — wake + generate OK (`Resumed 1 HCCL communicators after lightweight sleep`) | Wake **failed**: HCCL timeout on `sleep_lifecycle_anchor` AllReduce (`Communication_Error_Timeout EI0002`, ~1836s). Confirms multi-rank communicator lifecycle fragility even on destroy/restore + anchor path |
| L2 Graph binding | Replay without `capture_model`? | **Yes**: `acl_action_actual=inplace_keep`, `capture_model_called=False`; first generate after wake succeeded for 2 rounds | Not observed (wake did not complete) |
| L3 Joint | P1∧P2∧P3 simultaneously? | **P1∧P3 hold**. P2 holds for **overall sleep** (~42.8 GiB), but **not** as a Suspend-only HBM win (extra-cleanup ≈0). Report excerpt below | **P1 does not hold** (`hccl_action_actual` path = `fallback_destroy`). Wake correctness currently blocked at **L1** on this host |

#### TP=1 `Sleep cycle report` (round 1, verbatim)

```text
Sleep cycle report: {'round': 1, 'tp_size': 1,
  'hccl_policy_requested': 'lightweight', 'hccl_action_actual': 'suspend',
  'acl_policy_requested': 'inplace', 'acl_action_actual': 'inplace_keep',
  'hbm_free_before_bytes': 18296664064, 'hbm_free_after_sleep_bytes': 18265108480,
  'hbm_freed_bytes': 0, 'capture_model_called': False, 'wake_completed': True,
  'notes': 'suspended_comms=7'}
```

Round 2 identical pattern: `suspend` + `inplace_keep` + `capture_model_called=False` +
overall sleep freed ~42.82 GiB; wake generate OK.

#### TP=2 key lines

```text
Lightweight multi-NPU sleep via destroy/restore (destroyed_pgs=7).
Sleep mode released HCCL and attention workspace memory: 0.393 GiB.
Sleep mode (level=1) freed 42.54 GiB memory, 1.10 GiB memory is still in use.
... later wake ...
Communication_Error_Timeout(EI0002): ... tag[AllReduce_group_name_23] ...
user define information[sleep_lifecycle_anchor]
```

### 9.4 Clear conclusion (task closure statement)

1. **TP=1**: Under `lightweight` + `inplace`, **Suspend + no ACL recapture + successful multi-round wake/generate are proven**. Overall sleep still frees ~43 GiB (mostly CaMem). The Suspend/workspace extra-cleanup segment itself did **not** show a large HBM delta in this run (`hbm_freed_bytes=0`); P2 should be read as “sleep still releases memory”, not “Suspend alone is the dominant free”.
2. **TP≥2**: **P1 does not hold** — runtime correctly records destroy/restore under the same `lightweight` knob (`fallback_destroy` semantics). On this host, wake then hit an L1 HCCL timeout on the lifecycle anchor AllReduce, so multi-rank remains a compatibility boundary rather than a Suspend success story.
3. Blocking layer for the joint ideal on multi-rank is **L1 communicator/PG lifecycle** (not “ACL Graph must always recapture”). When L1 is fixed upstream, re-run the report gate before flipping TP≥2 to true Suspend.
