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
