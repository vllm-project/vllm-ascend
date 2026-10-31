# Acceptance Pack — Sleep Mode HCCL Lightweight + ACL Graph In-place

## Task

【vllm-ascend】[Sleep Mode] HCCL 通信域轻量化操作 + ACL Graph 原地恢复

## Implementation summary

- Unified lifecycle FSM in `sleep_mem_optimized.py` (ACL Graph → HCCL → CaMem; wake reverse).
- HCCL lightweight:
    - TP=1: `HcclCommSuspend` + `ProcessGroupHCCL.resume_hccl_comm` (torch_npu official resume path).
    - TP≥2: destroy/restore ProcessGroupHCCL under the same `sleep_hccl_policy=lightweight` (Suspend+Resume verify fails on current CANN/torch_npu; Suspend+destroy hangs generate).
- ACL Graph `inplace`: keep captured entries, free workspaces only; no mandatory `capture_model` on wake.
- Knobs: `sleep_hccl_policy`, `sleep_acl_graph_policy`, `sleep_allow_fallback`, `enable_sleep_mode_extra_cleanup`, `weight_nz_mode=0` for RL wake.

## Gates

| Gate | Result |
|------|--------|
| UT `test_sleep_mem_optimized.py` | 10 passed |
| TP1 eager lightweight+inplace | pass (prior final) |
| TP2 lightweight+inplace generate-after-wake | pass (destroy/restore multi path) |
| TP1 ACL Graph PIECEWISE inplace (enforce_eager=False) | **pass** — see `sleep_tp1_aclgraph_inplace.json` |
| Single feature commit | yes |

## Reproduce

```bash
export PATH=/usr/local/BinSheng/compiler/ccec_compiler/bin:$PATH
source /usr/local/Ascend/cann/set_env.sh
export VLLM_USE_MODELSCOPE=True VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
python3 deliverables/scripts/e2e_tp2_suspend.py
python3 deliverables/scripts/e2e_graph_inplace.py
```
