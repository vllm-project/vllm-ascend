# Task 1 — Native FULL Baseline and First Boundary

Date: 2026-08-25 (Asia/Shanghai)  
Server/container: server 1 / `whn_310b`  
Worktree: `/home/whn/vllm-ascend-full-minimal`  
Evidence root: `/home/whn/vllm_repair/full-minimal/task1-20260825`

## Frozen identity

- Worktree HEAD at collection: `4cf8c99592e07820293f7480e47c762f839fc301`.
- OpenSpec parent: `beb94e544b8481c179bfd4a94be80e1497c273ee`.
- Frozen production implementation base: `1a8feb60d1d642c87feccdb9d1aee5d273f7197a`.
- Branch: `fix/310p-dflash-full-graph-minimal`.
- Tracked status was clean. The only status entry was the intentionally untracked generated file `openspec/config.yaml`.
- Runtime package metadata: vLLM `0.24.0+empty`, vLLM Ascend `0.1.dev4221+gf62666eb2.d20260818`, torch `2.10.0+cpu`, torch-npu `2.10.0.post2`.
- Driver inventory reports `npu-smi 25.5.0`. Card 1 was selected only after a fresh idle snapshot at approximately 1.3 GiB and no vLLM process. Occupied cards and Alarm devices were not used.
- Dataset and model metadata hashes are recorded verbatim in `dataset-sha256.txt` and `model-metadata-sha256.txt`.

The serving environment prepended the installed wheel's `custom_transformer` OPP and `op_api/lib`, while Python imports were pinned to this worktree and `/home/whn/vllm-v0.24.0-clean`. This is required to keep the CANN custom-op environment valid; no source or model file was changed.

## Unit baseline

The unchanged focused command in `unit-baseline.log` passed:

```text
133 passed, 14 warnings in 1.14s
```

No production source was edited before or during this run.

## Matched 4B TP1 controls

All controls used Qwen3.5-4B target, Qwen3.5-4B-DFlash draft, DFlash K=15, card 1, GSM8K in fixed order, temperature 0, ignore EOS, concurrency 1, four measured requests, and output length 256. Only graph mode changed.

| Mode | Success | Output throughput | Acceptance length | Graph evidence |
|---|---:|---:|---:|---|
| Eager (`NONE`) | 4/4 | 36.0285 token/s | 7.3217 | Not applicable |
| Piecewise | 4/4 | 47.6903 token/s | 7.3217 | Capture 2/2 and `Replaying aclgraph` |
| FULL_DECODE_ONLY | 4/4 | 50.2208 token/s | 7.3217 | Capture 2/2 and `Replaying aclgraph` |

The detailed JSON, exact launch/benchmark commands, logs, graph evidence, and NPU snapshots are under `controls/{eager,piecewise,full_decode_only}/`. Runs were serialized. After Piecewise, the driver retained memory briefly after process exit; the FDO run was deliberately delayed until card 1 returned to its idle baseline.

## Unchanged native FULL reproduction

The reproduction used the same model and scheduling settings with `VLLM_LOGGING_LEVEL=DEBUG`, configured `cudagraph_mode=FULL`, capture sizes `[160,16]`, and `--cudagraph-metrics`. The service never became healthy, so no request was sent.

The first boundary is deterministic and precedes any operator, address, or numerical failure:

1. API and EngineCore configuration both contain `cudagraph_mode=FULL` and capture descriptors 16 and 160.
2. The draft wrapper is created with `runtime_mode=FULL` (`server.log:745`).
3. The attention capability resolver reports:

   ```text
   CUDAGraphMode.FULL is not supported with AscendGDNAttentionBackend310
   backend (support: AttentionCGSupport.UNIFORM_BATCH); setting
   cudagraph_mode=FULL_DECODE_ONLY
   ```

   This is the first semantic divergence (`server.log:1249`).
4. Startup then captures two uniform-decode descriptors. Debug boundaries show the FDO dispatcher state, `dispatcher_mode=FULL_DECODE_ONLY`, target/draft component routes, rank 0, and descriptors `(tokens=160, reqs=10)` and `(tokens=16, reqs=1)`.
5. The wrappers were constructed under configured FULL before that late downgrade, so the exact-FDO manifest path did not retain the four target/draft entries. Startup validation therefore raises:

   ```text
   DFlashFullDecodeManifestError: missing FULL capture manifest entries:
   component=draft/rank=0/mode=FULL/tokens=16,
   component=draft/rank=0/mode=FULL/tokens=160,
   component=target/rank=0/mode=FULL/tokens=16,
   component=target/rank=0/mode=FULL/tokens=160
   ```

No `507xxx`, AICore, HCCL, graph-input-count, address, dtype, or numerical error occurs before this boundary.

## Hypothesis disposition

| Hypothesis | Status | Evidence |
|---|---|---|
| GDN advertises only `UNIFORM_BATCH`, causing native FULL to downgrade | Confirmed | First semantic divergence at `server.log:1249` |
| Late downgrade leaves wrapper/manifest policy inconsistent | Confirmed | Wrapper logs FULL before resolver downgrade; captured entries are absent from exact-FDO manifest |
| Existing Eager/Piecewise/FDO baseline is already broken | Disproved | All three matched controls are 4/4; graph modes genuinely replay |
| Missing custom-op runtime environment is the FULL defect | Disproved for this run | Same corrected vendor environment passes all controls; FULL reaches graph capture |
| FULL first fails due to mutable input address/shape/dtype | Not reached | Startup stops at capability/manifest policy boundary before real replay |
| Attention mask host transfer is the first blocker | Not reached | No native FULL request or qualified prefill capture occurs |
| `NonzeroV2`/boolean GDN state selection is the first blocker | Not reached | No such error before the capability/manifest failure |
| Chunk metadata host-list conversion is the first blocker | Not reached | No native FULL request or qualified chunked-prefill replay occurs |
| Dense-prefill recording is the first blocker | Not reached | Resolver prevents native FULL coverage before this can be tested |
| A new FULL-only operator is already justified | Not reached | No operator-level RED exists; operator work remains prohibited |

## Root-cause conclusion and next RED

The current first defect is policy/capability routing, not an operator implementation: the 310P GDN builder does not advertise native FULL for the exact 310P+DFlash+configured-FULL scope. The next production change must therefore be preceded by focused RED tests for the exact activation predicate and the exact-scope GDN capability override. Eager, Piecewise, FDO, non-DFlash, and non-310P must continue returning their baseline behavior.

Only after that boundary is GREEN will hardware be rerun to expose the next first failure. No attention, buffer, or operator change is justified by Task 1.
