# [Refact MRv2][310P] MRv2 Qwen3/Qwen3.5 TP + ACL Graph + W8A8/W8A8SC adaptation

This document is the community-merge companion for the first 310P Model Runner V2
(MRv2) drop rebased onto current `vllm-ascend` main. Use it together with:

- [310P Model Runner V2 adaptation guide](ModelRunner_v2_310P_adaptation.md)
  (scope, acceptance, and staged delivery)
- [310P Model Runner V2 code-change log](ModelRunner_v2_310P_code_changes.md)
  (per-issue root cause and fix history)

Enable the path with `VLLM_USE_V2_MODEL_RUNNER=1`. The default remains Model
Runner V1 (`NPUModelRunner310`).

## Pull request summary (vllm-ascend template)

### What this PR does / why we need it?

310P cannot run the shared Ascend MRv2 path as-is: it has no Triton runtime,
Attention KV cache must be allocated as `ACL_FORMAT_FRACTAL_NZ`, and ACL Graph
capture must record direct NPU operators rather than mainline graph-task handles.

This PR adds a 310P MRv2 stack under `vllm_ascend/_310p/worker/v2/` plus narrow
shared MRv2 extension points so the following acceptance surface works on 310P:

> Qwen3 / Qwen3.5 dense, VL, MoE, and hybrid models with **TP + ACL Graph
> (`FULL_DECODE_ONLY`) + W8A8 / W8A8SC / W8A8-Dynamic**, without prefix cache
> or MTP. Qwen3 dense accuracy must not regress relative to MRv1 on the same
> checkpoint and TP size.

Prefix cache, full sampling postprocessing, and MTP stay rejected at startup and
are deferred to a follow-up 310P MRv2 release. Disabling `VLLM_USE_V2_MODEL_RUNNER`
must leave V1 behavior unchanged.

### Does this PR introduce _any_ user-facing change?

Yes. Setting `VLLM_USE_V2_MODEL_RUNNER=1` on 310P selects `NPUModelRunner310V2`
instead of `NPUModelRunner310` for supported Qwen3 and Qwen3.5 workloads. The
default remains MRv1.

### How was this patch tested?

- **Unit tests added/updated:** `tests/ut/_310p/test_model_runner_v2_310p.py`,
  `tests/ut/_310p/quantization/test_w8a8_dynamic_310.py`,
  `tests/ut/_310p/quantization/test_w8a8_dynamic_tp_310.py`,
  `tests/ut/_310p/quantization/test_w8a8sc_310.py`,
  `tests/ut/_310p/quantization/test_modelslim_config_310.py`,
  `tests/ut/_310p/attention/test_attention_v1_310.py`,
  `tests/ut/_310p/ops/test_qwen3vl_310.py`,
  `tests/ut/worker/test_attn_utils_v2.py`, and
  `tests/ut/quantization/test_modelslim_config.py`.
- **E2E tests added:** `tests/e2e/pull_request/one_card/_310p/test_model_runner_v2_310p.py`,
  `tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_310p.py`, and
  `tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_moe_310p.py`.
- **Local review (this workspace):** static code review and diff reconciliation
  against current main; no 310P hardware in this environment.
- **310P server validation (required before merge):** run the unit and E2E commands
  in Section 4 on Ascend 310P hardware and capture real-weight serve evidence for
  Qwen3 dense TP + graph + W8A8/W8A8SC accuracy checks.

## Review boundary

The adaptation guide defined the dense/VL W8A8/W8A8SC baseline. This PR expands
that acceptance surface to W8A8-Dynamic, Qwen3 MoE, Qwen3.5 dense/hybrid and
MoE, embedding pooling, and explicit compatibility without vLLM PR #43048. The
support matrix below is authoritative when that expanded scope differs from the
earlier planning baseline.

## 1. Feature scope

```text
Core 310P MRv2 stack
  _310p/worker_310p.py          explicit V1/V2 runner selection
  _310p/attention/metadata_builder.py
                                graph-safe host/device attention metadata
  _310p/worker/v2/model_runner.py
                                TP input/state flow, NZ KV allocation,
                                hybrid cache handling, sampling writeback
  _310p/worker/v2/{block_table,states,rope,model_state,aclgraph,sampler,
                   kv_block_zeroer,feature_support,kernel_registry}.py
                                Triton-free 310P implementations
  worker/v2/{model_runner,attn_utils,model_states}/
                                narrow shared extension points
  patch/platform/patch_use_v2_model_runner.py
                                explicit opt-in and 310P HAS_TRITON bypass
  patch/worker/patch_v2/        310P class substitution
  _310p/quantization/           MoE [E,K,N], dynamic TP bias, tid2eid,
                                W8A8SC TP bias, static-MoE error
  tests/ut/_310p/              runner and quantization contracts
  tests/e2e/pull_request/       TP/graph/model-family acceptance cases
  docs/source/developer_guide/  design, change history, and PR notes
```

### Requirement-to-implementation map

- **Tensor Parallel:** 310P consumes scheduler CPU mirrors for request indices,
  positions, sequence lengths, block tables, and slot mappings; quantized
  row-parallel bias is applied only on TP rank 0.
- **ACL Graph:** `ModelAclGraphManager310` captures decode with resident device
  buffers. Prefill-no-cache keeps host `seq_lens`, while paged/splitfuse decode
  uses device `seq_lens`; pageable H2D copies are excluded from capture.
- **Quantization:** dense linears support W8A8, W8A8SC, and W8A8-Dynamic.
  Grouped MoE supports W8A8-Dynamic expert descriptions only and converts
  expert weights to `[E,K,N]` before NZ.
- **Qwen3/Qwen3.5 families:** standard attention, multimodal MRoPE, MoE, and
  hybrid Full-Attention + GDN/Mamba model states have dedicated acceptance
  cases.
- **Compatibility:** MRV1 remains the default. Non-310P shared behavior keeps
  its existing block-table ABI, and vLLM PR #43048 is optional rather than a
  runtime prerequisite.

Design rules for reviewers:

1. 310P differences stay in `_310p/`. Shared MRv2 only grows class-level
   hooks or overridable methods. Do not scatter `is_310p()` on hot paths.
2. No dependency on unmerged upstream work. Triton paths are replaced by
   substituting classes and module attributes, the mechanisms this repo
   already uses, so the plugin imports cleanly on vLLM main. `kernel_registry`
   only opts into [vLLM #43048](https://github.com/vllm-project/vllm/pull/43048)
   if that dispatcher is present, and is a no-op today.
3. V1 must not import `_310p/worker/v2/`, and 310P must not import
   `worker/v2/block_table.py`, which defines a Triton kernel at import time.
4. Slot mapping, positions, and seq lens are built from CPU mirrors. Device
   tensors are not copied back with `.cpu()` / `.item()` in the request path.
5. Attention KV cache is created with `torch_npu.empty_with_format(...,
   ACL_FORMAT_FRACTAL_NZ)`. `view()` / `reshape()` cannot produce physical NZ.
6. ACL Graph replay uses preallocated device buffers. Pageable H2D must not
   be captured.

## 3. Support matrix (first release)

Status: **verified** = real-weight TP + graph serve evidence was supplied from
the 310P server; **e2e-ready** = the test has landed but was not run in this
Windows workspace; **known failing** = real-weight serve currently fails and
must not be advertised as supported; **code-compatible** = the path exists but
still needs 310P acceptance.

| Model | TP | ACL Graph `FULL_DECODE_ONLY` | Quantization boundary |
| --- | --- | --- | --- |
| Qwen3-8B | verified TP + graph | verified | W8A8 verified; W8A8SC verified TP1/TP2 + graph (`sharded_state`); W8A8-Dynamic TP1/TP2 e2e-ready |
| Qwen3-Embedding-8B | verified TP2 | verified | pooling TP2 graph regression added |
| Qwen3-VL-2B-Instruct | verified TP1/TP2 | verified (encoder eager, decode graph) | quantized VL checkpoint pending |
| Qwen3-VL-4B-Instruct | verified TP1 | verified (encoder eager, decode graph) | W8A8SC TP1 + graph verified (`sharded_state`; no TP2 shard on disk) |
| Qwen3-VL-8B-Instruct | verified TP1/TP2 | verified (encoder eager, decode graph) | W8A8SC TP2 + graph verified (`sharded_state`) |
| Qwen3-32B | — | — | W8A8SC TP4 + graph verified (`sharded_state`; no TP2 shard) |
| Qwen3-30B-A3B | verified TP2 | verified | W8A8-Dynamic expert checkpoint e2e-ready; static W8A8/W8A8SC experts unsupported |
| Qwen3.5-2B | verified TP + graph | verified | Qwen3.5-2B-W8A8 verified TP2 + graph (310P dynamic linear uses ND fp16 dequant) |
| Qwen3.5-4B | FP16 code-compatible; post-§31 hardware re-validation required | FP16 hardware re-validation required | Qwen3.5-4B-W8A8 verified with TP + graph |
| Qwen3.5-9B | — | — | Qwen3.5-9B-W8A8 verified TP2 + graph |
| Qwen3.5-27B | verified TP4 | verified | quantized checkpoint pending |
| Qwen3.5-35B-A3B | verified TP2; TP4 on this server is memory-bound | verified TP2 | Qwen3.5-MoE FP16; local `Qwen3.5-VL-35B-A3B` is JANG, not Ascend |
| Qwen3.5-VL-2B (local checkpoint) | invalid artifact | invalid artifact | MLX affine 8-bit; not an Ascend/ModelSlim checkpoint |

Quantization boundary (layer registry, shared by V1/V2):

| Layer | W8A8 | W8A8SC | W8A8-Dynamic |
| --- | --- | --- | --- |
| Dense linear | supported | supported | supported; fixed int8 activation contract |
| MoE experts | unsupported on 310P grouped operator | unsupported on 310P grouped operator | supported; weights are transposed to `[E,K,N]` before NZ |

W4A8, EP, PP, DP, CP, LoRA, KV transfer, sleep mode, structured output, and
non-greedy sampling are rejected. Expert Parallel is out of 310P V1 scope
and remains rejected here.

### Acceptance interpretation

- The verified list reflects the supplied 310P real-weight serve results; the
  repository changes alone do not upgrade an `e2e-ready` or `known failing`
  entry to verified.
- Qwen3.5-35B-A3B FP16 TP2 + graph is the Qwen3.5-MoE claim. Local
  `Qwen3.5-VL-35B-A3B` is a JANG/MLX dump and must not be used as acceptance.
- Qwen3.5-2B-W8A8 compiled NZ `npu_quant_matmul` aicore-faults on 310P
  (`QuantBatchMatmulV3_NZ_NZ` kernel 21). 310P W8A8-Dynamic *linear* keeps
  ND weights and dequants to fp16; MoE experts still use grouped-matmul NZ.
- Local `Qwen3.5-VL-2B` is MLX affine 8-bit. ModelSlim no longer claims
  `bits`+`group_size` dumps as `"ascend"`.
- Static W8A8/W8A8SC expert descriptions are an explicit operator boundary,
  not a missing registry alias. Quantized MoE checkpoints must describe
  experts as W8A8-Dynamic.
- W8A8SC dense/VL dumps are pre-sharded compressed ModelSlim artifacts.
  Serve with `--quantization ascend --load-format sharded_state` and a TP
  size that matches the shard folder (8B/VL-8B TP2, VL-4B TP1 only, 32B
  TP4 only). MRv2 reuses `AscendW8A8SCLinearMethod310`.

## 4. Test inventory

### Unit tests (CPU)

```bash
pytest -sv tests/ut/_310p/test_model_runner_v2_310p.py
pytest -sv tests/ut/_310p/test_block_table_310p.py
pytest -sv tests/ut/_310p/quantization/test_w8a8_dynamic_310.py
pytest -sv tests/ut/_310p/quantization/test_w8a8_dynamic_tp_310.py
pytest -sv tests/ut/_310p/quantization/test_w8a8sc_310.py
pytest -sv tests/ut/_310p/quantization/test_modelslim_config_310.py
pytest -sv tests/ut/worker/test_attn_utils_v2.py
```

Covered contracts: Triton gate skip on 310P only, no Triton or dispatcher
import from the 310P block tables, NumPy slot mapping across cache groups,
first-release config rejects, NZ KV allocation, capture `seq_lens` refresh,
FULL-graph padding, hybrid model-state routing, W8A8SC `tp_rank != 0`
quant-bias zeroing, W8A8-Dynamic nonzero-rank bias suppression, MoE
`tid2eid` forwarding, and the static-quant error hint.

`test_block_table_310p.py` and `test_attn_utils_v2.py` are pre-existing
regression suites from the core branch; they remain mandatory because this
hardening delta changes the shared/non-310P block-table compatibility surface.

Local static verification completed with `compileall`, `ruff check`, and
`git diff --check`. The Windows virtual environment does not contain `pytest`,
so the unit commands above still need to run in the Linux development image.

### E2E (310P hardware)

```bash
# one card: dense/hybrid TP1 + graph; W8A8/W8A8-Dynamic; W8A8SC sharded_state; VL
pytest -sv tests/e2e/pull_request/one_card/_310p/test_model_runner_v2_310p.py

# four cards: TP2 dense/hybrid/VL/embedding; TP2 W8A8/W8A8SC; TP4 27B / 32B-W8A8SC
pytest -sv tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_310p.py

# four cards: Qwen3-30B-A3B TP2 eager/graph; Qwen3.5-35B-A3B TP4 eager/graph
pytest -sv tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_moe_310p.py
```

Existing 310P V1 files under `tests/e2e/pull_request/{one,four}_card/_310p/`
must keep passing with `VLLM_USE_V2_MODEL_RUNNER` unset.

The W8A8-Dynamic cases require access to
`vllm-ascend/Qwen3-8B-W8A8-Dynamic`. Before running the MoE quantized case,
verify that `vllm-ascend/Qwen3-30B-A3B-W8A8` describes expert weights as
`W8A8_DYNAMIC`; static W8A8/W8A8SC expert descriptions are intentionally
rejected on 310P.

### Serve smoke (greedy, no prefix cache)

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --cudagraph-mode full_decode_only \
  --no-enable-prefix-caching \
  --max-model-len 8192 --port 8000

# VL, MoE, and Qwen3.5 follow the same flags. Hybrid models also need:
#   --mamba-ssm-cache-dtype float16
# Quantized dense:
#   --quantization ascend
```

Pass criteria for each model: `/v1/models` 200; two consecutive greedy
requests 200 with non-empty output; logs contain
`run 310P full ACL Graph with num_tokens=...`; no Triton compile/invoke.

## 5. User-facing change

Yes, when `VLLM_USE_V2_MODEL_RUNNER=1` on 310P:

- Engine uses `NPUModelRunner310V2`.
- Prefix cache / MTP / non-greedy sampling fail at startup or first request
  with `NotImplementedError`.
- Default (`VLLM_USE_V2_MODEL_RUNNER` unset or `0`) is unchanged V1.

## 6. Out of scope (do not block this PR)

- Prefix cache and MTP (second 310P MRv2 release).
- Temperature / top-k / top-p / penalties / logprobs / grammar.
- Real-weight W8A8-Dynamic accuracy sign-off for combinations not marked
  verified above. The runtime contract and unit/E2E coverage are included.
- Static W8A8/W8A8SC MoE schemes (no 310P operator; registry refuses them).
- Qwen3.5 image input, Gemma4, EP, and piecewise ACL Graph as a new 310P
  product mode.

## 7. Reviewer checklist

- [ ] Treat `mrv2_310p_ys` and `mrv2_310p_820` as one PR lineage.
- [ ] Shared MRv2 changes remain narrow extension points.
      `worker/v2/block_table.py` is restored to the existing multi-group
      Triton ABI, with no 310P branch in the non-310P hot path.
- [ ] Nothing imports `vllm.model_executor.triton_dispatcher` unconditionally.
- [ ] 310P V1 path does not import `_310p/worker/v2/`.
- [ ] First-release guards still reject prefix cache, MTP, EP, PP/DP/CP,
      LoRA, KV transfer, sleep mode.
- [ ] Attention KV is NZ at allocation time; Mamba/GDN state stays ND with
      contiguous per-state stride.
- [ ] `FULL_DECODE_ONLY` capture/replay uses resident device `seq_lens` /
      `query_start_loc` for decode/splitfuse, host `seq_lens` for
      PrefillNoCache.
- [ ] Dense quantized row-parallel bias is rank-0-only; MoE weights are
      `[E,K,N]` before NZ and preserve `tid2eid`.
- [ ] New E2E files live under `*_310p.py` so CI routes them to 310P runners.
- [ ] MRV1 regression files pass with `VLLM_USE_V2_MODEL_RUNNER` unset.
- [ ] Qwen3.5-35B-A3B either passes its acceptance cases or is removed from
      the first-release support claim.
- [ ] No new environment variable except the existing
      `VLLM_USE_V2_MODEL_RUNNER`.
- [ ] Commit messages follow Conventional Commits and are signed off.

Suggested PR title:

```text
[Feat][310P] Add Qwen3/Qwen3.5 MRv2 with TP, ACL Graph, and W8A8 variants
```

## 8. Copy-paste PR description

### What this PR does / why we need it?

- Adds the first Ascend 310P Model Runner V2 path for Qwen3 dense, embedding,
  VL, and MoE models plus Qwen3.5 dense/hybrid and MoE models.
- Supports Tensor Parallel execution and ACL Graph `FULL_DECODE_ONLY` using
  fixed-address NPU buffers, 310P FRACTAL_NZ attention KV cache, and
  Triton-free CPU-mirror metadata preparation.
- Supports dense W8A8, W8A8SC, and W8A8-Dynamic. Quantized grouped MoE uses
  W8A8-Dynamic experts, `[E,K,N]` weight layout before NZ, rank-aware bias,
  and preserved expert mapping.
- Removes the hard dependency on the unmerged vLLM Triton dispatcher PR
  [#43048](https://github.com/vllm-project/vllm/pull/43048). The first release
  uses class/module replacement and keeps an optional future registration
  bridge.
- Keeps Model Runner V1 as the default and rejects unadapted features instead
  of silently changing behavior.

### Does this PR introduce _any_ user-facing change?

Yes. On Ascend 310P, setting `VLLM_USE_V2_MODEL_RUNNER=1` selects
`NPUModelRunner310V2`. The first release supports greedy requests without
prefix caching or MTP. Unsupported parallel modes, non-greedy postprocessing,
LoRA, KV transfer, and sleep mode fail explicitly. With the environment
variable unset, the existing Model Runner V1 path is unchanged.

### How was this patch tested?

- Added CPU unit coverage for runner selection and feature guards, NumPy slot
  mapping, hybrid cache allocation/state, graph replay metadata,
  dispatcher-free imports, W8A8-Dynamic layouts and TP bias, W8A8SC TP bias,
  `tid2eid`, and static-MoE error handling.
- Added 310P E2E cases for one-card/four-card dense and hybrid models,
  Qwen3-VL image requests, Qwen3 embedding, Qwen3/Qwen3.5 MoE, quantized
  dense/MoE paths, TP, chunked prefill, and `FULL_DECODE_ONLY`. The cases send
  consecutive requests to exercise state cleanup and graph replay.
- Local `compileall`, Ruff, IDE diagnostics, and `git diff --check` pass.
- Real-weight TP + graph serve evidence is recorded in the support matrix.
  Entries marked `e2e-ready` or `known failing` still require the listed 310P
  Linux validation before merge. Qwen3.5-4B FP16 also requires the post–GDN
  state-stride-fix re-validation recorded in the code-change log.

## 9. Follow-up after merge

1. Second release: prefix cache (block reuse is already implemented;
   startup reject is the remaining product gate).
2. Second release: greedy-plus sampling and MTP speculator under
   `_310p/worker/v2/spec_decode/`.
3. Run the 310P accuracy and throughput acceptance matrix for W8A8-Dynamic
   Dense and MoE checkpoints; keep the fixed contract unless hardware evidence
   requires a reviewed operator change.
4. Optional: restore 310P W8A8-Dynamic linear NZ `npu_quant_matmul` once GE
   can compile `QuantBatchMatmulV3` for small KV shards (Qwen3.5-2B TP2 N=256)
   without kernel 21. Current path is ND fp16 dequant for linears only.
5. Reduce `NPUModelRunner310V2.initialize_kv_cache` duplication by hooking
   `_allocate_kv_cache_tensors` in the shared runner.
6. If [vLLM #43048](https://github.com/vllm-project/vllm/pull/43048) merges
   and the relevant upstream kernels become pluggable, register only those
   310P implementations in `kernel_registry.KERNEL_IMPLS`; until then the
   class-level fallbacks remain the compatibility path.
