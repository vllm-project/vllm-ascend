# vLLM-Ascend CI Failure Analysis Report

## Overview

| Item | Value |
|:---|:---|
| **Run URL** | https://github.com/vllm-project/vllm-ascend/actions/runs/22551808471 |
| **Run Date** | 2025-03-01 |
| **Good Commit (pinned)** | `15d76f74e2fdb12a95ea00f0ca283acf6219a2b7` |
| **Bad Commit (tested)** | `6290470843c131681e3e1318ae71070a34f33225` |
| **Total Failed Jobs** | 4 / 9 |
| **Distinct Issues Found** | 4 code bugs + 0 env flakes |

## Failed Jobs Summary

| Job | Conclusion | Failed Tests |
|:---|:---|:---|
| e2e-test / multicard-4-full (0) | failure | 13 tests |
| e2e-test / multicard-2-full (0) | failure | 20 tests |
| e2e-test / singlecard-full (0) | failure | 10 tests |
| e2e-test / singlecard-full (1) | failure | 10 tests |

## Issue Analysis

### Issue 1: CudagraphDispatcher.dispatch() Missing 'disable_full' Parameter

| Item | Detail |
|:---|:---|
| **Category** | Code Bug |
| **Error Type** | TypeError |
| **Affected Tests** | 53 tests (all failed tests) |
| **Root Cause Commit** | `1d532f9d8fb2` — "[DP] Only use DP padding when cudagraphs are actually used (#34102)" |
| **Changed File** | `vllm/v1/cudagraph_dispatcher.py` |
| **Impact in vllm-ascend** | `vllm_ascend/worker/model_runner_v1.py` |

**Error Message:**
```
TypeError: CudagraphDispatcher.dispatch() got an unexpected keyword argument 'disable_full'
```

**Explanation:**
Upstream vLLM removed the `disable_full` parameter from `CudagraphDispatcher.dispatch()` in commit 1d532f9d8fb2. The new signature only accepts `num_tokens`, `uniform_decode`, `has_lora`, `num_active_loras`, `valid_modes`, and `invalid_modes`. However, vllm-ascend's `model_runner_v1.py` still passes `disable_full=disable_full` when calling the dispatcher.

**Fix Suggestion:**
Remove the `disable_full` parameter from all calls to `cudagraph_dispatcher.dispatch()` in `vllm_ascend/worker/model_runner_v1.py`. The upstream implementation now uses `valid_modes` and `invalid_modes` to control which cudagraph modes are allowed.

---

### Issue 2: NoneType Comparison in compilation_time Aggregation

| Item | Detail |
|:---|:---|
| **Category** | Code Bug |
| **Error Type** | TypeError |
| **Affected Tests** | 31 tests |
| **Root Cause Commit** | `7b346ba8ed54` — "[Bugfix] Propagate compilation_time from workers to main process for TP>1 (#35503)" |
| **Changed File** | `vllm/v1/executor/abstract.py` |
| **Impact in vllm-ascend** | `vllm_ascend/worker/` (worker implementations) |

**Error Message:**
```
TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'
```

**Error Context:**
```python
File "/vllm-workspace/vllm/vllm/v1/executor/abstract.py", line 124, in initialize_from_config
    self.vllm_config.compilation_config.compilation_time = max(
                                                           ^^^^
```

**Explanation:**
Upstream commit 7b346ba8ed54 added logic to aggregate `compilation_time` from workers using `max()`. However, if workers don't properly initialize or return `compilation_time`, the `max()` call receives `None` values and fails. This is likely because vllm-ascend's worker implementations don't set `compilation_time` on the worker base class.

**Fix Suggestion:**
Ensure vllm-ascend workers properly initialize and propagate `compilation_time`. Check `vllm_ascend/worker/worker_v1.py` and related worker classes to ensure they set `self.compilation_time` or return it in the appropriate method that the executor calls.

---

### Issue 3: AscendMMEncoderAttention Missing 'sequence_lengths' Parameter

| Item | Detail |
|:---|:---|
| **Category** | Code Bug |
| **Error Type** | TypeError |
| **Affected Tests** | 46 tests |
| **Root Cause Commit** | `9c3fe9936b92` — "Flashinfer cuDNN backend for Qwen3 VL ViT attention (#34580)" |
| **Changed File** | `vllm/model_executor/layers/attention/mm_encoder_attention.py` |
| **Impact in vllm-ascend** | `vllm_ascend/ops/mm_encoder_attention.py` |

**Error Message:**
```
TypeError: AscendMMEncoderAttention.forward_oot() got an unexpected keyword argument 'sequence_lengths'
```

**Explanation:**
Upstream vLLM added a new `sequence_lengths` parameter to all platform-specific forward methods (`forward_cuda`, `forward_cpu`, `forward_xpu`, `forward_native`) in `mm_encoder_attention.py` to support the FlashInfer cuDNN backend. This parameter is now passed to all platform dispatches, but vllm-ascend's `forward_oot()` method doesn't accept it.

**Fix Suggestion:**
Add `sequence_lengths=None` parameter to the `forward_oot()` method signature in `vllm_ascend/ops/mm_encoder_attention.py`:

```python
def forward_oot(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,  # Add this parameter
) -> torch.Tensor:
```

---

### Issue 4: Missing '_C.rms_norm' Custom Op

| Item | Detail |
|:---|:---|
| **Category** | Code Bug |
| **Error Type** | AttributeError |
| **Affected Tests** | 24 tests |
| **Root Cause Commit** | `66c1751d13b7` — "[compile] Cleanup: Remove unnecessary +rms_norm forcing for sequence parallelism (#35410)" |
| **Changed File** | `vllm/config/vllm.py` |
| **Impact in vllm-ascend** | `vllm_ascend/ascend_config.py` or config overrides |

**Error Message:**
```
AttributeError: '_OpNamespace' '_C' object has no attribute 'rms_norm'
```

**Explanation:**
Upstream commit 66c1751d13b7 changed the logic for when `+rms_norm` is forced in the custom ops list. The old logic was:
- If TP=1: disable SP
- If `-rms_norm` in custom_ops: warn but continue
- Otherwise: append `+rms_norm`

The new logic is more complex and checks for pipeline parallelism and graph partitioning. This change means that in some configurations, `+rms_norm` is no longer automatically added, causing code to reference `torch.ops._C.rms_norm` which doesn't exist on Ascend.

**Fix Suggestion:**
In vllm-ascend's config override (likely in `vllm_ascend/ascend_config.py` or wherever VllmConfig is patched), explicitly force `+rms_norm` to be added to `custom_ops` for configurations that need it. This ensures the Ascend custom op is registered and used instead of the CUDA one.

```python
# In vllm-ascend config override
if "+rms_norm" not in self.compilation_config.custom_ops:
    self.compilation_config.custom_ops.append("+rms_norm")
```

---

## Summary Table

| # | Error | Category | Upstream Commit | Affected Tests | Fix |
|:--|:---|:---|:---|:---|:---|
| 1 | CudagraphDispatcher 'disable_full' parameter | Code Bug | 1d532f9d8fb2 | 53 | Remove disable_full from dispatch() calls |
| 2 | NoneType comparison in compilation_time | Code Bug | 7b346ba8ed54 | 31 | Initialize compilation_time in workers |
| 3 | MMEncoderAttention 'sequence_lengths' parameter | Code Bug | 9c3fe9936b92 | 46 | Add sequence_lengths=None to forward_oot() |
| 4 | Missing '_C.rms_norm' custom op | Code Bug | 66c1751d13b7 | 24 | Force +rms_norm in config override |

---

## Recommended Actions

1. **Fix Issue 1:** Remove `disable_full` parameter from `cudagraph_dispatcher.dispatch()` calls in `vllm_ascend/worker/model_runner_v1.py`
2. **Fix Issue 2:** Ensure vllm-ascend workers properly initialize and return `compilation_time`
3. **Fix Issue 3:** Add `sequence_lengths=None` parameter to `AscendMMEncoderAttention.forward_oot()` in `vllm_ascend/ops/mm_encoder_attention.py`
4. **Fix Issue 4:** Force `+rms_norm` in custom_ops list in vllm-ascend's config override
5. **Test:** Run the full test suite to verify all fixes work correctly
6. **Submit PR:** Create a pull request with all fixes and reference this analysis report

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
