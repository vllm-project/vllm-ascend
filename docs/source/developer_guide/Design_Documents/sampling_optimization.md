# Sampling Optimization Detailed Design

## 1. Overview

This document describes the implementation plan for sampling optimization in
the v1 Ascend model runner. The goal is to move the sampling path to the
upstream state-driven sampler architecture, make all logits-row mappings
explicit, and then layer distributed and fused execution on top.

The implementation is split into a staged rollout. Phase 1 establishes the
sampler bridge in small reviewable PRs, while later phases add distributed and
fused execution:

1. **Phase 1a: sampling foundation**: add configuration parsing, the
   `V1SamplingContext` contract, and model-runner-side context construction.
   This PR must not route sampling through the new path.
2. **Phase 1b: opt-in normal decode bridge**: route normal decode through the
   new adapter only when sampling optimization is explicitly enabled. Logprobs
   and speculative decoding remain on the existing path.
3. **Phase 1c: default logits processing**: add full sampling-parameter
   processing for the bridge, including mixed request parameters and skip mode.
4. **Phase 1d: logprobs**: add raw/processed logprob support for the bridge.
5. **Phase 1e: speculative decoding**: add expanded logits, rejection sampling,
   and spec-decode output shaping on top of the same mapping contract.
6. **Phase 2: batch-parallel sampling**: partition sampling work across
   data-parallel ranks and gather compact sampler outputs.
7. **Phase 3: fused logits processing**: add fused kernels for the default
   logits processing stages.

This staging keeps the first code PR small and low risk. Speculative decoding,
logprobs, and normal decode still share the same mapping contract, but each
runtime behavior is reviewed and validated in a separate landing.

## 2. Architecture

![Sampling optimization architecture](../../assets/sampling_optimization/architecture.svg)

**Key design decisions**:

- The sampling path is based on logits-row mapping, not request index
  coincidence.
- `SamplerOutput` uses the upstream class name directly; import paths describe
  whether a type comes from v1 outputs or the worker sampler implementation.
- The default logits processing stage delegates orchestration to upstream
  `Sampler.apply_sampling_params` where possible. Ascend-specific kernels are
  used behind the state adapters.
- `gumbel_sample` replaces exponential sampling in the adapter path. It
  samples from the same categorical distribution but does not preserve identical
  random draws.
- Speculative decoding, logprobs, and normal decode share the same
  `V1SamplingContext` contract, but they land in separate Phase 1 PRs.

## 3. Configuration

Sampling optimization has three design-level knobs in the target design. They
do not need to land in the same PR. Phase 1a only needs the top-level opt-in
switch; later PRs can add or activate the more specific knobs when their
runtime behavior lands.

```python
class SamplingConfig:
    """Configuration for sampling path optimization in v1 model runner."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        self.enable_sampling_v2: bool = config.get(
            "enable_sampling_v2", False)
        self.enable_batch_parallel: bool = config.get(
            "enable_batch_parallel", False)
        self.logits_processing_mode: str = config.get(
            "logits_processing_mode", "default")
        self._validate()

    def _validate(self):
        if self.logits_processing_mode not in ("default", "skip", "fused"):
            raise ValueError(
                "logits_processing_mode must be 'default', 'skip', or 'fused', "
                f"got '{self.logits_processing_mode}'")
```

`enable_sampling_v2` gates Phase 1 runtime behavior and defaults to `False`.
It only controls the path where `model_runner_v1` is connected to sampling
operators introduced for `model_runner_v2`. If vLLM Ascend uses
`model_runner_v2` directly in the future, this switch will have no effect on
`model_runner_v2`. Phase 1a may introduce the config object and build the
future sampler context, but it must not route any sampling call through the new
path.

`logits_processing_mode="default"` and `"skip"` belong to Phase 1c.
`enable_batch_parallel` belongs to Phase 2. `logits_processing_mode="fused"`
belongs to Phase 3 and should raise a clear error until implemented.

## 4. Phase 1: Sampling Refactor

Phase 1 replaces the sampling implementation while preserving the model runner
contract expected by bookkeeping, async scheduling, pipeline parallelism, and
structured outputs. It is intentionally split into sub-phases so each PR has a
small behavioral surface.

| Step | Scope | Explicitly out of scope |
|------|-------|-------------------------|
| 1a | config, package layout, context construction, unit tests | sampler routing, kernels, E2E behavior |
| 1b | opt-in normal decode bridge and output formatting | logprobs, spec decode, batch parallel, fused processing |
| 1c | default and skip logits processing, mixed request parameters | logprobs, speculative decoding |
| 1d | raw/processed logprobs, sampled-token-only logprobs, top-k/full-vocab forms | speculative decoding |
| 1e | expanded logits, strict rejection, spec-decode bookkeeping | batch parallel and fused processing |

Phase 1 is complete only when all of the following paths work through the new
adapter:

- normal decode
- default and skip logits processing modes
- mixed batches where different requests have different sampling parameters
- raw and processed logprobs
- `max_num_logprobs == 0` sampled-token-only logprobs
- speculative decoding / expanded logits

### 4.1 Phase 1a: Foundation PR

Phase 1a is a preparation PR. It creates the public contract for later PRs and
instantiates that contract from the v1 model runner, while keeping the existing
sampler path and outputs unchanged.

Phase 1a should include:

- `SamplingConfig` parsing with `enable_sampling_v2=False` by
  default
- `vllm_ascend/worker/v1/sample/` package initialization
- `V1SamplingContext` and mapping validation utilities
- `NPUModelRunner` construction of `self._v1_sampling_context` from logits-row
  inputs when the opt-in switch is enabled
- unit tests for config parsing and mapping validation

Phase 1a should not include:

- `GpuSamplerBridge` runtime routing
- new sampling, logprobs, rejection-sampling, or E2E behavior

### 4.2 Mapping Contract

`V1SamplingContext` is the boundary object between v1 model runner state and the
new sampler pipeline.

**File**: `vllm_ascend/worker/v1/sample/sampling_context.py`

```python
@dataclass
class V1SamplingContext:
    # [num_logits] - maps each logits row to active request index.
    expanded_idx_mapping: torch.Tensor

    # [num_logits] CPU mirror of expanded_idx_mapping for Python-side checks.
    idx_mapping_np: np.ndarray

    # [num_logits] - token position for each logits row.
    pos: torch.Tensor

    # [num_logits] - input token ID for each logits row.
    input_ids: torch.Tensor

    # [num_logits] - local row position inside its request's expanded group.
    expanded_local_pos: torch.Tensor

    # [num_reqs + 1] cumulative logits rows per request, used by logprobs.
    cu_num_logits_np: np.ndarray | None

    # True when logits rows are not identity-mapped to requests.
    expanded_logits: bool

    # Number of active requests.
    num_reqs: int

    # Active request IDs ordered by request index.
    req_ids: tuple[str, ...] | None = None
```

The core invariant is:

```text
idx_mapping_np[row] = request_index_that_owns_logits_row
```

Examples:

```text
normal decode:
  num_reqs = 3
  logits rows = [req0, req1, req2]
  idx_mapping_np = [0, 1, 2]

expanded/speculative decode:
  num_reqs = 2
  logits rows = [req0_base, req0_draft0, req1_base]
  idx_mapping_np = [0, 0, 1]
```

All per-request tensors, such as temperature, top-p, penalties, and seeds, are
indexed through `expanded_idx_mapping` when applied to logits rows. Phase 1a
validates the contract and stores it on the model runner; later PRs consume it
from the sampler adapter.

### 4.3 Building Context From v1 Logits Rows

The model runner already knows which flattened token positions produced logits.
Phase 1a starts capturing that information explicitly in `sample_tokens()`
instead of relying on `[:num_reqs]` slicing. It builds the same context shape
for normal decode and speculative decode, but does not route either path through
the new sampler yet. Phase 1b consumes the normal-decode context, and Phase 1e
consumes the expanded speculative rows.

```python
@staticmethod
def from_model_runner_inputs(
    num_reqs: int,
    positions_at_logits: torch.Tensor,
    input_ids_at_logits: torch.Tensor,
    req_indices_at_logits: torch.Tensor,
    device: torch.device,
    req_ids: tuple[str, ...] | None = None,
    expanded_local_pos: torch.Tensor | None = None,
    cu_num_logits_np: np.ndarray | None = None,
    idx_mapping_np: np.ndarray | None = None,
) -> "V1SamplingContext":
    ...
```

The caller is responsible for selecting `positions_at_logits` and
`input_ids_at_logits` at exactly the same rows as the logits tensor.

For normal decode, `req_indices_at_logits` is usually `torch.arange(num_reqs)`.
For speculative decoding, the mapping can contain multiple rows per request.
`idx_mapping_np` may be supplied by the model runner to avoid an immediate
device-to-host copy for Python-side validation. The first speculative
implementation may require expanded rows to be grouped by request; if a future
speculative path produces interleaved rows, it must provide an explicit
`expanded_local_pos` and logprobs grouping that preserves correctness.

### 4.4 Phase 1b: Opt-in Normal Decode Bridge

**File**: `vllm_ascend/worker/v1/sample/adapter.py`

Phase 1b introduces the adapter for normal decode only. The adapter owns:

- consuming `V1SamplingContext` from the model runner
- the Phase 1b supported logits processing mode
- gumbel sampling
- output formatting

The bridge is activated only when `enable_sampling_v2=True`. It must
fall back to the existing sampler or fail with a clear error for unsupported
Phase 1b features such as logprobs or speculative decoding.

```python
class V1SamplerAdapter:

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
    ) -> SamplerOutput:
        processed_logits = self._logits_processor.apply(
            logits, sampling_metadata, ctx, self._num_speculative_tokens)
        sampled = self._sample(processed_logits, sampling_metadata, ctx)
        return SamplerOutput(
            sampled_token_ids=self._format_sampled_token_ids(sampled, ctx),
            logprobs_tensors=None,
        )
```

Sampling uses Ascend gumbel sampling:

```python
sampled = gumbel_sample(
    logits=processed_logits,
    idx_mapping=ctx.expanded_idx_mapping,
    temperature=temperature,
    seed=seeds,
    pos=ctx.pos,
    apply_temperature=False,
)
```

Temperature has two meanings:

- `None` in `SamplingMetadata.temperature`: all requests are greedy. The
  adapter provides a zero temperature tensor to gumbel sampling.
- tensor value `0.0`: the corresponding request is greedy.

Seeds are cached by request ID, not by request slot. When a request moves in
the batch, its seed must move with the request.

### 4.5 Phase 1c: Logits Processing

**File**: `vllm_ascend/worker/v1/sample/logits_processor.py`

Phase 1b may start with the smallest safe processing surface for normal
decode. Phase 1c completes two logits processing modes:

- `default`: full sampling-parameter processing, using upstream sampler
  orchestration and Ascend-compatible state adapters.
- `skip`: bypass logits processing and return logits in FP32 form. If the
  input logits are already FP32, skip mode may return the original tensor
  without copying. This is useful for workloads that do not use penalties, bad
  words, logit bias, min-p/top-k/top-p, or custom logits processors.

The default pipeline should reuse upstream sampler orchestration:

```python
class LogitsProcessor:

    def _apply_default(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
        num_speculative_tokens: int,
    ) -> torch.Tensor:
        return self._apply_sampling_params_bridge.apply(
            logits, sampling_metadata, ctx, num_speculative_tokens)
```

The bridge exposes the state methods expected by upstream
`Sampler.apply_sampling_params`:

```text
logit_bias_state.apply_logit_bias
penalties_state.apply_penalties
bad_words_state.apply_bad_words
sampling_states.apply_temperature
sampling_states.apply_min_p
sampling_states.apply_top_k_top_p
```

Each state method adapts v1 `SamplingMetadata` and `V1SamplingContext` to the
Ascend implementation. This keeps the stage order aligned with upstream while
still allowing Ascend-specific kernels and mapping-aware fallbacks.

The default stage order is:

1. logit bias stage: allowed token IDs and non-argmax-invariant processors
2. penalties: repetition, frequency, presence
3. bad words
4. temperature
5. argmax-invariant processors, including min-p
6. top-k / top-p

The logits processor must always copy raw logits into a separate FP32 working
tensor before mutation:

```python
processed_logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
```

This preserves raw logits for `raw_logprobs` and avoids aliasing bugs caused by
in-place processing.

Skip mode does not need a defensive copy because it does not mutate logits. It
only converts non-FP32 inputs:

```python
return logits if logits.dtype == torch.float32 else logits.to(dtype=torch.float32)
```

Skip mode does not silently claim semantic equivalence. It should warn once per
category when active requests use parameters that would be ignored:

```text
penalties
bad_words
allowed_token_ids / logit_bias / non-argmax-invariant processors
argmax-invariant processors such as min-p
top-k / top-p
```

### 4.6 Phase 1d: Logprobs

Logprobs land after normal decode and logits processing so the PR can focus on
output semantics and raw/processed logits selection.

The adapter supports:

- `max_num_logprobs is None`: no logprobs
- `max_num_logprobs == 0`: sampled-token-only logprobs
- `max_num_logprobs > 0`: sampled token plus top-k logprobs
- `max_num_logprobs == -1`: full vocabulary logprobs when supported by the
  existing v1 output contract

`logprobs_mode` selects the logits source:

```text
raw_logprobs       -> raw model logits
processed_logprobs -> processed logits after sampling parameters
```

Phase 1d should first cover normal decode. The `cu_num_logits` form needed by
expanded speculative rows is added in Phase 1e or guarded behind tests that
exercise expanded rows.

### 4.7 Phase 1e: Speculative Decoding

Speculative decoding lands after normal decode and logprobs because it expands
the logits-row mapping and touches rejection-sampling bookkeeping.

For expanded logits:

- `num_logits` can be larger than `num_reqs`
- `sampled_token_ids` can have shape `[num_reqs, max_num_generated]`
- unused cells are filled with a sentinel value consumed by existing
  bookkeeping logic
- `cu_num_logits_np` is passed to logprobs when rows are expanded

The model runner must pass an explicit row-to-request mapping for every logits
row. If the current upstream metadata does not expose such a mapping, Phase 1e
must add the mapping at the model-runner boundary rather than inferring it from
row counts.

Request rows should be grouped for the first implementation:

```text
valid:   [0, 0, 0, 1, 1, 2, 2]
invalid: [0, 1, 0, 1, 2, 2, 0]
```

If interleaved speculative rows become necessary, the context must carry enough
information for penalties, bad words, and logprobs to compute per-request
history without relying on contiguous groups.

For expanded logits with logprobs, `LogprobsTensors` must include
`cu_num_logits` so the engine can map logprob rows back to requests.

Strict rejection sampling should be the first Phase 1e landing. Probabilistic
rejection sampling and drafter `draft_logits` plumbing may be split into a
follow-up PR if they make the review too broad.

### 4.8 Model Runner Integration

**File**: `vllm_ascend/worker/model_runner_v1.py`

The model runner integration is staged.

Phase 1a builds `self._v1_sampling_context` after `execute_model_state` is
unpacked. This keeps the future sampler data flow visible in the first PR
without changing the current sampler call:

```python
self._v1_sampling_context = self._maybe_build_v1_sampling_context(
    positions,
    spec_decode_metadata,
)
```

`_maybe_build_v1_sampling_context()` derives `positions_at_logits`,
`input_ids_at_logits`, request-index mapping, cumulative row counts for
expanded rows, and request IDs from v1 runner state. The normal-decode mapping
is identity. The speculative mapping is expanded from
`spec_decode_metadata.num_draft_tokens`, so the data contract is ready before
Phase 1e starts consuming it.

Phase 1b routes `_sample()` through the adapter only when
`enable_sampling_v2=True` and the request is in the supported normal
decode surface:

```python
return self._v1_sampler_adapter(
    logits=logits,
    sampling_metadata=sampling_metadata,
    ctx=self._v1_sampling_context,
)
```

Phase 1d extends the integration to logprobs. Phase 1e extends adapter
consumption to speculative decoding and adds spec-decode output parsing.

The adapter path must not assume `logits.shape[0] == num_reqs`. If a staged
code path cannot provide the logits-row mapping yet, it should stay on the
existing sampler path or fail explicitly rather than silently falling back to
request-order assumptions.

### 4.9 Phase 1 Test Plan

| Step | Test | File | Description |
|------|------|------|-------------|
| 1a | SamplingConfig parsing | `tests/ut/sample/test_sampling_config.py` | Defaults and modes; no routing |
| 1a | Sampling context identity | `tests/ut/sample/test_v1_sampling_context.py` | Normal decode mapping |
| 1a | Sampling context expanded | `tests/ut/sample/test_v1_sampling_context.py` | Expanded row validation |
| 1a | Runner context construction | `tests/ut/worker/test_model_runner_v1.py` | Opt-in context build; no routing |
| 1b | Adapter normal decode | `tests/ut/sample/test_v1_sampler_adapter.py` | Sampling and formatting |
| 1b | Model runner routing | `tests/ut/worker/test_model_runner_v1.py` | Disabled path and opt-in path |
| 1b | E2E normal decode | `tests/e2e/singlecard/test_sampling_optimization.py` | Greedy and stochastic smoke coverage |
| 1c | Logits processing default | `tests/ut/sample/test_logits_processor.py` | Upstream stage delegation |
| 1c | Logits processing skip | `tests/ut/sample/test_logits_processor.py` | Processing bypass and warnings |
| 1c | FP32 working copy | `tests/ut/sample/test_logits_processor.py` | Raw logits are not mutated |
| 1c | Mixed parameters | `tests/ut/sample/test_logits_processor.py` | Per-request parameter coverage |
| 1d | Logprobs | `tests/ut/sample/test_v1_sampler_adapter.py` | None, 0, top-k, full-vocab, raw and processed sources |
| 1d | E2E logprobs | `tests/e2e/singlecard/test_sampling_optimization_logprobs.py` | Raw and processed logprobs |
| 1e | Speculative mapping | `tests/ut/sample/test_v1_sampler_adapter.py` | Expanded rows and output formatting |
| 1e | Rejection sampler | `tests/ut/sample/test_rejection_sampler.py` | Strict rejection coverage |
| 1e | E2E speculative decode | `tests/e2e/singlecard/test_sampling_optimization_spec.py` | Spec decode smoke coverage |

## 5. Phase 2: Batch-Parallel Sampling

Phase 2 distributes row-wise sampling work across data-parallel ranks. It is
built on the Phase 1 mapping contract, so it works for normal and expanded
logits.

### 5.1 BatchParallelSampler

**File**: `vllm_ascend/worker/v1/sample/batch_parallel.py`

```python
class BatchParallelSampler:
    """Slice logits rows locally, run sampling, then gather compact outputs."""

    def slice(
        self,
        logits: torch.Tensor,
        ctx: V1SamplingContext,
    ) -> tuple[torch.Tensor, V1SamplingContext]:
        ...

    def maybe_gather(
        self,
        sampled: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        original_ctx: V1SamplingContext,
    ) -> tuple[torch.Tensor, LogprobsTensors | None]:
        ...
```

The local context is a sliced `V1SamplingContext`. All fields that are indexed by
logits row must be sliced together:

```text
logits
expanded_idx_mapping
idx_mapping_np
pos
input_ids
expanded_local_pos
```

### 5.2 Partitioning

Normal decode can use contiguous logits-row partitioning:

```python
start = num_logits * dp_rank // dp_world_size
end = num_logits * (dp_rank + 1) // dp_world_size
```

For expanded logits, request groups must remain intact unless the expanded
history logic explicitly supports interleaving. Therefore the initial
batch-parallel implementation partitions by request group, then converts the
request range to logits-row ranges using `cu_num_logits_np`.

```python
start_req = num_reqs * dp_rank // dp_world_size
end_req = num_reqs * (dp_rank + 1) // dp_world_size
start = cu_num_logits_np[start_req]
end = cu_num_logits_np[end_req]
```

This keeps penalties, bad words, sampled output formatting, and logprobs
grouping consistent.

### 5.3 Gather

Batch parallel gathers compact outputs, not full logits.

Gathered tensors include:

- sampled token IDs
- optional logprob token IDs
- optional logprob values
- optional logprob ranks or metadata required by the v1 output contract

For uneven partitions, each rank pads to a common row count before
`all_gather_into_tensor`, then truncates to the global row count after gather.

For expanded logits, gathered sampled rows are formatted back to
`[num_reqs, max_num_generated]` after the full result is available.

### 5.4 Interactions

**Async scheduling**: gather completes inside sampling before bookkeeping
consumes `SamplerOutput`.

**Pipeline parallelism**: PP broadcasts sampled tokens after `sample_tokens()`.
The broadcast therefore sees full-batch sampled outputs.

**Structured outputs**: grammar bitmasks are applied to logits before sampling.
Batch parallel slices already-masked logits.

**Logprobs**: Phase 2 must gather logprobs in the same row order as sampled
tokens. The gather code should treat `LogprobsTensors` as part of the sampling
output, not as a later CPU-side fixup.

### 5.5 Phase 2 Test Plan

| Test | Description |
|------|-------------|
| Row partition | Even and uneven normal decode partitions |
| Request-group partition | Expanded rows for one request stay on the same rank |
| Single-rank no-op | No gather or slicing overhead for world size 1 |
| Gather sampled tokens | Full result matches sequential sampling row order |
| Gather logprobs | Logprob tensors match sequential sampling |
| Multi-rank greedy E2E | Batch-parallel output matches default greedy output |
| Multi-rank stochastic E2E | Fixed-seed stochastic output is valid and deterministic |
| Multi-rank speculative E2E | Expanded logits work with request-group partitioning |

## 6. Phase 3: Fused Logits Processing

Phase 3 optimizes the default logits processing pipeline by reducing kernel
launches and full-vocabulary memory traffic.

### 6.1 Fused Kernel Scope

The fused implementation targets the same semantics as Phase 1 default mode.
It is not allowed to drop parameter combinations supported by default mode.

The intended split is:

```text
Kernel 1: logit bias + allowed tokens + min tokens + penalties + bad words
Kernel 2: temperature + min-p + top-k/top-p
```

The split can be adjusted if profiling shows a better boundary, but the fused
path must remain semantically equivalent to default mode.

### 6.2 FusedLogitsProcessor

**File**: `vllm_ascend/worker/v1/sample/fused_logits.py`

```python
class FusedLogitsProcessor:
    """Fused Ascend kernels for logits processing."""

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
        num_speculative_tokens: int,
    ) -> torch.Tensor:
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        logits = self._fused_preprocessing(
            logits, sampling_metadata, ctx, num_speculative_tokens)
        logits = self._fused_filtering(logits, sampling_metadata, ctx)
        return logits
```

`LogitsProcessor.apply()` selects the fused implementation when
`logits_processing_mode == "fused"`.

### 6.3 Validation Requirements

The fused path must:

- match default mode within float32 tolerance
- preserve raw logits for `raw_logprobs`
- support expanded logits and request mapping
- support mixed parameter batches
- support batch-parallel slicing
- be validated across supported Ascend hardware and CANN versions

### 6.4 Phase 3 Test Plan

| Test | Description |
|------|-------------|
| Fused vs default | Output diff within tolerance for all parameter combinations |
| Raw logits preservation | `raw_logprobs` unaffected by fused processing |
| Expanded logits | Speculative rows produce the same masks and sampled outputs |
| Batch parallel + fused | Fused mode works after row/request-group slicing |
| Mixed batch | Some requests use penalties/bad words/top-p, others do not |
| Performance benchmark | Kernel count and latency vs default mode |

## 7. File Manifest

| Phase | File | Action | Description |
|-------|------|--------|-------------|
| 1a | `vllm_ascend/ascend_config.py` | Modify | Add opt-in `SamplingConfig`; no routing behavior |
| 1a | `vllm_ascend/worker/model_runner_v1.py` | Modify | Build `V1SamplingContext` without sampler routing |
| 1a | `vllm_ascend/worker/v1/sample/__init__.py` | Create | Package init |
| 1a | `vllm_ascend/worker/v1/sample/sampling_context.py` | Create | `V1SamplingContext` and validation helpers |
| 1a | `tests/ut/sample/test_sampling_config.py` | Create | Config parsing tests |
| 1a | `tests/ut/sample/test_v1_sampling_context.py` | Create | Sampling context tests |
| 1a | `tests/ut/worker/test_model_runner_v1.py` | Modify | Model-runner context construction tests |
| 1b | `vllm_ascend/worker/v1/sample/adapter.py` | Create | Opt-in normal decode adapter |
| 1b | `vllm_ascend/worker/model_runner_v1.py` | Modify | Opt-in adapter routing for normal decode |
| 1b | `tests/ut/sample/test_v1_sampler_adapter.py` | Create | Adapter, seed, and output-formatting tests |
| 1b | `tests/ut/worker/test_model_runner_v1.py` | Modify | Disabled-path and opt-in routing tests |
| 1b | `tests/e2e/singlecard/test_sampling_optimization.py` | Create | Normal decode E2E |
| 1c | `vllm_ascend/worker/v1/sample/logits_processor.py` | Create | Default and skip logits processing |
| 1c | `tests/ut/sample/test_logits_processor.py` | Create | Default, skip, mixed-parameter tests |
| 1d | `vllm_ascend/worker/v1/sample/adapter.py` | Modify | Raw/processed logprobs and logprob output formatting |
| 1d | `tests/ut/sample/test_v1_sampler_adapter.py` | Modify | Logprobs tests |
| 1d | `tests/e2e/singlecard/test_sampling_optimization_logprobs.py` | Create | Logprobs E2E |
| 1e | `vllm_ascend/worker/v1/sample/rejection_sampler.py` | Create | Strict speculative rejection sampler |
| 1e | `vllm_ascend/worker/model_runner_v1.py` | Modify | Expanded logits-row mapping and spec-decode output parsing |
| 1e | `vllm_ascend/spec_decode/eagle_proposer.py` | Modify | Optional draft logits plumbing |
| 1e | `tests/ut/sample/test_rejection_sampler.py` | Create | Rejection sampler tests |
| 1e | `tests/e2e/singlecard/test_sampling_optimization_spec.py` | Create | Speculative decode E2E |
| 2 | `vllm_ascend/worker/v1/sample/batch_parallel.py` | Create | Batch-parallel slicing and gather |
| 2 | `tests/ut/sample/test_batch_parallel.py` | Create | Partition and gather tests |
| 2 | `tests/e2e/multicard/test_batch_parallel_sampling.py` | Create | Multi-rank sampling tests |
| 3 | `vllm_ascend/worker/v1/sample/fused_logits.py` | Create | Fused logits kernels |
| 3 | `vllm_ascend/worker/v1/sample/logits_processor.py` | Modify | Wire fused mode |
| 3 | `tests/ut/sample/test_fused_logits.py` | Create | Fused correctness tests |
| 3 | `tests/e2e/singlecard/test_fused_sampling.py` | Create | Fused E2E and benchmark coverage |

## 8. Risks and Mitigations

### 8.1 Hidden `num_logits == num_reqs` Assumptions

**Risk**: Normal decode hides incorrect request-index assumptions that break as
soon as speculative decoding expands logits rows.

**Mitigation**: Phase 1a introduces the explicit mapping contract before any
runtime routing. Phase 1b routes only normal decode. Phase 1e enables expanded
rows only after the model runner provides a row-to-request mapping for every
logits row. Any code path that cannot provide it must stay on the existing
sampler path or fail explicitly.

### 8.2 Logprobs and Speculative Decoding Coupling

**Risk**: Implementing logprobs and speculative decoding in separate PRs could
require revisiting the same mapping and output-shaping code.

**Mitigation**: Both PRs use the `V1SamplingContext` introduced in Phase 1a.
Phase 1d covers normal-decode logprobs first. Phase 1e then adds expanded
`cu_num_logits` coverage with speculative-decoding tests.

### 8.3 Raw Logits Mutation

**Risk**: In-place logits processing can corrupt `raw_logprobs`.

**Mitigation**: Default and fused modes must mutate only a separate FP32
working tensor. Skip mode must not mutate logits and may reuse the original
FP32 tensor.

### 8.4 Gumbel vs Exponential Sampling

**Risk**: Gumbel and exponential sampling generate different concrete token
draws even though they represent the same categorical distribution.

**Mitigation**: Correctness tests should compare deterministic greedy behavior
exactly and stochastic behavior statistically or with fixed adapter seeds where
appropriate.

### 8.5 Upstream State Reuse on Ascend

**Risk**: Upstream state objects may depend on CUDA-specific behavior.

**Mitigation**: Reuse upstream orchestration, not CUDA-only storage. The bridge
implements the state methods with Ascend-compatible tensors and kernels.

### 8.6 Batch-Parallel Communication Overhead

**Risk**: Gather overhead may outweigh sampling savings for small batches.

**Mitigation**: Gather only compact sampler outputs. Keep batch parallel under
its own design knob and validate latency across batch sizes and DP world sizes.

### 8.7 Fused Kernel Coverage

**Risk**: Fused mode may accidentally support fewer parameter combinations than
default mode.

**Mitigation**: Fused mode must be tested against default mode with mixed
parameter batches, expanded logits, logprobs modes, and batch-parallel slicing.
