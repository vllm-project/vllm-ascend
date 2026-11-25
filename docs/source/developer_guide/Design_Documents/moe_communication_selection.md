# MoE Communication Selection

MoE communication selection is implemented by `select_moe_comm_method` in
`vllm_ascend/ascend_forward_context.py`. The hardware profile supplies a
`MoECommPolicy`; the selector combines it with the model, EP topology, validated
Ascend configuration, and the current token count. The selected enum resolves to
an implementation in `vllm_ascend/ops/fused_moe/moe_comm_method.py`.

## Selection Order

Apply these common gates before a hardware-specific policy:

1. A dense model returns `None`, not an MoE communication implementation.
2. EP disabled, or an EP group of size one, selects `ALLGATHER` through a
   fast path that does not query token capacity or the hardware policy.
3. LoRA with EP selects `ALLTOALL`. Its adapters cannot be inserted into the
   fused dispatch-FFN-combine operation.
4. A hardware policy of `ALLGATHER` selects `ALLGATHER`.
5. Otherwise, dispatch to the registered hardware policy below.

`ALLGATHER` names the implementation, not a guarantee that every invocation
executes an all-gather collective. For example, a non-sequence-parallel path can
avoid a redundant gather.

| Hardware Policy | Default Hardware | Selection After Common Gates |
| --- | --- | --- |
| `CAPACITY_AND_EXPERT_DENSITY` | A2 | Prefer eligible CANN MegaMoe; otherwise prefer fused mode with EP size at most 32. Without a fused path, use `MC2` only when tokens fit the reserved capacity, EP size is at least 16, and experts per EP rank is at most 24. Otherwise use `ALLGATHER`. |
| `FUSED_OR_CAPACITY` | A3 | Prefer eligible CANN MegaMoe; otherwise prefer fused mode with EP size at most 32. Without a fused path, use `MC2` within capacity and `ALLTOALL` above it. |
| `CAPACITY_AND_WORLD_SIZE` | A5 | Prefer supported fused mode. An MTP draft with unsupported MoE quantization skips the fused path. Otherwise use `MC2` within capacity when EP size exceeds one; above capacity use `ALLGATHER` when EP size is at most router top-k, or `ALLTOALL` otherwise. |
| `ALLGATHER` | 310P | Use `ALLGATHER`. |

The EP group spans DP, PCP, and TP ranks, but excludes PP stages. Expert density
and router top-k comparisons must use that group, not a world size including PP.
For A2/A3, CANN MegaMoe additionally requires the corresponding hardware
capability, an available supported operator, enabled fused mode, no LoRA, and
`1 < EP size <= 64`. Configuration validation can reject or disable a fused
mode before the selector is called.

The selector has no prefill/decode argument. A mixed chunked-prefill batch is
selected by token count and capacity, without classifying the whole batch as
prefill or decode. Deployment role can affect **buffer provisioning**, which is
a separate concern from phase-dependent selection.

## Configuration And Capacity

Use the typed [additional configuration](../../user_guide/configuration/additional_config.md):

```bash
vllm serve /models/model \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --additional-config '{"enable_fused_mc2": 1}'
```

| User Setting | Meaning |
| --- | --- |
| `enable_fused_mc2=0` | Disable fused mode; the hardware policy can still select ordinary MC2. |
| `enable_fused_mc2=1` | Request dispatch-FFN-combine, subject to hardware and model eligibility. |
| `enable_fused_mc2=2` | Request CANN MegaMoe where available and eligible. |
| `enable_fused_mc2=3` | Request MegaMoe with shared-expert fusion; supported hardware, model, and operator are required. |
| `enable_prefill_mc2=true` | Provision capacity from the scheduler token budget rather than only the decode graph/request budget. |
| `mega_moe_max_tokens` | Receive/output allocation budget, distinct from the per-rank input-token limit. |

Modes 2 and 3 are normalized internally to enabled fused mode with separate
MegaMoe capability state. Read the **validated** configuration, not only the
original CLI integer. Model-level SwiGLU limits and unsupported model/operator
combinations can disable the requested path; mode 3 also has explicit
initialization errors for unsupported hardware and models. The existence of a
draft model alone is not a reason to disable fused communication. The A5 draft
**quantization layout** guard remains necessary.

`set_mc2_tokens_capacity` calculates a batch token budget `B` from scheduler,
target graph, or request/speculative-query limits. PCP profiling and the target
graph shape grid also influence this budget. For TP size `T`, the effective
reserved token capacity is:

```text
per_rank = min(ceil(B / T), operator_input_limit)
mc2_tokens_capacity = per_rank * T
```

Current per-rank input limits are 512 for ordinary MC2, 512 for
dispatch-FFN-combine, and 16384 for CANN MegaMoe. Equal numeric limits for two
operators do not make their buffers or contracts interchangeable. MegaMoe's
receive-token budget is also separate from these input limits and depends on
expert routing load. Increasing a scheduler budget or setting fused mode does
not prove that its actual dispatched load fits all buffers.

`num_tokens=None` is used during profiling and treated as within the ordinary
MC2 capacity check. Fused eligibility has priority over that check; it is not an
unconditional promise to fall back when a fused operator exceeds its own
allocation limits. Keep startup/profile budgets and operator constraints aligned.

## Migration And Verification

The historical device-identity switch has been split into small policy
functions registered by the hardware profile. Phase-independent token/capacity
selection remains explicit. A2 now also has fused-path capability checks, and
A5 excludes PP from its EP-size calculation and checks draft quantization.
These changes must not be described as deleting all draft-related checks.

The legacy `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable is no longer
supported. Replace it with `additional_config.enable_fused_mc2`. Do not restore
an environment-variable fallback or make conflicting channels compete.

Run policy, configuration, and implementation tests:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
    tests/ut/test_ascend_forward_context.py \
    tests/ut/test_ascend_config.py \
    tests/ut/ops/test_moe_comm_method.py \
    tests/ut/device/test_hardware_profile.py
```

Mocked hardware-policy tests do not establish hardware availability. For each
relevant real hardware/model/quantization/topology combination, record the
resolved configuration, actual implementation calls, initialization and mixed
prefill/decode execution, and output/accuracy results. Test both sides of the
capacity boundary, EP-off/EP-on, and supported fused modes.

For performance comparisons, keep the model weights, TP/DP/EP, quantization,
token inputs, concurrency, graph mode, cache state, warmup, and software/runtime
fixed. Compare the baseline and candidate for the **same setting**, preferably
with interleaved repetitions. Record TTFT, latency, throughput, communication
time, and peak memory. Comparing EP-off, ordinary MC2, and fused MC2 against one
another is a strategy comparison, not evidence that a refactor has no regression.
Trace/profiler instrumentation should run separately from timed measurements.
