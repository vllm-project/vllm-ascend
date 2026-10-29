# Exact MoE expert offload: MRV2 eager and decode ACL graph

Model Runner V2 (MRV2) owns expert loading, execution and cleanup. The configuration
field remains `use_v2_model_runner`.

Status: single-worker implementation snapshot. Prepared native H2D and checkpoint-native DSpark numerical and performance acceptance are recorded below. Decode ACL graph adds replayed expert loading and capacity filtering. Both scheduler modes are accepted; current asynchronous scheduling evidence is bounded as described below.

## Pinned implementation baseline

| Input | Commit |
| --- | --- |
| vLLM Ascend main, locked on October 12, 2026 | `1c6ec402ec584f1eac0173b6318539128d74cf7c` |
| Matching vLLM, from `.github/vllm-main-verified.commit` | `8cc9aa5ad350f6f80cf3bebb298d410803ff75a9` |
| Committed feature reference | `6f0fc035c63fead688a262d1dee1dd6ec8dc03e6` |

The implementation was written from the committed feature behavior and the pinned main interfaces. It does not reuse the previous workspace's uncommitted adaptation.

## Configuration and restrictions

Expert offload is disabled by default. Enable target expert offload using the existing Ascend additional configuration:

```json
{
  "expert_offload_config": {
    "enabled": true,
    "num_device_experts": 32
  }
}
```

The slot count must satisfy `top_k <= num_device_experts < num_experts` for every target routed expert layer. Unknown keys, string booleans, non-integral slot counts, and invalid capacities fail early.

Model compatibility is checked per routed-expert layer, not by a target model-name
allowlist. Ascend 950 (A5) hardware is required. The adapters accept W4A8 MXFP
(group size 32, gated SiLU/SITU), W8A8 MXFP8 (group size 32, gated SiLU),
W8A8 INT8 (channel scales/offsets, gated SiLU), and native block FP8 checkpoints
that the existing scheme executes as MXFP8 on A5. Each layer must expose complete
gate/up/down tensors, an expert/shard checkpoint loader and native post-processing.
All target routed layers must be
offloaded, and their layouts must be compatible with the single shared pool.
Raw tensor shapes and dtypes are checked before allocating owned host storage;
native physical NZ storage is checked during preparation and transfer. Shared and
dense model weights follow their existing paths.

Full-model acceptance recorded below covers the locally stored
DeepSeek-V4-Flash-0731 checkpoint. GLM-5.3-Flash (`Glm5Next`) and
Qwen3.8-Flash-Next (`Qwen4Exp`) now have expert-format adapters for their native
MoE paths. GLM's checkpoint-native MTP composition is described below. This
feature requires Model Runner V2. Full-model accuracy and memory acceptance are
checkpoint-specific; passing format and configuration checks does not establish
them. Start validation without `speculative_config`, then validate the combined
configuration against the same checkpoint with speculative decoding disabled.
Passing the capability checks for another model
does not establish full-model hardware or accuracy acceptance for that model.

Required settings are MRV2 and single-worker execution with TP/DP/PP/PCP/DCP sizes equal to one. Use `enforce_eager=True` for eager, or compilation mode `NONE` with `FULL_DECODE_ONLY` for ACL graph. Compiled/piecewise or mixed graphs, expert parallelism, EPLB, dual batch overlap, LoRA, sleep mode, online weight reload, generic CPU weight offload, and PD disaggregation are unsupported. Biased experts and sequence-parallel expert execution are also rejected. Use normal `auto`, `safetensors`, or `pt` checkpoint loading; dummy and preprocessed weight-cache loaders are unsupported.

### Why configurations are restricted

The following checks describe the current implementation. They are not claims
that a combination is fundamentally impossible. A capability failure is an
implementation constraint; a missing hardware result is a validation boundary.

| Area | Current requirement and reason | Extension work |
| --- | --- | --- |
| Expert format | W4A8 MXFP, W8A8 MXFP8, W8A8 INT8 or block FP8 converted through native MXFP8; complete tensors and supported gated activation | Validate each checkpoint and native physical-copy/GEMM path on A5 |
| Worker ownership | TP/DP/PP/PCP/DCP equal one; slots and maps are worker-local | Define distributed ownership, communication and failure recovery |
| EP/EPLB and mixed placement | Other expert placement/remapping must not invalidate the residency map | Coordinate placement changes with resident slots and host bytes |
| DBO, shared-expert overlap and alternate fused dispatch | One pool and one execution scope are reused; alternate dispatch integration is absent | Add independently owned execution buffers and overlap/dispatch acceptance |
| Sleep and online/RL reload | Captured addresses and prepared bytes have no sleep/wake or rebuild protocol | Drain, rebuild and recapture resources before reuse |
| Generic CPU offload and PD | Parameter ownership and split-worker lifecycle are not integrated | Define one ownership and cleanup protocol |
| LoRA, expert bias and sequence parallel | Their slot/weight-view integration is not implemented | Add composition, numerical and lifecycle coverage before enabling |
| Checkpoint loader | `auto`, `safetensors` or `pt` must report every expert/shard | Adapt completeness tracking for other loaders |
| Shared full-expert pool | All adopted target layers require homogeneous specifications | Introduce bounded pools for distinct layout families |
| Optional speculative decoding | Checkpoint-native DeepSeek V4 DSpark or GLM-5.3-Flash MTP; eager draft and fixed verification | Validate other draft architectures and draft graphs separately |
| Other target model names | Accepted only when every layer satisfies the adapter contract | Run matching checkpoint, eager/graph and offload-disabled acceptance |

DSpark is optional: expert offload can run without `speculative_config`, including
with multiple requests. In that case, target-model compatibility is determined by
the routed-expert capability checks above. The draft architecture restrictions only
apply when speculative decoding is configured; they are not a target-model allowlist.

DSpark KV-cache input preparation follows the main implementation unchanged.
Expert offload only manages target expert bindings and runner-owned cleanup.
Draft experts remain fully resident on the NPU and use the existing native
loading and execution paths, without offload bindings or offload-owned pinned storage.
The runner scopes offload to the target loading tag. Native draft loading tags
exclude drafts even when DSpark retains the target ModelConfig; multimodal
language submodels remain eligible when they clone ModelConfig under the target tag.

### Manager lifecycle

`REGISTERING` collects loaded layer bindings. Successful `finalize_loading` adopts
target layers and initializes the shared pool, then enters `READY`.
The separate `executing` flag prevents overlapping execution in `READY`.
Loading, transfer or forward failure enters `FAILED`; execution cannot resume on
that manager. `close()` drains owned work, releases bindings, staging and the pool,
and enters `CLOSED`. Cleanup is repeatable; recovery constructs a new manager.

### Asynchronous scheduling

The configuration accepts both synchronous and asynchronous scheduling. Offload execution scopes, native H2D completion and target graph replay retain their existing serialization and fences; enabling asynchronous scheduling does not make expert transfers nonblocking.

Historical validation of `fe34a97da` on the local 0731 checkpoint includes target-only execution with the original prepared pinned storage. Target-plus-DSpark validation used 128 slots in each model, FP8 KV, a 3 GiB KV budget, a six-token target graph and eager fixed drafting. That run required a separate bounded pinned-staging workaround after full host pinned allocation failed. It passed five greedy requests, including 65536 input and 1024 output tokens, with matching synchronous/asynchronous output hashes.

An October 23 follow-up on Ascend 950PR used the same PR baseline with separate bounded pinned staging and allocator prewarm. Nine shared requests matched between synchronous and asynchronous DSpark execution, including actual prefix-cache hits and partial/full draft rejection. Nine shared requests also matched with DSA attention overlap enabled. With DSpark disabled, four concurrent requests matched their serial references; mixed batches, graph-capacity eager fallback, cancellation recovery and a pure-eager reference were covered. These DSpark runs used one request at a time; this is a validation boundary, not a `max_num_seqs` restriction.

These resource workarounds are not included in this guard-removal change. The original direct-pinned DSpark asynchronous path and acceptance after migration to current main remain pending. The diagnostic runs do not establish an asynchronous throughput improvement.

### Checkpoint-native DSpark

DeepSeek V4 uses the checkpoint's own DSpark draft, with eager drafting and fixed verification. Expert offload does not impose a single-request limit. Decode graph eligibility is checked against the padded token count and resident expert capacity; batches beyond that bound use the existing eager fallback. GLM MTP is described below. External drafts, other speculative architectures, draft graphs and adaptive verification remain rejected. For the locally tested 0731 checkpoint, the draft block size is five:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}'
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6]}'
```

The existing `is_draft_layer` role is forwarded through the upstream routed-expert
constructor arguments. Draft layers skip offload configuration before weight
creation: all their experts load on the NPU and use native quantization
post-processing. They create neither `HostExpertStore` nor raw/prepared pinned
expert buffers. The runner adopts only target bindings after loading; DSpark
requires no offload manager or extra execution scope. The resident-slot setting
therefore applies only to target layers. Keeping all draft experts resident
increases NPU memory demand compared with the historical draft-offload policy.
The current policy still requires full-checkpoint acceptance under the selected
NPU memory budget; historical results below used the previous policy.

Uniform verification descriptors can contain several tokens per request. Capture and dispatch include every verification and padding row in the existing capacity bound. Five draft tokens plus the new target token require at least 36 slots at top-k six for a six-token target graph. Adding the draft also increases model and KV memory; a slot count that fits ordinary decode is not guaranteed to fit DSpark. This integration does not enable LRC, expert prefetch or hot-expert preload.

### Checkpoint-native GLM-5.3-Flash MTP

GLM-5.3-Flash can combine target expert offload with its checkpoint's own
`Glm5NextMTPModel` head. The draft stays fully resident on the NPU. Both
synchronous and asynchronous scheduling are accepted, including multiple requests.
Use eager drafting and fixed verification:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16]}'
--additional-config '{"expert_offload_config":{"enabled":true,"num_device_experts":68}}'
```

The GLM deployment guide's `deepseek_mtp` method is a legacy alias that vLLM
normalizes to `mtp`; it still loads GLM's own head. The target must use the
`glm5_next_text` configuration and the draft must come from the same checkpoint.
`enforce_eager` inside `speculative_config` applies to the draft, while the target
can use decode graphs.

For three draft tokens, verification contains four tokens per request. At top-k
eight, 68 resident experts admit four-token and eight-token target graphs. A
four-request verification batch has 16 tokens and a worst-case 128 expert routes,
so it uses the existing eager capacity fallback. Graph filtering does not reduce
`max_num_seqs`. Allow additional NPU memory for the full-resident draft and its KV
cache; the target resident count alone does not bound total model memory.

Full-checkpoint MRV2 runs have exercised expert transfers, target graph replay,
concurrent eager fallback and native MTP acceptance. Exact greedy equivalence
with MTP-disabled decoding has not passed for every mixed batch; numerical
acceptance remains pending.

## Serving example

Expose one idle NPU and install the pinned versions before running:

```bash
bash examples/run_moe_expert_offload.sh /models/DeepSeek-V4-Flash-0731
```

The script enables expert offload with 16 resident slots, a fixed 2 GiB KV budget, BF16 KV, and graph candidates `[1, 2, 4]`. Pass extra `vllm serve` arguments to override these example defaults. If BF16 KV is incompatible with the selected environment, append `--kv-cache-dtype fp8` and use the same KV dtype in the numerical reference. The pinned DeepSeek V4 path uses `--block-size 32`. Remove graph compilation settings and add `--enforce-eager` for eager acceptance.

To reproduce the accepted single-card 84-slot DSpark configuration, use the same pinned source versions and CANN 9.2 environment, expose one idle NPU, and run:

```bash
bash examples/run_moe_expert_offload.sh /models/DeepSeek-V4-Flash-0731 \
    --kv-cache-dtype fp8 --max-model-len 262144 \
    --max-num-batched-tokens 2048 --max-num-seqs 1 \
    --kv-cache-memory-bytes 2415919104 --gpu-memory-utilization 0.97 \
    --enable-prefix-caching \
    --additional-config '{"expert_offload_config":{"enabled":true,"num_device_experts":84}}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6]}'
```

This configuration uses FP8 KV, a 2.25 GiB KV budget and a six-token target verification graph. It keeps eager drafting, serialized execution and the default exact offload policy, without LRC or expert prefetch. The benchmark uses 65536 input tokens, 1024 output tokens, five formal requests, concurrency one and an actual 90% prefix-cache hit rate; these are benchmark-client settings rather than server arguments. The checkpoint path must resolve to local disk.

## Loading and execution

The routed-expert owner configures its quantization wrapper before weight creation. Enabled layers allocate raw expert Parameters on CPU and retain the original checkpoint loader. A wrapper records accepted expert/shard loads and checks that every weight and scale is present. Registered CPU Parameters own the checkpoint copies; the finalized pinned host store does not retain mmap-backed checkpoint views.

For enabled layers, the quantization wrapper disables upstream automatic device loading. It reuses the original W4A8 MXFP post-processing on a bounded set of resident experts. During loading, it also prepares every logical expert's native weight and scale storage on CPU, converting one expert at a time on NPU and retaining its physical bytes in owned pinned buffers. The complete raw CPU layer is then released, retaining one raw prototype per component for initial pool allocation. It preserves Parameter metadata while reducing the resident expert dimension. The V2 runner adopts these layer bindings after parent model loading, checks the shared pool layout, and allocates one full-expert prefill pool. The pool's persistent device allocation is included in `model_memory_usage` before KV-cache budgeting.

During decode, the planner protects selected resident experts and deterministically assigns misses to free or unselected slots. All four prepared tensor components are transferred together, without runtime post-processing or temporary device bundles. The map is published after transfer completion. A batched host route read replaces scalar device synchronization loops.

On NPU, transfers copy complete physical expert storage blocks directly from pinned CPU buffers into slots with asynchronous ACL host-to-device memcpy. Ordinary tensor slice copies cannot handle the NZ internal weight format on 950DT. The entire batch is validated before any submission, checking device, dtype, shape, strides, format, storage offsets, pinned ownership and capacities. Copies include NZ padding and preserve packed scale layouts. The caller's stream orders copies after previous consumers and is synchronized once per nonempty batch before map publication. Partially submitted batches are drained on failure. CPU tests retain ordinary tensor copies.

Prefill, mixed batches, profiling, and decode batches with more unique experts than slots use the reusable full-expert pool. Phase one requires homogeneous pool specifications, including activation and expert tensor shapes/dtypes. The full pool can consume one complete layer plus format-conversion transients; this implementation does not guarantee that an arbitrary model fits simply because its slot count is reduced.

The kernels receive an explicit weight view and a private all-gather dispatcher with the matching physical expert count. Both GEMMs see the same view. Original routing and prepare/finalize behavior are retained, and temporary communication context is restored in `finally`. Prefill does not rebind resident Parameters or change their addresses.

The manager is owned by the runner. MRV2 batch metadata determines prefill versus decode. Executions are serialized; a failed copy or forward invalidates the manager so stale or partially replaced experts cannot be used by another request. Worker shutdown drains and closes offload resources while preserving parent shutdown behavior.

## Decode ACL graph

Capture and dispatch both filter the actual padded descriptors against `num_tokens * top_k <= num_device_experts` in every offloaded layer. This bound includes padding routes; it does not depend on observed expert overlap. With 16 slots and top-k 6, only 1/2-token candidates are captured. A 3-token batch padded to a 4-token candidate selects eager before replay. Prefill and mixed batches remain eager and use the full-expert pool when required.

Each layer owns pinned route and mapping staging for each captured shape, with stable resident weights and device mapping addresses. The graph records `D2H routes -> host callback -> H2D mapping -> both GEMMs`. Every replay computes a new transactional plan. The callback explicitly enters inference mode, selects its device, directly copies the prepared bundle on a separate load stream, and publishes the mapping after successful completion. It never synchronizes the executing graph stream.

The graph manager subscribes the actual NPU capture stream, independent of upstream CUDA stream caching. The runner fences replay after attention updates are submitted and checks callback errors before returning hidden states. A transfer failure invalidates the manager and rejects that output. Shutdown drains work, unsubscribes report streams, and releases staging. Offload execution remains serialized even when asynchronous scheduling is enabled.

## Tests and acceptance

The new CPU tests cover strict configuration, randomized residency conservation, loader completeness and ownership, bundle/view consistency, pool reuse, stable resident addresses, failure invalidation, and lifecycle composition. Hook tests execute isolated bodies from the actual runner, quantization wrapper, and routed-expert source; they do not substitute for importing and exercising the full matching vLLM runner.

In an installed Ascend/vLLM development environment:

```bash
pytest -q tests/ut/expert_offload
pytest -q tests/ut/worker/test_model_runner_v2.py
pytest -q tests/ut/quantization/test_method_adapters.py
pytest -q tests/e2e/pull_request/one_card/test_expert_offload_a5.py
bash format.sh ci
```

The Windows development workspace has CPU Torch and lacks installed vLLM and Torch NPU. Its isolated test launcher bypasses the plugin package initializer and parent UT bootstrap to run dependency-light tests. Full runner UTs and the repository-wide CI command require the matching development environment.

Configuration, layout validation, residency, prefill-pool reuse and lifecycle
failure boundaries belong to the CPU UTs. The single A5 hardware file keeps two
regular cases: real NZ expert replacement during graph replay, and real
checkpoint/runner integration. The graph case compares both GEMM outputs with
fully resident experts across repeated evictions, checks stable addresses and
rejects injected callback failures. Full-pool prefill is exercised through the
runner case rather than a separate hardware test.

The MRV2 integration test writes a deterministic small Qwen3 MoE HF checkpoint
with packed expert weights, then uses normal safetensors loading and the actual
worker. It checks completed loader adoption, filtered padded graph sizes, prefill,
graph decode, oversized decode fallback, stable slot addresses, cleanup and token/
log-probability agreement with an offload-disabled eager reference. This tests a
small generated checkpoint, not production Qwen3 MoE accuracy. A test-only worker
extension observes actual dispatch decisions without replacing the loader or
forcing manager state. Its worker extension is defined in the same test file.
The file is registered in the curated A5 suite and explicitly routed to A5;
the `_a5` filename alone does not select hardware.

CPU UTs cover target-only offload bindings, preservation of the native DSpark
draft path, and loading/cleanup boundaries. Full DeepSeek checkpoint DSpark
validation is maintained as separate hardware acceptance evidence. The two A5
automated cases above do not establish full-model DSpark acceptance; record the
checkpoint, source commit, configuration and results for that validation
separately from CI and the historical snapshots below.

### October 12, 2026 acceptance snapshot

These results describe the preceding synchronous device-copy implementation. Prepared native H2D acceptance is recorded in the October 13 snapshot. The current A5 regressions reject runtime format conversion and exercise both GEMMs and graph replay.

The snapshot uses one 950DT NPU, the local DeepSeek-V4-Flash-0731 checkpoint, 16 expert slots, BF16 KV, a fixed 2 GiB KV budget, and the pinned source commits above. The CI A5 nightly provides CANN 9.2.0 (`V100R001C25B114`), Torch 2.10.0 and Torch NPU 2.10.0.post4. The installed custom operators were rebuilt against CANN 9.2.

| Check | Result |
| --- | --- |
| Expert offload, V2 runner and quantization adapter UTs | 204 passed |
| Real NPU NZ copy, pool refill and graph replay/failure tests | 3 passed |
| Captured descriptors, requested `[1, 2, 4]` | Only 1/2-token uniform decode graphs |
| Full-model request coverage | 3 serial short/long/short, 2 mixed, 4 concurrent; 33 execution steps, 12 graph replays |
| Numerical reference | 13 logits tensors of shape `[1, 129280]`, finite, bitwise equal, maximum absolute error 0; matching generated text |
| Loading allocated peak | 23.342 GiB |
| Allocated after warmup, including fixed KV budget | 23.326 GiB |
| Runtime allocated peak | 23.513 GiB |
| Phase transitions | Prefill to decode graph and decode graph to prefill passed |
| Capacity overflow | Four concurrent requests completed through eager fallback |

The numerical reference stages all 256 experts of each layer through the reusable full-expert pool and uses the same quantization/post-processing. A whole-model fully resident copy does not fit on one card. This reference validates exact expert selection; its memory is not a fully resident baseline. The numerical tolerance was fixed at `rtol=atol=1e-3` before the test; the observed difference is zero. Results apply to this checkpoint/environment and BF16 KV; FP8 KV and asynchronous scheduling are not claimed as validated. No throughput improvement is claimed.

For another NPU/checkpoint pair or a community submission, also run:

- Compare expert outputs and logits with the same fully resident quantized model, using tolerances established before testing.
- Exercise prefill, chunked prefill, mixed batches, alternating decode routes, capacity overflow, and prefill-to-decode transitions.
- Verify actual NPU NZ formats and scale strides when transferring individual experts into resident and full-pool tensors.
- Measure loading peak, persistent allocated/reserved memory, prefill conversion peak, and resulting KV-cache capacity.
- Report TTFT, inter-token latency, throughput, route synchronization, and H2D/format-conversion cost.
- Run offload-disabled regressions and the full matching runner/quantization UTs.

### October 13, 2026 native H2D acceptance snapshot

The same 950DT/CANN 9.2 environment and local 0731 checkpoint were tested with 96 resident experts per layer, V2, FP8 KV, a fixed 2 GiB KV budget, 262144 maximum context, 2048-token prefill chunks and a one-token decode graph. LRC, expert prefetch and speculative decoding were disabled in this snapshot.

| Check | Result |
| --- | --- |
| Expert offload UTs | 126 passed before the DSpark extension |
| Real NPU NZ copy, both GEMMs and graph tests | 3 passed |
| Numerical comparison with preceding eager transfer path | 13 finite logits tensors, bitwise equal, maximum absolute error 0; matching output |
| Phase transitions | Short/long/short requests, 17 execution steps, 9 graph replays; both prefill/decode transitions passed |
| Loading allocated peak | 66.180 GiB |
| Allocated after warmup | 66.166 GiB |
| Benchmark runtime allocated peak | 66.799 GiB |
| Benchmark reserved peak | 67.262 GiB |
| Sampled physical HBM use, including platform overhead | 85912 / 86016 MiB |
| AIS Bench formal requests | 5 successful, each 65536 input and 1024 output tokens, concurrency 1 |
| Actual prefix-cache hit rate | 90%; 294912 hits / 327680 queried tokens |
| Mean TPOT, preceding path to prepared H2D | 355.405 ms to 74.272 ms, 4.79 times faster |
| Mean TTFT | 90.386 s to 12.339 s |

The old and new benchmark request inputs were verified identical by SHA256. All five new TPOT values were 71.800-76.092 ms, so the below-50-ms target was not reached. This is a comparison with the immediately preceding PR01 implementation, not a verified comparison with the historical September 28 DSpark run. The full-model numerical reference for this snapshot uses the preceding eager implementation with the same 96 slots. The physical HBM peak is sampled every 15 seconds, while allocator peaks are worker counters. These performance results do not include DSpark.

### October 13, 2026 DSpark numerical acceptance

The same local checkpoint and environment were tested with 48 slots in both target and draft expert layers, FP8 KV, maximum context 262144 and a fixed 2.25 GiB KV budget. DSpark uses five speculative tokens with eager drafting; target verification uses a six-token decode graph. LRC, expert prefetch and adaptive verification are disabled.

The eager reference forces both target and draft MoE through the reusable complete expert pool. Three serial short/long/short requests, including a 3009-token prompt split into 2048-token chunks, each generated 32 tokens. The offload graph path reproduced all generated tokens and all 34 finite logits tensors bitwise, with maximum absolute error zero (`rtol=atol=1e-3` fixed before comparison). The six-token target graph replayed 30 times. All three generated prefixes also matched the first four greedy tokens saved from the ordinary non-speculative run. This is bounded numerical evidence for this checkpoint, not a full accuracy evaluation or a 96-slot performance result.

The related checks passed 141 expert-offload UTs, 94 runner/quantization UTs and 137 DSpark/speculator UTs. Source formatting and whitespace checks passed.

### October 13, 2026 DSpark performance acceptance

The 96-slot DSpark configuration failed during startup before KV allocation: an additional 768 MiB allocation encountered 67.21 GiB active memory and only 50.18 MiB free. The approved performance configuration uses 84 resident experts in each of the 43 target and three draft MoE layers, five speculative tokens, eager drafting and a six-token target verification graph. Maximum context remains 262144 with FP8 KV and a fixed 2.25 GiB KV budget; LRC, expert prefetch, adaptive verification and asynchronous scheduling remain disabled. The benchmark uses the local disk checkpoint.

| Check | Result |
| --- | --- |
| AIS Bench formal requests | 5 successful, each 65536 input and 1024 output tokens, concurrency 1 |
| Actual prefix-cache hit rate | 90%; 294912 hits / 327680 queried tokens |
| Mean TPOT | 40.548 ms |
| Per-request TPOT | 45.921 / 31.839 / 45.691 / 32.487 / 46.801 ms; all below 50 ms |
| P90 TPOT | 46.449 ms |
| Mean TTFT / end-to-end latency | 15.886 s / 57.366 s |
| Loading allocated peak | 63.569 GiB |
| Allocated after warmup | 63.826 GiB |
| Benchmark allocated / reserved peak | 64.489 / 65.092 GiB |
| Sampled physical HBM peak | 83715 / 86016 MiB, including platform overhead |
| Post-benchmark 84-slot output check | Short/3009-token chunked long/short, 32 tokens each; all match the 48-slot eager full-pool reference |

TPOT was recalculated from the raw AIS timing arrays using `(last_response_time - first_response_time) / (output_tokens - 1)`. DSpark streams multiple accepted tokens per response, so the number of response timestamps is not the output token count. The five prompt SHA256 values match the preceding non-speculative benchmark. That benchmark used 96 slots and a 2 GiB KV budget, so the difference is not a controlled measurement of DSpark alone. The historical September 28 result has not been independently reproduced. The 34-tensor logits comparison used 48 slots; the 84-slot follow-up compares generated text and token strings without logits instrumentation. These checks are bounded numerical evidence rather than a full accuracy evaluation.

Prepared native H2D removes per-miss format conversion and temporary device-to-device copies. Broader asynchronous scheduling validation, LRC, prediction, MemFabric, EP, and prefill pipelining remain separate contributions.
