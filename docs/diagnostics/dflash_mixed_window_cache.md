# Mixed-window DFlash cache repair

## Scope and layout

This restores the cache repair for Ascend V2 with mixed Full/SWA DFlash and
Mamba/GDN target state. The layout and capacity repair leaves target-only,
all-full DFlash, and all-sliding DFlash paths unchanged. It does not change sampling, GDN rollback, attention
masks, model weights, or the official four SWA layers / one full layer / 2048
window configuration. Unsupported layouts fail at initialization.

Ascend stores contiguous conv/SSM planes at the start of each shared raw pool
and contiguous K/V planes at its tail. Equal `page_size_padded` alone is not
enough when the Full and SWA physical block sizes differ. In the reported TP2
geometry, only 12 physical blocks are needed for old SWA block 1 K/V writes to
overlap full-attention V blocks 10/11 despite their distinct block IDs.

The repair derives the attention storage block size from the SSM plane bytes
divided by the K/V row bytes. For this TP2 geometry, SWA storage changes from
128 to 1536 tokens; the kernel still uses 128-token blocks and the attention
window remains 2048. With conv page bytes `C=102400`, SSM page bytes
`S=1572864`, and common page bytes `P=C+2*S`, the actual layout becomes
`[all conv, all K/SSM, all V]`. Distinct physical IDs then own disjoint ranges.
Startup validates actual pointers, dtype, contiguity, and shared-arena offsets.

Separately, the supplied NPU probe passed scatter writes but failed FIA reads
at kernel block IDs 65536 and 65543. For that geometry, block 65536 starts at
element `2**32`. The repair conservatively bounds both the kernel-block count
and plane-element count, without claiming which CANN internal index caused
the failure. It also limits explicit overrides to the real profiled memory
budget and reruns the planner before allocation. The address-derived limit
is 5461 physical blocks for the reported geometry; memory or a smaller override
may reduce it further. This is an operator workaround, not a CANN kernel fix.

Both context prewrites and ordinary cache writes mask the entire reserved
physical null block and out-of-range slots. A 1536-token null block includes
kernel blocks 0 through 11. No per-step debug logging or CPU synchronization
is added.

## Validation

Run in the normal vLLM-Ascend development environment:

```bash
pytest -q tests/ut/worker/test_dflash_cache_layout.py tests/ut/worker/test_dflash_cache.py tests/ut/worker/test_dflash_cache_views.py
```

The first two suites cover the byte-level alias and planner contracts; the
metadata suite substitutes dependency fixtures and is not runtime integration.
The view suite uses real Torch tensors, real vLLM specs and the production
materializer. None replaces NPU end-to-end validation.

Deploy all changed Python modules together and restart all workers. Keep the
official draft config unchanged. First retain `--num-gpu-blocks-override 4096`
and `--enforce-eager`, replaying the previously failing GPQA request order and
warmup at concurrency 1, then 32. Exercise long outputs and block reuse with
different requests; a single short curl is insufficient. After this passes,
remove the override and repeat, then test graph mode.

Expected startup markers for the reported TP2 geometry:

```text
DFlash mixed cache layout aligned: ... block_size=128 -> 1536, sliding_window=2048 ...
DFlash mixed cache guard: physical_blocks=... -> ...
DFlash mixed cache plan ready: physical_blocks=... (memory/address safe)
DFlash mixed cache views verified: ... conv_bytes=102400, plane_bytes=1572864
```

The guard warning only appears when capacity is reduced. A startup layout
exception means validation rejected the configuration; absence of an exception
does not prove generation accuracy. Record raw responses, `finish_reason`,
completion-token counts and server logs. High acceptance alone is not a pass.

If repetition remains, compare the first differing output token against the
target-only baseline at an identical integer-token prefix, including top
logprobs. Replaying a prefix that has already looped for thousands of tokens
cannot establish that the earlier transition into the loop was correct.
This patch restores the cache repair only; it does not add repetition penalties
or claim to resolve all causes of repetitive generation.

## Capacity and performance after the repair

The address guard can reduce the number of resident requests even with an
unchanged `--max-num-seqs`: that flag is an upper bound, not reserved capacity.
The usable block count is bounded by the original plan (including an explicit
override), the profiled memory budget, the FIA address bound, and the other
workers' limits. A large memory budget does not override the FIA bound.
Conversely, changing SWA storage from 128 to 1536 tokens does not by itself
prove lower capacity: the old small SWA view already advertised a padded page.
The number of cache groups, padding, and request lengths also matter.

The startup-only `DFlash mixed cache capacity:` log reports each worker's
`original_planned_blocks`, `budget_limit_blocks`, `fia_limit_blocks`, `override`,
`effective_blocks`, `pool_bytes`, `budget_bytes`, and group layouts. The original
`budget_limit_blocks` uses the original pool width; after regrouping,
`effective_budget_limit_blocks` uses the selected pool width. Use these with
Running/Waiting, cache usage, and preemption counts. The group block sizes are
storage sizes, not the 128-token attention kernel block size. Do not infer a
request concurrency limit by multiplying `effective_blocks` by 1536: requests
consume blocks from several groups with different retention rules.

Higher acceptance length is not a latency guarantee. Approximately,
`decode time/token = (target verification + draft + preparation time) / acc_len`.
End-to-end latency also includes queueing, prefill, and the number of output
tokens. In particular, repetitive generations can inflate both output length
and acceptance length. The target's full-attention layers still attend to long
contexts; adding SWA to the draft does not eliminate that cost.

The `fix_swa3` hot-path changes preserve all layout and capacity protections:

- Vectorize context mapping and graph padding in the Ascend V2 DFlash input
  kernel. Previously, even a decode step cleared the padded query-slot buffer
  one element at a time, for every draft cache group. The mixed draft uses
  multiple groups, so it repeats this preparation more often than all-full.
  The signatures, query/sample generation, padding values, and ownership of
  writes are unchanged in both supported upstream variants.
- Fuse the two bounds comparisons, AND, and selection in the mixed-cache slot
  guard into one NPU kernel. Context prewrites and ordinary writes both retain
  the complete null-block and out-of-range checks. Results are not cached
  across steps, and source mappings are not modified. CPU and nonstandard
  tensor layouts retain the original PyTorch expression.

These first-stage changes reduce preparation work; by themselves they do not
raise the resident-request limit or change grouping. The additional changes
below address capacity and metadata overhead without removing the safety cap.
The vectorized input preparation also applies to unmixed V2 DFlash; the fused
write guard remains scoped to the mixed cache repair.

### Budget-aware grouping and exact metadata reuse

The reported run on vLLM `a97dacb710` has 69 singleton groups: 16 target Full,
4 draft SWA, 1 draft Full, and 48 target Mamba. At a page size of 3,248,128
bytes, the FIA cap of 5461 blocks limits the one-column pool to
17,738,027,008 bytes, despite a 26,556,098,560-byte budget. Approximately
8.21 GiB is therefore unavailable to the scheduler.

The planner now compares wider groups with the original layout, keeping exact
cache specs, group metadata, and target/draft ownership separate. It chooses
a candidate only if it strictly improves the initialization admission-capacity
estimate within the same memory, FIA, and explicit override limits. It does
not blindly maximize group width. For the supplied budget, the metadata test
selects 35 groups in two pool columns and 4087 physical blocks per column:
26,550,198,272 bytes in total. The smaller block count does **not** indicate
less capacity: the pool is twice as wide and requests consume fewer groups.
The storage block remains 1536, kernel block 128, and SWA window 2048.

Loaded draft layer names are carried in the worker spec RPC and copied into
the local planner call before upstream merges the dictionaries. This correctly
sets `is_eagle_group` on draft groups without flagging the target Mamba groups.
The warning that all groups are treated as draft should disappear. This is a
prefix-cache lookup classification fix; the warning did not mean that Mamba
layers were being executed as draft layers. With no explicit loaded identity,
the planner retains the previous safe layout rather than guessing layer names.

Two additional preparation optimizations preserve exact lengths and masks:

- Ordinary V2 parallel-drafting attention builders share one exact device
  `seq_lens.tolist()` result within a single metadata build. They do not use
  CPU upper bounds. Every group receives its own list for FIA padding, and
  the next build always reads fresh device lengths.
- FULL-graph DFlash replay consumes the metadata already built in the same
  `propose` call when real and padded request/token counts match. Capture,
  profile, dummy, and mismatched-descriptor paths rebuild. References are
  cleared on normal return and exceptions; nothing is reused across steps.

These metadata optimizations also apply to other ordinary V2 parallel-drafting
attention / V2 DFlash configurations, respectively. Grouping and draft-role
propagation are scoped to mixed-window DFlash. No sampling, weights, attention
window, or cache-address protection is changed. Runtime speedup and output
equivalence still require NPU validation; CPU contract tests cannot establish
either.

For approximately the supplied budget, verify the startup markers below before
running another full GPQA evaluation. Actual counts depend on the new profiling
budget; the final view check must still report all 69 layers, not 35 layers.

```text
DFlash mixed cache regrouped: groups=69 -> 35, pool_width=1 -> 2, physical_blocks=4087, draft_groups=3 (same specs and FIA limit)
DFlash mixed cache plan ready: physical_blocks=4087 (memory/address safe)
DFlash mixed cache views verified: layers=69, physical_blocks=4087, ...
```

Keep the reported FULL-graph mode, concurrency 32, and no explicit block
override for the performance comparison. The automatic FIA cap remains active.
First check a previously failing long-output GPQA subset (including request
reuse and padding at concurrency 1 and 32), then the 198-case run. Disable
per-step DFlash/Mamba debug output when timing. Save total generated tokens,
not only case progress or acceptance length, and compare the same mixed model
on the old and new code before comparing with all-full attention.

## Performance validation

First run the existing cache regression suites and the new input/slot suites,
then the NPU tests listed below. CPU emulation checks exact integer semantics;
it cannot validate Triton compilation, NPU latency, or graph execution.

```bash
python tests/ut/worker/test_dflash_cache.py
python tests/ut/worker/test_dflash_cache_planning.py  # includes the grouping suites
python tests/ut/attention/test_exact_seq_lens_cache.py
python tests/ut/worker/test_dflash_prepare_inputs.py  # includes the metadata-reuse suites
```

Compare `fix_swa2` mixed against `fix_swa3` mixed first, keeping the model files,
NPU/CANN/vLLM versions, graph mode, block override, batch-token limit, request
order, sampling settings, and warmup identical. Restart between configurations
and run them sequentially on otherwise idle devices. Disable per-step debug
instrumentation for timing. Use all-full only as a third control; comparing
different attention configurations alone does not isolate the code change.

For a short fixed-workload check, this Linux command uses the local vLLM
benchmark client. Set the port to the current service port. Run once per
configuration with a distinct result filename, first at concurrency 1, then
8 and 32. Repeat with longer input/output lengths after the short test passes.

```bash
vllm bench serve --backend openai-chat --host 127.0.0.1 --port 2088 --endpoint /v1/chat/completions --model qwen3.6 --tokenizer /data/weights/Qwen3.6-27B --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 --num-prompts 32 --max-concurrency 1 --request-rate inf --seed 0 --temperature 0 --top-p 1 --ignore-eos --save-result --result-dir ./dflash-perf --result-filename fix_swa3_mixed_c1.json
```

`--ignore-eos` deliberately fixes the generated length for a performance-only
test; do not use it for GPQA quality acceptance. Random prompts also do not
represent GPQA's acceptance distribution. Separately replay the same GPQA
subset and then the full dataset with normal stopping and the original output
limit. Compare actual total completion tokens, output tokens/s, TTFT, TPOT,
end-to-end latency, Running/Waiting, preemptions, and `finish_reason`. Report
repetition or length-limit terminations separately. A higher `acc_len`, an
absence of garbage, or a lower number of completed cases alone is insufficient
to conclude that the performance or quality regression has been fixed.
